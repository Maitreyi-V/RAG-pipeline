#!/usr/bin/env python3
"""
Step 5: the passage index the app reads, built with the LOCKED Step 4 system.

Two files, no transcript text (the text stays in the unit grids):

  results/step5_index_<split>.csv     one row per passage
      unit_id, video_file, language, speaker, tradition, unit_index,
      primary_verse (BG 2.x or OFF), theme_id, theme_name,
      p_primary          the decoded emission probability of the primary state
                         ((1 - P_off) * the verse's share of P_verse); ranks the
                         passages inside a segment
      p_off              P_off from the off-verse scorer
      secondary_verses   other verses on the passage: the detector's anchors
                         plus any second verse the locked model emitted
      segment_id
      word_start, word_end, lecture_words   word offsets from n_words
      pos_start, pos_end   word offsets / lecture length. APPROXIMATE position:
                         time = pos * lecture duration (durations not in the
                         repo yet; nothing here assumes a speaking rate)

  results/step5_segments_<split>.csv  one row per contiguous run of one primary
      segment_id, video_file, primary_verse, theme_id, first_unit, last_unit,
      dwell (passages in the run = the coverage-depth score),
      effective_dwell (sum of p_primary over the run), pos_start, pos_end

Speaker names exist only for the new speakers (data/lectures_new.csv); the 39
original lectures get "original Kannada speaker" / "original English speaker".

Only the tuning (dev) split is run before the 29 Oct freeze.

Usage:
    python scripts/step5_index.py                 # tuning lectures
    python scripts/step5_index.py --split test    # ONLY at the freeze
"""
import argparse
import csv
import os
import sys
from collections import Counter

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

from evalkit import (LECTURES_NEW, UNITS, UNITS_HELDOUT, load_csv, load_gold,  # noqa: E402
                     read_split, score)
from catalogue import THEMES  # noqa: E402
from step4_viterbi import OFF, near_f1, predict_locked, theme_f1  # noqa: E402

ORIGINAL_SPEAKER = {"kn": "original Kannada speaker", "en": "original English speaker"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["dev", "test", "heldout"], default="dev")
    ap.add_argument("--alpha", choices=["a05", "a0"], default="a05")
    args = ap.parse_args()

    grid = {u["unit_id"]: u for u in load_csv(UNITS) + load_csv(UNITS_HELDOUT)}
    speaker = {r["video_file"]: r["speaker"] for r in load_csv(LECTURES_NEW)}
    theme = {r["verse_id"]: (r["theme_id"], r["theme_name"])
             for r in load_csv(THEMES)}

    videos = read_split("kn", args.split) + read_split("en", args.split)
    by_video = {}
    for u in grid.values():
        if u["video_file"] in videos:
            by_video.setdefault(u["video_file"], []).append(u)
    lecture_units = {v: [u["unit_id"] for u in sorted(us, key=lambda u: int(u["unit_index"]))]
                     for v, us in by_video.items()}

    z = np.load(os.path.join(HERE, "results", "step2_features.npz"))
    refs = list(z["refs"])
    row = {u: i for i, u in enumerate(z["unit_ids"])}
    decoded = predict_locked(lecture_units, args.alpha)

    passages, segments = [], []
    for v in videos:
        us = lecture_units[v]
        lang = grid[us[0]]["language"]
        words = np.cumsum([0] + [int(grid[u]["n_words"]) for u in us])
        total = int(words[-1])
        seg = None
        for k, (u, (state, extra, p_state, p_off)) in enumerate(zip(us, decoded[v])):
            primary = "OFF" if state == OFF else refs[state]
            tid, tname = theme.get(primary.replace("BG ", ""), ("", ""))
            anchors = [j for j in np.flatnonzero(z["anchor"][row[u]]) if j != state]
            secondary = sorted(set(anchors) | set(extra))
            if seg is None or seg["primary_verse"] != primary:
                seg = dict(segment_id=f"{v}_s{len(segments):03d}", video_file=v,
                           primary_verse=primary, theme_id=tid, first_unit=u,
                           last_unit=u, dwell=0, effective_dwell=0.0,
                           pos_start=round(words[k] / total, 4))
                segments.append(seg)
            seg["last_unit"], seg["dwell"] = u, seg["dwell"] + 1
            seg["effective_dwell"] = round(seg["effective_dwell"] + p_state, 3)
            seg["pos_end"] = round(words[k + 1] / total, 4)
            passages.append(dict(
                unit_id=u, video_file=v, language=lang,
                speaker=speaker.get(v, ORIGINAL_SPEAKER[lang]),
                tradition=grid[u]["tradition"], unit_index=grid[u]["unit_index"],
                primary_verse=primary, p_primary=round(p_state, 4), p_off=round(p_off, 4),
                theme_id=tid, theme_name=tname,
                secondary_verses=";".join(refs[j] for j in secondary),
                segment_id=seg["segment_id"], word_start=int(words[k]),
                word_end=int(words[k + 1]), lecture_words=total,
                pos_start=round(words[k] / total, 4), pos_end=round(words[k + 1] / total, 4)))

    for name, rows_ in (("index", passages), ("segments", segments)):
        out = os.path.join(HERE, "results", f"step5_{name}_{args.split}.csv")
        with open(out, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows_[0]))
            w.writeheader()
            w.writerows(rows_)
        print(f"  wrote -> {os.path.relpath(out, HERE)}  ({len(rows_)} rows)")

    # ---- summary ----------------------------------------------------------
    print(f"\n  {len(videos)} lectures, {len(passages)} passages, {len(segments)} segments")
    for lang in ("kn", "en"):
        ps = [p for p in passages if p["language"] == lang]
        sg = [s for s in segments if grid[s["first_unit"]]["language"] == lang]
        vs = [s for s in sg if s["primary_verse"] != "OFF"]
        d = Counter(s["dwell"] for s in vs)
        print(f"  {lang}: passages {len(ps)}  off-verse {sum(p['primary_verse'] == 'OFF' for p in ps)}"
              f"  with secondary verses {sum(bool(p['secondary_verses']) for p in ps)}"
              f"  verse segments {len(vs)}  distinct verses {len({s['primary_verse'] for s in vs})}")
        print(f"      dwell (passages per verse segment): "
              + "  ".join(f"{k}:{d[k]}" for k in sorted(d)))

    if args.split == "dev":
        # in-sample sanity check: the locked models were fit on these lectures
        lectures, gold, _ = load_gold()
        pred = {u: ([] if st == OFF else [refs[st]]) + [refs[j] for j in extra]
                for v in videos for u, (st, extra, _, _) in zip(lecture_units[v], decoded[v])}
        print("\n  IN-SAMPLE check (models were fit on these lectures; not a result):")
        for lang in ("kn", "en"):
            vids = [v for v in read_split(lang, "dev") if v in lectures]
            print(f"    {lang}: strict {score(gold, pred, lectures, vids)['F1']:.3f}  "
                  f"+-1 {near_f1(gold, pred, lectures, vids)[2]:.3f}  "
                  f"theme {theme_f1(gold, pred, lectures, vids):.3f}")


if __name__ == "__main__":
    main()
