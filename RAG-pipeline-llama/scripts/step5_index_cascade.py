#!/usr/bin/env python3
"""
Step 5 (cascade): the passage index the app reads, with verse labels from the IMPROVED 3-LAYER system
(Sol, then Cue digression veto, then Cue fill), for ALL 59 lectures.

Does not touch the frozen step5_index.py. Writes results/step5_index_all.csv and step5_segments_all.csv
in the same columns the app already reads (gitaground.py needs no change).

Inputs
  results/sol_labels_59.csv     unit_id, video_file, refs   (Sol answer per passage, ';'-joined 'BG 2.x')
  results/step2_features.npz    string anchors (secondary verses); must be rebuilt for all 2,583 passages
  data/units_v2.csv, data/units_heldout_v1.csv, data/lectures_new.csv, data/verse_themes.csv

Layers (as improved3l_55.py; keep rule on the 9 tuning lectures with Sol: Kannada = Sol + Cue fill,
English = Cue veto + Sol, English fill dropped)
  English : Cue off-verse with P_off >= 0.95 -> OFF; otherwise Sol's label
  Kannada : Sol's label; if Sol says NONE and Cue names a verse with conf >= 0.5 -> Cue's verse
  Cue comes from improved3l_55.cue_all() for the 55 lectures it covers. Lectures it does not cover
  (video_11..video_14: no teacher P_off yet) get Sol alone, flagged layer "2 Sol (no Cue)".

p_primary (ranks passages inside a segment; the cascade itself gives no probability)
  cascade verse == Cue's verse : Cue's decoded p_state
  cascade verse != Cue's verse : 1 - Cue P_off   (OFF: P_off)
  no Cue for the lecture       : 1.0 (flat; the app multiplies by cosine, so ranking still works)

Usage
    python scripts/step5_index_cascade.py              # full: needs the rebuilt features + Cue models
    python scripts/step5_index_cascade.py --no-cue     # Sol alone for every lecture (no models needed)
"""
import argparse
import csv
import os
import sys
from collections import Counter

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "scripts"))

from evalkit import LECTURES_NEW, UNITS, UNITS_HELDOUT, load_csv  # noqa: E402
from catalogue import THEMES  # noqa: E402

FILL, VETO = 0.5, 0.95
ORIGINAL_SPEAKER = {"kn": "original Kannada speaker", "en": "original English speaker"}
RES = os.path.join(HERE, "results")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-cue", action="store_true", help="Sol alone, no Cue layers (needs no models)")
    ap.add_argument("--sol", default=os.path.join(RES, "sol_labels_59.csv"))
    ap.add_argument("--out-suffix", default="all")
    a = ap.parse_args()

    grid = {u["unit_id"]: u for u in load_csv(UNITS) + load_csv(UNITS_HELDOUT)}
    speaker = {r["video_file"]: r["speaker"] for r in load_csv(LECTURES_NEW)}
    theme = {r["verse_id"]: (r["theme_id"], r["theme_name"]) for r in load_csv(THEMES)}
    sol = {r["unit_id"]: [x for x in r["refs"].split(";") if x] for r in load_csv(a.sol)}

    by_video = {}
    for u in grid.values():
        by_video.setdefault(u["video_file"], []).append(u)
    lectures = {v: [u["unit_id"] for u in sorted(us, key=lambda u: int(u["unit_index"]))]
                for v, us in sorted(by_video.items())}
    missing = [u for us in lectures.values() for u in us if u not in sol]
    assert not missing, f"{len(missing)} passages have no Sol label, e.g. {missing[:3]}"

    # string anchors -> secondary verses
    z = np.load(os.path.join(RES, "step2_features.npz"))
    refs_all = list(z["refs"])
    frow = {u: i for i, u in enumerate(z["unit_ids"])}
    if not a.no_cue:
        miss_f = [u for us in lectures.values() for u in us if u not in frow]
        assert not miss_f, (f"step2_features.npz lacks {len(miss_f)} passages: rebuild it "
                            "(python scripts/step2_features.py) before running without --no-cue")

    cue = conf = poff = {}
    if not a.no_cue:
        import improved3l_55 as I
        from evalkit import load_gold
        lect_gold, _, _ = load_gold()
        cue, conf, poff, _ = I.cue_all(lect_gold)

    passages, segments = [], []
    layer_count = Counter()
    for v, us in lectures.items():
        lang = grid[us[0]]["language"]
        words = np.cumsum([0] + [int(grid[u]["n_words"]) for u in us])
        total = int(words[-1])
        seg = None
        has_cue = (not a.no_cue) and all(u in cue for u in us)
        for k, u in enumerate(us):
            labels, layer = list(sol[u]), "2 Sol" if has_cue else "2 Sol (no Cue)"
            if has_cue:
                if lang == "kn" and not labels and cue[u] and conf[u] >= FILL:
                    labels, layer = list(cue[u]), "3 Cue fill"
                if lang == "en" and not cue[u] and poff[u] >= VETO:
                    labels, layer = [], "1 Cue veto"
            layer_count[layer] += 1
            primary = labels[0] if labels else "OFF"
            tid, tname = theme.get(primary.replace("BG ", ""), ("", ""))
            if has_cue:
                if primary == "OFF":
                    pp = poff[u]
                elif cue[u] and cue[u][0] == primary:
                    pp = conf[u]
                else:
                    pp = 1.0 - poff[u]
            else:
                pp = 1.0
            anchors = []
            if u in frow:
                anchors = [refs_all[j] for j in np.flatnonzero(z["anchor"][frow[u]])]
            secondary = sorted({r for r in anchors + labels[1:] if r != primary and r != "OFF"},
                               key=lambda r: int(r.split(".")[1]))
            if seg is None or seg["primary_verse"] != primary:
                seg = dict(segment_id=f"{v}_s{len(segments):03d}", video_file=v, primary_verse=primary,
                           theme_id=tid, first_unit=u, last_unit=u, dwell=0, effective_dwell=0.0,
                           pos_start=round(words[k] / total, 4))
                segments.append(seg)
            seg["last_unit"], seg["dwell"] = u, seg["dwell"] + 1
            seg["effective_dwell"] = round(seg["effective_dwell"] + pp, 3)
            seg["pos_end"] = round(words[k + 1] / total, 4)
            passages.append(dict(
                unit_id=u, video_file=v, language=lang,
                speaker=speaker.get(v, ORIGINAL_SPEAKER[lang]),
                tradition=grid[u]["tradition"], unit_index=grid[u]["unit_index"],
                primary_verse=primary, p_primary=round(pp, 4),
                p_off=round(poff[u], 4) if has_cue else "",
                theme_id=tid, theme_name=tname, secondary_verses=";".join(secondary),
                segment_id=seg["segment_id"], word_start=int(words[k]), word_end=int(words[k + 1]),
                lecture_words=total, pos_start=round(words[k] / total, 4),
                pos_end=round(words[k + 1] / total, 4)))

    for name, rows_ in (("index", passages), ("segments", segments)):
        out = os.path.join(RES, f"step5_{name}_{a.out_suffix}.csv")
        with open(out, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows_[0]))
            w.writeheader()
            w.writerows(rows_)
        print(f"  wrote -> {os.path.relpath(out, HERE)}  ({len(rows_)} rows)")

    vs = [s for s in segments if s["primary_verse"] != "OFF"]
    print(f"\n  {len(lectures)} lectures, {len(passages)} passages, {len(segments)} segments, "
          f"{len({s['primary_verse'] for s in vs})} distinct verses")
    print("  deciding layer: " + "  ".join(f"{k}: {n}" for k, n in sorted(layer_count.items())))


if __name__ == "__main__":
    main()
