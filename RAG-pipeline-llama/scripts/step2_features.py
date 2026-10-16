#!/usr/bin/env python3
"""
Step 2 (core): passage x verse evidence for every unit in both unit grids.

For every (passage t, verse v) pair, v = BG 2.1 .. 2.72, this saves:

  2A  string     rapidfuzz partial_ratio / 100 of the verse against the WHOLE
                 passage (Kannada: spelling_key on both sides; English: romanised
                 verse vs normalised passage, as roman_detector does)
      anchor     1 if the existing detector fires v on this passage
                 (Kannada ShlokaDetector 0.12 / 45, English Roman 55)
  2B  cos_aX     cosine of the passage to the verse's meaning profile mu_v
      z_aX       that cosine as a z-score over the 72 verses on this passage
      gain_aX    cos(p, mu_v) - cos(p, mu_{v-1});  0 for BG 2.1
                 X = alpha, the weight on the verse text (a0 = 0, a05 = 0.5)
  2C  recency    1 / (1 + passages since v was last detected in this lecture),
                 counting detections BEFORE t only; 0 if never detected before t
      next       1 if v-1 was among the verses detected at the most recent
                 earlier anchor passage in this lecture
      last       1 if v itself was among them (the verse plain carry would hold)

Meaning profile: average BGE-M3 embedding of every anchor passage of v plus the
next 2 passages in its lecture, over the 39 original lectures' DETECTIONS (no
gold labels). The lecture being scored is left out; new-speaker lectures use all
39. mu_v = normalise((1 - alpha) * corpus_profile + alpha * verse_text), where
verse_text is the same verse document embed_scores.py calls "expanded"
(Devanagari + Kannada script + translations). If v has no anchor outside the
scored lecture, its profile falls back to the embedding of the Sanskrit text
alone (no translation). Two options, compared with --compare: --stop-window
(stop the 2 following passages at a detection of another verse) and
--per-language (profiles from same-language lectures only).

Passage-level, for the off-verse state (2D; kept only if the Step 3 ablation
shows a gain):
      adj_cos            cosine of passage t-1 and t (1 for a lecture's first)
      sanskrit_density   ShlokaDetector.sanskrit_density of the whole passage
                         (counts Kannada-script characters only)

Then prints, on the TUNING (dev) lectures only, how well each feature separates
gold (unit, verse) pairs from all other pairs: ROC AUC and average precision.

Usage:
    python scripts/step2_features.py --compare
    python scripts/step2_features.py [--stop-window] [--per-language]
"""
import argparse
import contextlib
import io
import os
import re
import sys
from collections import defaultdict

import numpy as np
from rapidfuzz import fuzz

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

from evalkit import UNITS, UNITS_HELDOUT, load_csv, load_gold, read_split  # noqa: E402
from embed_scores import unit_cache  # noqa: E402

OUT = os.path.join(HERE, "results", "step2_features.npz")
ALPHAS = {"a0": 0.0, "a05": 0.5}
FOLLOW = 2          # passages after an anchor that also feed the profile


def verse_docs():
    """One text per verse: Devanagari + Kannada script + English translations
    (what embed_scores.py calls "expanded")."""
    from verse_database import VERSES_BG_CH2
    from verse_translations import VERSE_TRANSLATIONS
    return ["\n".join([v["devanagari"], v["kannada"]] + list(VERSE_TRANSLATIONS[v["ref"]]))
            for v in VERSES_BG_CH2]


def normalise(x):
    n = np.linalg.norm(x, axis=-1, keepdims=True)
    return np.divide(x, n, out=np.zeros_like(x), where=n > 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--compare", action="store_true",
                    help="print alpha = 0 AUC for the four profile variants and stop")
    ap.add_argument("--stop-window", action="store_true",
                    help="stop a profile window at a detection of a different verse")
    ap.add_argument("--per-language", action="store_true",
                    help="build profiles from same-language lectures only")
    args = ap.parse_args()
    from roman_detector import RomanShlokaDetector, normalize_asr
    from shloka_detector import ShlokaDetector, spelling_key
    from verse_database import VERSES_BG_CH2

    refs = [v["ref"] for v in VERSES_BG_CH2]
    assert refs == [f"BG 2.{i}" for i in range(1, 73)], "verse order is not 2.1..2.72"
    V = len(refs)

    # ---- units of both grids, grouped by lecture in spoken order -------------
    units, emb = [], []
    original = set()
    for path in (UNITS, UNITS_HELDOUT):
        grid = load_csv(path)
        z = np.load(unit_cache(path), allow_pickle=False)
        if list(z["unit_ids"]) != [u["unit_id"] for u in grid]:
            sys.exit(f"embedding cache does not match {path}; run embed_scores.py")
        units += grid
        emb.append(z["emb"])
        if path == UNITS:
            original = {u["video_file"] for u in grid}
    E = np.concatenate(emb).astype(np.float32)
    row = {u["unit_id"]: i for i, u in enumerate(units)}
    lectures = defaultdict(list)
    for u in sorted(units, key=lambda u: (u["video_file"], int(u["unit_index"]))):
        lectures[u["video_file"]].append(row[u["unit_id"]])
    print(f"  {len(units)} units, {len(lectures)} lectures "
          f"({len(original)} original, {len(lectures) - len(original)} new-speaker)")

    # ---- 2A: continuous string score + detector anchor flag ------------------
    kn_det = ShlokaDetector(sanskrit_threshold=0.12, match_threshold=45)
    en_det = RomanShlokaDetector(match_threshold=55)
    kn_keys = [spelling_key(v["kannada"]) for v in VERSES_BG_CH2]
    en_keys = [en_det.verses[r] for r in refs]
    string = np.zeros((len(units), V), dtype=np.float32)
    anchor = np.zeros((len(units), V), dtype=np.int8)
    for i, u in enumerate(units):
        if u["language"] == "kn":
            hay, keys = spelling_key(u["text"]), kn_keys
            with contextlib.redirect_stdout(io.StringIO()):
                hits = [d["verse_ref"] for d in kn_det.detect(u["text"])]
        else:
            hay, keys = re.sub(r"\s+", " ", normalize_asr(u["text"])).strip(), en_keys
            hits = [d["verse_ref"] for d in en_det.detect(u["text"])]
        string[i] = [fuzz.partial_ratio(k, hay) / 100 for k in keys] if hay else 0
        for r in hits:
            anchor[i, refs.index(r)] = 1
    print(f"  2A: {int(anchor.sum())} detector anchors on "
          f"{int((anchor.sum(1) > 0).sum())} passages")

    # ---- 2B: meaning profiles, leave the scored lecture out -----------------
    from embeddings import embed_texts
    T = normalise(np.asarray(embed_texts(verse_docs()), dtype=np.float32))
    S = normalise(np.asarray(embed_texts([v["devanagari"] + "\n" + v["kannada"]
                                          for v in VERSES_BG_CH2]), dtype=np.float32))
    lang_of = {u["video_file"]: u["language"] for u in units}
    ctx = dict(E=E, lectures=lectures, original=original, lang_of=lang_of,
               anchor=anchor, T=T, S=S)

    if args.compare:
        variants = {(st, pl): meaning_features(**ctx, stop=st, per_lang=pl)[0]
                    for st in (False, True) for pl in (False, True)}
        compare(units, row, variants, refs)
        return

    feats, n_fallback = meaning_features(**ctx, stop=args.stop_window,
                                         per_lang=args.per_language)
    print(f"  2B stop_window={args.stop_window} per_language={args.per_language}: "
          f"(passage, verse) pairs using the Sanskrit-only fallback: {n_fallback}")

    # ---- 2C: observable recitation memory -----------------------------------
    recency = np.zeros((len(units), V), dtype=np.float32)
    nxt = np.zeros((len(units), V), dtype=np.int8)
    last = np.zeros((len(units), V), dtype=np.int8)
    for rows in lectures.values():
        last_seen = np.full(V, -1)
        last_set = []
        for pos, i in enumerate(rows):
            seen = last_seen >= 0
            recency[i, seen] = 1 / (1 + pos - last_seen[seen])
            last[i, last_set] = 1
            for v in last_set:
                if v + 1 < V:
                    nxt[i, v + 1] = 1
            hits = np.flatnonzero(anchor[i])
            if len(hits):
                last_seen[hits] = pos
                last_set = list(hits)

    # ---- passage-level signals for the off-verse state (2D, kept only if
    # the Step 3 ablation shows a gain) ---------------------------------------
    adj_cos = np.ones(len(units), dtype=np.float32)      # first passage: 1 (no drop)
    for rows in lectures.values():
        for a, b in zip(rows, rows[1:]):
            adj_cos[b] = E[a] @ E[b]
    density = np.array([kn_det.sanskrit_density(u["text"]) for u in units],
                       dtype=np.float32)

    feats.update(string=string, anchor=anchor, recency=recency, next=nxt, last=last,
                 adj_cos=adj_cos, sanskrit_density=density)
    np.savez_compressed(OUT, unit_ids=np.array([u["unit_id"] for u in units]),
                        refs=np.array(refs), **feats)
    print(f"  wrote -> {os.path.relpath(OUT, HERE)}  features: {sorted(feats)}")

    report(units, row, feats, refs)


def anchor_profile_sums(E, lectures, original, anchor, stop=False):
    """Per original lecture: {L: (72 x dim) sum of embeddings}, {L: (72,) count}
    over every anchor passage of v plus the next FOLLOW passages."""
    V = anchor.shape[1]
    lec_sum = {L: np.zeros((V, E.shape[1])) for L in original}
    lec_cnt = {L: np.zeros(V) for L in original}
    for L in original:
        rows = lectures[L]
        for pos, i in enumerate(rows):
            for v in np.flatnonzero(anchor[i]):
                for k, j in enumerate(rows[pos:pos + 1 + FOLLOW]):
                    others = np.flatnonzero(anchor[j])
                    if stop and k > 0 and any(o != v for o in others):
                        break
                    lec_sum[L][v] += E[j]
                    lec_cnt[L][v] += 1
    return lec_sum, lec_cnt


def meaning_features(E, lectures, original, lang_of, anchor, T, S, stop, per_lang):
    """cos / z / gain to the meaning profile mu_v, for each alpha.

    stop      : stop adding the passages after an anchor of v as soon as one of
                them has a detection of a different verse
    per_lang  : profiles from original lectures of the scored lecture's language
                only (default: all 39 pooled)
    A verse with no corpus anchor left (after leave-one-out) falls back to the
    embedding of its Sanskrit text alone (Devanagari + Kannada script, no
    translation), so alpha = 0 stays translation-free.
    Returns (feats, number of (passage, verse) pairs that used the fallback).
    """
    V = anchor.shape[1]
    lec_sum, lec_cnt = anchor_profile_sums(E, lectures, original, anchor, stop)

    n = len(anchor)
    feats = {f"{k}_{a}": np.zeros((n, V), dtype=np.float32)
             for a in ALPHAS for k in ("cos", "z", "gain")}
    n_fallback = 0
    for L, rows in lectures.items():
        pool = [M for M in original if M != L and (not per_lang or lang_of[M] == lang_of[L])]
        s = sum(lec_sum[M] for M in pool)
        c = sum(lec_cnt[M] for M in pool)
        base = np.where((c > 0)[:, None], normalise(s.astype(np.float32)), S)
        n_fallback += int((c == 0).sum()) * len(rows)
        P = E[rows]
        for a, alpha in ALPHAS.items():
            mu = normalise((1 - alpha) * base + alpha * T)
            cos = P @ mu.T
            feats[f"cos_{a}"][rows] = cos
            feats[f"z_{a}"][rows] = (cos - cos.mean(1, keepdims=True)) / cos.std(1, keepdims=True)
            feats[f"gain_{a}"][rows, 1:] = cos[:, 1:] - cos[:, :-1]
    return feats, n_fallback


def tuning_targets(units, row, refs):
    """Rows, gold (passage x verse) matrix and lecture ids for the dev lectures."""
    lectures, gold, _ = load_gold()
    out = {}
    for lang in ("kn", "en"):
        vids = [v for v in read_split(lang, "dev") if v in lectures]
        us = [u for v in vids for u in lectures[v]]
        y = np.zeros((len(us), len(refs)), dtype=np.int8)
        for k, u in enumerate(us):
            for r in gold[u]:
                if r in refs:
                    y[k, refs.index(r)] = 1
        out[lang] = (vids, [row[u] for u in us], y)
    return out


def compare(units, row, variants, refs):
    """alpha = 0 AUC on the tuning lectures for each profile variant."""
    from sklearn.metrics import roc_auc_score
    print("\n" + "=" * 78)
    print("alpha = 0 profile variants, TUNING lectures, AUC (Sanskrit-only fallback in all)")
    print("=" * 78)
    tgt = tuning_targets(units, row, refs)
    print(f"  {'stop_window':12} {'per_language':13}" + "".join(
        f"{lang + ' ' + k:>10}" for lang in ("kn", "en") for k in ("cos", "z", "gain")))
    for (st, pl), f in variants.items():
        cells = []
        for lang in ("kn", "en"):
            _, rows, y = tgt[lang]
            cells += [roc_auc_score(y.ravel(), f[f"{k}_a0"][rows].ravel())
                      for k in ("cos", "z", "gain")]
        print(f"  {str(st):12} {str(pl):13}" + "".join(f"{x:>10.3f}" for x in cells))


def report(units, row, feats, refs):
    """How well each feature separates gold pairs from the rest, tuning lectures only."""
    from sklearn.metrics import average_precision_score, roc_auc_score

    tgt = tuning_targets(units, row, refs)
    order = ["string", "anchor", "cos_a0", "z_a0", "gain_a0",
             "cos_a05", "z_a05", "gain_a05", "recency", "next"]
    print("\n" + "=" * 78)
    print("TUNING lectures only: gold (passage, verse) pairs vs all other pairs")
    print("AUC 0.5 = no separation, 1.0 = perfect.  AP = average precision "
          "(chance = positive rate)")
    print("=" * 78)
    for lang in ("kn", "en"):
        vids, rows, y2 = tgt[lang]
        y = y2.ravel()
        print(f"\n  {lang}: {len(vids)} lectures, {len(rows)} passages, "
              f"{int(y.sum())} positive pairs of {len(y)} "
              f"(positive rate {y.mean():.4f})")
        print(f"    {'feature':10} {'AUC':>6} {'AP':>6}")
        for f in order:
            x = feats[f][rows].ravel().astype(np.float64)
            print(f"    {f:10} {roc_auc_score(y, x):>6.3f} {average_precision_score(y, x):>6.3f}")

    print("\n" + "=" * 78)
    print("TUNING lectures: do the passage-level signals pick out OFF-VERSE passages?")
    print("(target = passage with no gold verse; scores flipped so higher = off-verse)")
    print("=" * 78)
    for lang in ("kn", "en"):
        _, rows, y2 = tgt[lang]
        off = (y2.sum(1) == 0).astype(int)
        print(f"  {lang}: {int(off.sum())} off-verse of {len(off)} passages")
        if 0 < off.sum() < len(off):
            for f in ("adj_cos", "sanskrit_density"):
                x = -feats[f][rows].astype(np.float64)
                print(f"    {f:17} AUC {roc_auc_score(off, x):.3f}   "
                      f"mean {(-x).mean():.3f}  max {(-x).max():.3f}")


if __name__ == "__main__":
    main()
