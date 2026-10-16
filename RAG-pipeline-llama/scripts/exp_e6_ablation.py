#!/usr/bin/env python3
"""
Experiment E6: ablation of the locked Step 4 system on the 9 TUNING lectures.

Same protocol as step4_viterbi.py: leave-one-lecture-out; for every held-out
lecture the scorers, transitions and (gamma, beta_off, tau) are re-fit / re-chosen
on the other 8, with the part removed. alpha 0.5.

  full                 the locked system
  - string match       verse scorer without string, anchor; off-verse scorer
                       without max_string, any_anchor. NOTE the memory features
                       and the held-verse similarity are computed FROM the string
                       detections, so detections still enter through them.
  - meaning match      verse scorer without cos, z, gain; off-verse scorer
                       without max_cos and the held-verse similarity
  - lecture memory     verse scorer without recency, next, last; off-verse
                       scorer without max_recency
  - off-verse state    off-verse allowed only before the lecture's first detection
  - lecture decoding   the full scorer alone, no Viterbi (verses with P >= 0.5)

Also reports segmentation accuracy: Pk and WindowDiff (lower is better).

Writes results/exp_e6_ablation.txt.

Usage:
    python scripts/exp_e6_ablation.py
"""
import os
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

from evalkit import score, verse_number  # noqa: E402
from step4_viterbi import near_f1, run, theme_f1  # noqa: E402

OUT = os.path.join(HERE, "results", "exp_e6_ablation.txt")


def labels(refs_per_passage, primary=False):
    """one label per passage, 0 = off-verse. primary=True: the first output verse
    (Viterbi's state; the rest are secondary verses). Otherwise the lowest Chapter 2
    verse (gold, carry and the scorer have no primary verse)."""
    if primary:
        return [verse_number(r[0]) or 0 if r else 0 for r in map(list, refs_per_passage)]
    return [min((n for n in map(verse_number, r) if n), default=0) for r in refs_per_passage]


def seg_errors(ref, hyp):
    """Pk and WindowDiff (errors, windows) for one lecture; k = half the mean gold
    segment length (Beeferman et al. 1999; Pevzner & Hearst 2002)."""
    rb = [int(a != b) for a, b in zip(ref, ref[1:])]
    hb = [int(a != b) for a, b in zip(hyp, hyp[1:])]
    k = max(1, round(len(ref) / (2 * (sum(rb) + 1))))
    n = len(ref) - k
    if n <= 0:
        return 0, 0, 0
    pk = sum((sum(rb[i:i + k]) == 0) != (sum(hb[i:i + k]) == 0) for i in range(n))
    wd = sum(sum(rb[i:i + k]) != sum(hb[i:i + k]) for i in range(n))
    return pk, wd, n


def segmentation(gold, pred, lectures, vids, primary=False):
    """pooled Pk, WindowDiff over the lectures (lower is better)"""
    pk = wd = n = 0
    for v in vids:
        a, b, c = seg_errors(labels(gold[u] for u in lectures[v]),
                             labels((pred.get(u, ()) for u in lectures[v]), primary))
        pk, wd, n = pk + a, wd + b, n + c
    return pk / n, wd / n

VARIANTS = [
    ("full (locked)", {}),
    ("- string match", dict(drop_verse=("string", "anchor"), drop_off=("max_string", "any_anchor"))),
    ("- meaning match", dict(drop_verse=("cos", "z", "gain"), drop_off=("max_cos",),
                             off_active=False)),
    ("- lecture memory", dict(drop_verse=("recency", "next", "last"), drop_off=("max_recency",))),
    ("- off-verse state", dict(off_after_anchor=False)),
]


def main():
    rows = []
    for name, kw in VARIANTS:
        order, gold, lectures, tune = run(alphas=("a05",), verbose=False, **kw)
        systems = dict(order)
        rows.append((name, systems["Viterbi (alpha 0.5)"]))
        if name.startswith("full"):
            scorer = ("- lecture decoding", systems["scorer alone (alpha 0.5)"])
            carry_rows = [("plain carry (reference)", systems["plain carry"]),
                          ("carry, best anchor (reference)", systems["carry, best anchor only"])]
    rows.append(scorer)
    rows += carry_rows

    lines = ["E6 ablation, 9 TUNING lectures, leave-one-lecture-out, alpha 0.5",
             "Themes: data/verse_themes.csv (domain expert). Settings re-chosen per fold.",
             "",
             f"  {'system':32} {'kn strict':>9} {'kn +-1':>7} {'kn theme':>8}   "
             f"{'en strict':>9} {'en +-1':>7} {'en theme':>8}"]
    seg_lines = [f"  {'system':32} {'kn Pk':>7} {'kn WD':>7}   {'en Pk':>7} {'en WD':>7}"]
    full = None
    for name, pred in rows:
        cells = []
        for L in ("kn", "en"):
            v = tune[L]
            cells += [score(gold, pred, lectures, v)["F1"], near_f1(gold, pred, lectures, v)[2],
                      theme_f1(gold, pred, lectures, v)]
        full = full or cells
        viterbi_row = "reference" not in name and "decoding" not in name
        sg = [x for L in ("kn", "en")
              for x in segmentation(gold, pred, lectures, tune[L], primary=viterbi_row)]
        seg_lines.append(f"  {name:32} {sg[0]:>7.3f} {sg[1]:>7.3f}   {sg[2]:>7.3f} {sg[3]:>7.3f}")
        lines.append(f"  {name:32} {cells[0]:>9.3f} {cells[1]:>7.3f} {cells[2]:>8.3f}   "
                     f"{cells[3]:>9.3f} {cells[4]:>7.3f} {cells[5]:>8.3f}")
        if not name.startswith("full") and "reference" not in name:
            d = [c - f for c, f in zip(cells, full)]
            lines.append(f"  {'   change vs full':32} {d[0]:>+9.3f} {d[1]:>+7.3f} {d[2]:>+8.3f}   "
                         f"{d[3]:>+9.3f} {d[4]:>+7.3f} {d[5]:>+8.3f}")
    lo = [x for L in ("kn", "en") for x in segmentation(gold, rows[0][1], lectures, tune[L])]
    lines += ["", "Segmentation (lower is better): Pk and WindowDiff, one label per passage,",
              "k = half the mean gold segment length per lecture, errors pooled over lectures.",
              "Label: Viterbi rows use the decoded state (primary verse; secondary verses are",
              "not segments); gold, carry and the scorer alone use the lowest Chapter 2 verse",
              "on the passage (they have no primary verse); 0 = off-verse."] + seg_lines
    lines += [f"  (full system with the lowest-verse rule instead: kn {lo[0]:.3f} / {lo[1]:.3f}, "
              f"en {lo[2]:.3f} / {lo[3]:.3f})"]
    lines += ["", "Memory features and held-verse similarity are computed from the string",
              "detections, so '- string match' does not remove detections entirely.",
              "The plan's row 'w/o boundary & depth features' has no counterpart: no 2D",
              "feature is in the locked system (none helped on tuning); '- off-verse state'",
              "is the closest row."]
    text = "\n".join(lines)
    print(text)
    with open(OUT, "w", encoding="utf-8") as f:
        f.write(text + "\n")
    print(f"\n  wrote -> {os.path.relpath(OUT, HERE)}")


if __name__ == "__main__":
    main()
