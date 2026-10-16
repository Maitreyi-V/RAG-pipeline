#!/usr/bin/env python3
"""
One comparison on the 9 TUNING lectures (leave-one-lecture-out, alpha 0.5):
the locked Step 4 system with / without
  resume  : 145-state model, "off-verse while holding verse v" can resume v, v+1, v+2
  held_z  : z-score of the carried verse added to the off-verse scorer

Prints P, R, strict / within-1 / theme F1 per language, the transition bucket
counts, and how many of the 7 gold off-verse passages inside video_04's
2.16 segment (step5_segments_dev.csv, video_04_s027) are decoded as OFF.

Usage:
    python scripts/step4_resume_compare.py
"""
import csv
import os
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

from evalkit import score  # noqa: E402
from step4_viterbi import near_f1, run, theme_f1  # noqa: E402

VARIANTS = [("locked", {}), ("+ resume", dict(resume=True)),
            ("+ held_z", dict(held_z=True)), ("+ resume + held_z", dict(resume=True, held_z=True))]


def main():
    seg = [r["unit_id"] for r in csv.DictReader(open(
        os.path.join(HERE, "results", "step5_index_dev.csv"), encoding="utf-8"))
        if r["segment_id"] == "video_04_s027"]
    out = []
    for name, kw in VARIANTS:
        info = {}
        order, gold, lectures, tune = run(alphas=("a05",), verbose=False, info=info, **kw)
        pred = dict(order)["Viterbi (alpha 0.5)"]
        stories = [u for u in seg if not gold[u]]
        out.append((name, pred, info, sum(1 for u in stories if not pred[u]), len(stories),
                    sum(1 for u in seg if gold[u] and not pred[u])))

    print("=" * 104)
    print("Step 4 resume / held_z, 9 TUNING lectures, leave-one-lecture-out, alpha 0.5")
    print("=" * 104)
    print(f"  {'system':20} {'lang':4} {'P':>6} {'R':>6} {'strict':>7} {'+-1':>6} {'theme':>6}")
    for name, pred, *_ in out:
        for L in ("kn", "en"):
            s = score(gold, pred, lectures, tune[L])
            print(f"  {name:20} {L:4} {s['P']:>6.3f} {s['R']:>6.3f} {s['F1']:>7.3f} "
                  f"{near_f1(gold, pred, lectures, tune[L])[2]:>6.3f} "
                  f"{theme_f1(gold, pred, lectures, tune[L]):>6.3f}")
    print("\n  video_04, 2.16 segment of the locked index (52 passages):")
    for name, _, info, hit, n, lost in out:
        print(f"    {name:20} gold off-verse decoded OFF: {hit} of {n}   "
              f"gold 2.16 passages decoded OFF: {lost}")
    print("\n  transition bucket counts, all 9 tuning lectures (+1 smoothing):")
    for name, _, info, *_ in out[:2]:
        print(f"    {name:20} {info['counts']}")
    print("\n  settings chosen per held-out lecture (gamma, beta_off, tau):")
    for name, _, info, *_ in out:
        print(f"    {name:20} " + "  ".join(f"{h}:{g}/{b}/{t}" for h, g, b, d, t in info["chosen"]))


if __name__ == "__main__":
    main()
