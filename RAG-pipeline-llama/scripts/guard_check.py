#!/usr/bin/env python3
"""
E3 guard check (one run, TUNING lectures, leave-one-lecture-out): v2 with only
confidence-5 DIGRESSION answers as digression evidence. DIGRESSION answers with
confidence 3-4 get P_off = 0.5 (no evidence either way); VERSE answers and
confidence-5 DIGRESSION keep soft_target. Adopt only if English strict F1 does
not drop AND fewer gold-verse passages are decoded off-verse.

Usage:
    python scripts/guard_check.py
"""
import csv
import os
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

from evalkit import score  # noqa: E402
from phase2_decoder import column  # noqa: E402
from step4_viterbi import near_f1, run, theme_f1  # noqa: E402


def main():
    teacher = column("results/silver/silver_labels.csv", "soft_target")
    weak = {r["unit_id"] for r in csv.DictReader(open(os.path.join(HERE, "results/silver/silver_labels.csv")))
            if r["label"] == "DIGRESSION" and r["confidence"] != "5"}
    guard = {u: (0.5 if u in weak else p) for u, p in teacher.items()}
    lines = [f"Guard check — v2 (teacher P_off, 145-state), TUNING LOLO; {len(weak)} DIGRESSION answers "
             "with confidence < 5 set to P_off 0.5",
             f"  {'P_off':14} {'KN strict':>9} {'EN strict':>9} {'EN +-1':>7} {'EN theme':>8} "
             f"{'EN verse->OFF':>13} {'EN off->OFF':>11} {'teacher flags on EN verse (P>0.5)':>34}"]
    for name, src in (("teacher", teacher), ("teacher+guard", guard)):
        order, gold, lectures, tune = run(alphas=("a05",), verbose=False, resume=True, off_override=src)
        pred = dict(order)["Viterbi (alpha 0.5)"]
        en = tune["en"]
        us = [u for v in en for u in lectures[v]]
        verse = [u for u in us if gold[u]]
        off = [u for u in us if not gold[u]]
        lines.append(f"  {name:14} {score(gold, pred, lectures, tune['kn'])['F1']:>9.3f} "
                     f"{score(gold, pred, lectures, en)['F1']:>9.3f} {near_f1(gold, pred, lectures, en)[2]:>7.3f} "
                     f"{theme_f1(gold, pred, lectures, en):>8.3f} "
                     f"{sum(1 for u in verse if not pred[u]):>6}/{len(verse):<6} "
                     f"{sum(1 for u in off if not pred[u]):>5}/{len(off):<5} "
                     f"{sum(1 for u in verse if src[u] > 0.5):>27}/{len(verse)}")
    out = "\n".join(lines)
    print(out)
    open(os.path.join(HERE, "results", "student", "guard_check.txt"), "w").write(out + "\n")


if __name__ == "__main__":
    main()
