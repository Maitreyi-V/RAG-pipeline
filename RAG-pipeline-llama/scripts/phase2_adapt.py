#!/usr/bin/env python3
"""
Work plan Phase 2, last step: per-lecture adaptation of two scalars (stay
probability, off-verse rate; MAP with a Beta / Dirichlet prior, see
step4_viterbi.adapt_counts). The prior strength kappa is chosen together with
gamma, beta_off and tau by leave-one-lecture-out on the 9 TUNING lectures.
Rows without adaptation are the Phase 2 rows (regression check). No test gold.

Usage:
    python scripts/phase2_adapt.py
"""
import os
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

from evalkit import score  # noqa: E402
from phase2_decoder import column  # noqa: E402
from step4_viterbi import near_f1, run, theme_f1  # noqa: E402


def main():
    teacher = column("results/silver/silver_labels.csv", "soft_target")
    lines = ["Phase 2 adaptation — TUNING lectures, leave-one-lecture-out, alpha 0.5 (no test gold)",
             f"  {'P_off':8} {'decoder':10} {'adapt':6} {'KN strict':>9} {'EN strict':>9} {'EN +-1':>7} "
             f"{'EN theme':>8}   {'EN off->OFF':>11}   EN strict per lecture | kappa per held-out lecture"]
    for name, src in (("old", {}), ("teacher", {"off_override": teacher})):
        for dec, kw in (("73-state", {}), ("145-state", {"resume": True})):
            for adapt in (False, True):
                info = {}
                order, gold, lectures, tune = run(alphas=("a05",), verbose=False, info=info,
                                                  adapt=adapt, **src, **kw)
                pred = dict(order)["Viterbi (alpha 0.5)"]
                en = tune["en"]
                off = [u for v in en for u in lectures[v] if not gold[u]]
                per = ", ".join(f"{v} {score(gold, pred, lectures, [v])['F1']:.3f}" for v in en)
                kap = ", ".join(f"{v}:{k:g}" for v, k in info["kappa"].items()) if adapt else ""
                lines.append(f"  {name:8} {dec:10} {'yes' if adapt else 'no':6} "
                             f"{score(gold, pred, lectures, tune['kn'])['F1']:>9.3f} "
                             f"{score(gold, pred, lectures, en)['F1']:>9.3f} "
                             f"{near_f1(gold, pred, lectures, en)[2]:>7.3f} {theme_f1(gold, pred, lectures, en):>8.3f}"
                             f"   {sum(1 for u in off if not pred[u]):>3}/{len(off):<7}   {per}"
                             + (f" | {kap}" if kap else ""))
                print(lines[-1], flush=True)
    out = "\n".join(lines)
    open(os.path.join(HERE, "results", "student", "phase2_adapt.txt"), "w").write(out + "\n")


if __name__ == "__main__":
    main()
