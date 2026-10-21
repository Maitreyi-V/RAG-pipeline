#!/usr/bin/env python3
"""
Work plan Phase 1, check before the student: does better off-verse evidence help
the decoder? P_off is replaced by the silver teacher's soft_target
(results/silver/silver_labels.csv); everything else is unchanged and gamma,
beta_off, tau are re-chosen by leave-one-lecture-out on the 9 TUNING lectures.
No test gold is used.

Rows: current system (old P_off) and teacher-direct, each with the 73-state and
the 145-state resume decoder. alpha 0.5.

Usage:
    python scripts/teacher_direct_check.py
"""
import csv
import os
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

from evalkit import score  # noqa: E402
from step4_viterbi import near_f1, run, theme_f1  # noqa: E402


def main():
    soft = {r["unit_id"]: float(r["soft_target"]) for r in
            csv.DictReader(open(os.path.join(HERE, "results", "silver", "silver_labels.csv"), encoding="utf-8"))}
    rows = []
    for name, kw in (("current P_off, 73-state", {}),
                     ("current P_off, 145-state resume", dict(resume=True)),
                     ("teacher P_off, 73-state", dict(off_override=soft)),
                     ("teacher P_off, 145-state resume", dict(off_override=soft, resume=True))):
        info = {}
        order, gold, lectures, tune = run(alphas=("a05",), verbose=False, info=info, **kw)
        pred = dict(order)["Viterbi (alpha 0.5)"]
        cells = []
        for L in ("kn", "en"):
            cells += [score(gold, pred, lectures, tune[L])["F1"], near_f1(gold, pred, lectures, tune[L])[2],
                      theme_f1(gold, pred, lectures, tune[L])]
        off_ok = sum(1 for v in tune["en"] for u in lectures[v] if not gold[u] and not pred[u])
        off_n = sum(1 for v in tune["en"] for u in lectures[v] if not gold[u])
        settings = sorted({(g, b, t) for _, g, b, d, t in info["chosen"]})
        rows.append((name, cells, off_ok, off_n, settings))
    print("TUNING lectures, leave-one-lecture-out, alpha 0.5 (no test gold)")
    print(f"  {'system':34} {'KN strict':>9} {'KN +-1':>7} {'KN theme':>8}   {'EN strict':>9} {'EN +-1':>7} "
          f"{'EN theme':>8}   EN off-verse passages decoded OFF")
    for name, c, ok, n, st in rows:
        print(f"  {name:34} {c[0]:>9.3f} {c[1]:>7.3f} {c[2]:>8.3f}   {c[3]:>9.3f} {c[4]:>7.3f} {c[5]:>8.3f}   "
              f"{ok}/{n}")
    print("  settings chosen per fold (gamma, beta_off, tau):")
    for name, *_, st in rows:
        print(f"    {name:34} {st}")


if __name__ == "__main__":
    main()
