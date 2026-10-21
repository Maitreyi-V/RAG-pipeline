#!/usr/bin/env python3
"""
Work plan Phase 2: the decoder with different off-verse evidence.

P_off sources: old scorer (the locked system), teacher (silver soft_target),
student (results/student/student_poff.csv, column p_final: out-of-fold, the
student chosen by silver loss), and the old scorer with the student's or the
teacher's logit as a 6th feature (refitted on tuning gold, leave-one-lecture-out). Decoders: 73-state and 145-state resume.
gamma, beta_off, tau re-chosen by leave-one-lecture-out on the 9 TUNING lectures.
No test gold.

Usage:
    python scripts/phase2_decoder.py
"""
import csv
import math
import os
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

from evalkit import score  # noqa: E402
from step4_viterbi import near_f1, run, theme_f1  # noqa: E402


def column(path, col):
    return {r["unit_id"]: min(0.98, max(0.02, float(r[col]))) for r in
            csv.DictReader(open(os.path.join(HERE, path), encoding="utf-8"))}


def logit(d):
    return {u: math.log(p / (1 - p)) for u, p in d.items()}


def main():
    teacher = column("results/silver/silver_labels.csv", "soft_target")
    student = column("results/student/student_poff.csv", "p_final")
    sources = {"old": {},
               "teacher": {"off_override": teacher},
               "student": {"off_override": student},
               "old+student": {"off_extra": logit(student)},
               "old+teacher": {"off_extra": logit(teacher)}}
    lines = ["Phase 2 — TUNING lectures, leave-one-lecture-out, alpha 0.5 (no test gold)",
             "old+X = old off-verse scorer with X's logit as a 6th feature, refitted on tuning gold",
             f"  {'P_off':12} {'decoder':10} {'KN strict':>9} {'KN +-1':>7} {'EN strict':>9} {'EN +-1':>7} "
             f"{'EN theme':>8}   {'EN off->OFF':>11}   EN strict per lecture"]
    for name, src in sources.items():
        for dec, kw in (("73-state", {}), ("145-state", {"resume": True})):
            kw = dict(kw, **src)
            order, gold, lectures, tune = run(alphas=("a05",), verbose=False, **kw)
            pred = dict(order)["Viterbi (alpha 0.5)"]
            kn = (score(gold, pred, lectures, tune["kn"])["F1"], near_f1(gold, pred, lectures, tune["kn"])[2])
            en = (score(gold, pred, lectures, tune["en"])["F1"], near_f1(gold, pred, lectures, tune["en"])[2],
                  theme_f1(gold, pred, lectures, tune["en"]))
            off = [u for v in tune["en"] for u in lectures[v] if not gold[u]]
            per = ", ".join(f"{v} {score(gold, pred, lectures, [v])['F1']:.3f}" for v in tune["en"])
            lines.append(f"  {name:12} {dec:10} {kn[0]:>9.3f} {kn[1]:>7.3f} {en[0]:>9.3f} {en[1]:>7.3f} "
                         f"{en[2]:>8.3f}   {sum(1 for u in off if not pred[u]):>3}/{len(off):<7}   {per}")
    out = "\n".join(lines)
    print(out)
    open(os.path.join(HERE, "results", "student", "phase2_decoder.txt"), "w").write(out + "\n")


if __name__ == "__main__":
    main()
