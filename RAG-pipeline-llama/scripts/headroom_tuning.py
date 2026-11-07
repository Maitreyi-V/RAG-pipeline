#!/usr/bin/env python3
"""
Headroom check for combining GPT-6.1 Sol with GitaGround-Cue, TUNING lectures only
(no test gold). Pre-registered in docs/experiment_log.md.

  Sol   gpt-6.1-sol end to end (llm_frontier_api.py send --lectures tuning)
  Cue   v2 primary, leave-one-lecture-out on the 9 tuning lectures (each lecture decoded
        with settings fitted on the other 8), exactly as in phase3_hybrid.tuning_rule()

Per language:
  - outcome table: gold on/off x Cue on/off x Sol NONE / correct (shares a gold verse) / wrong
  - oracle arbitration: per passage take Sol's or Cue's verse set, whichever has fewer strict
    errors (fp + fn); tie -> Sol. An upper bound on any per-passage choice between the two,
    not a system.
  - strict F1 (evalkit.score): Sol alone, Cue (LOLO), oracle; gain = oracle - Sol alone

-> results/headroom_sol_tuning.txt

Usage:
    python scripts/headroom_tuning.py
"""
import csv
import os
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

from evalkit import load_csv, score  # noqa: E402
from llm_frontier import DIR, parse  # noqa: E402
from step4_viterbi import run  # noqa: E402

SOL = os.path.join(DIR, "answers_gpt-6.1-sol", "tuning")


def errors(g, p):
    return len(p - g) + len(g - p)


def main():
    teacher = {r["unit_id"]: min(0.98, max(0.02, float(r["soft_target"]))) for r in
               csv.DictReader(open(os.path.join(HERE, "results/silver/silver_labels.csv"), encoding="utf-8"))}
    order, gold, lectures, tune = run(alphas=("a05",), verbose=False, resume=True, off_override=teacher)
    cue = dict(order)["Viterbi (alpha 0.5)"]
    sol, missing = {}, 0
    for L in ("kn", "en"):
        for v in tune[L]:
            ids = [(r["passage_id"], r["unit_id"]) for r in load_csv(os.path.join(DIR, "tuning", f"{v}_ids.csv"))]
            p, prob = parse(os.path.join(SOL, f"{v}.txt"), ids)
            missing += len(prob["missing"])
            for u in lectures[v]:
                sol[u] = p.get(u, [])                    # missing passage = NONE
    lines = [f"Headroom check, TUNING lectures only: GPT-6.1 Sol vs GitaGround-Cue (v2, LOLO); "
             f"Sol passages missing: {missing}"]
    for L, lname in (("kn", "Kannada"), ("en", "English")):
        vs = tune[L]
        units = [u for v in vs for u in lectures[v]]
        table = {}
        oracle = {}
        for u in units:
            g, s, c = set(gold[u]), set(sol[u]), set(cue[u])
            so = "NONE" if not s else "correct" if s & g else "wrong"
            key = ("on" if g else "off", "on" if c else "off", so)
            table[key] = table.get(key, 0) + 1
            oracle[u] = list(c) if errors(g, c) < errors(g, s) else list(s)
        f_sol, f_cue, f_or = (score(gold, p, lectures, vs) for p in (sol, cue, oracle))
        lines += ["", f"{lname} tuning: {len(vs)} lectures ({', '.join(vs)}), {len(units)} passages, "
                      f"{sum(1 for u in units if not gold[u])} gold off-verse",
                  f"  {'gold':5} {'Cue':5} {'Sol':8} {'passages':>8}"]
        for key in sorted(table):
            lines.append(f"  {key[0]:5} {key[1]:5} {key[2]:8} {table[key]:>8}")
        for name, f in (("Sol alone", f_sol), ("Cue (LOLO)", f_cue), ("oracle Sol/Cue", f_or)):
            lines.append(f"  {name:15} strict F1 {f['F1']:.3f}   tp {f['tp']} fp {f['fp']} fn {f['fn']}")
        lines.append(f"  oracle gain over Sol alone: {f_or['F1'] - f_sol['F1']:+.3f}")
        lines.append("  per lecture, strict F1 Sol / Cue / oracle: " + ", ".join(
            f"{v} {score(gold, sol, lectures, [v])['F1']:.3f} / {score(gold, cue, lectures, [v])['F1']:.3f}"
            f" / {score(gold, oracle, lectures, [v])['F1']:.3f}" for v in vs))
    text = "\n".join(lines)
    print(text)
    open(os.path.join(HERE, "results", "headroom_sol_tuning.txt"), "w").write(text + "\n")


if __name__ == "__main__":
    main()
