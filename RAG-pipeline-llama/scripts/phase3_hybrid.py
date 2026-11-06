#!/usr/bin/env python3
"""
Phase 3 v3 hybrid: the LLM proposes the verse, the decoder keeps the structure.

The v2 primary system (teacher P_off, 145-state resume decoder) with one more
piece of evidence: gpt-5-mini's end-to-end labels (llm_e2e.py, same prompt and
effort as before) as a verse vote. bonus_share multiplies the voted verses'
verse-scorer value by e^lambda before decoupling (total verse mass stays 1 - P_off,
but the voted verse becomes more peaked, so the path can also leave off-verse). lambda is chosen with gamma, beta_off and tau by leave-one-lecture-out
on the 9 TUNING lectures (lambda = 0 = ignore the vote). No test gold.

  (no flag)  LOLO report on the tuning lectures, lambda fixed 0 vs chosen
  --save     also fit on all 9 tuning lectures and write results/v3_hybrid145.pkl

Usage:
    python scripts/phase3_hybrid.py
    python scripts/phase3_hybrid.py --save
"""
import argparse
import csv
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

from evalkit import load_csv, score  # noqa: E402
from llm_e2e import REVIEW  # noqa: E402
from llm_pilot import parse  # noqa: E402
from step4_viterbi import FEATS, near_f1, run, theme_f1  # noqa: E402

LLM = "gpt-5-mini"
OUT = os.path.join(HERE, "results", "v3_hybrid145.pkl")


def llm_votes():
    """{unit_id: [verse index]} from every lecture gpt-5-mini answered (tuning + test + held-out)"""
    refs = list(np.load(FEATS)["refs"])
    folder = os.path.join(REVIEW, LLM)
    votes = {}
    for f in sorted(os.listdir(folder)):
        if not f.endswith("_ids.csv"):
            continue
        v = f[:-len("_ids.csv")]
        if not os.path.exists(os.path.join(folder, f"{v}.txt")):
            continue
        ids = [(r["passage_id"], r["unit_id"]) for r in load_csv(os.path.join(folder, f))]
        p = parse(os.path.join(folder, f"{v}.txt"), ids)
        p = p[0] if isinstance(p, tuple) else p
        for _, u in ids:
            votes[u] = [refs.index(r) for r in p.get(u, []) if r in refs]
    return votes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--save", action="store_true")
    args = ap.parse_args()
    teacher = {r["unit_id"]: min(0.98, max(0.02, float(r["soft_target"]))) for r in
               csv.DictReader(open(os.path.join(HERE, "results/silver/silver_labels.csv"), encoding="utf-8"))}
    votes = llm_votes()
    lines = [f"Phase 3 v3 hybrid — TUNING lectures, leave-one-lecture-out, alpha 0.5 (no test gold)",
             f"verse vote: {LLM} end-to-end labels on {len(votes)} passages; P_off = teacher; 145-state resume",
             f"  {'system':28} {'KN strict':>9} {'KN +-1':>7} {'EN strict':>9} {'EN +-1':>7} {'EN theme':>8}"
             f"   EN strict per lecture"]
    for name, kw in (("v2 teacher 145 (lambda 0)", {}), ("v3 hybrid (lambda chosen)", {"verse_bonus": votes})):
        info = {}
        order, gold, lectures, tune = run(alphas=("a05",), verbose=False, resume=True, off_override=teacher,
                                          info=info, save=OUT if (args.save and kw) else None, **kw)
        pred = dict(order)["Viterbi (alpha 0.5)"]
        kn = (score(gold, pred, lectures, tune["kn"])["F1"], near_f1(gold, pred, lectures, tune["kn"])[2])
        en = (score(gold, pred, lectures, tune["en"])["F1"], near_f1(gold, pred, lectures, tune["en"])[2],
              theme_f1(gold, pred, lectures, tune["en"]))
        per = ", ".join(f"{v} {score(gold, pred, lectures, [v])['F1']:.3f}" for v in tune["en"])
        lines.append(f"  {name:28} {kn[0]:>9.3f} {kn[1]:>7.3f} {en[0]:>9.3f} {en[1]:>7.3f} {en[2]:>8.3f}   {per}")
        if kw:
            lines.append("  lambda chosen per held-out lecture: "
                         + ", ".join(f"{c[0]} {c[-1]}" for c in info["chosen"]))
            kn_per = ", ".join(f"{v} {score(gold, pred, lectures, [v])['F1']:.3f}" for v in tune["kn"])
            lines.append(f"  KN strict per lecture: {kn_per}")
    out = "\n".join(lines)
    print(out)
    open(os.path.join(HERE, "results", "phase3_hybrid_tuning.txt"), "w").write(out + "\n")


if __name__ == "__main__":
    main()
