#!/usr/bin/env python3
"""
Step 1 diagnostic, TUNING lectures only (no test gold): where GitaGround-Cue and a frontier
LLM disagree, who is right, split by the signal Cue had. Descriptive; spec in docs/experiment_log.md.

  Cue     v2 primary, leave-one-lecture-out on the 9 tuning lectures (as headroom_tuning.py)
  LLMs    Sol (answers_gpt-6.1-sol/tuning), Opus 5.5 chat (tuning/answers),
          Gemini 3.1 Pro API low (answers_api_gemini-3.1-pro-preview_low/tuning)
  disagree  Cue's verse set != the LLM's verse set
  right     the system's set equals the gold set (zero strict errors); in a disagreement at most
            one of the two can be right
  signal rows
    anchor     the string detector fired on the passage (FEATS "anchor", any verse)
    conf>=0.9  no anchor, and Cue's scorer probability for its own label >= 0.9:
               P_off if Cue says off-verse, else (1 - P_off) * Pv(s) / sum Pv  (the decoupled
               emission of the chosen state; the Viterbi decoder has no posterior)
    neither    the rest
  type rows   Cue off / LLM verse;  Cue verse / LLM NONE;  both verses, different sets

-> results/diag_disagreement_tuning.txt

Usage:
    python scripts/diag_disagreement_tuning.py
"""
import csv
import os
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "scripts"))

import numpy as np  # noqa: E402
from evalkit import load_csv  # noqa: E402
from llm_frontier import DIR, parse  # noqa: E402
from step4_viterbi import FEATS, run  # noqa: E402

CONF = 0.9
LLMS = (("Sol", os.path.join(DIR, "answers_gpt-6.1-sol", "tuning")),
        ("Opus", os.path.join(DIR, "tuning", "answers")),
        ("Gemini", os.path.join(DIR, "answers_api_gemini-3.1-pro-preview_low", "tuning")))
SIGNALS = ("anchor fired", f"no anchor, conf>={CONF}", "neither")
TYPES = ("Cue off / LLM verse", "Cue verse / LLM NONE", "both verses, different")


def main():
    teacher = {r["unit_id"]: min(0.98, max(0.02, float(r["soft_target"]))) for r in
               csv.DictReader(open(os.path.join(HERE, "results/silver/silver_labels.csv"), encoding="utf-8"))}
    info = {}
    order, gold, lectures, tune = run(alphas=("a05",), verbose=False, resume=True, off_override=teacher, info=info)
    cue = dict(order)["Viterbi (alpha 0.5)"]
    probs = info["probs_a05"]
    z = np.load(FEATS)
    refs = list(z["refs"])
    row = {u: i for i, u in enumerate(z["unit_ids"])}

    signal, conf = {}, {}
    for L in tune:
        for v in tune[L]:
            pv, po = probs[v]
            for k, u in enumerate(lectures[v]):
                if cue[u]:
                    s = refs.index(cue[u][0])
                    conf[u] = (1 - po[k]) * pv[k, s] / pv[k].sum()
                else:
                    conf[u] = po[k]
                signal[u] = (SIGNALS[0] if z["anchor"][row[u]].any() else
                             SIGNALS[1] if conf[u] >= CONF else SIGNALS[2])

    lines = [f"Step 1 diagnostic, TUNING lectures only: Cue (v2, LOLO) vs each frontier LLM where their verse "
             f"sets differ; right = set equals gold. Cells: passages / Cue right / LLM right / both wrong."]
    for name, folder in LLMS:
        llm, missing = {}, 0
        for L in tune:
            for v in tune[L]:
                ids = [(r["passage_id"], r["unit_id"]) for r in load_csv(os.path.join(DIR, "tuning", f"{v}_ids.csv"))]
                p, prob = parse(os.path.join(folder, f"{v}.txt"), ids)
                missing += len(prob["missing"])
                for u in lectures[v]:
                    llm[u] = p.get(u, [])
        lines += ["", f"=== {name} ({os.path.relpath(folder, DIR)}; {missing} passages missing = NONE) ==="]
        for L, lname in (("kn", "Kannada"), ("en", "English")):
            units = [u for v in tune[L] for u in lectures[v]]
            agree = [u for u in units if set(cue[u]) == set(llm[u])]
            dis = [u for u in units if set(cue[u]) != set(llm[u])]

            def cell(us):
                c = sum(1 for u in us if set(cue[u]) == set(gold[u]))
                m = sum(1 for u in us if set(llm[u]) == set(gold[u]))
                return f"{len(us):>5} {c:>6} {m:>6} {len(us) - c - m:>6}"

            def kind(u):
                return TYPES[0] if not cue[u] else TYPES[1] if not llm[u] else TYPES[2]

            lines += ["", f"  {lname}: {len(units)} passages; agree {len(agree)} "
                          f"({sum(1 for u in agree if set(cue[u]) == set(gold[u]))} right); disagree {len(dis)}",
                      f"    {'':34}{'psg':>5} {'Cue ok':>6} {'LLM ok':>6} {'both x':>6}"]
            for s in SIGNALS:
                lines.append(f"    {s:34}" + cell([u for u in dis if signal[u] == s]))
            lines.append(f"    {'all disagreements':34}" + cell(dis))
            lines.append("    by type:")
            for t in TYPES:
                lines.append(f"    {t:34}" + cell([u for u in dis if kind(u) == t]))
            lines.append("    signal x type:")
            for s in SIGNALS:
                for t in TYPES:
                    us = [u for u in dis if signal[u] == s and kind(u) == t]
                    if us:
                        lines.append(f"    {s[:14] + ' | ' + t:34}" + cell(us))
    text = "\n".join(lines)
    print(text)
    with open(os.path.join(HERE, "results", "diag_disagreement_tuning.txt"), "w", encoding="utf-8") as f:
        f.write(text + "\n")


if __name__ == "__main__":
    main()
