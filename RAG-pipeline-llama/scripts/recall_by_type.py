#!/usr/bin/env python3
"""
Recall by mention type on the 42 lectures, from stored outputs only (analysis spec in
docs/experiment_log.md, 7d7799f). Descriptive: nothing is designed or chosen from it.

  systems  Gemini 3.1 Pro (API, low), GPT-6.1 Sol, Claude Opus 5.5 (single chat run),
           GitaGround-Cue (v2 primary), GitaGround-Local, Plain Carry
  splits   E2 Kannada test, E2 English test, E3 Kannada, E3 English
  main     recitation (T1+T2) vs non-recitation (T3+T4+T5): strict recall, 95% lecture CI
  appendix T1..T5 strict recall with n; n < 20 marked

-> results/recall_by_type.txt

Usage:
    python scripts/recall_by_type.py
"""
import csv
import os
import random
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, "scripts"))
sys.path.insert(0, HERE)

import numpy as np  # noqa: E402
from baselines import carry  # noqa: E402
from evalkit import load_gold, read_split  # noqa: E402
from freeze_grounding import MIN_LECTURES_FOR_CI  # noqa: E402
from score_by_tier_units import TIERS  # noqa: E402
from step4_viterbi import FEATS, OFF, OUT, predict_locked  # noqa: E402

N_BOOT = 10000
GROUPS = (("recitation T1+T2", ("T1", "T2")), ("non-recitation T3+T4+T5", ("T3", "T4", "T5")))
LLM_CSV = (("Gemini 3.1 Pro (API, low)", "llm_frontier_gemini31pro_api_low.csv"),
           ("GPT-6.1 Sol (API, high)", "llm_frontier_gpt61sol.csv"),
           ("Claude Opus 5.5 (chat, high)", "llm_frontier_opus55.csv"))


def read_llm(name):
    with open(os.path.join(HERE, "results", name), encoding="utf-8") as f:
        return {r["unit_id"]: [x for x in r["refs"].split(";") if x] for r in csv.DictReader(f)}


def hits(gold, pred, units, tiers):
    """(hit, n) over gold (passage, verse) pairs whose type is in tiers"""
    h = n = 0
    for u in units:
        p = set(pred.get(u, ()))
        for v, t in gold[u].items():
            if t in tiers:
                n += 1
                h += v in p
    return h, n


def recall_ci(gold, pred, lectures, vids, tiers):
    per = [hits(gold, pred, lectures[v], tiers) for v in vids]
    h, n = (sum(x) for x in zip(*per))
    point = h / n if n else float("nan")
    if len(vids) < MIN_LECTURES_FOR_CI:
        return point, n, None
    rng = random.Random(0)
    draws = []
    for _ in range(N_BOOT):
        s = [per[rng.randrange(len(per))] for _ in per]
        sh, sn = sum(x[0] for x in s), sum(x[1] for x in s)
        draws.append(sh / sn if sn else 0.0)
    draws.sort()
    return point, n, (draws[int(0.025 * N_BOOT)], draws[min(int(0.975 * N_BOOT), N_BOOT - 1)])


def main():
    lectures, gold, _ = load_gold()
    splits = [("E2 Kannada test", read_split("kn", "test")), ("E2 English test", read_split("en", "test")),
              ("E3 Kannada", read_split("kn", "heldout")), ("E3 English", read_split("en", "heldout"))]
    vids = [v for _, vs in splits for v in vs]
    z = np.load(FEATS)
    refs = list(z["refs"])
    row = {u: i for i, u in enumerate(z["unit_ids"])}
    anchors = {u: [refs[j] for j in np.flatnonzero(z["anchor"][row[u]])] for u in row}

    def decoded(path):
        dec = predict_locked({v: lectures[v] for v in vids}, "a05", path=path)
        return {u: ([] if st == OFF else [refs[st]]) + [refs[j] for j in extra]
                for v in vids for u, (st, extra, _, _) in zip(lectures[v], dec[v])}

    systems = {name: read_llm(f) for name, f in LLM_CSV}
    systems["GitaGround-Cue"] = decoded(os.path.join(HERE, "results", "v2_teacher145.pkl"))
    systems["GitaGround-Local"] = decoded(OUT)
    systems["Plain Carry"] = {}
    for v in vids:
        systems["Plain Carry"].update(carry(lectures[v], anchors))

    lines = [f"Recall by mention type, strict (gold pair hit), 42 lectures; 95% CI resampling lectures "
             f"({N_BOOT} resamples, seed 0), none under {MIN_LECTURES_FOR_CI} lectures. Descriptive only."]
    lines += ["", "MAIN: recitation (T1+T2) vs non-recitation (T3+T4+T5)"]
    for sname, vs in splits:
        units = [u for v in vs for u in lectures[v]]
        n = {g: hits(gold, {}, units, ts)[1] for g, ts in GROUPS}
        lines += ["", f"{sname}: {len(vs)} lectures; mentions: recitation {n[GROUPS[0][0]]}, non-recitation "
                      f"{n[GROUPS[1][0]]} ({n[GROUPS[1][0]] / sum(n.values()):.0%} non-recitation)",
                  f"  {'system':30}" + "".join(f"{g:>30}" for g, _ in GROUPS)]
        for name, pred in systems.items():
            cells = []
            for _, ts in GROUPS:
                pt, _, ci = recall_ci(gold, pred, lectures, vs, ts)
                cells.append(f"{pt:.3f}" + (f" [{ci[0]:.3f},{ci[1]:.3f}]" if ci else ""))
            lines.append(f"  {name:30}" + "".join(f"{c:>30}" for c in cells))
    lines += ["", "APPENDIX: each type, strict recall (n per cell; * = n < 20, not interpreted)"]
    for sname, vs in splits:
        units = [u for v in vs for u in lectures[v]]
        n = {t: hits(gold, {}, units, (t,))[1] for t in TIERS}
        lines += ["", f"{sname}: " + ", ".join(f"{t} n={n[t]}{'*' if n[t] < 20 else ''}" for t in TIERS),
                  f"  {'system':30}" + "".join(f"{t:>8}" for t in TIERS)]
        for name, pred in systems.items():
            cells = []
            for t in TIERS:
                h, m = hits(gold, pred, units, (t,))
                cells.append("-" if not m else f"{h / m:.2f}{'*' if m < 20 else ''}")
            lines.append(f"  {name:30}" + "".join(f"{c:>8}" for c in cells))
    text = "\n".join(lines)
    print(text)
    with open(os.path.join(HERE, "results", "recall_by_type.txt"), "w", encoding="utf-8") as f:
        f.write(text + "\n")


if __name__ == "__main__":
    main()
