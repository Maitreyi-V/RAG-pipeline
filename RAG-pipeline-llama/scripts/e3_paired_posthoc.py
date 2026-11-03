#!/usr/bin/env python3
"""
POST-HOC analysis (not pre-registered): paired lecture-bootstrap tests on the
FROZEN E3 (unseen-speaker) outputs. Nothing is re-run or re-tuned; the systems
are the frozen files results/step4_locked.pkl (GitaGround-Local) and
results/v2_teacher145.pkl (GitaGround-Cue), as in freeze_grounding.py --v2.

For pooled new Kannada (9 lectures) and pooled new English (3 lectures):
each GitaGround system minus Plain Carry and minus Best-Anchor Carry, strict
F1 difference with a 95% paired lecture-bootstrap CI (10,000 resamples, seed 0)
and per-lecture wins / ties / losses (freeze_grounding.paired).

Writes results/e3_paired_posthoc.txt.

Usage:
    python scripts/e3_paired_posthoc.py
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

from baselines import carry  # noqa: E402
from evalkit import load_gold, read_split  # noqa: E402
from freeze_grounding import f1_strict, paired, strict_counts  # noqa: E402
from step4_viterbi import FEATS, OFF, OUT, predict_locked  # noqa: E402


def main():
    lectures, gold, _ = load_gold()
    z = np.load(FEATS)
    refs = list(z["refs"])
    row = {u: i for i, u in enumerate(z["unit_ids"])}
    anchors = {u: [refs[j] for j in np.flatnonzero(z["anchor"][row[u]])] for u in row}
    best = {u: ([max(a, key=lambda r: z["string"][row[u], refs.index(r)])] if a else [])
            for u, a in anchors.items()}
    groups = [("New Kannada (pooled)", [v for v in read_split("kn", "heldout") if v in lectures]),
              ("New English (pooled)", [v for v in read_split("en", "heldout") if v in lectures])]
    scope = [v for _, vs in groups for v in vs]

    def decoded(path):
        dec = predict_locked({v: lectures[v] for v in scope}, "a05", path=path)
        return {u: ([] if st == OFF else [refs[st]]) + [refs[j] for j in extra]
                for v in scope for u, (st, extra, _, _) in zip(lectures[v], dec[v])}
    systems = {"GitaGround-Local": decoded(OUT),
               "GitaGround-Cue": decoded(os.path.join(HERE, "results", "v2_teacher145.pkl"))}
    baselines = {"Plain Carry": {}, "Best-Anchor Carry": {}}
    for v in scope:
        baselines["Plain Carry"].update(carry(lectures[v], anchors))
        baselines["Best-Anchor Carry"].update(carry(lectures[v], best))

    lines = ["POST-HOC (not pre-registered): paired lecture bootstrap on the frozen E3 outputs",
             "strict F1 difference A - B, 95% CI from 10,000 paired lecture resamples (seed 0); W/T/L per lecture",
             f"  {'group':22} {'A':18} {'B':18} {'F1 A':>6} {'F1 B':>6} {'diff':>7} {'95% CI':>17} {'W/T/L':>7}"]
    for gname, vs in groups:
        for a, pa in systems.items():
            for b, pb in baselines.items():
                fa = f1_strict([sum(x) for x in zip(*(strict_counts(gold, pa, lectures[v]) for v in vs))])
                fb = f1_strict([sum(x) for x in zip(*(strict_counts(gold, pb, lectures[v]) for v in vs))])
                d, ci, _, (w, t, lo), _ = paired(gold, lectures, pa, pb, vs, 10000)
                lines.append(f"  {gname:22} {a:18} {b:18} {fa:>6.3f} {fb:>6.3f} {d:>+7.3f} "
                             f"[{ci[0]:+.3f},{ci[1]:+.3f}] {f'{w}/{t}/{lo}':>7}")
    out = "\n".join(lines)
    print(out)
    open(os.path.join(HERE, "results", "e3_paired_posthoc.txt"), "w").write(out + "\n")


if __name__ == "__main__":
    main()
