#!/usr/bin/env python3
"""
Freeze run for E4: does the system work on verses it was never trained on?

The locked system is re-fit on the 9 tuning lectures using ONLY verses
2.1 .. 2.36 (step4_viterbi.fit_restricted: scorer weights and transition
counts; gamma, beta_off, tau kept). Both that model and the full locked model
then decode the lectures, and are scored on the passages whose GOLD labels
include a verse in 2.37 .. 2.72.

  --split dev       DRY RUN on the tuning lectures (in-sample; few such passages)
  --split test / heldout   the single freeze run

Writes results/freeze_e4_<split>.txt.

Usage:
    python scripts/freeze_e4.py --split dev
"""
import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

from evalkit import _prf, load_gold, read_split, verse_number  # noqa: E402
from freeze_grounding import (f1_near, f1_strict, near_counts, strict_counts,  # noqa: E402
                              theme_counts, with_ci)
from step4_viterbi import FEATS, OFF, fit_restricted, predict_locked  # noqa: E402

SPLIT_AT = 36
RES = os.path.join(HERE, "results")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["dev", "test", "heldout"], required=True)
    ap.add_argument("--n-boot", type=int, default=1000)
    args = ap.parse_args()

    lectures, gold, _ = load_gold()
    refs = list(np.load(FEATS)["refs"])
    e4 = fit_restricted(SPLIT_AT)
    lines = [f"FREEZE RUN — E4 unseen verses, split = {args.split}"
             + ("   (DRY RUN: IN-SAMPLE, not a result)" if args.split == "dev" else ""),
             f"E4 model: {e4['trained_on']}; transition counts {e4['counts']}",
             f"Scored on passages whose gold includes a verse in 2.{SPLIT_AT + 1}-2.72."]

    for L, lname in (("kn", "Kannada"), ("en", "English")):
        vids = [v for v in read_split(L, args.split) if v in lectures]
        unseen = {v: [u for u in lectures[v]
                      if any((verse_number(r) or 0) > SPLIT_AT for r in gold[u])] for v in vids}
        vids = [v for v in vids if unseen[v]]
        n = sum(len(unseen[v]) for v in vids)
        lines += ["", f"{lname}: {n} passages in {len(vids)} lectures"]
        if not n:
            continue
        lines.append(f"  {'model':28} {'P':>6} {'R':>6}  {'strict F1 [CI]':22} {'+-1 F1 [CI]':22} "
                     f"{'theme F1 [CI]':22}")
        for name, model in (("trained on 2.1-2.36 (E4)", e4), ("locked, all verses", None)):
            dec = predict_locked({v: lectures[v] for v in vids}, model=model)
            pred = {u: ([] if st == OFF else [refs[st]]) + [refs[j] for j in extra]
                    for v in vids for u, (st, extra, _, _) in zip(lectures[v], dec[v])}
            tp, fp, fn = [sum(x) for x in zip(*(strict_counts(gold, pred, unseen[v]) for v in vids))]
            P, R, _ = _prf(tp, fp, fn)
            cells = []
            for counts, f1 in ((strict_counts, f1_strict), (near_counts, f1_near),
                               (theme_counts, f1_strict)):
                point, ci = with_ci([counts(gold, pred, unseen[v]) for v in vids], f1, args.n_boot)
                cells.append(f"{point:.3f}" + (f" [{ci[0]:.3f},{ci[1]:.3f}]" if ci else ""))
            lines.append(f"  {name:28} {P:>6.3f} {R:>6.3f}  {cells[0]:22} {cells[1]:22} {cells[2]:22}")

    text = "\n".join(lines)
    print(text)
    out = os.path.join(RES, f"freeze_e4_{args.split}.txt")
    with open(out, "w", encoding="utf-8") as f:
        f.write(text + "\n")
    print(f"\n  wrote -> {os.path.relpath(out, HERE)}")


if __name__ == "__main__":
    main()
