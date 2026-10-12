#!/usr/bin/env python3
"""
Where do the errors of plain carry come from?

Runs unbounded carry (keep the last string-match detection until the next one)
and sorts every wrong (unit, verse) pair into one cause, per speaker:

  false positives
    carried into empty     the unit grounds nothing; the verse was carried in
    carried past a switch  the unit grounds a different verse; no anchor fired
    wrong anchor           string matching itself fired the wrong verse here
  false negatives
    no anchor yet          nothing is active: no detection so far in the lecture
    stale verse            another verse is active (the other side of a missed switch)
    extra verse            the right verse is active too, but the unit grounds more than one

"carried into empty" is what a departure detector would fix; "carried past a
switch", "stale verse" and "no anchor yet" are what a better anchor would fix.

Usage:
    python scripts/error_breakdown.py --language en
"""
import argparse
import os
from collections import Counter

from baselines import carry
from evalkit import HERE, LECTURES_NEW, load_csv, load_gold, load_pred, read_split

RES = os.path.join(HERE, "results")
FP = ["carried into empty", "carried past a switch", "wrong anchor"]
FN = ["no anchor yet", "stale verse", "extra verse"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--language", choices=["kn", "en"], required=True)
    args = ap.parse_args()
    L = args.language

    lectures, gold, _ = load_gold()
    anchors = load_pred(os.path.join(RES, "results_v3", f"final_l1_{L}.csv"))
    anchors.update(load_pred(os.path.join(RES, f"heldout_l1_{L}.csv")))
    speaker = {r["video_file"]: r["speaker"] for r in load_csv(LECTURES_NEW)}

    held = [v for v in read_split(L, "heldout") if v in lectures]
    groups = [("original test set", read_split(L, "test"))]
    for sp in sorted({speaker[v] for v in held}):
        groups.append((sp, [v for v in held if speaker[v] == sp]))

    print("=" * 96)
    print(f"ERROR BREAKDOWN, UNBOUNDED CARRY — {L}")
    print("=" * 96)
    for name, videos in groups:
        c, tp = Counter(), 0
        for v in videos:
            us = lectures[v]
            pred = carry(us, anchors)
            for u in us:
                g, p, fired = set(gold[u]), set(pred[u]), set(anchors.get(u, ()))
                tp += len(g & p)
                for verse in p - g:
                    if verse in fired:
                        c["wrong anchor"] += 1
                    elif not g:
                        c["carried into empty"] += 1
                    else:
                        c["carried past a switch"] += 1
                for verse in g - p:
                    if not p:
                        c["no anchor yet"] += 1
                    elif p & g:
                        c["extra verse"] += 1
                    else:
                        c["stale verse"] += 1
        n_fp, n_fn = sum(c[k] for k in FP), sum(c[k] for k in FN)
        print(f"\n  {name}   correct {tp}   false positives {n_fp}   false negatives {n_fn}")
        for label, keys, tot in (("FP", FP, n_fp), ("FN", FN, n_fn)):
            for k in keys:
                share = f"{c[k] / tot:.0%}" if tot else "-"
                print(f"    {label}  {k:24} {c[k]:4}  {share:>5}")


if __name__ == "__main__":
    main()
