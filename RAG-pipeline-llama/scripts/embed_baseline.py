#!/usr/bin/env python3
"""
Embedding baseline with a proper abstention rule.

The frozen embedding row used an absolute cosine threshold and fired on 69-77%
of units whatever they contained. That is a thresholding failure, not evidence
that embeddings fail: BGE-M3 cosines between any two on-topic texts sit in a
narrow high band, so an absolute cut barely separates "this verse" from "some
Gita-flavoured talk".

This predicts the top-1 verse and abstains by one of three criteria:

  abs     top-1 similarity >= t                  (the old rule, for comparison)
  margin  top-1 minus top-2 similarity >= t      (is one verse clearly ahead?)
  z       z-score of top-1 among all 72 >= t     (does it stand out at all?)

t is chosen on DEV (max F1) and TEST is scored once at that t. The fire rate is
reported overall and on gold-empty units, which is the number that shows
whether the detector can tell "no verse here".

Usage:
    python scripts/embed_scores.py                       # once
    python scripts/embed_baseline.py --language en --criterion margin
    python scripts/embed_baseline.py --language kn --criterion z --rep trans
"""
import argparse
import csv
import os
import sys

import numpy as np

from evalkit import HERE, SPANS, UNITS, bootstrap, load_gold, read_split, score

SCORES = os.path.join(HERE, "results", "embed_scores.npz")


def statistic(M, criterion):
    """One abstention score per unit (row of M)."""
    top = np.sort(M, axis=1)
    if criterion == "abs":
        return top[:, -1]
    if criterion == "margin":
        return top[:, -1] - top[:, -2]
    return (top[:, -1] - M.mean(axis=1)) / M.std(axis=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--language", choices=["kn", "en"], required=True)
    ap.add_argument("--criterion", choices=["abs", "margin", "z"], default="margin")
    ap.add_argument("--rep", choices=["expanded", "trans"], default="expanded")
    ap.add_argument("--scores", default=SCORES)
    ap.add_argument("--spans", default=SPANS)
    ap.add_argument("--units", default=UNITS)
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--out", help="predictions CSV at the chosen threshold")
    args = ap.parse_args()

    if not os.path.exists(args.scores):
        sys.exit(f"{args.scores} not found -- run scripts/embed_scores.py first")

    lectures, gold, _ = load_gold(args.spans, args.units)
    dev = [v for v in read_split(args.language, "dev") if v in lectures]
    test = [v for v in read_split(args.language, "test") if v in lectures]

    z = np.load(args.scores, allow_pickle=False)
    M, refs = z[args.rep], list(z["refs"])
    row = {u: i for i, u in enumerate(z["unit_ids"])}
    stat, top1 = statistic(M, args.criterion), M.argmax(axis=1)

    def run(t):
        return {u: ([refs[top1[row[u]]]] if stat[row[u]] >= t else [])
                for v in dev + test for u in lectures[v]}

    # candidate thresholds: every 2nd percentile of the statistic on dev
    dev_vals = np.array([stat[row[u]] for v in dev for u in lectures[v]])
    best = None
    for t in np.percentile(dev_vals, np.arange(0, 100, 2)):
        f1 = score(gold, run(t), lectures, dev)["F1"]
        if best is None or f1 > best[1]:
            best = (float(t), f1)
    t = best[0]

    pred = run(t)
    s = score(gold, pred, lectures, test)
    ci = bootstrap(gold, pred, lectures, test, n=args.n_boot)
    empty = [u for v in test for u in lectures[v] if not gold[u]]
    fired_empty = sum(1 for u in empty if pred[u])

    print("=" * 70)
    print(f"EMBEDDING BASELINE — {args.language}   rep={args.rep}   criterion={args.criterion}")
    print("=" * 70)
    print(f"  threshold {t:.4f}  (max dev F1 = {best[1]:.3f} over {len(dev)} dev lectures)")
    print(f"  TEST ({len(test)} lectures, {s['units']} units):")
    for k in ("P", "R", "F1"):
        print(f"    {k:3} {s[k]:.3f}   95% CI [{ci[k][0]:.3f}, {ci[k][1]:.3f}]")
    print(f"    fire rate          {s['fired']}/{s['units']} ({s['fired'] / s['units']:.1%})")
    print(f"    fire rate on empty {fired_empty}/{len(empty)} "
          f"({fired_empty / max(len(empty), 1):.1%})")
    print("    per-tier recall (hit/n):  "
          + "   ".join(f"{k} {h}/{n}" for k, (h, n) in s["tiers"].items()))

    out = args.out or os.path.join(
        HERE, "results", f"embed_{args.language}_{args.rep}_{args.criterion}.csv")
    video_of = {u: v for v, us in lectures.items() for u in us}
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["unit_id", "video_file", "detected_verses"])
        for v in dev + test:
            for u in lectures[v]:
                w.writerow([u, video_of[u], ";".join(pred[u])])
    print(f"\n  wrote -> {os.path.relpath(out, HERE)}")


if __name__ == "__main__":
    main()
