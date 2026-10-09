#!/usr/bin/env python3
"""
Gated propagation: anchor with string matching, carry while the unit still
looks like the active verse.

propagate.py ends a carry by DISTANCE (--max-gap). This ends it by CONTENT.
A detected verse becomes active; on each following silent unit the active verse
is emitted only if the gate passes:

    gate(unit, verse) = how strongly this unit's embedding prefers the active
                        verse over the other 71, as a z-score

        z = (sim(unit, verse) - mean over all verses) / std over all verses

The z-score makes the gate RELATIVE: a unit that is vaguely similar to every
verse (which is most lecture text, and why absolute thresholds fire on 69% of
units) scores near 0 and fails.

One threshold tau controls everything:
    tau -> -inf   gate always passes  == unbounded carry
    tau -> +inf   gate never passes   == string matching alone
so sweeping tau traces the precision-recall curve between those two baselines.

--resume chooses what a failed gate does:
    off (default)  the verse is dropped until the next anchor
    on             the verse is withheld on this unit only and stays active,
                   so a speaker who digresses and returns is picked up again

tau is chosen on DEV (max F1). TEST is scored at that one tau; the full test
curve is written for the figure but must never be used to pick tau.

Usage:
    python scripts/embed_scores.py                       # once
    python scripts/gated_propagate.py --language en --base results/final_l1_en.csv
    python scripts/gated_propagate.py --language en --base results/final_l1_en.csv --resume
"""
import argparse
import csv
import os
import sys

import numpy as np

from baselines import carry
from evalkit import (HERE, SPANS, TIERS, UNITS, bootstrap, load_gold, load_pred,
                     read_split, score)

SCORES = os.path.join(HERE, "results", "embed_scores.npz")
TAUS = [round(float(t), 2) for t in np.arange(-1.5, 3.51, 0.25)]


def load_gate(path, rep):
    """Returns gate(unit_id, verse_ref) -> z-score of that verse for that unit."""
    z = np.load(path, allow_pickle=False)
    M = z[rep]
    Z = (M - M.mean(axis=1, keepdims=True)) / M.std(axis=1, keepdims=True)
    row = {u: i for i, u in enumerate(z["unit_ids"])}
    col = {r: i for i, r in enumerate(z["refs"])}

    def gate(u, v):
        return float(Z[row[u], col[v]]) if v in col else float("-inf")
    return gate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--language", choices=["kn", "en"], required=True)
    ap.add_argument("--base", required=True, help="string-matching predictions CSV (the anchors)")
    ap.add_argument("--scores", default=SCORES, help="matrix from embed_scores.py")
    ap.add_argument("--rep", choices=["expanded", "trans"], default="expanded",
                    help="verse representation the gate compares against")
    ap.add_argument("--resume", action="store_true",
                    help="a failed gate withholds the verse instead of dropping it")
    ap.add_argument("--tau", type=float, help="skip tuning and use this threshold")
    ap.add_argument("--spans", default=SPANS)
    ap.add_argument("--units", default=UNITS)
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--out", help="predictions CSV at the chosen tau")
    args = ap.parse_args()

    if not os.path.exists(args.scores):
        sys.exit(f"{args.scores} not found -- run scripts/embed_scores.py first")

    lectures, gold, _ = load_gold(args.spans, args.units)
    dev = [v for v in read_split(args.language, "dev") if v in lectures]
    test = [v for v in read_split(args.language, "test") if v in lectures]
    anchors = load_pred(args.base)
    gate = load_gate(args.scores, args.rep)

    def run(tau):
        pred = {}
        for v in dev + test:
            pred.update(carry(lectures[v], anchors, resume=args.resume,
                              keep=lambda u, verse: gate(u, verse) >= tau))
        return pred

    mode = "resume" if args.resume else "drop"
    print("=" * 78)
    print(f"GATED PROPAGATION — {args.language}   rep={args.rep}   failed gate -> {mode}")
    print("=" * 78)
    print(f"  {'tau':>6} | {'dev P':>6} {'dev R':>6} {'dev F1':>7} | "
          f"{'test P':>6} {'test R':>6} {'test F1':>7} {'fire':>5}")
    print("  " + "-" * 62)

    curve, best = [], None
    for tau in TAUS:
        pred = run(tau)
        d, t = score(gold, pred, lectures, dev), score(gold, pred, lectures, test)
        curve.append((tau, d, t))
        if best is None or d["F1"] > best[1]["F1"]:
            best = (tau, d)
    tau = args.tau if args.tau is not None else best[0]
    for x, d, t in curve:
        mark = "  <- chosen" if x == tau else ""
        print(f"  {x:>6.2f} | {d['P']:>6.3f} {d['R']:>6.3f} {d['F1']:>7.3f} | "
              f"{t['P']:>6.3f} {t['R']:>6.3f} {t['F1']:>7.3f} {t['fired']:>5}{mark}")

    pred = run(tau)
    s = score(gold, pred, lectures, test)
    ci = bootstrap(gold, pred, lectures, test, n=args.n_boot)
    src = "given" if args.tau is not None else f"max dev F1 over {len(dev)} dev lectures"
    print(f"\n  tau = {tau} ({src})")
    print(f"  TEST ({len(test)} lectures, {s['units']} units):")
    for k in ("P", "R", "F1"):
        print(f"    {k:3} {s[k]:.3f}   95% CI [{ci[k][0]:.3f}, {ci[k][1]:.3f}]")
    print(f"    fired on {s['fired']}/{s['units']} units")
    print("    per-tier recall (hit/n):  "
          + "   ".join(f"{t} {h}/{n}" for t, (h, n) in s["tiers"].items()))

    stem = f"gated_{args.language}_{args.rep}_{mode}"
    out = args.out or os.path.join(HERE, "results", f"{stem}.csv")
    video_of = {u: v for v, us in lectures.items() for u in us}
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["unit_id", "video_file", "detected_verses"])
        for v in dev + test:
            for u in lectures[v]:
                w.writerow([u, video_of[u], ";".join(pred[u])])
    curve_out = os.path.join(HERE, "results", f"{stem}_curve.csv")
    with open(curve_out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["tau", "dev_P", "dev_R", "dev_F1", "test_P", "test_R", "test_F1", "test_fired"]
                   + [f"test_{t}" for t in TIERS])
        for x, d, t in curve:
            w.writerow([x] + [f"{d[k]:.4f}" for k in ("P", "R", "F1")]
                       + [f"{t[k]:.4f}" for k in ("P", "R", "F1")] + [t["fired"]]
                       + [f"{h}/{n}" for h, n in (t["tiers"][k] for k in TIERS)])
    print(f"\n  wrote -> {os.path.relpath(out, HERE)}")
    print(f"  wrote -> {os.path.relpath(curve_out, HERE)}   (the PR curve)")


if __name__ == "__main__":
    main()
