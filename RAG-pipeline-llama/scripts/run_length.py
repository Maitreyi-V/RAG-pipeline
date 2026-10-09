#!/usr/bin/env python3
"""
Run-length analysis: how long does a verse actually stay "active"?

The old motivation for propagation was "consecutive grounded units share a verse
92% / 89% of the time". That is conditioned on BOTH units grounding something,
so it says nothing about carrying a verse across units that ground nothing.
This script measures what a carry policy really needs, from gold labels only:

  1. SURVIVAL   P(unit t+k grounds verse v | unit t grounds v), for each k.
                Unconditioned on anything in between -- empties count as misses.
  2. CARRY PRECISION (needs --base)
                Start from a unit where the base detector fired verse v and
                walk forward until the detector fires again. At distance k, how
                often is v really in gold? This is exactly the precision of a
                label carried k units, i.e. the curve a max-gap policy rides.
  3. RUN ENDINGS  For each maximal run of consecutive units grounding v:
                how long is it, and what comes next -- another verse, an empty
                unit, or the end of the lecture?

Usage:
    python scripts/run_length.py --language en
    python scripts/run_length.py --language en --split dev --base results/final_l1_en.csv
"""
import argparse
import csv
import os
from collections import Counter

from evalkit import HERE, SPANS, UNITS, load_gold, load_pred, read_split

KS = [1, 2, 3, 4, 5, 6, 8, 10, 15, 20, 30, 40, 50]


def survival(lectures, gold, videos):
    """{k: (still_grounding_v, total)} from every grounded (unit, verse)."""
    out = {k: [0, 0] for k in KS}
    for v in videos:
        us = lectures[v]
        for i, u in enumerate(us):
            for verse in gold[u]:
                for k in KS:
                    if i + k < len(us):
                        out[k][1] += 1
                        out[k][0] += verse in gold[us[i + k]]
    return out


def carry_precision(lectures, gold, pred, videos):
    """{k: (correct, total)} for a detected verse carried k silent units."""
    out = {k: [0, 0] for k in KS}
    for v in videos:
        us = lectures[v]
        for i, u in enumerate(us):
            for verse in pred.get(u, ()):
                for k in KS:
                    j = i + k
                    if j >= len(us) or any(pred.get(us[m]) for m in range(i + 1, j + 1)):
                        break               # lecture ended, or a new anchor took over
                    out[k][1] += 1
                    out[k][0] += verse in gold[us[j]]
    return out


def runs(lectures, gold, videos):
    """Maximal runs of consecutive units grounding the same verse."""
    lengths, endings = [], Counter()
    for v in videos:
        us = lectures[v]
        for i, u in enumerate(us):
            for verse in gold[u]:
                if i > 0 and verse in gold[us[i - 1]]:
                    continue                # not the start of a run
                j = i
                while j + 1 < len(us) and verse in gold[us[j + 1]]:
                    j += 1
                lengths.append(j - i + 1)
                if j + 1 == len(us):
                    endings["lecture ends"] += 1
                elif gold[us[j + 1]]:
                    endings["another verse"] += 1
                else:
                    endings["empty unit"] += 1
    return lengths, endings


def empty_stretches(lectures, gold, videos):
    """Lengths of runs of consecutive no-verse units."""
    out = []
    for v in videos:
        n = 0
        for u in lectures[v]:
            if gold[u]:
                if n:
                    out.append(n)
                n = 0
            else:
                n += 1
        if n:
            out.append(n)
    return sorted(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--language", choices=["kn", "en"], required=True)
    ap.add_argument("--split", choices=["dev", "test", "all"], default="all",
                    help="use 'dev' for anything a method will be tuned on")
    ap.add_argument("--base", help="predictions CSV; adds the carry-precision curve")
    ap.add_argument("--spans", default=SPANS)
    ap.add_argument("--units", default=UNITS)
    ap.add_argument("--out", help="CSV of the curves (default results/run_length_<lang>_<split>.csv)")
    args = ap.parse_args()

    lectures, gold, _ = load_gold(args.spans, args.units)
    videos = [v for v in read_split(args.language, args.split) if v in lectures]
    n_units = sum(len(lectures[v]) for v in videos)
    n_empty = sum(1 for v in videos for u in lectures[v] if not gold[u])

    print("=" * 70)
    print(f"RUN LENGTH — {args.language} / {args.split}")
    print("=" * 70)
    print(f"  lectures {len(videos)}   units {n_units}   "
          f"mean units/lecture {n_units / len(videos):.1f}   "
          f"empty units {n_empty} ({n_empty / n_units:.1%})")

    surv = survival(lectures, gold, videos)
    carry = carry_precision(lectures, gold, load_pred(args.base), videos) if args.base else None

    print(f"\n  {'k':>4} {'survival':>10} {'n':>6}" + (f" {'carry prec':>12} {'n':>6}" if carry else ""))
    print("  " + "-" * (22 + (20 if carry else 0)))
    table = []
    for k in KS:
        a, b = surv[k]
        if not b:
            continue
        line = f"  {k:>4} {a / b:>10.3f} {b:>6}"
        row = [k, f"{a / b:.4f}", b]
        if carry:
            c, d = carry[k]
            line += f" {c / d:>12.3f} {d:>6}" if d else f" {'-':>12} {0:>6}"
            row += [f"{c / d:.4f}" if d else "", d]
        print(line)
        table.append(row)

    lengths, endings = runs(lectures, gold, videos)
    lengths.sort()
    print(f"\n  verse runs: {len(lengths)}   median length {lengths[len(lengths) // 2]}   "
          f"mean {sum(lengths) / len(lengths):.1f}   max {lengths[-1]}")
    print(f"  runs of length 1: {sum(1 for x in lengths if x == 1) / len(lengths):.1%}")
    print("  what follows a run:")
    for k, c in endings.most_common():
        print(f"    {k:14} {c:4}  ({c / len(lengths):.1%})")

    gaps = empty_stretches(lectures, gold, videos)
    if gaps:
        print(f"\n  empty stretches: {len(gaps)}   median {gaps[len(gaps) // 2]}   "
              f"max {gaps[-1]}   of length >= 5: {sum(1 for g in gaps if g >= 5)}")
    else:
        print("\n  empty stretches: none")

    out = args.out or os.path.join(HERE, "results", f"run_length_{args.language}_{args.split}.csv")
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["k", "survival", "n"] + (["carry_precision", "n_carry"] if carry else []))
        w.writerows(table)
    print(f"\n  wrote -> {os.path.relpath(out, HERE)}")


if __name__ == "__main__":
    main()
