#!/usr/bin/env python3
"""
Inter-annotator agreement on UNIT-level verse-grounding annotation.

Supersedes compute_iaa.py, which was keyed on (video_file, verse_ref) from the
old verification workflow. This one reads the single append-only file written by
annotate_app.py and keys on unit_id.

Reports two agreement numbers:

  1. MENTION TYPE  — 6-way label per unit: NONE, T1..T5.
                     "Can two people agree on whether a verse is grounded here,
                     and how?" This is the headline kappa for the paper.
  2. VERSE IDENTITY — restricted to units where BOTH annotators said a verse is
                     present: did they pick the same verse?

Re-labelling is handled: the file is append-only, so for each (unit, annotator)
only the rows from that pair's most recent created_at are used.

Usage:
    python scripts/compute_iaa_units.py
    python scripts/compute_iaa_units.py --spans data/mention_spans.csv
    python scripts/compute_iaa_units.py --a "Meghana G" --b "Meghana S"
    python scripts/compute_iaa_units.py --video gita-1st_video
"""
import argparse
import csv
import os
import sys
from collections import defaultdict

LABELS = ["NONE", "T1", "T2", "T3", "T4", "T5"]
# Guideline tie-break rule 1: a unit takes its highest-fidelity form present.
FIDELITY = {"T1": 0, "T2": 1, "T3": 2, "T4": 3, "T5": 4, "NONE": 5}

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_SPANS = os.path.join(HERE, "data", "mention_spans.csv")


def load_rows(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def latest_per_pair(rows):
    """Append-only: keep only rows from the newest labelling event per (unit, annotator)."""
    newest = {}
    for r in rows:
        k = (r["unit_id"], r["annotator"])
        ts = r.get("created_at", "")
        if k not in newest or ts > newest[k]:
            newest[k] = ts
    return [r for r in rows if r.get("created_at", "") == newest[(r["unit_id"], r["annotator"])]]


def per_unit_view(rows, video=None):
    """-> {annotator: {unit_id: (label, frozenset(verses))}}"""
    grouped = defaultdict(lambda: defaultdict(list))
    for r in rows:
        if video and r["video_file"] != video:
            continue
        grouped[r["annotator"]][r["unit_id"]].append(r)

    view = {}
    for ann, units in grouped.items():
        view[ann] = {}
        for uid, rs in units.items():
            types = [(r.get("mention_type") or "").strip().upper() or "NONE" for r in rs]
            verses = {(r.get("verse_ref") or "").strip() for r in rs}
            verses.discard("NONE")
            verses.discard("")
            label = sorted(types, key=lambda t: FIDELITY.get(t, 9))[0]
            view[ann][uid] = (label, frozenset(verses))
    return view


def cohen_kappa(pairs, labels):
    """pairs: list of (a_label, b_label)."""
    n = len(pairs)
    if not n:
        return None, 0.0, 0.0
    po = sum(1 for a, b in pairs if a == b) / n
    ca = defaultdict(int)
    cb = defaultdict(int)
    for a, b in pairs:
        ca[a] += 1
        cb[b] += 1
    pe = sum((ca[l] / n) * (cb[l] / n) for l in labels)
    kappa = None if pe >= 1.0 else (po - pe) / (1 - pe)
    return kappa, po, pe


def interpret(k):
    if k is None:
        return "undefined (no variation in labels)"
    for lo, txt in [(0.8, "almost perfect"), (0.6, "substantial"), (0.4, "moderate"),
                    (0.2, "fair"), (0.0, "slight")]:
        if k >= lo:
            return txt
    return "poor (worse than chance)"


def matrix(pairs, labels, title):
    present = [l for l in labels if any(l in p for p in pairs)]
    if not present:
        return
    print(f"\n  {title}")
    print("      " + "".join(f"{l:>7}" for l in present) + "   (rows = A, cols = B)")
    counts = defaultdict(int)
    for a, b in pairs:
        counts[(a, b)] += 1
    for ra in present:
        print(f"  {ra:>4}" + "".join(f"{counts[(ra, cb)]:>7}" for cb in present))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spans", default=DEFAULT_SPANS)
    ap.add_argument("--a", help="annotator A (default: first two found)")
    ap.add_argument("--b", help="annotator B")
    ap.add_argument("--video", help="restrict to one video_file")
    args = ap.parse_args()

    if not os.path.exists(args.spans):
        sys.exit(f"No annotations at {args.spans}")

    rows = latest_per_pair(load_rows(args.spans))
    view = per_unit_view(rows, args.video)

    names = sorted(view)
    if not names:
        sys.exit("No annotations found" + (f" for video {args.video}" if args.video else ""))
    if args.a and args.b:
        a, b = args.a, args.b
    elif len(names) >= 2:
        a, b = names[0], names[1]
    else:
        print(f"Only one annotator present: {names[0]} ({len(view[names[0]])} units).")
        print("Cohen's kappa needs two people to annotate the SAME units.")
        sys.exit(1)
    for n in (a, b):
        if n not in view:
            sys.exit(f"Annotator {n!r} not found. Present: {names}")

    shared = sorted(set(view[a]) & set(view[b]))
    print("=" * 68)
    print("INTER-ANNOTATOR AGREEMENT — unit-level")
    print("=" * 68)
    print(f"  annotator A : {a}  ({len(view[a])} units)")
    print(f"  annotator B : {b}  ({len(view[b])} units)")
    if args.video:
        print(f"  video       : {args.video}")
    print(f"  overlapping : {len(shared)} units")
    if not shared:
        sys.exit("\nNo overlap — both annotators must label the SAME units.")

    # ---- 1. mention type -------------------------------------------------
    tp = [(view[a][u][0], view[b][u][0]) for u in shared]
    k, po, pe = cohen_kappa(tp, LABELS)
    print("\n" + "-" * 68)
    print("1. MENTION TYPE  (NONE + T1..T5)")
    print("-" * 68)
    print(f"  observed agreement : {po:.3f}   ({sum(1 for x, y in tp if x == y)}/{len(tp)})")
    print(f"  expected by chance : {pe:.3f}")
    print(f"  Cohen's kappa      : " + ("n/a" if k is None else f"{k:.3f}") + f"   -> {interpret(k)}")
    matrix(tp, LABELS, "confusion matrix")

    # ---- 2. verse identity ----------------------------------------------
    both = [u for u in shared if view[a][u][0] != "NONE" and view[b][u][0] != "NONE"]
    print("\n" + "-" * 68)
    print("2. VERSE IDENTITY  (units where both saw a verse)")
    print("-" * 68)
    if not both:
        print("  no units where both annotators marked a verse yet")
    else:
        exact = sum(1 for u in both if view[a][u][1] == view[b][u][1])
        overlap = sum(1 for u in both if view[a][u][1] & view[b][u][1])
        print(f"  units compared     : {len(both)}")
        print(f"  identical verse set: {exact}/{len(both)}  ({exact/len(both):.1%})")
        print(f"  any verse in common: {overlap}/{len(both)}  ({overlap/len(both):.1%})")

    # ---- 3. disagreements to adjudicate ----------------------------------
    dis = [u for u in shared
           if view[a][u][0] != view[b][u][0] or view[a][u][1] != view[b][u][1]]
    print("\n" + "-" * 68)
    print(f"3. DISAGREEMENTS TO ADJUDICATE  ({len(dis)})")
    print("-" * 68)
    for u in dis[:40]:
        la, va = view[a][u]
        lb, vb = view[b][u]
        print(f"  {u:26} A: {la:5} {sorted(va) or '-'}")
        print(f"  {'':26} B: {lb:5} {sorted(vb) or '-'}")
    if len(dis) > 40:
        print(f"  ... and {len(dis) - 40} more")

    print("\n" + "=" * 68)
    if k is not None:
        if k >= 0.6:
            print("  VERDICT: substantial agreement — proceed with the full pass.")
        elif k >= 0.4:
            print("  VERDICT: moderate — tighten the guideline where disagreements")
            print("           cluster (see the confusion matrix), then re-test.")
        else:
            print("  VERDICT: too low. STOP and fix the guideline before annotating")
            print("           further. Consider merging the tiers that confuse most.")
    print("=" * 68)


if __name__ == "__main__":
    main()
