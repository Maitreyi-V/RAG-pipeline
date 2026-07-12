#!/usr/bin/env python3
"""
Inter-annotator agreement (Cohen's kappa) on T1-T5 mention types.

Two annotators each fill in `mention_type` for the SAME (video_file, verse_ref)
rows -- give each their own copy of data/mention_types.csv. This script matches
rows by (video_file, verse_ref), computes Cohen's kappa on the overlap, prints a
confusion matrix, and lists disagreements to adjudicate.

Usage:
    python scripts/compute_iaa.py annotatorA.csv annotatorB.csv
    python scripts/compute_iaa.py A.csv B.csv --col mention_type

Reports go in the paper's dataset section. No external deps (pure stdlib).
"""
import csv, sys, argparse
from collections import defaultdict

LABELS = ["T1", "T2", "T3", "T4", "T5"]


def load(path, col):
    out = {}
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            key = (r["video_file"].strip(), r["verse_ref"].strip())
            val = (r.get(col) or "").strip().upper()
            if val:
                out[key] = val
    return out


def cohens_kappa(pairs):
    """pairs: list of (labelA, labelB). Returns (kappa, po, pe, n)."""
    n = len(pairs)
    if n == 0:
        return None, 0.0, 0.0, 0
    agree = sum(1 for a, b in pairs if a == b)
    po = agree / n
    ca, cb = defaultdict(int), defaultdict(int)
    for a, b in pairs:
        ca[a] += 1; cb[b] += 1
    pe = sum((ca[l] / n) * (cb[l] / n) for l in set(list(ca) + list(cb)))
    kappa = (po - pe) / (1 - pe) if pe != 1 else 1.0
    return kappa, po, pe, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("annotator_a")
    ap.add_argument("annotator_b")
    ap.add_argument("--col", default="mention_type")
    args = ap.parse_args()

    A = load(args.annotator_a, args.col)
    B = load(args.annotator_b, args.col)
    shared = sorted(set(A) & set(B))

    if not shared:
        print("No overlapping annotated rows. Did both annotators fill the same rows?")
        sys.exit(1)

    pairs = [(A[k], B[k]) for k in shared]
    kappa, po, pe, n = cohens_kappa(pairs)

    print("=" * 60)
    print(f"INTER-ANNOTATOR AGREEMENT  ({args.col})")
    print("=" * 60)
    print(f"  A: {args.annotator_a}  ({len(A)} labelled)")
    print(f"  B: {args.annotator_b}  ({len(B)} labelled)")
    print(f"  Overlap scored: {n}")
    print(f"  Observed agreement (po): {po:.4f}")
    print(f"  Expected by chance (pe): {pe:.4f}")
    print(f"  Cohen's kappa:           {kappa:.4f}")
    interp = ("poor" if kappa < 0.2 else "fair" if kappa < 0.4 else
              "moderate" if kappa < 0.6 else "substantial" if kappa < 0.8 else "almost perfect")
    print(f"  Interpretation (Landis-Koch): {interp}")

    # Confusion matrix (rows = A, cols = B)
    labels = LABELS + sorted({l for p in pairs for l in p} - set(LABELS))
    cm = {a: defaultdict(int) for a in labels}
    for a, b in pairs:
        cm[a][b] += 1
    print("\n  Confusion matrix (row=A, col=B):")
    print("        " + "  ".join(l.rjust(4) for l in labels))
    for a in labels:
        print(f"    {a:<4}" + "  ".join(str(cm[a][b]).rjust(4) for b in labels))

    # Disagreements to adjudicate
    disagree = [(k, A[k], B[k]) for k in shared if A[k] != B[k]]
    print(f"\n  Disagreements to adjudicate: {len(disagree)}")
    for (vid, vref), a, b in disagree:
        print(f"    {vid} {vref}:  A={a}  B={b}")


if __name__ == "__main__":
    main()
