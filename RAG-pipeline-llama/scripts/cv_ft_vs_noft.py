#!/usr/bin/env python3
"""
Paired lecture bootstrap, strict F1: fine-tuned CV system minus the same protocol without fine-tuning.
Reads stored predictions only (no re-decoding):
    results/cv_ft/predictions.csv   results/cv_noft/predictions.csv
Groups as in cv_decode.py: E2 Kannada / English test, E3 new Kannada / English (10,000 resamples, seed 0).

Usage:
    python scripts/cv_ft_vs_noft.py
"""
import argparse
import csv
import os
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

from evalkit import load_gold, read_split  # noqa: E402
from freeze_grounding import f1_strict, paired, strict_counts, with_ci  # noqa: E402


def load(path):
    return {r["unit_id"]: [x for x in r["refs"].split(";") if x]
            for r in csv.DictReader(open(path, newline="", encoding="utf-8"))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", default=os.path.join(HERE, "results", "cv_ft", "predictions.csv"))
    ap.add_argument("--b", default=os.path.join(HERE, "results", "cv_noft", "predictions.csv"))
    ap.add_argument("--out", default=os.path.join(HERE, "results", "cv_ft_vs_noft.txt"))
    args = ap.parse_args()
    lectures, gold, _ = load_gold()
    A, B = load(args.a), load(args.b)
    groups = [("E2 Kannada test", read_split("kn", "test")), ("E2 English test", read_split("en", "test")),
              ("E3 new Kannada", read_split("kn", "heldout")), ("E3 new English", read_split("en", "heldout"))]
    lines = ["Fine-tuned CV (A) minus CV without fine-tuning (B), strict F1; paired lecture bootstrap, 10,000 resamples"]
    for g, vs in groups:
        miss = [u for v in vs for u in lectures[v] if u not in A or u not in B]
        assert not miss, f"{g}: {len(miss)} passages missing from a predictions file"
        fa = with_ci([strict_counts(gold, A, lectures[v]) for v in vs], f1_strict, 10000)[0]
        fb = with_ci([strict_counts(gold, B, lectures[v]) for v in vs], f1_strict, 10000)[0]
        d, ci, le0, (w, t, lo), p = paired(gold, lectures, A, B, vs, 10000)
        lines.append(f"{g:18} ({len(vs):2} lectures)  A {fa:.3f}  B {fb:.3f}  A - B {d:+.3f} "
                     f"[{ci[0]:+.3f}, {ci[1]:+.3f}]  P(<=0) {le0:.3f}  W/T/L {w}/{t}/{lo}  sign p {p:.3f}")
    out = "\n".join(lines)
    print(out)
    open(args.out, "w", encoding="utf-8").write(out + "\n")


if __name__ == "__main__":
    main()
