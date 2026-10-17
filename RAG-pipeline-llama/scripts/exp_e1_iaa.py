#!/usr/bin/env python3
"""
E1: are the human labels reliable? First pass vs the blind second pass.

Pairs every passage re-labelled in the blind second pass (data/annotations_iaa/,
written by annotate_app.py "Second pass (blind)") with its first-pass labels in
data/mention_spans.csv, exactly as compute_iaa_units.py --second-pass does
(passages re-labelled by their own first annotator are dropped).

Reported per language and pooled, at three granularities:

  Cohen's kappa  one label per passage: its lowest Chapter 2 verse, or OFF
                 (73 states).  strict  labels equal
                               +-1     both OFF, or verses at most 1 apart
                               theme   same theme (data/verse_themes.csv), or both OFF
                 chance agreement uses each pass's label frequencies with the
                 same agreement rule.
  pairwise F1    first pass as gold, second pass as prediction, over (passage,
                 verse) pairs: strict, +-1, theme (as in freeze_grounding.py).

Every passage has exactly two annotators, so Cohen's kappa applies (Fleiss'
kappa is for more than two raters per item). Mention-type kappa (NONE, T1..T5)
stays in compute_iaa_units.py.

Writes results/exp_e1_iaa.txt.

Usage:
    python scripts/exp_e1_iaa.py
    python scripts/exp_e1_iaa.py --second-pass data/annotations_iaa
"""
import argparse
import glob
import os
import sys
from collections import Counter

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

from catalogue import load_themes  # noqa: E402
from compute_iaa_units import DEFAULT_SPANS, latest_per_pair, load_rows  # noqa: E402
from evalkit import verse_number  # noqa: E402
from freeze_grounding import (f1_near, f1_strict, near_counts, strict_counts,  # noqa: E402
                              theme_counts)

THEMES = load_themes()
OUT = os.path.join(HERE, "results", "exp_e1_iaa.txt")


def verse_sets(rows):
    """{unit_id: set of 'BG 2.x'} (Chapter 2 only; empty = no verse)"""
    out = {}
    for r in rows:
        refs = out.setdefault(r["unit_id"], set())
        v = (r.get("verse_ref") or "").strip()
        if verse_number(v):
            refs.add(v)
    return out


def label(refs):
    """lowest Chapter 2 verse number, or 0 = OFF"""
    return min((verse_number(r) for r in refs), default=0)


AGREE = {
    "strict": lambda a, b: a == b,
    "+-1": lambda a, b: a == b or (a and b and abs(a - b) <= 1),
    "theme": lambda a, b: a == b or (a and b and THEMES.get(a) == THEMES.get(b)),
}


def kappa(pairs, agree):
    n = len(pairs)
    po = sum(agree(a, b) for a, b in pairs) / n
    ca, cb = Counter(a for a, _ in pairs), Counter(b for _, b in pairs)
    pe = sum(ca[x] * cb[y] for x in ca for y in cb if agree(x, y)) / n / n
    return (po - pe) / (1 - pe) if pe < 1 else None, po


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spans", default=DEFAULT_SPANS)
    ap.add_argument("--second-pass", default=os.path.join(HERE, "data", "annotations_iaa"))
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.second_pass, "*.csv")))
    if not files:
        sys.exit(f"no second-pass files in {args.second_pass} yet")
    first = latest_per_pair(load_rows(args.spans))
    second = latest_per_pair([r for p in files for r in load_rows(p)])
    first_by = {r["unit_id"]: r["annotator"] for r in first}
    own = {r["unit_id"] for r in second if first_by.get(r["unit_id"]) == r["annotator"]}
    units = sorted({r["unit_id"] for r in second} - own)
    A = verse_sets(r for r in first if r["unit_id"] in units)
    B = verse_sets(r for r in second if r["unit_id"] in units)
    who = {r["unit_id"]: r["annotator"] for r in second}

    lines = [f"E1 — first pass vs blind second pass ({len(files)} file(s) in "
             f"{os.path.relpath(args.second_pass, HERE)})",
             f"passages re-labelled: {len(units)}"
             + (f"   (dropped {len(own)} re-labelled by their own first annotator)" if own else ""),
             "second-pass annotators: " + ", ".join(f"{k} {v}" for k, v in
                                                    Counter(who[u] for u in units).items())]
    groups = [("Kannada", [u for u in units if u.startswith("kn_")]),
              ("English", [u for u in units if u.startswith("en_")]),
              ("all", units)]
    lines.append(f"\n  {'group':8} {'n':>4}   {'kappa strict':>12} {'kappa +-1':>10} {'kappa theme':>12}"
                 f"   {'F1 strict':>9} {'F1 +-1':>7} {'F1 theme':>9}   observed agreement strict")
    for name, us in groups:
        if not us:
            lines.append(f"  {name:8} {0:>4}")
            continue
        pairs = [(label(A[u]), label(B[u])) for u in us]
        ks = [kappa(pairs, AGREE[k]) for k in ("strict", "+-1", "theme")]
        gold = {u: {r: "" for r in A[u]} for u in us}
        pred = {u: sorted(B[u]) for u in us}
        f = [f1_strict(strict_counts(gold, pred, us)), f1_near(near_counts(gold, pred, us)),
             f1_strict(theme_counts(gold, pred, us))]
        fmt = lambda k: "n/a" if k is None else f"{k:.3f}"      # noqa: E731
        lines.append(f"  {name:8} {len(us):>4}   {fmt(ks[0][0]):>12} {fmt(ks[1][0]):>10} "
                     f"{fmt(ks[2][0]):>12}   {f[0]:>9.3f} {f[1]:>7.3f} {f[2]:>9.3f}   {ks[0][1]:.3f}")
    text = "\n".join(lines)
    print(text)
    with open(OUT, "w", encoding="utf-8") as fh:
        fh.write(text + "\n")
    print(f"\n  wrote -> {os.path.relpath(OUT, HERE)}")


if __name__ == "__main__":
    main()
