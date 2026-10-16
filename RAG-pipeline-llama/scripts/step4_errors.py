#!/usr/bin/env python3
"""
Where do Step 4 (Viterbi, alpha 0.5) and plain carry go wrong on the ENGLISH
tuning lectures? Every passage falls in exactly one bucket:

  correct            predicted set == gold set
  verse on off       gold has no verse, a verse was predicted
  off on verse       gold has a verse, nothing was predicted (off-verse)
  off by one         no predicted verse is in gold, but every one is within +-1
                     of a gold verse (direction from the first predicted verse)
  partly right       some predicted verse is in gold, but the sets differ
                     (a verse missed or one extra)
  other wrong verse  no overlap with gold, at least one verse more than 1 away

Prints one row per Step 4 variant (the `last` feature, decoupled emissions,
the carry-verse cosine in the off-verse scorer), next to plain carry. The
unanchored next-step penalty delta was dropped: it was never chosen.

Usage:
    python scripts/step4_errors.py
"""
import os
import sys
from collections import Counter

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

from evalkit import verse_number  # noqa: E402
from step4_viterbi import run  # noqa: E402

BUCKETS = ["correct", "verse on off", "off on verse", "off by one",
           "partly right", "other wrong verse"]


def bucket(g, p):
    g, p = set(g), set(p)
    if g == p:
        return "correct", None
    if not g:
        return "verse on off", None
    if not p:
        return "off on verse", None
    if g & p:
        return "partly right", None
    gn = [verse_number(r) for r in g if verse_number(r)]
    pn = [verse_number(r) for r in p]
    if gn and all(x and min(abs(x - y) for y in gn) == 1 for x in pn):
        first = verse_number(sorted(p, key=verse_number)[0])
        near = min(gn, key=lambda y: abs(first - y))
        return "off by one", "+1" if first > near else "-1"
    return "other wrong verse", None


def buckets(gold, pred, us):
    return {u: bucket(gold[u], pred.get(u, ())) for u in us}


def main():
    from step4_viterbi import near_f1
    from evalkit import score

    off = dict(use_last=False, decouple=False, off_active=False)
    variants = [("Viterbi", off),
                ("+ last", {**off, "use_last": True}),
                ("+ last, decoupled", {**off, "use_last": True, "decouple": True}),
                ("+ last, off_active", {**off, "use_last": True, "off_active": True}),
                ("+ last, both (LOCKED)", {}),
                ("decoupled, no last", {**off, "decouple": True, "off_active": True})]
    rows = []
    carry_row = None
    for name, kw in variants:
        order, gold, lectures, tune = run(alphas=("a05",), verbose=False, **kw)
        systems = dict(order)
        if carry_row is None:
            carry_row = ("plain carry", systems["plain carry"])
            rows.append(carry_row)
        rows.append((name, systems["Viterbi (alpha 0.5)"]))

    us = [u for v in tune["en"] for u in lectures[v]]
    print("\n" + "=" * 100)
    print("Step 4 fixes, alpha 0.5, TUNING lectures (leave-one-lecture-out)")
    print("=" * 100)
    print(f"  {'system':22} {'kn strict':>9} {'kn +-1':>7} {'en strict':>9} {'en +-1':>7}   "
          f"English passages: {'correct':>7} {'+1 ahead':>8} {'-1 behind':>9} "
          f"{'off on verse':>12} {'verse on off':>12}")
    for name, pred in rows:
        f = []
        for L in ("kn", "en"):
            f += [score(gold, pred, lectures, tune[L])["F1"],
                  near_f1(gold, pred, lectures, tune[L])[2]]
        b = buckets(gold, pred, us)
        c = Counter(x[0] for x in b.values())
        d = Counter(x[1] for x in b.values() if x[0] == "off by one")
        print(f"  {name:22} {f[0]:>9.3f} {f[1]:>7.3f} {f[2]:>9.3f} {f[3]:>7.3f}   "
              f"{'':17} {c['correct']:>7} {d['+1']:>8} {d['-1']:>9} "
              f"{c['off on verse']:>12} {c['verse on off']:>12}")


if __name__ == "__main__":
    main()
