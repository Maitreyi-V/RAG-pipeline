#!/usr/bin/env python3
"""
Error decomposition v1 -> v2 on the FROZEN outputs (analysis only, nothing is
tuned): v1 = results/step4_locked.pkl, v2 = results/v2_teacher145.pkl, oracle
stopping = carry that stops / resumes exactly where gold does.

Passage classes (gold "no verse" = no reference at all, as in freeze_grounding):
  gold no-verse:  OK-off (predicted nothing) | FALSE-VERSE (predicted a verse)
  gold verse:     HIT (a gold verse predicted) | NEAR (+-1 only) | WRONG (other verse)
                  | FALSE-OFF (predicted nothing)
Strict micro-F1 counts (tp, fp, fn) are split by where they arise. Counterfactual:
strict F1 if every false verse on a no-verse passage were removed (perfect
off-verse handling, verse identity unchanged).

Usage:
    python scripts/error_decomp_v2.py --split test --lang en
"""
import argparse
import os
import sys
from collections import Counter

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

from baselines import carry  # noqa: E402
from evalkit import _prf, load_gold, read_split, verse_number  # noqa: E402
from step4_viterbi import FEATS, OFF, OUT, predict_locked  # noqa: E402

CLASSES = ["OK-off", "FALSE-VERSE", "HIT", "NEAR", "WRONG", "FALSE-OFF"]


def cls(g, p):
    if not g:
        return "FALSE-VERSE" if p else "OK-off"
    if not p:
        return "FALSE-OFF"
    if set(p) & set(g):
        return "HIT"
    gn = [x for x in map(verse_number, g) if x]
    if any(x is not None and any(abs(x - y) <= 1 for y in gn) for x in map(verse_number, p)):
        return "NEAR"
    return "WRONG"


def counts(gold, pred, us):
    c = Counter()
    for u in us:
        g, p = set(gold[u]), set(pred[u])
        side = "no-verse" if not g else "verse"
        c[("tp", side)] += len(p & g)
        c[("fp", side)] += len(p - g)
        c[("fn", side)] += len(g - p)
    return c


def f1(c, drop_fp_noverse=False):
    tp = c[("tp", "verse")] + c[("tp", "no-verse")]
    fp = c[("fp", "verse")] + (0 if drop_fp_noverse else c[("fp", "no-verse")])
    fn = c[("fn", "verse")] + c[("fn", "no-verse")]
    return _prf(tp, fp, fn)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test")
    ap.add_argument("--lang", default="en")
    args = ap.parse_args()
    lectures, gold, _ = load_gold()
    vids = [v for v in read_split(args.lang, args.split) if v in lectures]
    us = [u for v in vids for u in lectures[v]]
    z = np.load(FEATS)
    refs = list(z["refs"])
    row = {u: i for i, u in enumerate(z["unit_ids"])}
    anchors = {u: [refs[j] for j in np.flatnonzero(z["anchor"][row[u]])] for u in row}

    def decoded(path):
        dec = predict_locked({v: lectures[v] for v in vids}, "a05", path=path)
        return {u: ([] if st == OFF else [refs[st]]) + [refs[j] for j in extra]
                for v in vids for u, (st, extra, _, _) in zip(lectures[v], dec[v])}
    preds = {"v1": decoded(OUT), "v2": decoded(os.path.join(HERE, "results", "v2_teacher145.pkl"))}
    oracle = {}
    for v in vids:
        oracle.update(carry(lectures[v], anchors, keep=lambda u, r: r in gold[u], resume=True))
    preds["oracle"] = {u: list(oracle.get(u, ())) for u in us}

    n_nov = sum(1 for u in us if not gold[u])
    L = [f"Error decomposition, {args.lang} {args.split} ({len(vids)} lectures, {len(us)} passages, "
         f"{n_nov} with no gold verse) — frozen outputs, analysis only", "",
         "Passage classes:"]
    L.append(f"  {'system':8} " + " ".join(f"{c:>12}" for c in CLASSES))
    cl = {k: {u: cls(gold[u], p[u]) for u in us} for k, p in preds.items()}
    for k in preds:
        cc = Counter(cl[k].values())
        L.append(f"  {k:8} " + " ".join(f"{cc[c]:>12}" for c in CLASSES))

    L += ["", "Passage transitions v1 -> v2 (rows v1, columns v2; off-diagonal = changed):",
          f"  {'v1 \\ v2':12} " + " ".join(f"{c:>12}" for c in CLASSES)]
    tr = Counter((cl["v1"][u], cl["v2"][u]) for u in us)
    for a in CLASSES:
        L.append(f"  {a:12} " + " ".join(f"{tr[(a, b)]:>12}" for b in CLASSES))

    L += ["", "Strict micro-F1 counts by where they arise (verse = passage with a gold verse):",
          f"  {'system':8} {'tp':>5} {'fp@verse':>9} {'fp@no-verse':>12} {'fn@verse':>9} {'fn@no-verse':>12}"
          f" {'P':>6} {'R':>6} {'F1':>6} {'F1 if no fp@no-verse':>21}"]
    for k, p in preds.items():
        c = counts(gold, p, us)
        P, R, F = f1(c)
        L.append(f"  {k:8} {c[('tp', 'verse')] + c[('tp', 'no-verse')]:>5} {c[('fp', 'verse')]:>9} "
                 f"{c[('fp', 'no-verse')]:>12} {c[('fn', 'verse')]:>9} {c[('fn', 'no-verse')]:>12} "
                 f"{P:>6.3f} {R:>6.3f} {F:>6.3f} {f1(c, True)[2]:>21.3f}")

    L += ["", "Per lecture strict F1 and class changes v1 -> v2:"]
    for v in vids:
        vu = lectures[v]
        cv = {k: f1(counts(gold, preds[k], vu))[2] for k in ("v1", "v2", "oracle")}
        moved = Counter((cl["v1"][u], cl["v2"][u]) for u in vu if cl["v1"][u] != cl["v2"][u])
        top = ", ".join(f"{a}->{b} {n}" for (a, b), n in moved.most_common(4))
        L.append(f"  {v:28} v1 {cv['v1']:.3f}  v2 {cv['v2']:.3f}  oracle {cv['oracle']:.3f}  | {top}")
    out = "\n".join(L)
    print(out)
    open(os.path.join(HERE, "results", f"error_decomp_v2_{args.lang}_{args.split}.txt"), "w").write(out + "\n")


if __name__ == "__main__":
    main()
