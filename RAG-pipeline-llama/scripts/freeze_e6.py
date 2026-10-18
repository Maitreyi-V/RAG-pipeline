#!/usr/bin/env python3
"""
Freeze run for E6 on the TEST lectures: the ablation, once.

Each ablated system is fit exactly like the locked one, with the part removed:
scorers and transitions on all 9 tuning lectures, (gamma, beta_off, tau) chosen
by leave-one-lecture-out on them (step4_viterbi.run(save=...)). It is then
applied unchanged to the test lectures. The rows and what they remove are the
same as scripts/exp_e6_ablation.py (the tuning-lecture version):

  full (locked)        results/step4_locked.pkl
  - string match       string, anchor (verse scorer); max_string, any_anchor (off-verse)
  - meaning match      cos, z, gain; max_cos and the held-verse cosine
  - lecture memory     recency, next, last; max_recency
  - off-verse state    off-verse only before the lecture's first detection
  - lecture decoding   the locked verse scorer alone, P >= 0.5, no Viterbi
  plain carry / carry, best anchor   for reference

Per row: strict / within-1 / theme F1 with 95% lecture-bootstrap CIs, and
segmentation Pk / WindowDiff (lower is better; Viterbi rows labelled by the
decoded verse, the others by the lowest verse, as in exp_e6_ablation.py).

Writes results/freeze_e6_<split>.txt.

Usage:
    python scripts/freeze_e6.py --split test
"""
import argparse
import os
import pickle
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

from baselines import carry  # noqa: E402
from evalkit import load_gold, read_split  # noqa: E402
from exp_e6_ablation import VARIANTS, segmentation  # noqa: E402
from freeze_grounding import (f1_near, f1_strict, near_counts, strict_counts,  # noqa: E402
                              theme_counts, with_ci)
from step3_scorer import verse_X  # noqa: E402
from step4_viterbi import FEATS, OFF, OUT, V, predict_locked, run  # noqa: E402

RES = os.path.join(HERE, "results")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["test", "heldout"], required=True)
    ap.add_argument("--n-boot", type=int, default=1000)
    args = ap.parse_args()

    lectures, gold, _ = load_gold()
    z = np.load(FEATS)
    F = {k: z[k] for k in z.files}
    refs = list(F["refs"])
    row = {u: i for i, u in enumerate(F["unit_ids"])}
    groups = [(L, name, [v for v in read_split(L, args.split) if v in lectures])
              for L, name in (("kn", "Kannada"), ("en", "English"))]
    scope = {v: lectures[v] for _, _, vs in groups for v in vs}

    def decode(model):
        dec = predict_locked(scope, model=model)
        return {u: ([] if st == OFF else [refs[st]]) + [refs[j] for j in extra]
                for v in scope for u, (st, extra, _, _) in zip(scope[v], dec[v])}

    systems = []
    with open(OUT, "rb") as f:
        locked = pickle.load(f)["a05"]
    for name, kw in VARIANTS:
        if name.startswith("full"):
            model = locked
        else:
            with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as tmp:
                path = tmp.name
            run(alphas=("a05",), verbose=False, save=path, **kw)
            model = pickle.load(open(path, "rb"))["a05"]
            os.remove(path)
        systems.append((name, decode(model), True))

    scorer = {}
    for v, us in scope.items():
        rows_ = [row[u] for u in us]
        pv = locked["verse_model"].predict_proba(
            verse_X(F, rows_, "a05", locked["verse_cols"]))[:, 1].reshape(len(us), V)
        scorer.update({u: [refs[j] for j in np.flatnonzero(pv[k] >= 0.5)] for k, u in enumerate(us)})
    systems.append(("- lecture decoding", scorer, False))

    anchors = {u: [refs[j] for j in np.flatnonzero(F["anchor"][row[u]])] for us in scope.values() for u in us}
    best = {u: ([max(a, key=lambda r: F["string"][row[u], refs.index(r)])] if a else [])
            for u, a in anchors.items()}
    for name, anc in (("plain carry (reference)", anchors), ("carry, best anchor (reference)", best)):
        pred = {}
        for us in scope.values():
            pred.update(carry(us, anc))
        systems.append((name, pred, False))

    lines = [f"FREEZE RUN — E6 ablation, split = {args.split} (run once)",
             "Each ablated system fit on the 9 tuning lectures (settings by leave-one-lecture-out "
             "there), applied unchanged here.",
             f"CI: 95% lecture bootstrap, {args.n_boot} resamples (groups with >= 5 lectures)."]
    for L, gname, vids in groups:
        lines += ["", "=" * 120, f"{gname} {args.split}   lectures {len(vids)}   "
                  f"passages {sum(len(lectures[v]) for v in vids)}", "=" * 120,
                  f"  {'system':32} {'strict F1 [CI]':22} {'+-1 F1 [CI]':22} {'theme F1 [CI]':22} "
                  f"{'Pk':>6} {'WD':>6}"]
        full = None
        for name, pred, primary in systems:
            cells, points = [], []
            for counts, f1 in ((strict_counts, f1_strict), (near_counts, f1_near),
                               (theme_counts, f1_strict)):
                point, ci = with_ci([counts(gold, pred, lectures[v]) for v in vids], f1, args.n_boot)
                points.append(point)
                cells.append(f"{point:.3f}" + (f" [{ci[0]:.3f},{ci[1]:.3f}]" if ci else ""))
            pk, wd = segmentation(gold, pred, lectures, vids, primary=primary)
            lines.append(f"  {name:32} {cells[0]:22} {cells[1]:22} {cells[2]:22} {pk:>6.3f} {wd:>6.3f}")
            if full is None:
                full = points
            elif "reference" not in name:
                d = [p - f for p, f in zip(points, full)]
                lines.append(f"  {'   change vs full':32} {d[0]:>+22.3f} {d[1]:>+22.3f} {d[2]:>+22.3f}")
    text = "\n".join(lines)
    print(text)
    out = os.path.join(RES, f"freeze_e6_{args.split}.txt")
    with open(out, "w", encoding="utf-8") as f:
        f.write(text + "\n")
    print(f"\n  wrote -> {os.path.relpath(out, HERE)}")


if __name__ == "__main__":
    main()
