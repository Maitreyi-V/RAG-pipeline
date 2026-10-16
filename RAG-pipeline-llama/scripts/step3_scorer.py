#!/usr/bin/env python3
"""
Step 3: logistic-regression scorers on the Step 2 features, TUNING lectures only.

  verse scorer      one row per (passage, verse) pair, 7 shared features:
                    string, anchor, cos, z, gain, recency, next
                    -> P_emit(v | passage). The weights are the same for every
                    verse, so nothing is specific to one verse.
  off-verse scorer  one row per passage, 4 features: max string, max cos,
                    max recency (= 1 / (1 + passages since ANY detection)),
                    any anchor -> P(no verse | passage)

Both are StandardScaler + LogisticRegression with sklearn defaults. No class
weighting, so the probabilities stay calibrated for Viterbi (Step 4).

Checked on the 9 tuning lectures with leave-one-lecture-out (train on the
others, score the held-out one), for:
  pooled vs per-language verse weights
  alpha = 0.5 (main system) vs alpha = 0 (Experiment E5)
The off-verse scorer is pooled only: the Kannada tuning lectures contain one
off-verse passage, too few to train a Kannada-only model.

Reported: AUC and AP of the held-out probabilities, and strict F1 of the scorer
ALONE (emit every verse with P_emit >= 0.5; no Viterbi), next to plain carry on
the same lectures (carry unbounded on the Step 2 anchors).

The final models are then fit on all 9 tuning lectures and saved to
results/step3_scorers.pkl. Test and new-speaker lectures are not touched.

Usage:
    python scripts/step3_scorer.py
"""
import os
import pickle
import sys

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

from baselines import carry  # noqa: E402
from evalkit import load_gold, read_split, score  # noqa: E402

FEATS = os.path.join(HERE, "results", "step2_features.npz")
OUT = os.path.join(HERE, "results", "step3_scorers.pkl")
ALPHAS = ("a05", "a0")


VERSE_COLS = ("string", "anchor", "cos", "z", "gain", "recency", "next")


def verse_X(F, rows, a, cols=VERSE_COLS):
    """(len(rows) * 72, len(cols)) pair features; cos / z / gain use alpha a."""
    cols = [F[f"{c}_{a}" if c in ("cos", "z", "gain") else c] for c in cols]
    return np.stack([c[rows].astype(np.float64).ravel() for c in cols], axis=1)


def off_X(F, rows, a):
    """(len(rows), 4) passage features."""
    cols = [F["string"].max(1), F[f"cos_{a}"].max(1), F["recency"].max(1),
            F["anchor"].max(1)]
    return np.stack([c[rows].astype(np.float64) for c in cols], axis=1)


def model():
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))


def main():
    z = np.load(FEATS)
    F = {k: z[k] for k in z.files}
    refs = list(F["refs"])
    row = {u: i for i, u in enumerate(F["unit_ids"])}
    lectures, gold, _ = load_gold()

    # tuning lectures, gold matrices
    tune = {lang: [v for v in read_split(lang, "dev") if v in lectures] for lang in ("kn", "en")}
    lang_of = {v: lang for lang, vs in tune.items() for v in vs}
    rows, Y = {}, {}
    for v in lang_of:
        rows[v] = [row[u] for u in lectures[v]]
        y = np.zeros((len(lectures[v]), len(refs)), dtype=int)
        for k, u in enumerate(lectures[v]):
            for r in gold[u]:
                if r in refs:
                    y[k, refs.index(r)] = 1
        Y[v] = y

    def stack(vids, fn, a, target):
        X = np.concatenate([fn(F, rows[v], a) for v in vids])
        y = np.concatenate([target(Y[v]) for v in vids])
        return X, y

    pair_y = lambda y: y.ravel()                    # noqa: E731
    off_y = lambda y: (y.sum(1) == 0).astype(int)   # noqa: E731

    def to_pred(v, p):
        """P_emit matrix of one lecture -> {unit_id: [refs with p >= 0.5]}"""
        p = p.reshape(len(rows[v]), len(refs))
        return {u: [refs[j] for j in np.flatnonzero(p[k] >= 0.5)]
                for k, u in enumerate(lectures[v])}

    # ---- reference: plain carry on the Step 2 anchors ------------------------
    anchors = {u: [refs[j] for j in np.flatnonzero(F["anchor"][row[u]])]
               for v in lang_of for u in lectures[v]}
    print("=" * 84)
    print("STEP 3 on the TUNING lectures (leave-one-lecture-out)")
    print("=" * 84)
    carry_f1 = {}
    for lang, vids in tune.items():
        pred = {}
        for v in vids:
            pred.update(carry(lectures[v], anchors))
        carry_f1[lang] = score(gold, pred, lectures, vids)
    print("  reference, plain carry (unbounded, Step 2 anchors): strict F1  "
          + "   ".join(f"{lang} {s['F1']:.3f} (P {s['P']:.3f} R {s['R']:.3f})"
                       for lang, s in carry_f1.items()))

    # ---- verse scorer: LOLO ---------------------------------------------------
    print("\n  VERSE SCORER  (AUC / AP on held-out lecture pairs; F1 = scorer alone, "
          "P_emit >= 0.5)")
    print(f"  {'alpha':6} {'weights':10} {'lang':4} {'AUC':>6} {'AP':>6} "
          f"{'P':>6} {'R':>6} {'F1':>6}")
    for a in ALPHAS:
        for mode in ("pooled", "per-lang"):
            probs = {}
            for v in lang_of:
                pool = [w for w in lang_of if w != v
                        and (mode == "pooled" or lang_of[w] == lang_of[v])]
                m = model().fit(*stack(pool, verse_X, a, pair_y))
                probs[v] = m.predict_proba(verse_X(F, rows[v], a))[:, 1]
            for lang, vids in tune.items():
                y = np.concatenate([Y[v].ravel() for v in vids])
                p = np.concatenate([probs[v] for v in vids])
                pred = {}
                for v in vids:
                    pred.update(to_pred(v, probs[v]))
                s = score(gold, pred, lectures, vids)
                print(f"  {a:6} {mode:10} {lang:4} {roc_auc_score(y, p):>6.3f} "
                      f"{average_precision_score(y, p):>6.3f} "
                      f"{s['P']:>6.3f} {s['R']:>6.3f} {s['F1']:>6.3f}")

    # ---- off-verse scorer: LOLO, pooled ---------------------------------------
    print("\n  OFF-VERSE SCORER  (pooled; AUC / AP on held-out lecture passages)")
    for a in ALPHAS:
        probs = {}
        for v in lang_of:
            m = model().fit(*stack([w for w in lang_of if w != v], off_X, a, off_y))
            probs[v] = m.predict_proba(off_X(F, rows[v], a))[:, 1]
        for lang, vids in tune.items():
            y = np.concatenate([off_y(Y[v]) for v in vids])
            p = np.concatenate([probs[v] for v in vids])
            if 0 < y.sum() < len(y):
                print(f"  {a:6} {lang:4} off-verse {int(y.sum()):>3}/{len(y):<4} "
                      f"AUC {roc_auc_score(y, p):.3f}  AP {average_precision_score(y, p):.3f}"
                      f"  (chance AP {y.mean():.3f})")
            else:
                print(f"  {a:6} {lang:4} off-verse {int(y.sum())}/{len(y)}: cannot score")

    # ---- final models on all 9 tuning lectures --------------------------------
    names = ["string", "anchor", "cos", "z", "gain", "recency", "next"]
    off_names = ["max_string", "max_cos", "max_recency", "any_anchor"]
    final = {}
    print("\n  FINAL weights, fit on all tuning lectures (standardised features)")
    for a in ALPHAS:
        groups = {"pooled": list(lang_of), "kn": tune["kn"], "en": tune["en"]}
        for g, vids in groups.items():
            m = model().fit(*stack(vids, verse_X, a, pair_y))
            final[("verse", a, g)] = m
            w = m[-1].coef_[0]
            print(f"  verse {a:4} {g:7} " + " ".join(f"{n}={x:+.2f}" for n, x in zip(names, w)))
        m = model().fit(*stack(list(lang_of), off_X, a, off_y))
        final[("off", a, "pooled")] = m
        w = m[-1].coef_[0]
        print(f"  off   {a:4} pooled  " + " ".join(f"{n}={x:+.2f}" for n, x in zip(off_names, w)))
    with open(OUT, "wb") as f:
        pickle.dump(final, f)
    print(f"\n  wrote -> {os.path.relpath(OUT, HERE)}  ({len(final)} models)")


if __name__ == "__main__":
    main()
