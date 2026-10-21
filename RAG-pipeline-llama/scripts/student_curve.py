#!/usr/bin/env python3
"""
Learning curve for the student off-verse detector (silver-only training).

For each English tuning lecture, the final student (v1: logistic regression,
C = 0.001 as chosen by silver cross-validation) is trained on the silver labels
of randomly drawn OTHER lectures, adding lectures until about N passages are
reached (N = 375, 750, 1100, 1500), and scores that lecture. Three random draws
per size. Reported: AUC against gold on the 3 English tuning lectures (pooled),
mean and range over the draws. Tells whether the weaker inductive result
(374 tuning passages) is about the amount of training data.

Usage:
    python scripts/student_curve.py
"""
import os
import random
import sys

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

from evalkit import load_gold, read_split, verse_number  # noqa: E402
from student_detector import fit_lr, load, window_features  # noqa: E402

SIZES = [375, 750, 1100, 1500]
DRAWS = 3
C = 0.001


def main():
    from sklearn.metrics import roc_auc_score
    E, row, lectures, _, soft, _ = load()
    XW = window_features(E, row, lectures)
    _, gold, _ = load_gold()
    en_tune = read_split("en", "dev")
    lines = ["Student learning curve: English tuning AUC vs number of silver training passages",
             f"(v1, C={C}; training lectures drawn at random, never the scored lecture; {DRAWS} draws)",
             f"  {'target size':>11} {'actual sizes':>22} {'AUC mean':>9} {'AUC range':>16}"]
    for n in SIZES:
        aucs, sizes = [], []
        for d in range(DRAWS):
            rng = random.Random(1000 * n + d)
            y_all, p_all, used = [], [], []
            for v in en_tune:
                pool = [w for w in lectures if w != v]
                rng.shuffle(pool)
                chosen, total = [], 0
                for w in pool:
                    if total >= n:
                        break
                    chosen.append(w)
                    total += len(lectures[w])
                us = [u for w in chosen for u in lectures[w]]
                m = fit_lr(np.stack([XW[u] for u in us]), np.array([soft[u] for u in us]), C)
                p_all += list(m.predict_proba(np.stack([XW[u] for u in lectures[v]]))[:, 1])
                y_all += [int(not any(verse_number(r) for r in gold[u])) for u in lectures[v]]
                used.append(total)
            aucs.append(roc_auc_score(y_all, p_all))
            sizes += used
        lines.append(f"  {n:>11} {min(sizes):>10}-{max(sizes):<11} {np.mean(aucs):>9.3f} "
                     f"{min(aucs):>7.3f}-{max(aucs):<8.3f}")
    out = "\n".join(lines)
    print(out)
    open(os.path.join(HERE, "results", "student", "student_curve.txt"), "w").write(out + "\n")


if __name__ == "__main__":
    main()
