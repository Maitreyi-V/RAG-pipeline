"""Calibrate the off-verse classifier on each run's calib lectures and report scorer-only metrics.

For every run: Platt-scale off_logit on calib lectures (gold), apply to test lectures and write
scores/<run>/<lecture_id>_cal.npz with
  p_off_ft      : [T]     calibrated P_off from the fine-tuned classifier
  verse_feature : [T, 72] per-passage z-score of the verse logit (the 9th feature for your LR)
Then prints, pooled over the test folds cv0..cv4 (and separately for e3), by language:
  verse top-1   : share of gold-verse passages whose argmax verse is in gold
  verse top-1±1 : same, allowing a one-verse miss
  off AUC       : off-verse classifier AUC against gold
These are diagnostics only; Strict F1 comes from your decoder after integration.
"""
import json, glob, os
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from common import load_passages, load_plan

lect, plan = load_passages(), load_plan()

def gold_off(lid):
    return np.array([int(not p["gold"]) for p in lect[lid]])

def report(name, lids, by_lang):
    for lang in ("kn", "en"):
        ls = [l for l in lids if lect[l][0]["lang"] == lang]
        if not ls:
            continue
        hit = hit1 = n = 0; ys, ps = [], []
        for l in ls:
            d = np.load(f"scores/{by_lang[l]}/{l}_cal.npz")
            for t, p in enumerate(lect[l]):
                ys.append(int(not p["gold"])); ps.append(d["p_off_ft"][t])
                if p["gold"]:
                    top = int(np.argmax(d["verse_feature"][t])) + 1
                    n += 1; hit += top in p["gold"]
                    hit1 += any(abs(top - g) <= 1 for g in p["gold"])
        auc = roc_auc_score(ys, ps) if 0 < sum(ys) < len(ys) else float("nan")
        print(f"{name:8} {lang}: {len(ls):2} lectures | verse top-1 {hit/max(1,n):.3f} "
              f"| top-1±1 {hit1/max(1,n):.3f} (n={n}) | off AUC {auc:.3f} (off={sum(ys)}/{len(ys)})")

done = {}
for run in plan["runs"]:
    if not os.path.isdir(f"scores/{run['name']}"):
        print(f"skip {run['name']}: no scores yet"); continue
    X = np.concatenate([np.load(f"scores/{run['name']}/{l}.npz")["off_logit"] for l in run["calib"]])
    y = np.concatenate([gold_off(l) for l in run["calib"]])
    platt = LogisticRegression(C=1e6).fit(X.reshape(-1, 1), y)
    for l in run["calib"] + run["test"]:
        d = np.load(f"scores/{run['name']}/{l}.npz")
        vl = d["verse_logit"]
        z = (vl - vl.mean(1, keepdims=True)) / (vl.std(1, keepdims=True) + 1e-6)
        np.savez(f"scores/{run['name']}/{l}_cal.npz",
                 p_off_ft=platt.predict_proba(d["off_logit"].reshape(-1, 1))[:, 1], verse_feature=z)
    for l in run["test"]:
        done[l] = run["name"]
    json.dump({"a": float(platt.coef_[0, 0]), "b": float(platt.intercept_[0])},
              open(f"scores/{run['name']}/platt.json", "w"))

cv = [l for l, r in done.items() if r.startswith("cv")]
e3 = [l for l, r in done.items() if r == "e3"]
e2_test = [l for l in cv if lect[l][0]["split"] == "e2"]
if cv: report("CV-all", cv, done)
if e2_test: report("CV-E2", e2_test, done)
if e3: report("E3", e3, done)
