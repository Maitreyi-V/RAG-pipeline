"""Lecture-level 5-fold split of the 39 course lectures (tune + e2), stratified by language.

For test fold k: calibration fold = (k+1) % 5, training = the other 3 folds.
Run "e3": train on folds 1-4, calibrate on fold 0, test on the 12 unseen-speaker lectures.
Writes folds.json and prints the experiment-log entry. Run ONCE, commit folds.json before training.
"""
import json, datetime
from common import load_passages, SEED, rng

K = 5
lect = load_passages()
course = sorted(l for l, ps in lect.items() if ps[0]["split"] in ("tune", "e2"))
e3 = sorted(l for l, ps in lect.items() if ps[0]["split"] == "e3")
silver = sorted(l for l, ps in lect.items() if ps[0]["split"] == "silver")
r = rng(SEED)
fold = {}
for lang in ("kn", "en"):
    ls = [l for l in course if lect[l][0]["lang"] == lang]
    r.shuffle(ls)
    for i, l in enumerate(ls):
        fold[l] = i % K

def members(ks):
    return sorted(l for l in course if fold[l] in ks)

runs = []
for k in range(K):
    c = (k + 1) % K
    runs.append({"name": f"cv{k}", "test": members({k}), "calib": members({c}),
                 "train": members(set(range(K)) - {k, c}) + silver})
runs.append({"name": "e3", "test": e3, "calib": members({0}),
             "train": members({1, 2, 3, 4}) + silver})

plan = {"created": datetime.datetime.now().isoformat(timespec="seconds"), "seed": SEED,
        "k": K, "fold_of": fold, "runs": runs}
json.dump(plan, open("folds.json", "w"), indent=1)

print("=== EXPERIMENT LOG ENTRY (paste, then commit folds.json) ===")
print(f"date: {plan['created']}  seed: {SEED}  folds: {K}, lecture-level, stratified by language")
for k in range(K):
    ls = members({k})
    npas = sum(len(lect[l]) for l in ls)
    langs = {g: sum(lect[l][0]['lang'] == g for l in ls) for g in ('kn', 'en')}
    print(f"fold {k}: {len(ls)} lectures ({langs['kn']} kn, {langs['en']} en), {npas} passages: {', '.join(ls)}")
for run in runs:
    n = lambda ls: sum(len(lect[l]) for l in ls)
    print(f"run {run['name']}: train {len(run['train'])} lect/{n(run['train'])} pass | "
          f"calib {len(run['calib'])}/{n(run['calib'])} | test {len(run['test'])}/{n(run['test'])}")
print(f"silver lectures (train only, never calib/test): {len(silver)}")
print("model: BAAI/bge-reranker-v2-m3; tasks: verse scorer + off-verse classifier; "
      "epochs 2, lr 2e-5, batch 8 x accum 4, max_len 384, seed 42")
print("comparison set for go/no-go: the 30 E2 test lectures (frozen baselines are in-sample on tuning lectures)")
