"""Build training examples for one run (training lectures only).

verse task: (passage, verse text) -> 1 if verse in gold, else 0.
  verse passages : positives = all gold verses; negatives = primary±1, ±2, top-5 of the
                   current scorer (optional data/prior_scores/<lecture>.npy, shape [T,72]),
                   5 random verses.
  off passages   : negatives = held verse h and h+1 (last gold primary before the digression)
                   + 8 random verses.
off task:   (passage, neighbour context) -> 1 if off-verse (gold empty), else 0.
Gold examples get weight 1.0, silver examples --silver_weight.
"""
import argparse, json, os
import numpy as np
from common import load_passages, load_verses, verse_text, off_context, load_plan, rng, V

ap = argparse.ArgumentParser()
ap.add_argument("--run", required=True)
ap.add_argument("--silver_weight", type=float, default=0.4)
ap.add_argument("--out_dir", default="examples")
a = ap.parse_args()

lect, verses, plan = load_passages(), load_verses(), load_plan()
run = next(r for r in plan["runs"] if r["name"] == a.run)
held_out = set(run["calib"]) | set(run["test"])
assert not held_out & set(run["train"]), "train overlaps calib/test"
r = rng()
verse_ex, off_ex = [], []
for lid in run["train"]:
    ps = lect[lid]
    w = a.silver_weight if ps[0]["split"] == "silver" else 1.0
    prior_path = f"data/prior_scores/{lid}.npy"
    prior = np.load(prior_path) if os.path.exists(prior_path) else None
    if prior is not None and prior.shape != (len(ps), V):
        raise ValueError(f"{prior_path} has shape {prior.shape}, expected {(len(ps), V)}")
    held = None
    for t, p in enumerate(ps):
        lang, gold = p["lang"], p["gold"]
        off_ex.append({"a": p["text"], "b": off_context(ps, t), "y": int(not gold), "w": w,
                       "lecture_id": lid, "idx": t})
        if gold:
            neg = {gold[0] + d for d in (-2, -1, 1, 2)}
            if prior is not None:
                neg |= {int(i) + 1 for i in np.argsort(-prior[t])[:5]}
            pool = [v for v in range(1, V + 1) if v not in gold and v not in neg]
            neg |= set(r.sample(pool, 5))
            neg = {v for v in neg if 1 <= v <= V and v not in gold}
            for v in gold:
                verse_ex.append({"a": p["text"], "b": verse_text(verses, v, lang), "y": 1, "w": w,
                                 "lecture_id": lid, "idx": t, "v": v})
            held = gold[0]
        else:
            neg = {held, held + 1} if held else set()
            neg = {v for v in neg if 1 <= v <= V}
            pool = [v for v in range(1, V + 1) if v not in neg]
            neg |= set(r.sample(pool, 8))
        for v in sorted(neg):
            verse_ex.append({"a": p["text"], "b": verse_text(verses, v, lang), "y": 0, "w": w,
                             "lecture_id": lid, "idx": t, "v": v})

os.makedirs(a.out_dir, exist_ok=True)
for name, ex in (("verse", verse_ex), ("off", off_ex)):
    path = f"{a.out_dir}/{a.run}_{name}.jsonl"
    with open(path, "w", encoding="utf-8") as f:
        for e in ex:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")
    pos = sum(e["y"] for e in ex)
    print(f"{path}: {len(ex)} examples, {pos} positive ({pos/len(ex):.1%})")
