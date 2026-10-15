#!/usr/bin/env python3
"""
Shared loading + scoring for the unit-level grounding experiments.

Every experiment script (run_length, baselines, gated propagation) needs the
same three things, and they must agree with score_by_tier_units.py exactly:

  load_gold()   one current view of the annotation log, per lecture, in order
  score()       P / R / F1 over (unit, verse) pairs, plus per-tier hit/n
  bootstrap()   95% CI by resampling LECTURES, not passages

Why lectures: passages inside one lecture are not independent (the same verse
runs for many passages), so resampling passages would give intervals that are
far too narrow. The lecture is the unit that was actually sampled.
"""
import csv
import os
import random
import re
import sys
from collections import defaultdict

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from score_by_tier_units import FIDELITY, TIERS, latest_per_pair, load_csv  # noqa: E402

SPANS = os.path.join(HERE, "data", "mention_spans.csv")
UNITS = os.path.join(HERE, "data", "units_v2.csv")
UNITS_HELDOUT = os.path.join(HERE, "data", "units_heldout_v1.csv")
LECTURES_NEW = os.path.join(HERE, "data", "lectures_new.csv")
SPLITS = os.path.join(HERE, "results", "results_v3")


def load_gold(spans=SPANS, units=UNITS):
    """Returns (lectures, gold, lang).

    lectures : {video_file: [unit_id, ...]} reviewed units only, in spoken order
    gold     : {unit_id: {verse_ref: tier}}  -- {} means annotated "no verse"
    lang     : {video_file: "kn" | "en"}
    """
    rows = latest_per_pair(load_csv(spans))
    reviewed = {r["unit_id"] for r in rows}

    gold = {u: {} for u in reviewed}
    bad = []
    for r in rows:
        # one label in the log was typed "Bg 2.25"; refs are compared as strings
        v = re.sub(r"^bg\s*", "BG ", (r.get("verse_ref") or "").strip(), flags=re.I)
        if v in ("", "NONE"):
            continue
        t = (r.get("mention_type") or "").strip().upper()
        if t not in TIERS:
            bad.append((r["unit_id"], t[:30]))
        cur = gold[r["unit_id"]].get(v)
        if cur is None or FIDELITY.get(t, 9) < FIDELITY.get(cur, 9):
            gold[r["unit_id"]][v] = t
    if bad:
        print(f"  !! {len(bad)} gold row(s) with a malformed tier "
              f"(counted overall, in no tier): {bad}", file=sys.stderr)

    grid = load_csv(units)
    if units == UNITS and os.path.exists(UNITS_HELDOUT):
        grid += load_csv(UNITS_HELDOUT)          # new-speaker lectures
    by_video, lang = defaultdict(list), {}
    for u in grid:
        lang[u["video_file"]] = u["language"]
        if u["unit_id"] in reviewed:
            by_video[u["video_file"]].append((int(u["unit_index"]), u["unit_id"]))
    lectures = {v: [uid for _, uid in sorted(us)] for v, us in by_video.items()}
    return lectures, gold, lang


def read_split(language, split, splits_dir=SPLITS):
    """Lecture names for 'dev' / 'test' / 'all' from the frozen split files.

    'heldout' is the new-speaker set (data/lectures_new.csv): never tuned on,
    and not part of 'all'.
    """
    if split == "heldout":
        return [r["video_file"] for r in load_csv(LECTURES_NEW)
                if r["language"] == language and r["status"] == "ready"]
    names = []
    for s in (["dev", "test"] if split == "all" else [split]):
        with open(os.path.join(splits_dir, f"{language}_{s}.txt"), encoding="utf-8") as f:
            names += [v.strip() for v in f.read().replace("\n", ",").split(",") if v.strip()]
    return names


def load_pred(path):
    with open(path, newline="", encoding="utf-8") as f:
        return {r["unit_id"]: [v for v in (r["detected_verses"] or "").split(";") if v]
                for r in csv.DictReader(f)}


def verse_number(ref):
    """'BG 2.47' -> 47; None for anything outside chapter 2."""
    tail = ref.split(".")[-1]
    return int(tail) if ref.startswith("BG 2.") and tail.isdigit() else None


def _counts(gold, pred, units):
    """tp / fp / fn and per-tier (hit, n) over a list of units."""
    tp = fp = fn = fired = 0
    tiers = {t: [0, 0] for t in TIERS}
    for u in units:
        g, p = gold[u], set(pred.get(u, ()))
        fired += bool(p)
        tp += len(p & set(g))
        fp += len(p - set(g))
        fn += len(set(g) - p)
        for v, t in g.items():
            if t in tiers:
                tiers[t][1] += 1
                tiers[t][0] += v in p
    return tp, fp, fn, fired, tiers


def _prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return p, r, (2 * p * r / (p + r) if p + r else 0.0)


def score(gold, pred, lectures, videos):
    """Micro-averaged P/R/F1 over (unit, verse) pairs in the given lectures."""
    units = [u for v in videos for u in lectures[v]]
    tp, fp, fn, fired, tiers = _counts(gold, pred, units)
    p, r, f1 = _prf(tp, fp, fn)
    return {"P": p, "R": r, "F1": f1, "tp": tp, "fp": fp, "fn": fn,
            "units": len(units), "fired": fired,
            "tiers": {t: tuple(c) for t, c in tiers.items()}}


def bootstrap(gold, pred, lectures, videos, n=1000, seed=0):
    """95% percentile CI for P, R, F1, resampling lectures with replacement."""
    per = [_counts(gold, pred, lectures[v])[:3] for v in videos]
    rng = random.Random(seed)
    draws = {"P": [], "R": [], "F1": []}
    for _ in range(n):
        s = [per[rng.randrange(len(per))] for _ in per]
        p, r, f1 = _prf(*(sum(x[i] for x in s) for i in range(3)))
        draws["P"].append(p)
        draws["R"].append(r)
        draws["F1"].append(f1)
    out = {}
    for k, xs in draws.items():
        xs.sort()
        out[k] = (xs[int(0.025 * n)], xs[min(int(0.975 * n), n - 1)])
    return out
