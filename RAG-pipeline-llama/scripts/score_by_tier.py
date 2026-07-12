#!/usr/bin/env python3
"""
Re-score verse detection BROKEN DOWN BY mention type (T1-T5).

This produces the paper's headline table: for each detection paradigm, precision/
recall/F1 per mention tier -- showing string matching solves T1, embeddings recover
T2-T3, and everything collapses on T4-T5.

Inputs:
  1. A detection results JSON with a `per_video` list where each item has
     expected/detected verse lists. Handles BOTH formats in this repo:
       - Layer 1 (detector_kannada_full.json): keys expected_verses / detected_verses
       - Layer 3 (layer3_results.json):        keys expected / detected
  2. data/mention_types.csv with a filled `mention_type` column.

Usage:
  python scripts/score_by_tier.py results/detector_kannada_full.json --label "Layer1 (Kannada)"
  python scripts/score_by_tier.py layer3_results.json --label "Layer3 (English)"
  python scripts/score_by_tier.py results/layer4_gpt.json --tiers data/mention_types.csv

A gold mention is a TP if its verse is in that video's `detected` list, else FN.
Detected verses that are NOT gold mentions are counted as false positives and grouped
by verse (feeds the attractor-verse analysis, e.g. BG 2.39 / 2.45).
"""
import csv, json, sys, argparse, re
from collections import defaultdict

TIERS = ["T1", "T2", "T3", "T4", "T5"]


def norm_video(name):
    """Normalise the many filename conventions to a common key.
    'gita-1st video.txt' -> 'gita-1st_video' ; 'video_01.json' -> 'video_01'
    """
    n = re.sub(r"\.(txt|json)$", "", str(name).strip().lower())
    n = n.replace(" ", "_")
    return n


def load_tiers(path):
    tiers = {}          # (norm_video, verse) -> tier
    videos_seen = set()
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            v = norm_video(r["video_file"]); verse = r["verse_ref"].strip()
            t = (r.get("mention_type") or "").strip().upper()
            videos_seen.add(v)
            if t:
                tiers[(v, verse)] = t
    return tiers, videos_seen


def load_detections(path):
    d = json.load(open(path))
    pv = d.get("per_video", d if isinstance(d, list) else [])
    out = {}            # norm_video -> {"expected": set, "detected": set}
    for item in pv:
        v = norm_video(item.get("filename") or item.get("video_id") or item.get("video"))
        exp = item.get("expected_verses") or item.get("expected") or []
        det = item.get("detected_verses") or item.get("detected") or []
        out[v] = {"expected": set(exp), "detected": set(det)}
    return out


def prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("detections", help="detection results JSON (per_video with expected/detected)")
    ap.add_argument("--tiers", default="data/mention_types.csv")
    ap.add_argument("--label", default="paradigm")
    args = ap.parse_args()

    tiers, tier_videos = load_tiers(args.tiers)
    dets = load_detections(args.detections)

    if not tiers:
        print(f"WARNING: no rows in {args.tiers} have a mention_type filled in yet.")
        print("         Fill the tier column, then re-run. Showing untyped totals only.\n")

    # Per-tier TP/FN over gold mentions; FPs collected separately (no gold tier).
    tp = defaultdict(int); fn = defaultdict(int)
    untyped = 0; unmatched_videos = set()
    fp_by_verse = defaultdict(int); total_fp = 0

    for v, dd in dets.items():
        if v not in tier_videos and tiers:
            unmatched_videos.add(v)
        for verse in dd["expected"]:
            hit = verse in dd["detected"]
            t = tiers.get((v, verse))
            if t is None:
                untyped += 1
                t = "UNTYPED"
            (tp if hit else fn)[t] += 1
        for verse in dd["detected"] - dd["expected"]:
            fp_by_verse[verse] += 1; total_fp += 1

    print("=" * 64)
    print(f" DETECTION F1 BY MENTION TYPE  --  {args.label}")
    print(f" detections: {args.detections}")
    print("=" * 64)
    print(f"  {'tier':<8}{'TP':>5}{'FN':>5}{'recall':>9}   (precision/F1 use pooled FP below)")
    keys = TIERS + (["UNTYPED"] if untyped else [])
    tot_tp = tot_fn = 0
    for t in keys:
        if tp[t] + fn[t] == 0:
            continue
        _, r, _ = prf(tp[t], 0, fn[t])
        print(f"  {t:<8}{tp[t]:>5}{fn[t]:>5}{r:>9.3f}")
        tot_tp += tp[t]; tot_fn += fn[t]

    # Overall P/R/F1 (FP is not tier-specific, so precision is only meaningful pooled)
    P, R, F = prf(tot_tp, total_fp, tot_fn)
    print("  " + "-" * 40)
    print(f"  {'OVERALL':<8}{tot_tp:>5}{tot_fn:>5}{R:>9.3f}   FP={total_fp}  P={P:.3f}  F1={F:.3f}")

    if fp_by_verse:
        print("\n  Attractor verses (most-frequent false positives):")
        for verse, c in sorted(fp_by_verse.items(), key=lambda x: -x[1])[:8]:
            print(f"    {verse:<10} x{c}")

    if untyped:
        print(f"\n  NOTE: {untyped} gold mentions have no tier yet -> counted as UNTYPED.")
    if unmatched_videos:
        print(f"\n  NOTE: {len(unmatched_videos)} detection videos didn't match any tier CSV "
              f"video_file: {sorted(unmatched_videos)[:6]}")
        print("        Fix video_file naming in mention_types.csv so the join lands.")


if __name__ == "__main__":
    main()
