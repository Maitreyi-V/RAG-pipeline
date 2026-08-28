#!/usr/bin/env python3
"""
Dataset statistics + per-tier scoring for UNIT-level verse-grounding annotation.

Supersedes score_by_tier.py, which was keyed on (video_file, verse_ref) and
silently kept only one tier per verse per video.

Two modes:

  STATS (no detections given) — run this early and often:
      annotation coverage, tier distribution, mentions per video, a projection
      of final corpus size, and counts of the OOS: / ASR_DROP: note prefixes.

  SCORING (--detections results/detector_kannada_full.json):
      per-tier RECALL for a detector, plus overall precision/recall/F1.

Why per-tier RECALL and not F1: detector output is video-level (a set of verses
per video), so a false positive has no tier to attribute it to. Recall per tier
is the quantity the paper's claim is actually about — "string matching recovers
T1 but collapses on T5". Overall precision is reported separately.

Usage:
    python scripts/score_by_tier_units.py
    python scripts/score_by_tier_units.py --detections results/detector_kannada_full.json
    python scripts/score_by_tier_units.py --language kn --total-videos 30
"""
import argparse
import csv
import json
import os
import re
import sys
from collections import Counter, defaultdict

TIERS = ["T1", "T2", "T3", "T4", "T5"]
FIDELITY = {"T1": 0, "T2": 1, "T3": 2, "T4": 3, "T5": 4, "NONE": 5}

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_SPANS = os.path.join(HERE, "data", "mention_spans.csv")
DEFAULT_UNITS = os.path.join(HERE, "data", "units_v1.csv")


def norm_video(name):
    n = re.sub(r"\.(txt|json)$", "", str(name).strip().lower())
    return n.replace(" ", "_")


def load_csv(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def latest_per_pair(rows):
    """Resolve the append-only log to one current view.

    Keyed on (unit, annotator, VERSE): re-saving one verse must not shadow a
    different verse labelled on the same unit in a separate save. Then, per
    (unit, annotator), a NONE row and real verses are mutually exclusive --
    whichever was recorded later wins.
    """
    best = {}
    for r in rows:
        k = (r["unit_id"], r["annotator"], (r.get("verse_ref") or "").strip())
        if k not in best or r.get("created_at", "") > best[k].get("created_at", ""):
            best[k] = r
    by_ua = defaultdict(list)
    for r in best.values():
        by_ua[(r["unit_id"], r["annotator"])].append(r)
    out = []
    for rs in by_ua.values():
        nones = [r for r in rs if (r.get("verse_ref") or "").strip() == "NONE"]
        reals = [r for r in rs if (r.get("verse_ref") or "").strip() != "NONE"]
        if nones and reals:
            nn = max(r.get("created_at", "") for r in nones)
            nr = max(r.get("created_at", "") for r in reals)
            out.extend(nones if nn > nr else reals)
        else:
            out.extend(rs)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spans", default=DEFAULT_SPANS)
    ap.add_argument("--units", default=DEFAULT_UNITS)
    ap.add_argument("--detections", help="detector JSON with per_video results")
    ap.add_argument("--unit-detections",
                    help="CSV from scripts/detect_units.py (unit_id, detected_verses) "
                         "-- scores localisation, the real grounding task")
    ap.add_argument("--annotator", help="use only this annotator's labels")
    ap.add_argument("--language", choices=["kn", "en"], help="restrict to one language")
    ap.add_argument("--videos", help="comma-separated video_file list (dev/test split)")
    ap.add_argument("--total-videos", type=int, default=30,
                    help="corpus size to project towards (default 30)")
    args = ap.parse_args()

    if not os.path.exists(args.spans):
        sys.exit(f"No annotations at {args.spans}")

    rows = latest_per_pair(load_csv(args.spans))
    if args.annotator:
        rows = [r for r in rows if r["annotator"] == args.annotator]

    units = load_csv(args.units)
    lang_of = {u["unit_id"]: u["language"] for u in units}
    units_per_video = Counter(u["video_file"] for u in units)
    if args.language:
        rows = [r for r in rows if lang_of.get(r["unit_id"]) == args.language]
    if args.videos:
        keep = {v.strip() for v in args.videos.split(",") if v.strip()}
        rows = [r for r in rows if r["video_file"] in keep]

    if not rows:
        sys.exit("No annotation rows after filtering.")

    reviewed = {r["unit_id"] for r in rows}
    videos = sorted({r["video_file"] for r in rows})
    mentions = [r for r in rows if (r.get("verse_ref") or "").strip() not in ("", "NONE")]

    print("=" * 70)
    print("DATASET STATISTICS")
    print("=" * 70)
    print(f"  videos touched     : {len(videos)}")
    print(f"  units reviewed     : {len(reviewed)}")
    print(f"  verse mentions     : {len(mentions)}")
    print(f"  units with a verse : {len({r['unit_id'] for r in mentions})} "
          f"({len({r['unit_id'] for r in mentions})/max(len(reviewed),1):.1%} of reviewed)")
    print(f"  distinct verses    : {len({r['verse_ref'] for r in mentions})}")

    print("\n  per-video coverage (reviewed / total units):")
    done_videos = []
    for v in videos:
        n = len({r["unit_id"] for r in rows if r["video_file"] == v})
        tot = units_per_video.get(v, n)
        m = len([r for r in mentions if r["video_file"] == v])
        flag = "  <- complete" if n >= tot else ""
        if n >= tot:
            done_videos.append(v)
        print(f"    {v:20} {n:4}/{tot:<4}  mentions: {m:3}{flag}")

    print("\n  tier distribution:")
    dist = Counter((r.get("mention_type") or "").strip().upper() for r in mentions)
    for t in TIERS:
        c = dist.get(t, 0)
        bar = "#" * min(int(40 * c / max(len(mentions), 1)), 40)
        print(f"    {t}  {c:4}  ({c/max(len(mentions),1):5.1%})  {bar}")
    missing = [t for t in TIERS if dist.get(t, 0) == 0]
    if missing:
        print(f"    !! no examples yet for: {', '.join(missing)}")

    # projection
    if done_videos:
        per = sum(len([r for r in mentions if r["video_file"] == v]) for v in done_videos) / len(done_videos)
        print(f"\n  PROJECTION (from {len(done_videos)} fully-annotated video(s)):")
        print(f"    mean mentions/video : {per:.1f}")
        print(f"    projected at {args.total_videos} videos : ~{per*args.total_videos:.0f} mentions")
        if per * args.total_videos < 150:
            print("    !! thin for a benchmark — consider adding videos while it is cheap")
    else:
        print("\n  PROJECTION: no video fully annotated yet — finish one for an estimate.")

    # note prefixes
    notes = [(r.get("notes") or "").strip() for r in rows]
    pref = Counter()
    for n in notes:
        m = re.match(r"([A-Z_]+):", n)
        if m:
            pref[m.group(1)] += 1
    if pref:
        print("\n  flagged notes:")
        for k, c in pref.most_common():
            print(f"    {k+':':12} {c}")

    # ---------------- unit-level (localisation) scoring ----------------
    if args.unit_detections:
        with open(args.unit_detections, newline="", encoding="utf-8") as f:
            pred = {r["unit_id"]: {v for v in (r["detected_verses"] or "").split(";") if v}
                    for r in csv.DictReader(f)}

        # Only units a human actually reviewed can be scored: for those we know
        # the truth, including "no verse here" (the NONE rows).
        scored_units = reviewed & set(pred)
        gold_pairs = {}
        for r in mentions:
            if r["unit_id"] in scored_units:
                t = (r.get("mention_type") or "").strip().upper()
                k = (r["unit_id"], r["verse_ref"].strip())
                if k not in gold_pairs or FIDELITY.get(t, 9) < FIDELITY.get(gold_pairs[k], 9):
                    gold_pairs[k] = t
        pred_pairs = {(u, v) for u in scored_units for v in pred[u]}

        print("\n" + "=" * 70)
        print(f"UNIT-LEVEL (LOCALISATION) — {os.path.basename(args.unit_detections)}")
        print("=" * 70)
        print(f"  reviewed units scored : {len(scored_units)}")
        print(f"  gold mentions         : {len(gold_pairs)}")
        print(f"\n  {'tier':6} {'gold':>6} {'found':>6} {'recall':>8}")
        print("  " + "-" * 30)
        for t in TIERS:
            g = [k for k, tt in gold_pairs.items() if tt == t]
            if not g:
                print(f"  {t:6} {0:>6} {'-':>6} {'-':>8}")
                continue
            hit = sum(1 for k in g if k in pred_pairs)
            print(f"  {t:6} {len(g):>6} {hit:>6} {hit/len(g):>8.3f}")

        tp = len(set(gold_pairs) & pred_pairs)
        fn = len(gold_pairs) - tp
        fp = len(pred_pairs - set(gold_pairs))
        p = tp / (tp + fp) if tp + fp else 0.0
        r_ = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * p * r_ / (p + r_) if p + r_ else 0.0
        print("  " + "-" * 30)
        print(f"  OVERALL  TP={tp} FP={fp} FN={fn}   P={p:.3f} R={r_:.3f} F1={f1:.3f}")

        empty = {u for u in scored_units if not any(k[0] == u for k in gold_pairs)}
        spurious = sum(1 for u in empty if pred[u])
        print(f"\n  units gold-labelled 'no verse': {len(empty)}"
              f"   of which the detector fired on: {spurious}")

        top_fp = Counter(v for u, v in (pred_pairs - set(gold_pairs)))
        if top_fp:
            print("\n  attractor verses (most frequent false positives):")
            for v, c in top_fp.most_common(8):
                print(f"    {v:10} {c}")
        return

    # ---------------- video-level scoring ----------------
    if not args.detections:
        print("\n" + "=" * 70)
        print("  (stats only — pass --detections <file.json> for per-tier recall)")
        print("=" * 70)
        return

    det = json.load(open(args.detections, encoding="utf-8"))
    pv = det.get("per_video", det if isinstance(det, list) else [])
    detected = {}
    for item in pv:
        v = norm_video(item.get("filename") or item.get("video_id") or item.get("video"))
        detected[v] = set(item.get("detected_verses") or item.get("detected") or [])

    # gold mention = (video, verse) with its highest-fidelity tier
    gold = {}
    for r in mentions:
        key = (norm_video(r["video_file"]), r["verse_ref"].strip())
        t = (r.get("mention_type") or "").strip().upper()
        if key not in gold or FIDELITY.get(t, 9) < FIDELITY.get(gold[key], 9):
            gold[key] = t

    scored = [(v, verse, t) for (v, verse), t in gold.items() if v in detected]
    print("\n" + "=" * 70)
    print(f"PER-TIER RECALL — {os.path.basename(args.detections)}")
    print("=" * 70)
    if not scored:
        print("  no overlap between annotated videos and this detection file")
        return
    print(f"  gold mentions in scored videos: {len(scored)}"
          f"   (videos: {len({v for v,_,_ in scored})})")
    print(f"\n  {'tier':6} {'gold':>6} {'found':>6} {'recall':>8}")
    print("  " + "-" * 30)
    for t in TIERS:
        g = [(v, verse) for v, verse, tt in scored if tt == t]
        if not g:
            print(f"  {t:6} {0:>6} {'-':>6} {'-':>8}")
            continue
        hit = sum(1 for v, verse in g if verse in detected[v])
        print(f"  {t:6} {len(g):>6} {hit:>6} {hit/len(g):>8.3f}")

    allg = [(v, verse) for v, verse, _ in scored]
    tp = sum(1 for v, verse in allg if verse in detected[v])
    fn = len(allg) - tp
    goldset = defaultdict(set)
    for v, verse in allg:
        goldset[v].add(verse)
    fp = sum(len(detected[v] - goldset[v]) for v in goldset)
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    print("  " + "-" * 30)
    print(f"  OVERALL  TP={tp} FP={fp} FN={fn}   P={p:.3f} R={r:.3f} F1={f1:.3f}")
    print("\n  NOTE: detections are video-level, so a verse occurring in several")
    print("        units counts as found if detected anywhere in that video.")
    print("        Precision is not tier-attributable and is reported overall.")


if __name__ == "__main__":
    main()
