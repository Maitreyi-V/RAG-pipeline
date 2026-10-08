#!/usr/bin/env python3
"""
Discourse-continuity propagation over a base detector's predictions.

Gold annotation shows that when two consecutive units both ground a verse, they
share one 92% of the time (Kannada) / 89% (English). Every baseline here treats
units independently and discards that signal entirely.

This wraps any detector: where the base detector fires, its verse becomes the
"active" verse; where it is silent, the active verse is carried forward for up
to --max-gap units. A speaker expounding BG 2.29 for forty passages is still
expounding BG 2.29 even where no Sanskrit surfaces -- which is exactly the T4/T5
region every base method fails on.

Usage:
    python scripts/propagate.py results/final_l1_en.csv --max-gap 8 \
        --out results/final_prop_en.csv
"""
import argparse
import csv
import os

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
UNITS = os.path.join(HERE, "data", "units_v2.csv")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("base", help="predictions CSV from detect_units.py")
    ap.add_argument("--units", default=UNITS)
    ap.add_argument("--max-gap", type=int, default=8,
                    help="carry the active verse at most this many silent units")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    with open(args.units, newline="", encoding="utf-8") as f:
        units = list(csv.DictReader(f))
    with open(args.base, newline="", encoding="utf-8") as f:
        pred = {r["unit_id"]: [v for v in r["detected_verses"].split(";") if v]
                for r in csv.DictReader(f)}

    order = {}
    for u in units:
        order.setdefault(u["video_file"], []).append(u)
    for v in order:
        order[v].sort(key=lambda x: int(x["unit_index"]))

    out, n_prop = [], 0
    for video, us in order.items():
        active, gap = [], 0
        for u in us:
            uid = u["unit_id"]
            if uid not in pred:
                continue
            hits = pred[uid]
            if hits:
                active, gap = hits, 0
                out.append((uid, video, hits))
            elif active and gap < args.max_gap:
                gap += 1
                n_prop += 1
                out.append((uid, video, active))
            else:
                active = []
                out.append((uid, video, []))

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["unit_id", "video_file", "detected_verses"])
        for uid, video, hits in out:
            w.writerow([uid, video, ";".join(hits)])
    print(f"  propagated into {n_prop} silent units (max_gap={args.max_gap})")
    print(f"  wrote -> {os.path.relpath(args.out, HERE)}")


if __name__ == "__main__":
    main()
