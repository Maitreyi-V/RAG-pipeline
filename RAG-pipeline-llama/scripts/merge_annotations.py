#!/usr/bin/env python3
"""
Merge annotation files from several annotators into data/mention_spans.csv.

Each annotator runs annotate_app.py on their own machine and produces their own
data/mention_spans.csv. Collect those files, rename them per person, and merge.

Safe to run repeatedly: exact duplicate rows are dropped, and the master file is
backed up before every write.

Usage:
    python scripts/merge_annotations.py from_meghana_g.csv from_meghana_s.csv
    python scripts/merge_annotations.py *.csv --into data/mention_spans.csv
    python scripts/merge_annotations.py incoming.csv --dry-run
"""
import argparse
import csv
import os
import shutil
import sys
from collections import Counter
from datetime import datetime

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MASTER = os.path.join(HERE, "data", "mention_spans.csv")
COLS = ["unit_id", "video_file", "annotator", "verse_ref", "mention_type",
        "has_explicit_ref", "uncertain", "notes", "created_at"]


def read(path):
    if not os.path.exists(path):
        return []
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        missing = [c for c in COLS if c not in r]
        if missing:
            sys.exit(f"{path}: missing column(s) {missing} — is this a mention_spans file?")
    return rows


def key(r):
    return tuple((r.get(c) or "").strip() for c in COLS)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+", help="incoming annotator CSVs")
    ap.add_argument("--into", default=MASTER)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--split", action="store_true",
                    help="after merging, also rewrite data/annotations/<name>.csv "
                         "for every annotator (use while teammates still write to "
                         "the shared mention_spans.csv)")
    args = ap.parse_args()

    master = read(args.into)
    seen = {key(r) for r in master}
    print(f"master {os.path.relpath(args.into, HERE)}: {len(master)} rows, "
          f"{len({r['annotator'] for r in master})} annotator(s)")

    added, dupes = [], 0
    for path in args.files:
        rows = read(path)
        new = []
        for r in rows:
            k = key(r)
            if k in seen:
                dupes += 1
            else:
                seen.add(k)
                new.append({c: r.get(c, "") for c in COLS})
        who = Counter(r["annotator"] for r in rows)
        vids = sorted({r["video_file"] for r in rows})
        print(f"\n  {os.path.basename(path)}")
        print(f"    rows      : {len(rows)}  (new: {len(new)}, duplicate: {len(rows)-len(new)})")
        print(f"    annotators: {dict(who)}")
        print(f"    videos    : {', '.join(vids)}")
        if not who:
            print("    !! no rows")
        if "" in who:
            print("    !! blank annotator name — fix before merging")
        added += new

    if not added and not args.split:
        print("\nNothing new to merge.")
        return
    if not added:
        print("\nNothing new to merge — re-splitting existing rows.")

    combined = master + added
    print(f"\nresult: {len(combined)} rows "
          f"({len(master)} existing + {len(added)} new, {dupes} duplicates skipped)")
    print(f"        {len({r['annotator'] for r in combined})} annotators, "
          f"{len({r['unit_id'] for r in combined})} units")

    ov = Counter()
    for r in combined:
        ov[r["unit_id"]] = ov[r["unit_id"]]
    per_unit = {}
    for r in combined:
        per_unit.setdefault(r["unit_id"], set()).add(r["annotator"])
    shared = sum(1 for a in per_unit.values() if len(a) > 1)
    print(f"        units with 2+ annotators (kappa-eligible): {shared}")
    if not shared:
        print("        !! no overlap yet — two people must annotate the SAME video")

    if args.dry_run:
        print("\n--dry-run: nothing written.")
        return

    if os.path.exists(args.into):
        bak = args.into + "." + datetime.now().strftime("%Y%m%d-%H%M%S") + ".bak"
        shutil.copy2(args.into, bak)
        print(f"\nbackup: {os.path.relpath(bak, HERE)}")

    with open(args.into, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLS)
        w.writeheader()
        w.writerows(combined)
    print(f"wrote  : {os.path.relpath(args.into, HERE)}")

    if args.split:
        import re
        outdir = os.path.join(HERE, "data", "annotations")
        os.makedirs(outdir, exist_ok=True)
        by_ann = {}
        for r in combined:
            by_ann.setdefault(r["annotator"], []).append(r)
        print("\nsplit into per-annotator files:")
        for name, rs in sorted(by_ann.items()):
            slug = re.sub(r"[^a-z0-9]+", "_", name.strip().lower()).strip("_") or "unnamed"
            p = os.path.join(outdir, slug + ".csv")
            with open(p, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=COLS)
                w.writeheader()
                w.writerows(rs)
            print(f"  {name:14} {len(rs):4} rows -> data/annotations/{slug}.csv")


if __name__ == "__main__":
    main()
