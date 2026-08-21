#!/usr/bin/env python3
"""
Export annotation units to readable Markdown, one file per video.

For annotators who want to read a whole video offline, for the domain expert
adjudicating disagreements, and for reviewing units before annotating in the app.

Usage:
    python scripts/export_units.py                    # everything
    python scripts/export_units.py --language kn      # Kannada only
    python scripts/export_units.py --video kannada_07 # one video
    python scripts/export_units.py --out data/exports
"""
import argparse
import csv
import os
from collections import defaultdict

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_UNITS = os.path.join(HERE, "data", "units_v1.csv")
DEFAULT_OUT = os.path.join(HERE, "data", "exports")

LEGEND = ("Tag each unit: **T1** full Sanskrit recitation · **T2** partial Sanskrit · "
          "**T3** translation · **T4** paraphrase · **T5** allusion · "
          "otherwise **no verse here**.\n\n"
          "Notes prefixes: `OOS:` verse outside Chapter 2 · "
          "`ASR_DROP:` recitation audible but missing from transcript.\n")

TRADITION = {"kn": "Kannada / Dvaita", "en": "English / Advaita"}


def mmss(sec):
    m, s = divmod(int(float(sec)), 60)
    return f"{m}:{s:02d}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--units", default=DEFAULT_UNITS)
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--language", choices=["kn", "en"])
    ap.add_argument("--video")
    args = ap.parse_args()

    with open(args.units, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    by_video = defaultdict(list)
    for r in rows:
        if args.language and r["language"] != args.language:
            continue
        if args.video and r["video_file"] != args.video:
            continue
        by_video[r["video_file"]].append(r)

    if not by_video:
        raise SystemExit("No units matched.")

    os.makedirs(args.out, exist_ok=True)
    total_u = total_w = 0
    for vid in sorted(by_video):
        units = sorted(by_video[vid], key=lambda r: int(r["unit_index"]))
        lang = units[0]["language"]
        words = sum(int(u["n_words"]) for u in units)
        path = os.path.join(args.out, f"{vid}_units.md")
        with open(path, "w", encoding="utf-8") as f:
            f.write(f"# {vid} — annotation units (v1)\n\n")
            f.write(f"{len(units)} units · {words} words · {TRADITION.get(lang, lang)}\n\n")
            f.write(LEGEND)
            f.write("\n---\n\n")
            for u in units:
                head = f"### {int(u['unit_index'])+1}. `{u['unit_id']}`  ({u['n_words']} words)"
                if u["time_start"] not in ("", None):
                    head += f"  ·  {mmss(u['time_start'])}–{mmss(u['time_end'])}"
                f.write(head + "\n\n" + u["text"].strip() + "\n\n")
        print(f"  {vid:20} {len(units):4} units  {words:7} words  ->  {os.path.relpath(path, HERE)}")
        total_u += len(units)
        total_w += words

    print(f"\n  {len(by_video)} file(s) · {total_u} units · {total_w} words")


if __name__ == "__main__":
    main()
