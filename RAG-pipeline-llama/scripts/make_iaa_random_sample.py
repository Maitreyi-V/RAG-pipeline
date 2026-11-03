#!/usr/bin/env python3
"""
E1b: a RANDOM sample for a second blind annotation pass (the E1 sample was the
first N passages of 7 lectures).

Sample: 40 Kannada + 60 English passages drawn at random (seed 2026) from all
51 lectures, excluding the 209 passages already double-annotated
(data/annotations_iaa/). Each passage's first annotator is the annotator of its
first-pass labels in data/mention_spans.csv (for a passage with no label row,
the lecture's main annotator). The second annotator is one of Maitreyi,
Prerana, Meghana S who is not the first annotator, assigned so that the load
is as even as possible (ties: alphabetical).

Writes data/iaa_random_sample.csv (no transcript text). Labels are made in
annotate_app.py with "Second pass (random sample)" and saved to
data/annotations_iaa_random/; score with
    python scripts/exp_e1_iaa.py --second-pass data/annotations_iaa_random --out results/exp_e1_iaa_random.txt

Usage:
    python scripts/make_iaa_random_sample.py
"""
import csv
import glob
import os
import random
import sys
from collections import Counter

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

from evalkit import UNITS, UNITS_HELDOUT, load_csv  # noqa: E402

SEED = 2026
N = {"kn": 40, "en": 60}
SECOND = ["Maitreyi", "Meghana S", "Prerana"]
OUT = os.path.join(HERE, "data", "iaa_random_sample.csv")


def main():
    units = load_csv(UNITS) + load_csv(UNITS_HELDOUT)
    done = {r["unit_id"] for f in glob.glob(os.path.join(HERE, "data", "annotations_iaa", "*.csv"))
            for r in csv.DictReader(open(f, encoding="utf-8"))}
    spans = list(csv.DictReader(open(os.path.join(HERE, "data", "mention_spans.csv"), encoding="utf-8")))
    first_of_unit = {}
    for r in spans:
        first_of_unit.setdefault(r["unit_id"], r["annotator"])
    main_of_lecture = {}
    for v in {u["video_file"] for u in units}:
        c = Counter(r["annotator"] for r in spans if r["video_file"] == v)
        main_of_lecture[v] = c.most_common(1)[0][0] if c else ""
    rng = random.Random(SEED)
    load = Counter()
    rows = []
    for lang in ("kn", "en"):
        pool = sorted(u["unit_id"] for u in units if u["language"] == lang and u["unit_id"] not in done)
        by_id = {u["unit_id"]: u for u in units}
        for uid in sorted(rng.sample(pool, N[lang]), key=lambda x: (by_id[x]["video_file"], int(by_id[x]["unit_index"]))):
            u = by_id[uid]
            first = first_of_unit.get(uid, main_of_lecture[u["video_file"]])
            second = min((a for a in SECOND if a != first), key=lambda a: (load[a], a))
            load[second] += 1
            rows.append(dict(unit_id=uid, video_file=u["video_file"], language=lang,
                             unit_index=u["unit_index"], first_annotator=first, second_annotator=second))
    with open(OUT, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {os.path.relpath(OUT, HERE)}: {len(rows)} passages "
          f"({sum(r['language'] == 'kn' for r in rows)} kn, {sum(r['language'] == 'en' for r in rows)} en) "
          f"from {len({r['video_file'] for r in rows})} lectures")
    print("  second annotator load:", dict(load))
    print("  first annotators:", dict(Counter(r["first_annotator"] for r in rows)))


if __name__ == "__main__":
    main()
