#!/usr/bin/env python3
"""
Five lecture-level folds over all 51 lectures (work plan, Phase 0).

Each speaker's lectures are shuffled with a fixed seed and dealt to the folds in
turn, continuing from where the previous speaker stopped, so every fold gets a
similar number of lectures and each speaker is spread over as many folds as it
has lectures. Speakers are taken in a fixed order (most lectures first).

Writes data/folds_v1.csv: video_file, speaker, language, split, passages, fold.

Usage:
    python scripts/make_folds.py
"""
import csv
import os
import random
import sys
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

from evalkit import LECTURES_NEW, load_csv, load_gold, read_split  # noqa: E402

SEED = 2026
K = 5
OUT = os.path.join(HERE, "data", "folds_v1.csv")


def main():
    lectures, _, lang = load_gold()
    new_speaker = {r["video_file"]: r["speaker"] for r in load_csv(LECTURES_NEW)}
    split = {v: s for L in ("kn", "en") for s in ("dev", "test", "heldout") for v in read_split(L, s)}

    def speaker(v):
        return new_speaker.get(v) or ("original Kannada speaker" if lang[v] == "kn"
                                      else "original English speaker")

    by_speaker = defaultdict(list)
    for v in sorted(lectures):
        by_speaker[speaker(v)].append(v)
    rng = random.Random(SEED)
    fold, nxt = {}, 0
    for sp in sorted(by_speaker, key=lambda s: (-len(by_speaker[s]), s)):
        vids = sorted(by_speaker[sp])
        rng.shuffle(vids)
        for v in vids:
            fold[v] = nxt % K
            nxt += 1

    with open(OUT, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["video_file", "speaker", "language", "split", "passages", "fold"])
        for v in sorted(fold, key=lambda v: (fold[v], speaker(v), v)):
            w.writerow([v, speaker(v), lang[v], split.get(v, ""), len(lectures[v]), fold[v]])

    print(f"  wrote -> {os.path.relpath(OUT, HERE)}  ({len(fold)} lectures, seed {SEED})")
    for k in range(K):
        vids = [v for v in fold if fold[v] == k]
        print(f"  fold {k}: {len(vids):2} lectures, {sum(len(lectures[v]) for v in vids):4} passages | "
              + ", ".join(f"{s.split(' (')[0]} {n}" for s, n in sorted(Counter(speaker(v) for v in vids).items())))


if __name__ == "__main__":
    main()
