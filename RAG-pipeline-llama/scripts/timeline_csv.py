#!/usr/bin/env python3
"""
Timeline figure data for one lecture (frozen outputs, no text): per passage the
gold verse(s), Plain Carry, GitaGround-Local, GitaGround-Cue, and the LLM cue's
P_off. Verses are written as numbers (47 = BG 2.47), several joined by ';',
off-verse as OFF.

Writes results/figures/timeline_<lecture>.csv.

Usage:
    python scripts/timeline_csv.py --lecture video_09
"""
import argparse
import csv
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

from baselines import carry  # noqa: E402
from evalkit import load_gold, verse_number  # noqa: E402
from step4_viterbi import FEATS, OFF, OUT, predict_locked  # noqa: E402


def cell(refs):
    ns = sorted({n for n in map(verse_number, refs) if n})
    return ";".join(str(n) for n in ns) if ns else "OFF"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lecture", default="video_09")
    v = ap.parse_args().lecture
    lectures, gold, _ = load_gold()
    us = lectures[v]
    z = np.load(FEATS)
    refs = list(z["refs"])
    row = {u: i for i, u in enumerate(z["unit_ids"])}
    anchors = {u: [refs[j] for j in np.flatnonzero(z["anchor"][row[u]])] for u in us}

    def decoded(path):
        dec = predict_locked({v: us}, "a05", path=path)[v]
        return [([] if st == OFF else [refs[st]]) + [refs[j] for j in extra] for st, extra, _, _ in dec]
    local = decoded(OUT)
    cue_path = os.path.join(HERE, "results", "v2_teacher145.pkl")
    cue = decoded(cue_path)
    p_off = [p for _, _, _, p in predict_locked({v: us}, "a05", path=cue_path)[v]]
    plain = carry(us, anchors)
    out = os.path.join(HERE, "results", "figures", f"timeline_{v}.csv")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["passage_idx", "gold", "plain_carry", "local", "cue", "p_off_cue"])
        for k, u in enumerate(us):
            w.writerow([k + 1, cell(gold[u]), cell(plain.get(u, [])), cell(local[k]), cell(cue[k]), f"{p_off[k]:.2f}"])
    print(f"wrote {os.path.relpath(out, HERE)} ({len(us)} passages)")


if __name__ == "__main__":
    main()
