#!/usr/bin/env python3
"""
Pilot: a frontier LLM labels a WHOLE lecture at once (run by hand in claude.ai).

  prepare   writes, per lecture, one text file to paste into a NEW chat: the
            instructions, the 72-verse catalogue, and the numbered passages
            (P001, P002, ...), plus the passage-id -> unit_id map
  score     reads the answers saved as results/<lecture>.txt and scores them
            (strict / within-1 / theme F1) next to our locked system and plain
            carry on the same lectures

Lectures, fixed by rule before anything is run (seed below):
  prompt check  kannada_10 (tuning; the shortest lecture) - format check only
  test          video_09 (the English test lecture with the most no-verse passages),
                one English and one Kannada test lecture drawn at random

Files contain transcript text, so they go outside the repo:
  /Users/maitreyi/Documents/Capstone/review/llm_pilot/

Usage:
    python scripts/llm_pilot.py prepare
    python scripts/llm_pilot.py score
"""
import argparse
import csv
import os
import random
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

from evalkit import UNITS, load_csv, load_gold, read_split  # noqa: E402

OUT = "/Users/maitreyi/Documents/Capstone/review/llm_pilot"
SEED = 2026
PROMPT_CHECK = "kannada_10"

INSTRUCTIONS = """You will label a transcript of ONE lecture on the Bhagavad Gita, Chapter 2.

The transcript is split into numbered passages (about 100 words each), in the order they were spoken. Below the passages' instructions you will find a catalogue of all 72 verses of Chapter 2.

For EACH passage, decide which verse(s) of Chapter 2 the speaker is reciting, quoting, translating, explaining or paraphrasing IN THAT PASSAGE.
- Use the whole lecture as context: speakers often explain one verse over several passages without repeating its words.
- A passage can have more than one verse.
- If a passage is about no verse of Chapter 2 (introduction, a story, a digression, an audience question, a verse from another chapter or text), label it NONE.
- Use verse numbers 2.1 to 2.72 only.

OUTPUT FORMAT - follow exactly. One line per passage, every passage from P001 to {last}, in order, inside ONE code block, nothing else before or after:
```
P001|2.47
P002|2.47;2.48
P003|NONE
```
"""


def lectures_for_pilot():
    lectures, gold, _ = load_gold()
    en = [v for v in read_split("en", "test") if v in lectures]
    kn = [v for v in read_split("kn", "test") if v in lectures]
    hardest = max(en, key=lambda v: sum(1 for u in lectures[v] if not gold[u]))
    rng = random.Random(SEED)
    return [PROMPT_CHECK, hardest,
            rng.choice(sorted(v for v in en if v != hardest)), rng.choice(sorted(kn))]


def catalogue():
    from roman_detector import romanize
    from verse_database import VERSES_BG_CH2
    from verse_translations import VERSE_TRANSLATIONS
    lines = ["VERSE CATALOGUE (Bhagavad Gita Chapter 2)"]
    for v in VERSES_BG_CH2:
        lines.append(f"\n{v['ref'].replace('BG ', '')}\n"
                     f"  Devanagari: {v['devanagari']}\n"
                     f"  Kannada script: {v['kannada']}\n"
                     f"  Roman: {re.sub(r'\s+', ' ', romanize(v['devanagari'])).strip()}\n"
                     f"  Translation: {VERSE_TRANSLATIONS[v['ref']][0]}")
    return "\n".join(lines)


def prepare():
    os.makedirs(os.path.join(OUT, "results"), exist_ok=True)
    lectures, _, _ = load_gold()
    text = {u["unit_id"]: u["text"] for u in load_csv(UNITS)}
    cat = catalogue()
    for v in lectures_for_pilot():
        us = lectures[v]
        ids = [f"P{k + 1:03d}" for k in range(len(us))]
        body = "\n\n".join(f"{i}: {text[u]}" for i, u in zip(ids, us))
        with open(os.path.join(OUT, f"{v}_PASTE_THIS.txt"), "w", encoding="utf-8") as f:
            f.write(INSTRUCTIONS.format(last=ids[-1]) + "\n" + cat
                    + f"\n\nLECTURE PASSAGES ({len(us)} passages)\n\n" + body + "\n")
        with open(os.path.join(OUT, f"{v}_ids.csv"), "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["passage_id", "unit_id"])
            w.writerows(zip(ids, us))
        print(f"  {v:16} {len(us):4} passages -> {v}_PASTE_THIS.txt")
    print(f"  written to {OUT}")


def parse(path, ids):
    """-> ({unit_id: [refs]}, problems)"""
    m = dict(ids)
    pred, bad, outside = {}, [], 0
    for line in open(path, encoding="utf-8"):
        line = line.strip().strip("`")
        if not line:
            continue
        mm = re.match(r"^(P\d{3})\s*\|\s*(.+)$", line)
        if not mm or mm.group(1) not in m:
            bad.append(line[:60])
            continue
        refs = []
        for x in re.split(r"[;,\s]+", mm.group(2).strip()):
            if x.upper() == "NONE" or not x:
                continue
            n = re.match(r"^2\.(\d{1,2})$", x)
            if n and 1 <= int(n.group(1)) <= 72:
                refs.append(f"BG 2.{int(n.group(1))}")
            else:
                outside += 1
        pred[m[mm.group(1)]] = refs
    missing = [p for p, u in ids if u not in pred]
    return pred, dict(unparsed=bad, missing=missing, outside_ch2=outside)


def score():
    from baselines import carry
    from freeze_grounding import f1_near, f1_strict, near_counts, strict_counts, theme_counts
    from step4_viterbi import FEATS, OFF, predict_locked
    lectures, gold, _ = load_gold()
    z = np.load(FEATS)
    refs = list(z["refs"])
    row = {u: i for i, u in enumerate(z["unit_ids"])}
    for v in lectures_for_pilot():
        path = os.path.join(OUT, "results", f"{v}.txt")
        if not os.path.exists(path):
            print(f"\n  {v}: no results/{v}.txt yet")
            continue
        ids = [(r["passage_id"], r["unit_id"]) for r in load_csv(os.path.join(OUT, f"{v}_ids.csv"))]
        llm, problems = parse(path, ids)
        us = lectures[v]
        dec = predict_locked({v: us})[v]
        ours = {u: ([] if st == OFF else [refs[st]]) + [refs[j] for j in extra]
                for u, (st, extra, _, _) in zip(us, dec)}
        anchors = {u: [refs[j] for j in np.flatnonzero(z["anchor"][row[u]])] for u in us}
        plain = carry(us, anchors)
        tag = "  (prompt check, tuning lecture)" if v == PROMPT_CHECK else ""
        print(f"\n  {v}   {len(us)} passages{tag}")
        print(f"    parsing: {len(problems['missing'])} passage(s) missing, "
              f"{len(problems['unparsed'])} unparsed line(s), "
              f"{problems['outside_ch2']} verse(s) outside 2.1-2.72")
        for name, pred in (("frontier LLM (chat)", llm), ("ours (locked)", ours), ("plain carry", plain)):
            print(f"    {name:20} strict {f1_strict(strict_counts(gold, pred, us)):.3f}   "
                  f"+-1 {f1_near(near_counts(gold, pred, us)):.3f}   "
                  f"theme {f1_strict(theme_counts(gold, pred, us)):.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["prepare", "score"])
    args = ap.parse_args()
    prepare() if args.step == "prepare" else score()


if __name__ == "__main__":
    main()
