#!/usr/bin/env python3
"""
Gemini 3.1 Pro (API, low): run 1 vs run 2, passage by passage, all 55 lectures.
Reads the two label files written by gemini_runs_55.py.

  outcome of a run on a passage (labels compared with gold as verse sets)
    correct          same set as gold (a gold no-verse passage answered NONE included)
    extra verse      all gold verses named, plus one or more extra
    missed a verse   some gold verses named, none extra, at least one gold verse missing
    partly right     shares a verse with gold but also misses one and adds one
    one verse off    no shared verse, but a named verse is within 1 of a gold verse
    wrong verse      no shared verse, none within 1
    said NONE        gold has a verse, the run answered NONE (or the line was unreadable)
    false verse      gold is a digression (no verse), the run named a verse
  change             both right / both wrong / run 1 right only / run 2 right only

-> results/gemini_runs/run1_vs_run2_passages.csv (one row per passage)
   results/gemini_runs/run1_vs_run2_summary.txt

Usage:
    python scripts/gemini_runs_compare.py
"""
import csv
import os
import sys
from collections import Counter

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "scripts"))

from evalkit import load_gold, verse_number  # noqa: E402
from gemini_runs_55 import OUT, TUNE, groups  # noqa: E402

OUTCOMES = ["correct", "extra verse", "missed a verse", "partly right", "one verse off", "wrong verse",
            "said NONE", "false verse"]


def outcome(g, p):
    g, p = set(g), set(p)
    if g == p:
        return "correct"
    if not g:
        return "false verse"
    if not p:
        return "said NONE"
    if g < p:
        return "extra verse"
    if p < g:
        return "missed a verse"
    if g & p:
        return "partly right"
    gn = [verse_number(x) for x in g]
    if any(n is not None and any(m is not None and abs(n - m) <= 1 for m in gn) for n in map(verse_number, p)):
        return "one verse off"
    return "wrong verse"


def short(refs):
    return ";".join(sorted((r.replace("BG ", "") for r in refs), key=lambda x: int(x.split(".")[1]))) or "NONE"


def main():
    lectures, gold, _ = load_gold()
    labels = {}
    for k in ("run1_earlier_low_55", "run2_rerun_low_55"):
        with open(os.path.join(OUT, f"{k}.csv"), newline="", encoding="utf-8") as f:
            labels[k[:4]] = {r["unit_id"]: [x for x in r["refs"].split(";") if x] for r in csv.DictReader(f)}
    G = groups()
    group_of = {v: g for g, vs in G for v in vs}
    rows = []
    for g, vs in G:
        for v in vs:
            for i, u in enumerate(lectures[v], start=1):
                o1, o2 = outcome(gold[u], labels["run1"][u]), outcome(gold[u], labels["run2"][u])
                ch = ("both right" if o1 == o2 == "correct" else "run 1 right only" if o1 == "correct" else
                      "run 2 right only" if o2 == "correct" else "both wrong")
                rows.append(dict(group=g, lecture=v, passage=f"P{i:03d}", unit_id=u, gold=short(gold[u]),
                                 run1=short(labels["run1"][u]), run2=short(labels["run2"][u]),
                                 run1_outcome=o1, run2_outcome=o2, same_label=set(labels["run1"][u]) == set(labels["run2"][u]),
                                 change=ch))
    with open(os.path.join(OUT, "run1_vs_run2_passages.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    lines = ["Gemini 3.1 Pro (API, low): run 1 (7 Nov) vs run 2 (16 Nov), passage by passage, 55 lectures",
             "correct = the run's verse set equals gold; per-passage table: run1_vs_run2_passages.csv", "",
             "1. RIGHT / WRONG PER PASSAGE", "",
             f"  {'group':24}{'psg':>6}{'both right':>12}{'run 1 only':>12}{'run 2 only':>12}{'both wrong':>12}"
             f"{'same label':>12}{'run 1 right':>13}{'run 2 right':>13}"]
    sets = G + [("All 42 test", [v for g, vs in G[2:6] for v in vs]), ("All 55", [v for _, vs in G for v in vs])]
    for g, vs in sets:
        rs = [r for r in rows if r["lecture"] in vs]
        c = Counter(r["change"] for r in rs)
        n1 = sum(r["run1_outcome"] == "correct" for r in rs)
        n2 = sum(r["run2_outcome"] == "correct" for r in rs)
        lines.append(f"  {g:24}{len(rs):>6}{c['both right']:>12}{c['run 1 right only']:>12}{c['run 2 right only']:>12}"
                     f"{c['both wrong']:>12}{sum(r['same_label'] for r in rs):>12}{n1:>13}{n2:>13}")
    lines += ["", "2. WHAT WENT WRONG (passages per outcome)", "",
              f"  {'group':24}{'run':>5}" + "".join(f"{o:>16}" for o in OUTCOMES)]
    for g, vs in sets:
        rs = [r for r in rows if r["lecture"] in vs]
        for k in ("run1", "run2"):
            c = Counter(r[f"{k}_outcome"] for r in rs)
            lines.append(f"  {g if k == 'run1' else '':24}{k[-1]:>5}" + "".join(f"{c[o]:>16}" for o in OUTCOMES))
    lines += ["", "3. HOW THE OUTCOME CHANGED, ALL 55 (run 1 outcome -> run 2 outcome, changed passages only)", ""]
    c = Counter((r["run1_outcome"], r["run2_outcome"]) for r in rows if r["run1_outcome"] != r["run2_outcome"])
    for (a, b), n in c.most_common():
        lines.append(f"  {a:16} -> {b:16} {n:>5}")
    lines += ["", "4. PER LECTURE (passages: run 1 right only / run 2 right only / labels differ)", ""]
    for g, vs in G:
        lines.append(f"  -- {g} --")
        for v in vs:
            rs = [r for r in rows if r["lecture"] == v]
            c = Counter(r["change"] for r in rs)
            lines.append(f"  {v:26}{len(rs):>5}   run 1 only {c['run 1 right only']:>3}   run 2 only {c['run 2 right only']:>3}"
                         f"   differ {sum(not r['same_label'] for r in rs):>3}")
    text = "\n".join(lines)
    open(os.path.join(OUT, "run1_vs_run2_summary.txt"), "w", encoding="utf-8").write(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
