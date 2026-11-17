#!/usr/bin/env python3
"""
Gemini 3.1 Pro (API, thinking low), end to end, on all 55 lectures: one results file per run.

  run1  the earlier run (7 Nov, 4a628fb; tuning 7be6186; 4 new lectures new4_gemini3l.py)
        answers: review/llm_frontier/answers_api_gemini-3.1-pro-preview_low/ (+ tuning/)
  run2  the re-run (16 Nov, same paste files, model and settings; post-hoc)
        answers: review/llm_frontier/answers_api_gemini-3.1-pro-preview_low_rerun/ (+ tuning/)

Scoring as llm_frontier.py: a line that does not parse is ignored and a missing passage = NONE;
strict / +-1 / theme F1 micro over (passage, verse) pairs; groups pooled; cost $2 / $12 per 1M
(output includes thinking), from each lecture's usage file.

-> results/gemini_runs/run1_earlier_low_55.txt, run2_rerun_low_55.txt (+ .csv labels per passage)

Usage:
    python scripts/gemini_runs_55.py
"""
import csv
import json
import os
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "scripts"))

from evalkit import load_csv, load_gold, read_split  # noqa: E402
from freeze_grounding import f1_near, f1_strict, near_counts, strict_counts, theme_counts  # noqa: E402
from llm_frontier import DIR, parse  # noqa: E402

NEW4 = ["en_chinmaya_day06", "en_chinmaya_day07", "en_chinmaya_day08", "en_chinmaya_day09"]
TUNE = {"kn": ["gita-1st_video", "gita-4th_video", "kannada_07", "kannada_10", "kannada_13", "kannada_16"],
        "en": ["video_01", "video_04", "video_07"]}
ROOT = os.path.join(DIR, "answers_api_gemini-3.1-pro-preview_low")
RUNS = {"run1_earlier_low_55": ("Run 1: earlier run (7 Nov 2026, 4a628fb)", ROOT),
        "run2_rerun_low_55": ("Run 2: re-run (16 Nov 2026, same prompts and settings)", ROOT + "_rerun")}
OUT = os.path.join(HERE, "results", "gemini_runs")


def groups():
    return [("Tuning Kannada", TUNE["kn"]), ("Tuning English", TUNE["en"]),
            ("E2 Kannada test", read_split("kn", "test")), ("E2 English test", read_split("en", "test")),
            ("E3 new Kannada", read_split("kn", "heldout")),
            ("E3 new English", [v for v in read_split("en", "heldout") if v not in NEW4]),
            ("New 4 Chinmaya English", NEW4)]


def sub(v):
    return "tuning" if v in TUNE["kn"] + TUNE["en"] else "new4" if v in NEW4 else ""


def main():
    lectures, gold, _ = load_gold()
    G = groups()
    allv = [v for _, vs in G for v in vs]
    assert len(allv) == 55, len(allv)
    os.makedirs(OUT, exist_ok=True)
    for stem, (title, base) in RUNS.items():
        pred, per = {}, {}
        for v in allv:
            folder = os.path.join(base, "tuning") if sub(v) == "tuning" else base
            ids = [(r["passage_id"], r["unit_id"]) for r in load_csv(os.path.join(DIR, sub(v), f"{v}_ids.csv"))]
            p, prob = parse(os.path.join(folder, f"{v}.txt"), ids)
            pred |= {u: p.get(u, []) for u in lectures[v]}
            u = json.load(open(os.path.join(folder, f"{v}_usage.json")))["usage"]
            cost = u["promptTokenCount"] / 1e6 * 2 + (u.get("candidatesTokenCount", 0) + u.get("thoughtsTokenCount", 0)) / 1e6 * 12
            per[v] = (len(prob["missing"]), len(prob["unparsed"]), cost)

        def m(vs):
            us = [u for v in vs for u in lectures[v]]
            c = lambda cnt: [sum(x) for x in zip(*(cnt(gold, pred, lectures[v]) for v in vs))]  # noqa: E731
            dig = [u for u in us if not gold[u]]
            ver = [u for u in us if gold[u]]
            return (len(us), f1_strict(c(strict_counts)), f1_near(c(near_counts)), f1_strict(c(theme_counts)),
                    len(dig), sum(bool(pred[u]) for u in dig), len(ver), sum(not pred[u] for u in ver),
                    sum(per[v][0] for v in vs), sum(per[v][1] for v in vs), sum(per[v][2] for v in vs))

        head = (f"  {'':26}{'lect':>5}{'psg':>6}{'strict':>8}{'+-1':>7}{'theme':>7}"
                f"{'digressions':>13}{'false verse':>13}{'verse psg':>11}{'missed':>8}{'missing':>9}{'unparsed':>10}{'cost $':>9}")
        row = lambda name, n, x: (f"  {name:26}{n:>5}{x[0]:>6}{x[1]:>8.3f}{x[2]:>7.3f}{x[3]:>7.3f}"  # noqa: E731
                                  f"{x[4]:>13}{x[5]:>13}{x[6]:>11}{x[7]:>8}{x[8]:>9}{x[9]:>10}{x[10]:>9.3f}")
        lines = [f"Gemini 3.1 Pro (API, thinking low), end to end, all 55 lectures. {title}",
                 f"answers: {base}/ (tuning lectures in tuning/)",
                 "scoring as llm_frontier.py: unparsed line ignored, missing passage = NONE; "
                 "false verse = a verse named on a gold no-verse passage; missed = verse passage given NONE", "",
                 "1. GROUPS", "", head]
        for g, vs in G:
            lines.append(row(g, len(vs), m(vs)))
        test = [v for g, vs in G[2:6] for v in vs]
        lines += [row("All 9 tuning", 9, m(TUNE["kn"] + TUNE["en"])), row("All 42 test", 42, m(test)),
                  row("All 55", 55, m(allv)), "", "2. LECTURE BY LECTURE", "", head]
        for g, vs in G:
            lines.append(f"  -- {g} --")
            for v in vs:
                lines.append(row(v, 1, m([v])))
        text = "\n".join(lines)
        open(os.path.join(OUT, f"{stem}.txt"), "w", encoding="utf-8").write(text + "\n")
        with open(os.path.join(OUT, f"{stem}.csv"), "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["unit_id", "video_file", "refs"])
            for v in allv:
                for u in lectures[v]:
                    w.writerow([u, v, ";".join(pred[u])])
        print(text, "\n")


if __name__ == "__main__":
    main()
