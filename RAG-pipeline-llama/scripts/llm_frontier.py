#!/usr/bin/env python3
"""
Phase 3: a frontier LLM (Claude Opus 5.5, claude.ai chat, effort high) labels
all 42 evaluation lectures end to end. Pre-registered in docs/experiment_log.md.

  prepare   writes one paste file per lecture (instructions + the 72-verse
            catalogue, identical to the pilot's llm_pilot.catalogue(), + the
            numbered passages) and its passage-id -> unit_id map, plus a
            format-check file for kannada_10 (tuning lecture, not scored)
  score     reads the answers saved by hand as answers/<lecture>.txt and scores
            them like llm_e2e.py: E2 per language, E3 per new speaker and pooled;
            strict / +-1 / theme F1 with 95% lecture-bootstrap CI; paired lecture
            bootstrap (10,000) and W/T/L vs GitaGround-Cue and Plain Carry

Answer rules (pre-registered): one new chat per lecture; the paste file is sent
unchanged; the first complete answer counts. If the answer stops before the last
passage, ONE message "Continue" is allowed and both parts are saved in the same
answer file; nothing else is ever re-asked. Missing passages count as NONE; if a
passage appears twice, its first line counts.

Files contain transcript text, so they live outside the repo:
  /Users/maitreyi/Documents/Capstone/review/llm_frontier/

Usage:
    python scripts/llm_frontier.py prepare
    python scripts/llm_frontier.py prepare-tuning   # 9 tuning lectures -> review/llm_frontier/tuning/
    python scripts/llm_frontier.py score
    python scripts/llm_frontier.py score --model gemini31pro   # answers_gemini31pro/, Opus as extra row
    python scripts/llm_frontier.py score --model gemini38flash # answers_gemini38flash/, Opus + Gemini Pro rows
    python scripts/llm_frontier.py score --model gpt61sol      # answers_gpt-6.1-sol/ (llm_frontier_api.py), Opus row
"""
import argparse
import csv
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

from evalkit import LECTURES_NEW, UNITS, UNITS_HELDOUT, load_csv, load_gold, read_split  # noqa: E402
from llm_e2e import all_lectures  # noqa: E402
from llm_pilot import catalogue  # noqa: E402

DIR = "/Users/maitreyi/Documents/Capstone/review/llm_frontier"
ANSWERS = os.path.join(DIR, "answers")
TUNING_DIR = os.path.join(DIR, "tuning")              # 9 tuning lectures, v3 hybrid votes
NAME = "Claude Opus 5.5 (chat, high) end-to-end"
# score --model: answers folder, row name, output stem, LLM comparison rows (same lectures)
# (opus55 and gpt61sol = pre-registered 42-lecture runs; gemini31pro, gemini38flash = spot checks)
MODELS = {"opus55": (ANSWERS, NAME, "llm_frontier_opus55", ()),
          "gemini31pro": (os.path.join(DIR, "answers_gemini31pro"), "Gemini 3.1 Pro (AI Studio) end-to-end",
                          "llm_frontier_gemini31pro", ("opus55",)),
          "gemini38flash": (os.path.join(DIR, "answers_gemini38flash"), "Gemini 3.8 Flash (AI Studio) end-to-end",
                            "llm_frontier_gemini38flash", ("opus55", "gemini31pro")),
          "gpt61sol": (os.path.join(DIR, "answers_gpt-6.1-sol"), "GPT-6.1 Sol (API, high) end-to-end",
                       "llm_frontier_gpt61sol", ("opus55",)),
          "gemini31pro42": (os.path.join(DIR, "answers_gemini31pro_42"), "Gemini 3.1 Pro (AI Studio) end-to-end",
                            "llm_frontier_gemini31pro_42", ("opus55",))}
FORMAT_CHECK = "kannada_10"

INSTRUCTIONS = """Follow these instructions exactly. Do not ask questions, do not summarise, do not add commentary before or after the output, and do not request the lecture in parts. Your entire reply must be the single code block described under OUTPUT FORMAT.

You are labelling a transcribed lecture on Bhagavad Gita Chapter 2.

TASK
For every passage below, state which verse(s) of Chapter 2 the passage is explaining, or NONE if it is not explaining any verse (a story, anecdote, aside, announcement, or discussion of material outside Chapter 2).

RULES
- A passage explains a verse if it recites, translates, paraphrases, or expounds that verse's content. Passing mention of a verse while explaining another does not make it the passage's verse; list the verse being explained first, then any verse quoted in passing.
- A passage may explain more than one verse; list all verses, any order.
- Use only verses 2.1 to 2.72. Never output a verse outside this range.
- Every passage must get exactly one line, in order. Do not skip, merge, or renumber passages.
- If you reach an output limit before the last passage, stop cleanly at the end of a line; do not compress or summarise the remainder.

OUTPUT FORMAT
One code block. One line per passage:
<passage_id><TAB><verse list separated by commas, e.g. 2.47,2.48, or NONE>
"""


def prepare(tuning=False):
    """tuning=True: the 9 TUNING lectures (votes for the v3 hybrid), same instructions and
    catalogue, into TUNING_DIR (answers in TUNING_DIR/answers/)"""
    d = TUNING_DIR if tuning else DIR
    os.makedirs(os.path.join(d, "answers") if tuning else ANSWERS, exist_ok=True)
    lectures, _, _ = load_gold()
    units = {u["unit_id"]: u for u in load_csv(UNITS) + load_csv(UNITS_HELDOUT)}
    cat = catalogue()
    order = ([v for L in ("kn", "en") for v in read_split(L, "dev")] if tuning
             else [FORMAT_CHECK] + all_lectures())
    lines = []
    for n, v in enumerate(order):
        us = lectures[v]
        ids = [f"P{k + 1:03d}" for k in range(len(us))]
        lang = units[us[0]]["language"].upper()
        body = "\n\n".join(f"{i}: {units[u]['text']}" for i, u in zip(ids, us))
        tag = f"T{n + 1}" if tuning else "00_FORMAT_CHECK" if v == FORMAT_CHECK else f"{n:02d}"
        with open(os.path.join(d, f"{tag}_{v}_PASTE_THIS.txt"), "w", encoding="utf-8") as f:
            f.write(INSTRUCTIONS + "\n" + cat + f"\n\nLECTURE: {v}, {len(us)} passages, language {lang}\n\n"
                    + body + "\n")
        with open(os.path.join(d, f"{v}_ids.csv"), "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerows([("passage_id", "unit_id")] + list(zip(ids, us)))
        lines.append(f"{tag}\t{v}\t{lang}\t{len(us)} passages\tsave answer as answers/{v}.txt")
    with open(os.path.join(d, "ORDER.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"  wrote {len(order)} paste files to {d}")


def parse(path, ids):
    """-> ({unit_id: [refs]}, problems); accepts TAB, '|' or spaces after the id"""
    m = dict(ids)
    pred, bad, outside = {}, [], 0
    for line in open(path, encoding="utf-8"):
        line = line.strip().strip("`")
        if not line:
            continue
        mm = re.match(r"^(P\d{3})[\t |:]+(.+)$", line)
        if not mm or mm.group(1) not in m:
            bad.append(line[:60])
            continue
        if m[mm.group(1)] in pred:                       # duplicate: first line counts
            continue
        refs = []
        for x in re.split(r"[;,\s]+", mm.group(2).strip()):
            if not x or x.upper() == "NONE":
                continue
            n = re.match(r"^2\.(\d{1,2})$", x)
            if n and 1 <= int(n.group(1)) <= 72:
                refs.append(f"BG 2.{int(n.group(1))}")
            else:
                outside += 1
        pred[m[mm.group(1)]] = refs
    missing = [p for p, u in ids if u not in pred]
    return pred, dict(unparsed=bad, missing=missing, outside_ch2=outside)


def score(model="opus55"):
    from baselines import carry
    from freeze_grounding import (f1_near, f1_strict, near_counts, paired, strict_counts, theme_counts,
                                  with_ci)
    from step4_viterbi import FEATS, OFF, OUT, predict_locked
    lectures, gold, _ = load_gold()
    answers, me, stem, compare = MODELS[model]

    def read_answers(folder, only=None):
        pred, problems = {}, {}
        for v in all_lectures():
            path = os.path.join(folder, f"{v}.txt")
            if not os.path.exists(path) or (only is not None and v not in only):
                continue
            ids = [(r["passage_id"], r["unit_id"]) for r in load_csv(os.path.join(DIR, f"{v}_ids.csv"))]
            p, problems[v] = parse(path, ids)
            for u in lectures[v]:
                pred[u] = p.get(u, [])                   # missing passage = NONE
        return pred, problems
    pred, problems = read_answers(answers)
    vids = [v for v in all_lectures() if v in problems]
    if not vids:
        sys.exit(f"no answers in {answers} yet")
    with open(os.path.join(HERE, "results", f"{stem}.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["unit_id", "video_file", "refs"])
        for v in vids:
            for u in lectures[v]:
                w.writerow([u, v, ";".join(pred[u])])
    z = np.load(FEATS)
    refs = list(z["refs"])
    row = {u: i for i, u in enumerate(z["unit_ids"])}
    anchors = {u: [refs[j] for j in np.flatnonzero(z["anchor"][row[u]])] for u in row}

    def decoded(path):
        dec = predict_locked({v: lectures[v] for v in vids}, "a05", path=path)
        return {u: ([] if st == OFF else [refs[st]]) + [refs[j] for j in extra]
                for v in vids for u, (st, extra, _, _) in zip(lectures[v], dec[v])}
    systems = {me: pred}
    for k in compare:                                    # other LLMs, same lectures, as comparison rows
        systems[MODELS[k][1]] = read_answers(MODELS[k][0], set(vids))[0]
    systems |= {"GitaGround-Cue": decoded(os.path.join(HERE, "results", "v2_teacher145.pkl")),
               "GitaGround-Local": decoded(OUT), "Plain Carry": {}}
    for v in vids:
        systems["Plain Carry"].update(carry(lectures[v], anchors))
    speaker = {r["video_file"]: r["speaker"] for r in load_csv(LECTURES_NEW)}
    groups = []
    for split in ("test", "heldout"):
        for L, lname in (("kn", "Kannada"), ("en", "English")):
            vs = [v for v in read_split(L, split) if v in problems]
            if not vs:
                continue
            if split == "heldout":
                groups += [(f"E3 {lname}: {sp}", [v for v in vs if speaker[v] == sp])
                           for sp in sorted({speaker[v] for v in vs})]
                groups.append((f"E3 new {lname} (pooled)", vs))
            else:
                groups.append((f"E2 {lname} test", vs))
    lines = [f"Phase 3 — {me}, {len(vids)} of 42 lectures answered",
             "parsing: " + f"{sum(len(p['missing']) for p in problems.values())} passage(s) missing (scored as NONE), "
             f"{sum(len(p['unparsed']) for p in problems.values())} unparsed line(s), "
             f"{sum(p['outside_ch2'] for p in problems.values())} verse(s) outside 2.1-2.72"]
    lines += [f"  {v}: {len(p['missing'])} missing" for v, p in problems.items() if p["missing"]]
    for gname, vs in groups:
        units = [u for v in vs for u in lectures[v]]
        lines += ["", f"{gname}   lectures {len(vs)}   passages {len(units)}   no-verse "
                      f"{sum(1 for u in units if not gold[u])}",
                  f"  {'system':42} {'strict F1 [CI]':22} {'+-1 F1 [CI]':22} {'theme F1 [CI]':22}"]
        for name, p in systems.items():
            cells = []
            for counts, f1 in ((strict_counts, f1_strict), (near_counts, f1_near), (theme_counts, f1_strict)):
                point, ci = with_ci([counts(gold, p, lectures[v]) for v in vs], f1, 1000)
                cells.append(f"{point:.3f}" + (f" [{ci[0]:.3f},{ci[1]:.3f}]" if ci else ""))
            lines.append(f"  {name:42} {cells[0]:22} {cells[1]:22} {cells[2]:22}")
        lines.append(f"  paired, strict F1 (A - B), 10000 resamples:  {'diff':>7} {'95% CI':>17} "
                     f"{'win/tie/loss':>13}")
        short = me.split(" (")[0]
        for b in [s for s in systems if s not in (me, "GitaGround-Local")]:
            d, ci, _, (w, t, lo), _ = paired(gold, lectures, systems[me], systems[b], vs, 10000)
            lines.append(f"    {short + ' - ' + b.split(' (')[0]:41} {d:>+7.3f} [{ci[0]:+.3f},{ci[1]:+.3f}] "
                         f"{f'{w}/{t}/{lo}':>13}")
    text = "\n".join(lines)
    print(text)
    open(os.path.join(HERE, "results", f"{stem}.txt"), "w").write(text + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["prepare", "prepare-tuning", "score"])
    ap.add_argument("--model", choices=sorted(MODELS), default="opus55", help="score: whose answers")
    args = ap.parse_args()
    if args.step == "score":
        score(args.model)
    else:
        prepare(tuning=args.step == "prepare-tuning")


if __name__ == "__main__":
    main()
