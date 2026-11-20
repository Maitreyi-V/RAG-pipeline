#!/usr/bin/env python3
"""
Digression-only veto + skeleton prompt on the 9 TUNING lectures (Gemini 3.1 Pro API, thinking low).

  veto      a passage where Cue (leave-one-lecture-out on tuning, as cascade_tuning.load) decodes OFF with
            confidence (= P_off) >= VETO is resolved as NONE and never labelled by Gemini
  skeleton  the paste file of llm_frontier.prepare(tuning=True), rebuilt (checked identical to the saved
            T*_PASTE_THIS.txt), with the text of every resolved passage replaced by a one-line tag
            "[ALREADY LABELLED: NONE]" and the passage rule changed so that Gemini outputs a line only for
            the passages whose text is shown; everything else unchanged
  send      one call per lecture, same settings as llm_frontier_gemini_api.py (thinking low,
            maxOutputTokens 65536, API defaults otherwise); answers and usage outside the repo in
            review/llm_frontier/tuning_skeleton_veto/
  tokens    input tokens of the full and the skeleton prompt from the countTokens endpoint (no cost)
  score     final label = NONE on resolved passages, Gemini's line elsewhere (missing = NONE); strict F1
            per language vs Gemini full prompt: run 1 published (all51_gemini3l.txt at 00e717e: Kannada
            0.967, English 0.816; answers since deleted) and results/gemini_runs/gita_data.csv

-> results/skeleton_veto_tuning.txt

Usage:
    python scripts/skeleton_veto_tuning.py prepare
    python scripts/skeleton_veto_tuning.py send
    python scripts/skeleton_veto_tuning.py score
"""
import argparse
import csv
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "scripts"))

import numpy as np  # noqa: E402
from evalkit import UNITS, UNITS_HELDOUT, load_csv, load_gold  # noqa: E402
from freeze_grounding import f1_strict, strict_counts  # noqa: E402
from gemini_runs_55 import TUNE  # noqa: E402
from llm_frontier import DIR, INSTRUCTIONS, parse  # noqa: E402
from llm_frontier_gemini_api import MAX_OUT, URL, cost, key  # noqa: E402
from llm_pilot import catalogue  # noqa: E402
from step4_viterbi import FEATS, run  # noqa: E402

VETO = 0.9
MODEL, THINKING = "gemini-3.1-pro-preview", "low"
OUT = os.path.join(DIR, "tuning_skeleton_veto")
TUNING_DIR = os.path.join(DIR, "tuning")
SILVER_LABELS = os.path.join(HERE, "results", "silver", "silver_labels.csv")
TAG = "[ALREADY LABELLED: NONE]"
OLD_RULE = "- Every passage must get exactly one line, in order. Do not skip, merge, or renumber passages."
NEW_RULE = (f"- Some passages are already labelled; their text is not shown and they appear as \"{TAG}\". They are "
            "shown only so you can follow the order of the lecture. Do not output a line for them. Every other "
            "passage must get exactly one line, in order. Do not skip, merge, or renumber passages.")
PUBLISHED = {"kn": 0.967, "en": 0.816}
TUNE_ALL = TUNE["kn"] + TUNE["en"]


def resolved():
    """{unit_id: True if the veto resolves it}, Cue leave-one-lecture-out on tuning"""
    teacher = {r["unit_id"]: min(0.98, max(0.02, float(r["soft_target"])))
               for r in csv.DictReader(open(SILVER_LABELS, encoding="utf-8"))}
    info = {}
    order, _, lectures, _ = run(alphas=("a05",), verbose=False, resume=True, off_override=teacher, info=info)
    cue = dict(dict(order)["Viterbi (alpha 0.5)"])
    out = {}
    for v in TUNE_ALL:
        po = info["probs_a05"][v][1]
        for k, u in enumerate(lectures[v]):
            out[u] = (not cue[u]) and po[k] >= VETO
    return out


def build(v, lectures, units, cat, skip=None):
    us = lectures[v]
    ids = [f"P{k + 1:03d}" for k in range(len(us))]
    lang = units[us[0]]["language"].upper()
    body = "\n\n".join(f"{i}: {TAG if skip and skip[u] else units[u]['text']}" for i, u in zip(ids, us))
    head = INSTRUCTIONS if not skip else INSTRUCTIONS.replace(OLD_RULE, NEW_RULE)
    return head + "\n" + cat + f"\n\nLECTURE: {v}, {len(us)} passages, language {lang}\n\n" + body + "\n"


def saved_paste(v):
    return next(os.path.join(TUNING_DIR, f) for f in os.listdir(TUNING_DIR) if f.endswith(f"_{v}_PASTE_THIS.txt"))


def prepare():
    assert OLD_RULE in INSTRUCTIONS
    lectures, _, _ = load_gold()
    units = {u["unit_id"]: u for u in load_csv(UNITS) + load_csv(UNITS_HELDOUT)}
    cat = catalogue()
    skip = resolved()
    os.makedirs(OUT, exist_ok=True)
    for v in TUNE_ALL:
        if build(v, lectures, units, cat) != open(saved_paste(v), encoding="utf-8").read():
            sys.exit(f"rebuilt full paste file for {v} differs from the saved one; nothing written")
        with open(os.path.join(OUT, f"{v}_SKELETON.txt"), "w", encoding="utf-8") as f:
            f.write(build(v, lectures, units, cat, skip))
        n = sum(skip[u] for u in lectures[v])
        print(f"  {v:18} passages {len(lectures[v]):>4}  resolved by veto {n:>3}")
    json.dump({u: bool(x) for u, x in skip.items()}, open(os.path.join(OUT, "resolved.json"), "w"))
    print(f"  full paste files rebuilt identical; skeleton prompts in {OUT}")


def count_tokens(text, k):
    import requests
    r = requests.post(URL.format(MODEL).replace(":generateContent", ":countTokens"), headers={"x-goog-api-key": k},
                      json={"contents": [{"parts": [{"text": text}]}]}, timeout=(15, 120))
    r.raise_for_status()
    return r.json()["totalTokens"]


def send(workers=4):
    import requests
    k = key()
    todo = [v for v in TUNE_ALL if not os.path.exists(os.path.join(OUT, f"{v}.txt"))]
    print(f"  {len(todo)} lecture(s) to send to {MODEL}, thinking {THINKING}", flush=True)

    def one(v):
        text = open(os.path.join(OUT, f"{v}_SKELETON.txt"), encoding="utf-8").read()
        body = {"contents": [{"parts": [{"text": text}]}],
                "generationConfig": {"thinkingConfig": {"thinkingLevel": THINKING}, "maxOutputTokens": MAX_OUT}}
        t0 = time.time()
        for i in range(6):
            try:
                r = requests.post(URL.format(MODEL), headers={"x-goog-api-key": k}, json=body, timeout=(15, 1200))
                if r.status_code == 200:
                    break
                msg = f"HTTP {r.status_code} {r.text[:200]}"
                if r.status_code != 429 and r.status_code < 500:
                    print(f"  {v}: no answer, {msg}", flush=True)
                    return None
            except requests.RequestException as e:
                msg = type(e).__name__
            print(f"  {v}: no answer ({msg}), retry {i + 1}", flush=True)
            if i == 5:
                return None
            time.sleep(20 * (i + 1))
        j = r.json()
        cands = j.get("candidates", [])
        ans = "".join(p.get("text", "") for c in cands for p in c.get("content", {}).get("parts", [])
                      if not p.get("thought"))
        open(os.path.join(OUT, f"{v}.txt"), "w", encoding="utf-8").write(ans)
        u = j.get("usageMetadata", {})
        json.dump(dict(model=MODEL, model_version=j.get("modelVersion"), thinking_level=THINKING,
                       max_output_tokens=MAX_OUT, finish_reason=[c.get("finishReason") for c in cands], usage=u,
                       full_prompt_tokens=count_tokens(open(saved_paste(v), encoding="utf-8").read(), k),
                       seconds=round(time.time() - t0, 1)),
                  open(os.path.join(OUT, f"{v}_usage.json"), "w"), indent=1)
        print(f"  {v:18} in {u.get('promptTokenCount', 0):>7}  out {u.get('candidatesTokenCount', 0):>6} "
              f"(thinking {u.get('thoughtsTokenCount', 0)})  ${cost(MODEL, u):.3f}  {time.time() - t0:.0f}s", flush=True)
        return v

    with ThreadPoolExecutor(workers) as ex:
        done = [v for v in ex.map(one, todo) if v]
    print(f"  answered {len(done)}, failed {[v for v in todo if v not in done]}")


def score():
    lectures, gold, _ = load_gold()
    skip = json.load(open(os.path.join(OUT, "resolved.json")))
    with open(os.path.join(HERE, "results", "gemini_runs", "gita_data.csv"), newline="", encoding="utf-8") as f:
        gd = {r["unit_id"]: [x for x in r["refs"].split(";") if x] for r in csv.DictReader(f)}
    pred, rows, extra = {}, [], 0
    for v in TUNE_ALL:
        ids = [(r["passage_id"], r["unit_id"]) for r in load_csv(os.path.join(TUNING_DIR, f"{v}_ids.csv"))]
        p, prob = parse(os.path.join(OUT, f"{v}.txt"), ids)
        extra += sum(bool(skip[u]) and u in p for u in lectures[v])
        pred.update({u: [] if skip[u] else p.get(u, []) for u in lectures[v]})
        miss = sum(not skip[u] and u not in p for u in lectures[v])
        u = json.load(open(os.path.join(OUT, f"{v}_usage.json")))
        rows.append((v, len(lectures[v]), sum(skip[x] for x in lectures[v]), miss, len(prob["unparsed"]),
                     u["full_prompt_tokens"], u["usage"]["promptTokenCount"],
                     u["usage"].get("candidatesTokenCount", 0) + u["usage"].get("thoughtsTokenCount", 0),
                     cost(MODEL, u["usage"])))
    f1 = lambda p, vs: f1_strict([sum(x) for x in zip(*(strict_counts(gold, p, lectures[v]) for v in vs))])  # noqa: E731
    lines = [f"Digression-only veto (Cue OFF, P_off >= {VETO}) + skeleton prompt, Gemini 3.1 Pro API thinking low, "
             "9 tuning lectures (one run)",
             f"answers: {OUT}/; Cue = leave-one-lecture-out; {extra} line(s) given for resolved passages (ignored)", "",
             f"  {'lecture':18}{'psg':>5}{'resolved':>10}{'missing':>9}{'unparsed':>10}{'full in tok':>13}"
             f"{'skeleton in':>13}{'cut':>6}{'out+think':>11}{'$':>8}{'F1 skel':>9}{'F1 gita':>9}"]
    for v, n, r, m, up, fi, si, o, c in rows:
        lines.append(f"  {v:18}{n:>5}{r:>10}{m:>9}{up:>10}{fi:>13}{si:>13}{1 - si / fi:>6.0%}{o:>11}{c:>8.3f}"
                     f"{f1(pred, [v]):>9.3f}{f1(gd, [v]):>9.3f}")
    lines += ["", "  group            skeleton+veto   Gemini full: run 1 published / gita_data.csv   false verse on digressions "
              "skel / gita   input tokens full -> skeleton"]
    for name, lg in (("Tuning Kannada", "kn"), ("Tuning English", "en")):
        vs = TUNE[lg]
        dig = [u for v in vs for u in lectures[v] if not gold[u]]
        rs = [r for r in rows if r[0] in vs]
        fi, si = sum(r[5] for r in rs), sum(r[6] for r in rs)
        lines.append(f"  {name:16}{f1(pred, vs):>14.3f}   {PUBLISHED[lg]:>34.3f} / {f1(gd, vs):.3f}"
                     f"   {sum(bool(pred[u]) for u in dig):>23} / {sum(bool(gd[u]) for u in dig):<3} of {len(dig):<4}"
                     f"   {fi} -> {si} ({1 - si / fi:.0%} fewer)")
    lines.append(f"  cost of this run: ${sum(r[8] for r in rows):.3f}")
    text = "\n".join(lines)
    print(text)
    open(os.path.join(HERE, "results", "skeleton_veto_tuning.txt"), "w", encoding="utf-8").write(text + "\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["prepare", "send", "score"])
    {"prepare": prepare, "send": send, "score": score}[ap.parse_args().step]()
