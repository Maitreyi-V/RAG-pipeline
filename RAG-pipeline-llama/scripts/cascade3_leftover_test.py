#!/usr/bin/env python3
"""
3-layer cascade with Gemini, REAL leftover-only calls, on the 42 test lectures (post-hoc, not pre-registered).

The 3-layer design is the one in all51_gemini3l.py / cascade4_test.THREE for Gemini, chosen on tuning:
  Kannada  1 anchor -> anchor's verses   > 2 Cue conf >= 0.9 -> Cue's label   > Gemini
  English                                  2 Cue conf >= 0.9 -> Cue's label   > Gemini
Until now Gemini's labels came from its whole-lecture answers (every passage sent). Here Gemini is sent
ONLY the passages left by layers 1-2, each with up to 2 neighbours either side shown as context with their
accepted label: the round-1 leftover prompt of cascade_tuning.py (v1, no hints; 0.858 on English tuning),
built from the same 42 paste files as the whole-lecture runs; gemini-3.1-pro-preview, thinking low.

  send     prompts -> review/llm_frontier/test_leftover/<lecture>_LEFTOVER.txt
           answers -> review/llm_frontier/answers_api_gemini-3.1-pro-preview_low/test_leftover/
           an answer that exists is never re-asked
  score    strict F1: Gemini alone (whole lecture) vs 3L simulated (whole-lecture labels) vs 3L REAL
           (leftover answers; missing passage = NONE); paired lecture bootstrap 10,000, W/T/L;
           digressions; real cost: Gemini whole lecture vs Gemini leftover call + Cue teacher share
           (teacher cost as all51_gemini3l.py, shared out by passage count)

-> results/cascade3_leftover_test.txt

Usage:
    python scripts/cascade3_leftover_test.py send
    python scripts/cascade3_leftover_test.py score
"""
import argparse
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "scripts"))

import numpy as np  # noqa: E402
from all51_gemini3l import TEACHER_PASSAGES, TEACHER_TOKENS, gemini_cost  # noqa: E402
from cascade4_test import THREE  # noqa: E402
from cascade4_tuning import L1, L2, L4  # noqa: E402
from cascade_tuning import CONF, HEADER_NEW, HEADER_OLD, MODEL, NEIGHBOURS, RULE_NEW, RULE_OLD, THINKING  # noqa: E402
from evalkit import load_csv, load_gold, read_split  # noqa: E402
from freeze_grounding import f1_strict, paired, strict_counts  # noqa: E402
from llm_frontier import DIR, parse  # noqa: E402
from recall_by_type import read_llm  # noqa: E402
from step4_viterbi import FEATS, OFF, predict_locked  # noqa: E402

N_BOOT = 10000
PROMPTS = os.path.join(DIR, "test_leftover")
ANS_ROOT = os.path.join(DIR, "answers_api_gemini-3.1-pro-preview_low")
ANSWERS = os.path.join(ANS_ROOT, "test_leftover")
GROUPS = [("E2 Kannada test", "kn", "test"), ("E2 English test", "en", "test"),
          ("E3 Kannada", "kn", "heldout"), ("E3 English", "en", "heldout")]
NEW4 = {"en_chinmaya_day06", "en_chinmaya_day07", "en_chinmaya_day08", "en_chinmaya_day09"}   # not in the 42


def split(L, s):
    """the 42 test lectures: lectures_new.csv now also lists the 4 new Chinmaya lectures; they are left out"""
    return [v for v in read_split(L, s) if v not in NEW4]


def ids_of(v):
    return [(r["passage_id"], r["unit_id"]) for r in load_csv(os.path.join(DIR, f"{v}_ids.csv"))]


def fmt(refs):
    return ",".join(r.replace("BG ", "") for r in refs) or "NONE"


def setup():
    """-> lectures, gold, lang, layer, label (layers 1-2), gem (whole-lecture Gemini)"""
    lectures, gold, _ = load_gold()
    lang = {v: L for _, L, s in GROUPS for v in split(L, s)}
    z = np.load(FEATS)
    refs = list(z["refs"])
    row = {u: i for i, u in enumerate(z["unit_ids"])}
    dec = predict_locked({v: lectures[v] for v in lang}, "a05",
                         path=os.path.join(HERE, "results", "v2_teacher145.pkl"))
    layer, label = {}, {}
    for v, L in lang.items():
        for u, (st, extra, p_state, _) in zip(lectures[v], dec[v]):
            cue = ([] if st == OFF else [refs[st]]) + [refs[j] for j in extra]        # as all51_gemini3l
            anchor = [refs[j] for j in np.flatnonzero(z["anchor"][row[u]])]
            take = {L1: (bool(anchor), anchor), L2: (p_state >= CONF, cue)}
            la = next((x for x in THREE[("Gemini", L)] if take[x][0]), L4)
            layer[u] = la
            if la != L4:
                label[u] = take[la][1]
    gem = read_llm("llm_frontier_gemini31pro_api_low.csv")
    return lectures, gold, lang, layer, label, gem


def paste_file(v):
    return os.path.join(DIR, next(f for f in os.listdir(DIR) if re.match(rf"^\d\d_{re.escape(v)}_PASTE_THIS\.txt$", f)))


def build_prompt(v, lectures, layer, label):
    text = open(paste_file(v), encoding="utf-8").read()
    head, body = text.split("\nLECTURE: ", 1)
    assert HEADER_OLD in head and RULE_OLD in head, v
    head = head.replace(HEADER_OLD, HEADER_NEW).replace(RULE_OLD, RULE_NEW)
    passages = dict(re.findall(r"^(P\d{3}): (.*)$", body, flags=re.M))
    pid = {u: p for p, u in ids_of(v)}
    us = lectures[v]
    target = [k for k, u in enumerate(us) if layer[u] == L4]
    show = sorted({j for k in target for j in range(k - NEIGHBOURS, k + NEIGHBOURS + 1) if 0 <= j < len(us)})
    out, prev = [], None
    for j in show:
        if prev is not None and j != prev + 1:
            out.append("[...]")
        u = us[j]
        tag = "[LABEL]" if layer[u] == L4 else f"[CONTEXT, already labelled: {fmt(label[u])}]"
        out.append(f"{pid[u]} {tag}: {passages[pid[u]]}")
        prev = j
    lang = body.split("\n", 1)[0].rsplit("language ", 1)[-1]
    return (head + f"\nLECTURE: {v}, {len(target)} passages to label ([LABEL]), "
            f"{len(show) - len(target)} context passages, language {lang}\n\n" + "\n\n".join(out) + "\n"), len(target)


def send_one(v, prompt, n, k):
    import requests
    from llm_frontier_gemini_api import MAX_OUT, URL, cost
    body = {"contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"thinkingConfig": {"thinkingLevel": THINKING}, "maxOutputTokens": MAX_OUT}}
    t0, r = time.time(), None
    for i in range(6):
        try:
            r = requests.post(URL.format(MODEL), headers={"x-goog-api-key": k}, json=body, timeout=(15, 1200))
            if r.status_code == 200 or (r.status_code != 429 and r.status_code < 500):
                break
        except requests.RequestException as e:
            print(f"  {v}: no answer ({type(e).__name__}), retry {i + 1}", flush=True)
        time.sleep(20 * (i + 1))
    if r is None or r.status_code != 200:
        print(f"  {v}: no answer, {None if r is None else r.status_code} {'' if r is None else r.text[:200]}", flush=True)
        return
    j = r.json()
    cands = j.get("candidates", [])
    text = "".join(p.get("text", "") for c in cands for p in c.get("content", {}).get("parts", []) if not p.get("thought"))
    with open(os.path.join(ANSWERS, f"{v}.txt"), "w", encoding="utf-8") as f:
        f.write(text)
    u = j.get("usageMetadata", {})
    fin = [c.get("finishReason") for c in cands]
    json.dump(dict(model=MODEL, model_version=j.get("modelVersion"), thinking_level=THINKING, max_output_tokens=MAX_OUT,
                   finish_reason=fin, usage=u, passages_to_label=n, seconds=round(time.time() - t0, 1)),
              open(os.path.join(ANSWERS, f"{v}_usage.json"), "w"), indent=1)
    print(f"  {v:24} {n:>3} to label  {','.join(map(str, fin))}  in {u.get('promptTokenCount', 0)} "
          f"out {u.get('candidatesTokenCount', 0)} (thinking {u.get('thoughtsTokenCount', 0)})  ${cost(MODEL, u):.3f}",
          flush=True)


def send(lectures, lang, layer, label):
    from llm_frontier_gemini_api import key
    k = key()
    os.makedirs(PROMPTS, exist_ok=True)
    os.makedirs(ANSWERS, exist_ok=True)
    jobs = []
    for v in lang:
        prompt, n = build_prompt(v, lectures, layer, label)
        with open(os.path.join(PROMPTS, f"{v}_LEFTOVER.txt"), "w", encoding="utf-8") as f:
            f.write(prompt)
        if not n or os.path.exists(os.path.join(ANSWERS, f"{v}.txt")):
            print(f"  {v}: {n} to label, {'skipped (answer exists)' if n else 'nothing to send'}", flush=True)
            continue
        jobs.append((v, prompt, n))
    with ThreadPoolExecutor(4) as ex:
        list(ex.map(lambda a: send_one(*a, k), jobs))
    total = sum(gemini_cost(os.path.join(ANSWERS, f)) for f in os.listdir(ANSWERS) if f.endswith("_usage.json"))
    print(f"  answers in {ANSWERS}: ${total:.3f}")


def score(lectures, gold, lang, layer, label, gem):
    real, missing, unparsed = {}, 0, 0
    for v in lang:
        targets = [u for u in lectures[v] if layer[u] == L4]
        path = os.path.join(ANSWERS, f"{v}.txt")
        p, prob = parse(path, ids_of(v)) if (targets and os.path.exists(path)) else ({}, {"unparsed": []})
        unparsed += len(prob["unparsed"])
        for u in targets:
            missing += u not in p
            real[u] = p.get(u, [])
    sim = {u: label[u] if layer[u] != L4 else gem.get(u, []) for u in layer}
    rl = {u: label[u] if layer[u] != L4 else real[u] for u in layer}
    S = {"Gemini alone": gem, "3L simulated": sim, "3L REAL leftover": rl}
    per_psg = (TEACHER_TOKENS[0] / 1e6 * 0.25 + TEACHER_TOKENS[1] / 1e6 * 2) / TEACHER_PASSAGES
    cost = {}
    for v in lang:
        g = gemini_cost(os.path.join(ANS_ROOT, f"{v}_usage.json"))
        lp = os.path.join(ANSWERS, f"{v}_usage.json")
        left = gemini_cost(lp) if os.path.exists(lp) else 0.0
        c = per_psg * len(lectures[v])
        cost[v] = {"Gemini alone": g, "3L simulated": g + c, "3L REAL leftover": left + c, "leftover call only": left}
    f1 = lambda p, vs: f1_strict([sum(x) for x in zip(*(strict_counts(gold, p, lectures[v]) for v in vs))])  # noqa: E731
    ok = lambda p, u: set(p.get(u, ())) == set(gold[u])  # noqa: E731
    cols = list(S)
    lines = ["3-layer cascade with Gemini, REAL leftover-only Gemini calls, 42 test lectures "
             "(POST-HOC, not pre-registered; design = cascade4_test.THREE, chosen on tuning)",
             f"leftover answers: {missing} target passages missing (= NONE), {unparsed} unparsed lines", "",
             "1. LECTURE BY LECTURE  (strict F1; to Gemini = passages labelled by Gemini / all)", "",
             f"  {'lecture':24}{'psg':>4}{'to Gemini':>11}" + "".join(f"{c:>18}" for c in cols)
             + f"{'$ alone':>10}{'$ 3L real':>11}"]
    sets = [(g, split(L, s)) for g, L, s in GROUPS] + [("All 42 test", list(lang))]
    for g, vs in sets[:4]:
        lines.append(f"  -- {g} --")
        for v in vs:
            n = sum(layer[u] == L4 for u in lectures[v])
            lines.append(f"  {v:24}{len(lectures[v]):>4}{n:>11}" + "".join(f"{f1(S[c], [v]):>18.3f}" for c in cols)
                         + f"{cost[v]['Gemini alone']:>10.3f}{cost[v]['3L REAL leftover']:>11.3f}")
    lines += ["", "2. OVERVIEW (paired vs Gemini alone: diff [95% CI] W/T/L, 5+ lectures)", ""]
    for g, vs in sets:
        us = [u for v in vs for u in lectures[v]]
        n = sum(layer[u] == L4 for u in us)
        out = f"  {g:16}{len(vs):>3} lect  to Gemini {n}/{len(us)} ({n / len(us):.0%})  " + "  ".join(
            f"{c} {f1(S[c], vs):.3f}" for c in cols)
        lines.append(out)
        if len(vs) >= 5:
            for c in cols[1:]:
                d, ci, _, wtl, _ = paired(gold, lectures, S[c], gem, vs, N_BOOT)
                lines.append(f"      {c} - Gemini alone {d:+.3f} [{ci[0]:+.3f},{ci[1]:+.3f}]  W/T/L {wtl[0]}/{wtl[1]}/{wtl[2]}")
        dig = [u for u in us if not gold[u]]
        ver = [u for u in us if gold[u]]
        lines.append(f"      digressions {len(dig)}: false verse " + "  ".join(
            f"{c} {sum(bool(S[c].get(u)) for u in dig)}" for c in cols)
            + f"   | verse passages {len(ver)}: missed " + "  ".join(f"{c} {sum(not S[c].get(u) for u in ver)}" for c in cols))
        lt = [u for u in us if layer[u] == L4]
        lines.append(f"      leftover passages {len(lt)}: right with leftover call {sum(ok(rl, u) for u in lt)}, "
                     f"with whole-lecture call {sum(ok(gem, u) for u in lt)}")
        lines.append("      cost $: " + "  ".join(f"{c} {sum(cost[v][c] for v in vs):.3f}" for c in
                                                   ("Gemini alone", "leftover call only", "3L REAL leftover")))
    text = "\n".join(lines)
    print(text)
    with open(os.path.join(HERE, "results", "cascade3_leftover_test.txt"), "w", encoding="utf-8") as f:
        f.write(text + "\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["send", "score", "prompts"])
    a = ap.parse_args()
    lectures, gold, lang, layer, label, gem = setup()
    if a.step == "send":
        send(lectures, lang, layer, label)
    elif a.step == "prompts":                         # build and count only, no API call
        tot = 0
        for v in lang:
            p, n = build_prompt(v, lectures, layer, label)
            tot += n
            print(f"  {v:24} {n:>3}/{len(lectures[v])} to label, {len(p) // 4} tokens approx")
        print(f"  total to label {tot}")
    else:
        score(lectures, gold, lang, layer, label, gem)
