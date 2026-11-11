#!/usr/bin/env python3
"""
Cascade on the TUNING lectures (no test gold): cheap layers accept passages, Gemini labels the rest.
Spec in docs/experiment_log.md.

  layers (in order, per language)
    1 anchor        the string anchor fired -> the anchor's verses        (not used in English: fix 1)
    2 Cue conf>=0.9 Cue's scorer probability for its own label >= 0.9 -> Cue's label
                    (Cue = v2 primary, LOLO; confidence as diag_disagreement_tuning.py)
    3 Gemini        everything not accepted above
  keep rule (fix 2)  a layer is kept in a language only if, on that language's tuning lectures, its
                    label is right (set equals gold) on at least as many of the passages it would
                    accept as Gemini's full-lecture label; layers are checked in order, each on the
                    passages the kept layers before it left over
  simulate   step 1: layer 3 = Gemini's full-lecture labels (answers_api_gemini-3.1-pro-preview_low/tuning)
  send       step 2: one Gemini call per tuning lecture with only the layer-3 passages to label, each
             with up to 2 neighbours either side shown as context with their accepted label; same
             instructions and verse catalogue as the full-lecture paste file, adapted to say which
             passages to label; gemini-3.1-pro-preview, thinking low, as llm_frontier_gemini_api.py
             prompts  -> review/llm_frontier/tuning_leftover/<lecture>_LEFTOVER.txt
             answers  -> review/llm_frontier/answers_api_gemini-3.1-pro-preview_low/tuning_leftover/
  score      step 2: the real cascade (layer 3 = the leftover answers; a missing passage = NONE)
             vs the simulated cascade and Gemini alone, strict F1 per language and lecture

-> results/cascade_tuning.txt

Usage:
    python scripts/cascade_tuning.py simulate
    python scripts/cascade_tuning.py send
    python scripts/cascade_tuning.py score
"""
import argparse
import csv
import json
import os
import re
import sys
import time

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "scripts"))

import numpy as np  # noqa: E402
from evalkit import load_csv, score  # noqa: E402
from llm_frontier import DIR, parse  # noqa: E402
from score_by_tier_units import FIDELITY, TIERS  # noqa: E402
from step4_viterbi import FEATS, run  # noqa: E402

CONF = 0.9
NEIGHBOURS = 2
MODEL, THINKING = "gemini-3.1-pro-preview", "low"
GEMINI = os.path.join(DIR, "answers_api_gemini-3.1-pro-preview_low", "tuning")
PROMPTS = os.path.join(DIR, "tuning_leftover")
ANSWERS = os.path.join(DIR, "answers_api_gemini-3.1-pro-preview_low", "tuning_leftover")
LAYERS = ("1 anchor", f"2 Cue conf>={CONF}", "3 Gemini")
NO_ANCHOR_LAYER = ("en",)                                  # fix 1

HEADER_OLD = ("You are labelling a transcribed lecture on Bhagavad Gita Chapter 2.\n\nTASK\n"
              "For every passage below, state")
HEADER_NEW = ("You are labelling selected passages of a transcribed lecture on Bhagavad Gita Chapter 2.\n\n"
              "TASK\nThe passages below are shown in lecture order. A passage marked [LABEL] needs a label. "
              "A passage marked [CONTEXT, already labelled: ...] is shown only as context; its label has "
              "already been decided and must not be output. [...] marks passages left out.\n"
              "For every [LABEL] passage, state")
RULE_OLD = "- Every passage must get exactly one line, in order. Do not skip, merge, or renumber passages."
RULE_NEW = ("- Every [LABEL] passage must get exactly one line, in order. Do not skip, merge, or renumber "
            "passages. Output no line for [CONTEXT] passages.")


def load():
    teacher = {r["unit_id"]: min(0.98, max(0.02, float(r["soft_target"]))) for r in
               csv.DictReader(open(os.path.join(HERE, "results/silver/silver_labels.csv"), encoding="utf-8"))}
    info = {}
    order, gold, lectures, tune = run(alphas=("a05",), verbose=False, resume=True, off_override=teacher, info=info)
    cue = dict(order)["Viterbi (alpha 0.5)"]
    probs = info["probs_a05"]
    z = np.load(FEATS)
    refs = list(z["refs"])
    row = {u: i for i, u in enumerate(z["unit_ids"])}
    allv = [v for L in ("kn", "en") for v in tune[L]]
    gem, conf, anchor = {}, {}, {}
    for v in allv:
        p, _ = parse(os.path.join(GEMINI, f"{v}.txt"), ids_of(v))
        pv, po = probs[v]
        for k, u in enumerate(lectures[v]):
            gem[u] = p.get(u, [])
            conf[u] = (1 - po[k]) * pv[k, refs.index(cue[u][0])] / pv[k].sum() if cue[u] else po[k]
            anchor[u] = [refs[j] for j in np.flatnonzero(z["anchor"][row[u]])]
    return gold, lectures, tune, cue, gem, conf, anchor


def ids_of(v):
    return [(r["passage_id"], r["unit_id"]) for r in load_csv(os.path.join(DIR, "tuning", f"{v}_ids.csv"))]


def right(p, gold, u):
    return set(p) == set(gold[u])


def cascade(gold, lectures, tune, cue, gem, conf, anchor):
    """-> (layer of each passage, accepted label of layers 1-2, keep-rule lines)"""
    cand = {LAYERS[0]: (lambda u: bool(anchor[u]), lambda u: anchor[u]),
            LAYERS[1]: (lambda u: conf[u] >= CONF, lambda u: cue[u])}
    layer, label, lines = {}, {}, []
    for L in ("kn", "en"):
        left = [u for v in tune[L] for u in lectures[v]]
        for name in LAYERS[:2]:
            if name == LAYERS[0] and L in NO_ANCHOR_LAYER:
                lines.append(f"  {L} {name:18} dropped (fix 1)")
                continue
            take, lab = cand[name]
            acc = [u for u in left if take(u)]
            a, g = sum(right(lab(u), gold, u) for u in acc), sum(right(gem[u], gold, u) for u in acc)
            keep = a >= g
            lines.append(f"  {L} {name:18} would accept {len(acc):>3}: right {a:>3} vs Gemini {g:>3} -> "
                         f"{'KEEP' if keep else 'drop'}")
            if keep:
                for u in acc:
                    layer[u], label[u] = name, lab(u)
                left = [u for u in left if u not in layer]
        for u in left:
            layer[u] = LAYERS[2]
    return layer, label, lines


def tier(gold, u):
    return min(gold[u].values(), key=lambda t: FIDELITY.get(t, 9)) if gold[u] else "OFF"


def report(gold, lectures, tune, layer, label, last, gem, title):
    """layer table, by passage type, and the cascade's strict F1 with `last` as layer 3"""
    casc = {u: label[u] if layer[u] != LAYERS[2] else last.get(u, []) for u in layer}
    lines = ["", title]
    for L, lname in (("kn", "Kannada"), ("en", "English")):
        us = [u for v in tune[L] for u in lectures[v]]
        lines += [f"  {lname}: {len(us)} passages",
                  f"    {'layer':18}{'accepted':>14}{'cascade ok':>12}{'Gemini (full) ok':>18}"]
        for la in LAYERS:
            lu = [u for u in us if layer[u] == la]
            lines.append(f"    {la:18}{f'{len(lu)} ({len(lu) / len(us):.0%})':>14}"
                         f"{sum(right(casc[u], gold, u) for u in lu):>12}{sum(right(gem[u], gold, u) for u in lu):>18}")
        lines.append("    by passage type, accepted / cascade ok:   " + "   ".join(LAYERS))
        for t in TIERS + ["OFF"]:
            tu = [u for u in us if tier(gold, u) == t]
            if tu:
                lines.append(f"    {t:4} (n={len(tu):>3})  " + "      ".join(
                    f"{sum(layer[u] == la for u in tu):>4} / {sum(layer[u] == la and right(casc[u], gold, u) for u in tu):<4}"
                    for la in LAYERS))
        f_c, f_g = score(gold, casc, lectures, tune[L]), score(gold, gem, lectures, tune[L])
        reach = sum(layer[u] == LAYERS[2] for u in us)
        lines.append(f"    cascade strict F1 {f_c['F1']:.3f} (tp {f_c['tp']} fp {f_c['fp']} fn {f_c['fn']}) vs Gemini "
                     f"alone {f_g['F1']:.3f} ({f_c['F1'] - f_g['F1']:+.3f}); reaching Gemini {reach}/{len(us)} "
                     f"({reach / len(us):.0%})")
        lines.append("    per lecture, cascade / Gemini alone: " + ", ".join(
            f"{v} {score(gold, casc, lectures, [v])['F1']:.3f} / {score(gold, gem, lectures, [v])['F1']:.3f}"
            for v in tune[L]))
    return lines, casc


def fmt(refs):
    return ",".join(r.replace("BG ", "") for r in refs) or "NONE"


def build_prompt(v, lectures, layer, label):
    src = os.path.join(DIR, "tuning", next(f for f in os.listdir(os.path.join(DIR, "tuning"))
                                           if f.endswith(f"_{v}_PASTE_THIS.txt")))
    text = open(src, encoding="utf-8").read()
    head, body = text.split("\nLECTURE: ", 1)
    assert HEADER_OLD in head and RULE_OLD in head
    head = head.replace(HEADER_OLD, HEADER_NEW).replace(RULE_OLD, RULE_NEW)
    passages = dict(re.findall(r"^(P\d{3}): (.*)$", body, flags=re.M))
    ids = ids_of(v)
    pid = {u: p for p, u in ids}
    us = lectures[v]
    target = [k for k, u in enumerate(us) if layer[u] == LAYERS[2]]
    show = sorted({j for k in target for j in range(k - NEIGHBOURS, k + NEIGHBOURS + 1) if 0 <= j < len(us)})
    out, prev = [], None
    for j in show:
        if prev is not None and j != prev + 1:
            out.append("[...]")
        u = us[j]
        tag = "[LABEL]" if layer[u] == LAYERS[2] else f"[CONTEXT, already labelled: {fmt(label[u])}]"
        out.append(f"{pid[u]} {tag}: {passages[pid[u]]}")
        prev = j
    lang = body.split("\n", 1)[0].rsplit("language ", 1)[-1]
    return (head + f"\nLECTURE: {v}, {len(target)} passages to label ([LABEL]), "
            f"{len(show) - len(target)} context passages, language {lang}\n\n" + "\n\n".join(out) + "\n"), len(target)


def send(gold, lectures, tune, layer, label):
    import requests
    from llm_frontier_gemini_api import MAX_OUT, URL, cost, key
    k = key()
    os.makedirs(PROMPTS, exist_ok=True)
    os.makedirs(ANSWERS, exist_ok=True)
    for v in [v for L in ("kn", "en") for v in tune[L]]:
        prompt, n = build_prompt(v, lectures, layer, label)
        with open(os.path.join(PROMPTS, f"{v}_LEFTOVER.txt"), "w", encoding="utf-8") as f:
            f.write(prompt)
        if not n or os.path.exists(os.path.join(ANSWERS, f"{v}.txt")):
            print(f"  {v}: {n} to label, {'skipped (answer exists)' if n else 'nothing to send'}", flush=True)
            continue
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
            print(f"  {v}: no answer, {None if r is None else r.status_code} {'' if r is None else r.text[:200]}")
            continue
        j = r.json()
        cands = j.get("candidates", [])
        text = "".join(p.get("text", "") for c in cands for p in c.get("content", {}).get("parts", [])
                       if not p.get("thought"))
        with open(os.path.join(ANSWERS, f"{v}.txt"), "w", encoding="utf-8") as f:
            f.write(text)
        u = j.get("usageMetadata", {})
        fin = [c.get("finishReason") for c in cands]
        json.dump(dict(model=MODEL, model_version=j.get("modelVersion"), thinking_level=THINKING,
                       max_output_tokens=MAX_OUT, finish_reason=fin, usage=u, passages_to_label=n,
                       seconds=round(time.time() - t0, 1)), open(os.path.join(ANSWERS, f"{v}_usage.json"), "w"), indent=1)
        print(f"  {v:16} {n:>3} to label  {j.get('modelVersion')} {','.join(map(str, fin))}  in "
              f"{u.get('promptTokenCount', 0)} out {u.get('candidatesTokenCount', 0)} (thinking "
              f"{u.get('thoughtsTokenCount', 0)})  ${cost(MODEL, u):.3f}", flush=True)
    total = sum(cost(MODEL, json.load(open(os.path.join(ANSWERS, f)))["usage"])
                for f in os.listdir(ANSWERS) if f.endswith("_usage.json"))
    print(f"  leftover answers in {ANSWERS}: ${total:.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["simulate", "send", "score"])
    a = ap.parse_args()
    gold, lectures, tune, cue, gem, conf, anchor = load()
    layer, label, keep = cascade(gold, lectures, tune, cue, gem, conf, anchor)
    if a.step == "send":
        return send(gold, lectures, tune, layer, label)
    lines = ["Cascade, TUNING lectures only (fix 1: no anchor layer in English; fix 2: keep a layer only if "
             "it is right on at least as many of its passages as Gemini)", "", "keep rule:"] + keep
    sim, _ = report(gold, lectures, tune, layer, label, gem, gem,
                    "STEP 1, simulated: layer 3 = Gemini's full-lecture labels")
    lines += sim
    if a.step == "score":
        real, missing = {}, 0
        for v in [v for L in ("kn", "en") for v in tune[L]]:
            path = os.path.join(ANSWERS, f"{v}.txt")
            targets = [u for u in lectures[v] if layer[u] == LAYERS[2]]
            if not targets:
                continue
            p, prob = parse(path, ids_of(v)) if os.path.exists(path) else ({}, None)
            for u in targets:
                missing += u not in p
                real[u] = p.get(u, [])
        out, _ = report(gold, lectures, tune, layer, label, real, gem,
                        f"STEP 2, real: layer 3 = Gemini on the leftover passages only "
                        f"(+-{NEIGHBOURS} neighbours as context); {missing} leftover passages missing = NONE")
        lines += out
        lines += ["", "  leftover passages, Gemini right: real (leftover call) vs full lecture"]
        for L in ("kn", "en"):
            us = [u for v in tune[L] for u in lectures[v] if layer[u] == LAYERS[2]]
            lines.append(f"    {L}: {len(us)} passages, real {sum(right(real[u], gold, u) for u in us)}, "
                         f"full {sum(right(gem[u], gold, u) for u in us)}, labels differ on "
                         f"{sum(set(real[u]) != set(gem[u]) for u in us)}")
    text = "\n".join(lines)
    print(text)
    with open(os.path.join(HERE, "results", "cascade_tuning.txt"), "w", encoding="utf-8") as f:
        f.write(text + "\n")


if __name__ == "__main__":
    main()
