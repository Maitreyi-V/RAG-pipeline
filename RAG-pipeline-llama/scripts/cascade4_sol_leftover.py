#!/usr/bin/env python3
"""
Four-layer cascade with Sol, ENGLISH TUNING lectures: send Sol only the passages that reach layer 4
(real last layer), to measure the cost cut and whether Sol keeps its accuracy. Kannada keeps the
whole-lecture call (lectures are short), so its numbers and cost are those of cascade4_tuning.py.

  layers     as cascade4_tuning.py with Sol as layer 4 (English: layer 2 dropped, layer 3 kept)
  prompt     cascade_tuning.build_prompt v2: [LABEL] passages with +-2 neighbours as context, the
             nearest decided label before and after, "write each passage id exactly as shown"
  model      gpt-6.1-sol, Responses API, reasoning effort high (as the 42-lecture run)
             prompts -> review/llm_frontier/tuning_leftover4_sol/
             answers -> review/llm_frontier/answers_gpt-6.1-sol/tuning_leftover4/
  score      English strict F1 per lecture: Sol alone, 4-layer with Sol's whole-lecture labels,
             4-layer with the real leftover call (missing = NONE); cost of the three English
             lectures: Sol whole lecture vs Sol leftover call, plus gpt-5-mini

-> results/cascade4_sol_leftover_tuning.txt

Usage:
    python scripts/cascade4_sol_leftover.py send
    python scripts/cascade4_sol_leftover.py score
"""
import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "scripts"))

import cascade_tuning as C  # noqa: E402
from cascade4_tuning import L4, MINI, assign, candidates, labels  # noqa: E402
from evalkit import score  # noqa: E402
from llm_frontier import DIR, parse  # noqa: E402

MODEL, EFFORT = "gpt-6.1-sol", "high"
PRICE = (2.00, 10.00)
SOL = os.path.join(DIR, "answers_gpt-6.1-sol", "tuning")
PROMPTS = os.path.join(DIR, "tuning_leftover4_sol")
ANSWERS = os.path.join(DIR, "answers_gpt-6.1-sol", "tuning_leftover4")


def setup():
    gold, lectures, tune, cue, _, conf, anchor = C.load()
    mini, _ = labels(MINI, tune, lectures)
    sol, _ = labels(SOL, tune, lectures)
    us = [u for v in tune["en"] for u in lectures[v]]
    layer, out, keep = assign("en", us, sol, "Sol", gold, candidates(anchor, cue, conf, mini))
    # build_prompt marks layer == C.LAYERS[2] as [LABEL]; every other passage shows its accepted label
    marks = {u: C.LAYERS[2] if layer[u] == L4 else layer[u] for u in us}
    return gold, lectures, tune, sol, layer, out, marks, keep


def usage_cost(u):
    return u["input_tokens"] / 1e6 * PRICE[0] + u["output_tokens"] / 1e6 * PRICE[1]


def send():
    import openai
    from llm_frontier_api import client
    gold, lectures, tune, sol, layer, out, marks, keep = setup()
    c = client()
    os.makedirs(PROMPTS, exist_ok=True)
    os.makedirs(ANSWERS, exist_ok=True)
    for v in tune["en"]:
        prompt, n = C.build_prompt(v, lectures, marks, out, hints=True)
        with open(os.path.join(PROMPTS, f"{v}_LEFTOVER.txt"), "w", encoding="utf-8") as f:
            f.write(prompt)
        if os.path.exists(os.path.join(ANSWERS, f"{v}.txt")):
            print(f"  {v}: answer exists, skipped")
            continue
        t0, r = time.time(), None
        for i in range(6):
            try:
                r = c.responses.create(model=MODEL, input=prompt, reasoning={"effort": EFFORT})
                break
            except (openai.APIConnectionError, openai.RateLimitError, openai.InternalServerError) as e:
                print(f"  {v}: no answer ({type(e).__name__}), retry {i + 1}", flush=True)
                time.sleep(10 * (i + 1))
        if r is None:
            print(f"  {v}: no answer after 6 tries")
            continue
        with open(os.path.join(ANSWERS, f"{v}.txt"), "w", encoding="utf-8") as f:
            f.write(r.output_text or "")
        u = r.usage.model_dump()
        json.dump(dict(model=r.model, api="responses", effort=EFFORT, status=r.status, passages_to_label=n,
                       usage=u, seconds=round(time.time() - t0, 1)),
                  open(os.path.join(ANSWERS, f"{v}_usage.json"), "w"), indent=1)
        print(f"  {v:10} {n:>3} to label  {r.model} {r.status}  in {u['input_tokens']} out {u['output_tokens']}"
              f"  ${usage_cost(u):.3f}", flush=True)


def score_():
    gold, lectures, tune, sol, layer, out, marks, keep = setup()
    vs = tune["en"]
    real, missing = dict(out), 0
    for v in vs:
        p, _ = parse(os.path.join(ANSWERS, f"{v}.txt"), C.ids_of(v))
        for u in lectures[v]:
            if layer[u] == L4:
                missing += u not in p
                real[u] = p.get(u, [])
    lines = ["Four-layer cascade with Sol, ENGLISH tuning: real last layer (leftover passages only) vs "
             f"Sol's whole-lecture labels; leftover passages missing {missing} (= NONE)", "  keep rule:"] + keep
    lines += ["", f"  {'lecture':10}{'to Sol':>8}{'Sol alone':>11}{'4L whole':>10}{'4L real':>9}"
                  f"{'$ Sol whole':>13}{'$ Sol leftover':>16}{'$ mini':>8}"]
    tw = tl = tm = 0
    for v in vs + ["POOLED"]:
        vv = vs if v == "POOLED" else [v]
        n = sum(layer[u] == L4 for w in vv for u in lectures[w])
        if v != "POOLED":
            cw = usage_cost(json.load(open(os.path.join(SOL, f"{v}_usage.json")))["usage"])
            cl = usage_cost(json.load(open(os.path.join(ANSWERS, f"{v}_usage.json")))["usage"])
            um = json.load(open(os.path.join(MINI, f"{v}_usage.json")))["usage"]
            cm = um["prompt_tokens"] / 1e6 * 0.25 + um["completion_tokens"] / 1e6 * 2.0
            tw, tl, tm = tw + cw, tl + cl, tm + cm
        else:
            cw, cl, cm = tw, tl, tm
        f = [score(gold, p, lectures, vv)["F1"] for p in (sol, out, real)]
        lines.append(f"  {v:10}{n:>8}{f[0]:>11.3f}{f[1]:>10.3f}{f[2]:>9.3f}{cw:>13.3f}{cl:>16.3f}{cm:>8.3f}")
    us = [u for v in vs for u in lectures[v] if layer[u] == L4]
    lines += ["", f"  leftover passages right ({len(us)}): Sol whole-lecture "
                  f"{sum(set(sol[u]) == set(gold[u]) for u in us)}, Sol leftover call "
                  f"{sum(set(real[u]) == set(gold[u]) for u in us)}; labels differ on "
                  f"{sum(set(sol[u]) != set(real[u]) for u in us)}",
              f"  English cost: Sol alone ${tw:.3f}; 4-layer whole ${tw + tm:.3f}; 4-layer real ${tl + tm:.3f} "
              f"({(tl + tm) / tw - 1:+.0%} vs Sol alone)"]
    text = "\n".join(lines)
    print(text)
    with open(os.path.join(HERE, "results", "cascade4_sol_leftover_tuning.txt"), "w", encoding="utf-8") as f:
        f.write(text + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["send", "score"])
    send() if ap.parse_args().step == "send" else score_()


if __name__ == "__main__":
    main()
