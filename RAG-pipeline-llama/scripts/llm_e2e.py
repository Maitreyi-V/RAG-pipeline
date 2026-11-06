#!/usr/bin/env python3
"""
Phase 3: a cheap LLM labels whole lectures END TO END (verse identity and
off-verse, no structure), on all 42 test + held-out lectures.

Prompt: llm_pilot.INSTRUCTIONS + the 72-verse catalogue + the numbered passages
(unchanged from the frontier pilot). Model gpt-5-mini, reasoning effort "low"
(the same model and effort as the silver teacher, so the comparison isolates
"LLM as discourse cue + structure" vs "LLM end to end"). One call per lecture;
an API error is retried (no answer was received), an answer is never re-asked.
Passages missing from the answer count as NONE and are reported.

  run      --lectures dryrun | all        (dryrun = video_01, a tuning lecture)
  score    E2 (per language) and E3 (per new speaker and pooled): strict / +-1 /
           theme F1 with 95% lecture-bootstrap CI, and paired comparisons
           (freeze_grounding.paired) vs v2_teacher145, v1 frozen, plain carry

Local open-weight models: --model ollama:<name> (e.g. ollama:llama3.1:8b) sends the
same prompt to a local Ollama server (temperature 0, seed 0, context NUM_CTX tokens,
one lecture at a time); prompt tokens actually read are recorded, so a truncated
prompt is visible. Cost $0; wall time is recorded instead.

Raw answers (and the prompts) go outside the repo:
  /Users/maitreyi/Documents/Capstone/review/llm_e2e/<model>/
Parsed labels (no transcript text): results/llm_e2e_<model>.csv

Usage:
    python scripts/llm_e2e.py run --lectures dryrun
    python scripts/llm_e2e.py run --lectures all
    python scripts/llm_e2e.py run --lectures dev    # 9 tuning lectures (features for the v3 hybrid)
    python scripts/llm_e2e.py score
"""
import argparse
import csv
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

from evalkit import LECTURES_NEW, UNITS, UNITS_HELDOUT, load_csv, load_gold, read_split  # noqa: E402
from llm_pilot import INSTRUCTIONS, catalogue, parse  # noqa: E402

REVIEW = "/Users/maitreyi/Documents/Capstone/review/llm_e2e"
DRYRUN = ["video_01"]
PRICE = (0.25, 2.00)        # $ per 1M input / output tokens, as pasted 15 Oct (check the pricing page)
NUM_CTX = 65536


def folder(model):
    return model.replace("ollama:", "").replace(":", "-")


def ollama_call(name, prompt):
    import requests
    t0 = time.time()
    r = requests.post("http://localhost:11434/api/chat", timeout=7200, json=dict(
        model=name, stream=False, messages=[{"role": "user", "content": prompt}],
        options=dict(temperature=0, seed=0, num_ctx=NUM_CTX))).json()
    usage = dict(prompt_tokens=r.get("prompt_eval_count", 0), completion_tokens=r.get("eval_count", 0),
                 seconds=round(time.time() - t0, 1))
    return r["message"]["content"], r.get("model", name), r.get("done_reason", ""), usage


def all_lectures():
    return [v for s in ("test", "heldout") for L in ("kn", "en") for v in read_split(L, s)]


def run(model, effort, which, workers=4):
    local = model.startswith("ollama:")
    if local:
        workers = 1
    else:
        from dotenv import load_dotenv
        from openai import OpenAI
        load_dotenv(os.path.join(os.path.dirname(HERE), ".env"), override=True)
        client = OpenAI()
    lectures, _, _ = load_gold()
    text = {u["unit_id"]: u["text"] for u in load_csv(UNITS) + load_csv(UNITS_HELDOUT)}
    cat = catalogue()
    out = os.path.join(REVIEW, folder(model))
    os.makedirs(out, exist_ok=True)
    vids = (DRYRUN if which == "dryrun" else all_lectures() if which == "all"
            else [v for L in ("kn", "en") for v in read_split(L, "dev")])
    todo = [v for v in vids if not os.path.exists(os.path.join(out, f"{v}.txt"))]
    print(f"  {len(todo)} lecture(s) to send ({len(vids) - len(todo)} already answered) to {model}, effort {effort}")

    def one(v):
        us = lectures[v]
        ids = [f"P{k + 1:03d}" for k in range(len(us))]
        prompt = (INSTRUCTIONS.format(last=ids[-1]) + "\n" + cat + f"\n\nLECTURE PASSAGES ({len(us)} passages)\n\n"
                  + "\n\n".join(f"{i}: {text[u]}" for i, u in zip(ids, us)) + "\n")
        for i in range(6):
            try:
                if local:
                    content, served, finish, usage = ollama_call(model[len("ollama:"):], prompt)
                else:
                    r = client.chat.completions.create(model=model, reasoning_effort=effort,
                                                       messages=[{"role": "user", "content": prompt}],
                                                       max_completion_tokens=64000)
                    content, served, finish = r.choices[0].message.content, r.model, r.choices[0].finish_reason
                    usage = r.usage.model_dump()
                break
            except Exception as e:                 # no answer received: ask again
                if i == 5:
                    print(f"  {v}: FAILED {type(e).__name__}: {str(e)[:120]}")
                    return None
                time.sleep(3 * (i + 1))
        with open(os.path.join(out, f"{v}_ids.csv"), "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerows([("passage_id", "unit_id")] + list(zip(ids, us)))
        with open(os.path.join(out, f"{v}.txt"), "w", encoding="utf-8") as f:
            f.write(content or "")
        json.dump(dict(model=served, effort=None if local else effort, finish_reason=finish, usage=usage,
                       **(dict(temperature=0, seed=0, num_ctx=NUM_CTX) if local else {})),
                  open(os.path.join(out, f"{v}_usage.json"), "w"), indent=1)
        print(f"  {v:22} {served}  finish {finish:7} in {usage['prompt_tokens']:>7}  out "
              f"{usage['completion_tokens']:>6}" + (f"  {usage['seconds']}s" if local else ""), flush=True)
        return v, served, finish, usage["prompt_tokens"], usage["completion_tokens"]

    with ThreadPoolExecutor(workers) as ex:
        res = [x for x in ex.map(one, todo) if x]
    tin = sum(x[3] for x in res)
    tout = sum(x[4] for x in res)
    print(f"  total: {len(res)} answered, {tin} input + {tout} output tokens "
          + ("(local, $0)" if local else
             f"= ${tin / 1e6 * PRICE[0] + tout / 1e6 * PRICE[1]:.3f} at ${PRICE[0]}/${PRICE[1]} per 1M"))


def load_answers(model):
    lectures, _, _ = load_gold()
    out = os.path.join(REVIEW, folder(model))
    pred, problems = {}, {}
    for v in all_lectures():
        path = os.path.join(out, f"{v}.txt")
        if not os.path.exists(path):
            continue
        ids = [(r["passage_id"], r["unit_id"]) for r in load_csv(os.path.join(out, f"{v}_ids.csv"))]
        p, prob = parse(path, ids)
        for u in lectures[v]:
            pred[u] = p.get(u, [])                   # missing passage = NONE
        problems[v] = prob
    return pred, problems


def score(model):
    from baselines import carry
    from freeze_grounding import (f1_near, f1_strict, near_counts, paired, strict_counts, theme_counts,
                                  with_ci)
    from step4_viterbi import FEATS, OFF, OUT, predict_locked
    lectures, gold, _ = load_gold()
    pred, problems = load_answers(model)
    vids_all = [v for v in all_lectures() if v in problems]
    with open(os.path.join(HERE, "results", f"llm_e2e_{folder(model)}.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["unit_id", "video_file", "refs"])
        for v in vids_all:
            for u in lectures[v]:
                w.writerow([u, v, ";".join(pred[u])])
    z = np.load(FEATS)
    refs = list(z["refs"])
    row = {u: i for i, u in enumerate(z["unit_ids"])}
    anchors = {u: [refs[j] for j in np.flatnonzero(z["anchor"][row[u]])] for u in row}

    def decoded(path):
        dec = predict_locked({v: lectures[v] for v in vids_all}, "a05", path=path)
        return {u: ([] if st == OFF else [refs[st]]) + [refs[j] for j in extra]
                for v in vids_all for u, (st, extra, _, _) in zip(lectures[v], dec[v])}
    systems = {f"{model} end-to-end": pred,
               "v2 teacher 145 (primary)": decoded(os.path.join(HERE, "results", "v2_teacher145.pkl")),
               "v1 frozen (alpha 0.5)": decoded(OUT), "plain carry": {}}
    for v in vids_all:
        systems["plain carry"].update(carry(lectures[v], anchors))
    speaker = {r["video_file"]: r["speaker"] for r in load_csv(LECTURES_NEW)}
    groups = []
    for split in ("test", "heldout"):
        for L, name in (("kn", "Kannada"), ("en", "English")):
            vs = [v for v in read_split(L, split) if v in problems]
            if not vs:
                continue
            if split == "heldout":
                groups += [(f"E3 {name}: {sp}", [v for v in vs if speaker[v] == sp])
                           for sp in sorted({speaker[v] for v in vs})]
                groups.append((f"E3 new {name} (pooled)", vs))
            else:
                groups.append((f"E2 {name} test", vs))
    n_missing = sum(len(p["missing"]) for p in problems.values())
    n_bad = sum(len(p["unparsed"]) for p in problems.values())
    n_out = sum(p["outside_ch2"] for p in problems.values())
    tin = tout = 0
    for v in vids_all:
        u_ = json.load(open(os.path.join(REVIEW, folder(model), f"{v}_usage.json")))["usage"]
        tin, tout = tin + u_["prompt_tokens"], tout + u_["completion_tokens"]
    local = model.startswith("ollama:")
    lines = [f"Phase 3 — {model} end to end ("
             + (f"temperature 0, seed 0, num_ctx {NUM_CTX}" if local else "reasoning effort low")
             + f"), {len(vids_all)} lectures, one call each",
             f"parsing: {n_missing} passage(s) missing (scored as NONE), {n_bad} unparsed line(s), "
             f"{n_out} verse(s) outside 2.1-2.72; tokens {tin} in + {tout} out "
             + ("(local, $0)" if local else
                f"= ${tin / 1e6 * PRICE[0] + tout / 1e6 * PRICE[1]:.3f} at ${PRICE[0]}/${PRICE[1]} per 1M")]
    for gname, vs in groups:
        units = [u for v in vs for u in lectures[v]]
        lines += ["", f"{gname}   lectures {len(vs)}   passages {len(units)}   no-verse "
                      f"{sum(1 for u in units if not gold[u])}",
                  f"  {'system':32} {'strict F1 [CI]':22} {'+-1 F1 [CI]':22} {'theme F1 [CI]':22}"]
        for name, p in systems.items():
            cells = []
            for counts, f1 in ((strict_counts, f1_strict), (near_counts, f1_near), (theme_counts, f1_strict)):
                point, ci = with_ci([counts(gold, p, lectures[v]) for v in vs], f1, 1000)
                cells.append(f"{point:.3f}" + (f" [{ci[0]:.3f},{ci[1]:.3f}]" if ci else ""))
            lines.append(f"  {name:32} {cells[0]:22} {cells[1]:22} {cells[2]:22}")
        lines.append(f"  paired, strict F1 (A - B), 10000 resamples:  {'diff':>7} {'95% CI':>17} "
                     f"{'P(diff<=0)':>10} {'win/tie/loss':>13} {'sign p':>7}")
        a_ = f"{model} end-to-end"
        for b_ in ("v2 teacher 145 (primary)", "v1 frozen (alpha 0.5)", "plain carry"):
            d, ci, le0, (w, t, lo), p = paired(gold, lectures, systems[a_], systems[b_], vs, 10000)
            lines.append(f"    LLM - {b_:36} {d:>+7.3f} [{ci[0]:+.3f},{ci[1]:+.3f}] {le0:>10.3f} "
                         f"{f'{w}/{t}/{lo}':>13} {p:>7.3f}")
    text = "\n".join(lines)
    print(text)
    open(os.path.join(HERE, "results", f"llm_e2e_{folder(model)}.txt"), "w").write(text + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["run", "score"])
    ap.add_argument("--model", default="gpt-5-mini")
    ap.add_argument("--effort", default="low")
    ap.add_argument("--lectures", choices=["dryrun", "all", "dev"], default="dryrun")
    args = ap.parse_args()
    if args.step == "run":
        run(args.model, args.effort, args.lectures)
    else:
        score(args.model)


if __name__ == "__main__":
    main()
