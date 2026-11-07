#!/usr/bin/env python3
"""
Phase 3: an OpenAI frontier model labels lectures end to end through the API,
with the SAME paste files as the Opus 5.5 chat run (llm_frontier.py prepare),
sent unchanged, one API call per lecture (OpenAI Responses API).

  count   exact input tokens of the 42 paste files for --model (token-count
          endpoint, nothing is generated, no cost)
  send    --lectures probe   video_01 (TUNING lecture, T7 paste file; cost probe,
                             not one of the 42) -> answers_<model>/probe/
          --lectures test    the 30 E2 test lectures (ORDER.txt 01-30) -> answers_<model>/
          --lectures heldout the 12 E3 lectures (ORDER.txt 31-42)
          --lectures all     the 42 paste files in ORDER.txt order
          --lectures tuning  the 9 TUNING lectures (tuning/ORDER.txt, T1-T9) -> answers_<model>/tuning/

An API error with no answer is retried; an answer is never re-asked. A lecture
whose answer file exists is skipped. Answers and per-call usage live outside the
repo (they contain model output on transcript text):
  /Users/maitreyi/Documents/Capstone/review/llm_frontier/answers_<model>/

Usage:
    python scripts/llm_frontier_api.py count --model gpt-6.1-terra
    python scripts/llm_frontier_api.py send --model gpt-6.1-terra --effort high --lectures probe
    python scripts/llm_frontier_api.py send --model gpt-6.1-sol --effort high --lectures test
"""
import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

DIR = "/Users/maitreyi/Documents/Capstone/review/llm_frontier"
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROBE = ("video_01", os.path.join(DIR, "tuning", "T7_video_01_PASTE_THIS.txt"))
# $ per 1M input / output tokens, standard, short context (<=272K input), from
# developers.openai.com/api/docs/pricing on 7 Nov 2026
PRICE = {"gpt-6.1-sol": (2.00, 10.00), "gpt-6.1-terra": (1.00, 5.00)}


def client():
    from dotenv import load_dotenv
    from openai import OpenAI
    load_dotenv(os.path.join(os.path.dirname(HERE), ".env"), override=True)
    return OpenAI()


def paste_files(split=None):
    """the 42 scored lectures, ORDER.txt order (format check 00 excluded); split: test | heldout"""
    sys.path.insert(0, HERE)
    from evalkit import read_split
    keep = None if split is None else {v for L in ("kn", "en") for v in read_split(L, split)}
    out = []
    for line in open(os.path.join(DIR, "ORDER.txt"), encoding="utf-8"):
        tag, v = line.split("\t")[:2]
        if tag != "00_FORMAT_CHECK" and (keep is None or v in keep):
            out.append((v, os.path.join(DIR, f"{tag}_{v}_PASTE_THIS.txt")))
    return out


def tuning_files():
    """the 9 tuning lectures, tuning/ORDER.txt order"""
    d = os.path.join(DIR, "tuning")
    out = []
    for line in open(os.path.join(d, "ORDER.txt"), encoding="utf-8"):
        tag, v = line.split("\t")[:2]
        out.append((v, os.path.join(d, f"{tag}_{v}_PASTE_THIS.txt")))
    return out


def count(model):
    c = client()
    total = 0
    for v, path in paste_files():
        n = c.responses.input_tokens.count(model=model, input=open(path, encoding="utf-8").read()).input_tokens
        total += n
        print(f"  {v:24} {n:>8}")
    print(f"  total input tokens, 42 lectures, {model}: {total}")


def send(model, effort, which, workers=4):
    import openai
    c = client()
    out = os.path.join(DIR, f"answers_{model}") + {"probe": "/probe", "tuning": "/tuning"}.get(which, "")
    os.makedirs(out, exist_ok=True)
    todo = ([PROBE] if which == "probe" else tuning_files() if which == "tuning"
            else paste_files(None if which == "all" else which))
    todo = [(v, p) for v, p in todo if not os.path.exists(os.path.join(out, f"{v}.txt"))]
    print(f"  {len(todo)} lecture(s) to send to {model}, effort {effort}", flush=True)

    def one(item):
        v, path = item
        prompt = open(path, encoding="utf-8").read()
        t0 = time.time()
        for i in range(6):
            try:
                r = c.responses.create(model=model, input=prompt, reasoning={"effort": effort})
                break
            except (openai.APIConnectionError, openai.RateLimitError, openai.InternalServerError) as e:
                print(f"  {v}: no answer ({type(e).__name__}), retry {i + 1}", flush=True)
                if i == 5:                       # no answer received after 6 tries
                    return None
                time.sleep(10 * (i + 1))
        with open(os.path.join(out, f"{v}.txt"), "w", encoding="utf-8") as f:
            f.write(r.output_text or "")
        u = r.usage.model_dump()
        json.dump(dict(model=r.model, api="responses", effort=effort, status=r.status,
                       incomplete=r.incomplete_details.model_dump() if r.incomplete_details else None,
                       usage=u, seconds=round(time.time() - t0, 1)),
                  open(os.path.join(out, f"{v}_usage.json"), "w"), indent=1)
        print(f"  {v:24} {r.model}  {r.status}  in {u['input_tokens']:>7}  out {u['output_tokens']:>6} "
              f"(reasoning {u['output_tokens_details']['reasoning_tokens']})  {time.time() - t0:.0f}s", flush=True)
        return v

    with ThreadPoolExecutor(workers) as ex:
        done = [v for v in ex.map(one, todo) if v]
    failed = [v for v, _ in todo if v not in done]
    tin = tout = n = 0
    for f in sorted(os.listdir(out)):
        if f.endswith("_usage.json"):
            u = json.load(open(os.path.join(out, f)))["usage"]
            tin, tout, n = tin + u["input_tokens"], tout + u["output_tokens"], n + 1
    pin, pout = PRICE.get(model, (0, 0))
    print(f"  sent now: {len(done)} answered, {len(failed)} failed {failed}")
    print(f"  all answers in {out}: {n} lectures, {tin} input + {tout} output tokens"
          + (f" = ${tin / 1e6 * pin + tout / 1e6 * pout:.3f} at ${pin}/${pout} per 1M" if model in PRICE else ""))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["count", "send"])
    ap.add_argument("--model", required=True)
    ap.add_argument("--effort", default="high")
    ap.add_argument("--lectures", choices=["probe", "test", "heldout", "all", "tuning"], default="probe")
    a = ap.parse_args()
    count(a.model) if a.step == "count" else send(a.model, a.effort, a.lectures)


if __name__ == "__main__":
    main()
