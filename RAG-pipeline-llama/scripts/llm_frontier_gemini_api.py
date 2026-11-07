#!/usr/bin/env python3
"""
Phase 3: a Gemini model labels lectures end to end through the Gemini API
(REST, v1beta generateContent), with the SAME paste files as the Opus 5.5 chat
run and the GPT-6.1 Sol API run, sent unchanged, one call per lecture. Same
lecture sets and answer-folder layout as llm_frontier_api.py.

  send    --lectures probe   video_01 (TUNING lecture, T7 paste file; cost probe,
                             not one of the 42) -> answers_api_<model>_<thinking>/probe/
          --lectures test | heldout | all | tuning   as in llm_frontier_api.py

Settings: thinkingLevel = --thinking, maxOutputTokens 65536 (the model's limit),
everything else the API default. An API error with no answer (connection,
timeout, 429, 5xx) is retried; an answer is never re-asked. A lecture whose
answer file exists is skipped. Answers and per-call usage live outside the repo:
  /Users/maitreyi/Documents/Capstone/review/llm_frontier/answers_api_<model>_<thinking>/

Usage:
    python scripts/llm_frontier_gemini_api.py send --model gemini-3.1-pro-preview --thinking low --lectures probe
    python scripts/llm_frontier_gemini_api.py send --model gemini-3.1-pro-preview --thinking low --lectures all
"""
import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from llm_frontier_api import DIR, HERE, PROBE, paste_files, tuning_files  # noqa: E402

# $ per 1M input / output (output includes thinking), standard paid tier, prompts <= 200K,
# from aistudio.google.com/docs/pricing on 7 Nov 2026 (3.8 Flash: price through 31 Dec 2026)
PRICE = {"gemini-3.1-pro-preview": (2.00, 12.00), "gemini-3.8-flash": (0.75, 3.75)}
MAX_OUT = 65536
URL = "https://generativelanguage.googleapis.com/v1beta/models/{}:generateContent"


def key():
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(HERE), ".env"), override=True)
    return os.environ["GEMINI_API_KEY"]


def cost(model, u):
    """upper bound: cached input counted at the full input price"""
    pin, pout = PRICE.get(model, (0, 0))
    return u.get("promptTokenCount", 0) / 1e6 * pin + \
        (u.get("candidatesTokenCount", 0) + u.get("thoughtsTokenCount", 0)) / 1e6 * pout


def send(model, thinking, which, workers=4):
    import requests
    k = key()
    out = os.path.join(DIR, f"answers_api_{model}_{thinking}") + {"probe": "/probe", "tuning": "/tuning"}.get(which, "")
    os.makedirs(out, exist_ok=True)
    todo = ([PROBE] if which == "probe" else tuning_files() if which == "tuning"
            else paste_files(None if which == "all" else which))
    todo = [(v, p) for v, p in todo if not os.path.exists(os.path.join(out, f"{v}.txt"))]
    print(f"  {len(todo)} lecture(s) to send to {model}, thinking {thinking}", flush=True)

    def one(item):
        v, path = item
        body = {"contents": [{"parts": [{"text": open(path, encoding="utf-8").read()}]}],
                "generationConfig": {"thinkingConfig": {"thinkingLevel": thinking}, "maxOutputTokens": MAX_OUT}}
        t0 = time.time()
        for i in range(6):
            try:
                r = requests.post(URL.format(model), headers={"x-goog-api-key": k}, json=body, timeout=(15, 1200))
                if r.status_code == 200:
                    break
                msg = f"HTTP {r.status_code} {r.text[:200]}"
                if r.status_code != 429 and r.status_code < 500:
                    print(f"  {v}: no answer, {msg}", flush=True)
                    return None                  # request rejected (not rate limit / server): not retried
            except requests.RequestException as e:
                msg = type(e).__name__
            print(f"  {v}: no answer ({msg}), retry {i + 1}", flush=True)
            if i == 5:                           # no answer received after 6 tries
                return None
            time.sleep(20 * (i + 1))
        j = r.json()
        cands = j.get("candidates", [])
        text = "".join(p.get("text", "") for c in cands for p in c.get("content", {}).get("parts", [])
                       if not p.get("thought"))
        with open(os.path.join(out, f"{v}.txt"), "w", encoding="utf-8") as f:
            f.write(text)
        u = j.get("usageMetadata", {})
        fin = [c.get("finishReason") for c in cands]
        json.dump(dict(model=model, model_version=j.get("modelVersion"), api="gemini v1beta generateContent (REST)",
                       thinking_level=thinking, max_output_tokens=MAX_OUT, finish_reason=fin,
                       prompt_feedback=j.get("promptFeedback"), usage=u, seconds=round(time.time() - t0, 1)),
                  open(os.path.join(out, f"{v}_usage.json"), "w"), indent=1)
        print(f"  {v:24} {j.get('modelVersion')}  {','.join(map(str, fin))}  in {u.get('promptTokenCount', 0):>7}  "
              f"out {u.get('candidatesTokenCount', 0):>6} (thinking {u.get('thoughtsTokenCount', 0)})  "
              f"${cost(model, u):.3f}  {time.time() - t0:.0f}s", flush=True)
        return v

    with ThreadPoolExecutor(workers) as ex:
        done = [v for v in ex.map(one, todo) if v]
    failed = [v for v, _ in todo if v not in done]
    n, total = 0, 0.0
    for f in sorted(os.listdir(out)):
        if f.endswith("_usage.json"):
            n, total = n + 1, total + cost(model, json.load(open(os.path.join(out, f)))["usage"])
    print(f"  sent now: {len(done)} answered, {len(failed)} failed {failed}")
    print(f"  all answers in {out}: {n} lectures, ${total:.3f} at {PRICE.get(model)} per 1M in/out")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["send"])
    ap.add_argument("--model", required=True)
    ap.add_argument("--thinking", required=True, choices=["minimal", "low", "medium", "high"])
    ap.add_argument("--lectures", choices=["probe", "test", "heldout", "all", "tuning"], default="probe")
    a = ap.parse_args()
    send(a.model, a.thinking, a.lectures)


if __name__ == "__main__":
    main()
