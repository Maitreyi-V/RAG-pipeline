#!/usr/bin/env python3
"""
run_nvidia.py - run the pre-built lecture prompts through NVIDIA's free API
(build.nvidia.com, OpenAI-compatible) and save results in the same format as
the Gemma run: <lecture_id>.txt (raw reply, unchanged) + <lecture_id>_usage.json.

Same rules as the gpt-5-mini / Gemma runs:
  * plain user message, no system prompt, no schema / response_format
  * one call per lecture, an answer is NEVER re-asked
  * retry only when NO answer came back at all (connection / API error)
  * refusals, questions, bad formatting are saved as the answer

Usage:
  pip install openai
  export NVIDIA_API_KEY="nvapi-..."
  python run_nvidia.py --model moonshotai/kimi-k2.6 --out runs_kimi
  python run_nvidia.py --model z-ai/glm-5.2        --out runs_glm
  python run_nvidia.py --model moonshotai/kimi-k2.6 --out test --only kannada_10
"""
import os, sys, csv, json, time, argparse, glob
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True, help="exact model ID from build.nvidia.com")
ap.add_argument("--out", required=True, help="output folder")
ap.add_argument("--prompts", default=None, help="folder containing lectures.csv and prompts/ (auto-detected if omitted)")
ap.add_argument("--temperature", type=float, default=0.0)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--max-tokens", type=int, default=16384,
                help="reasoning models spend output tokens on thinking, so keep this generous")
ap.add_argument("--extra-body", default=None,
                help='JSON for the thinking setting, e.g. \'{"chat_template_kwargs": {"thinking": true}}\'. '
                     "Check the model's page on build.nvidia.com for the exact field. Recorded in usage.json.")
ap.add_argument("--thinking-label", default="default (not set)",
                help="human-readable thinking setting to record, e.g. 'on', 'off', 'high'")
ap.add_argument("--only", nargs="*", default=None, help="run only these lecture IDs (smoke test)")
ap.add_argument("--sleep", type=float, default=2.0, help="seconds between calls (rate limit ~40 req/min)")
args = ap.parse_args()

KEY = os.environ.get("NVIDIA_API_KEY")
if not KEY:
    sys.exit('ERROR: export NVIDIA_API_KEY="nvapi-..."')
try:
    from openai import OpenAI
except ImportError:
    sys.exit("Run: pip install openai")

client = OpenAI(base_url="https://integrate.api.nvidia.com/v1", api_key=KEY, timeout=3600)

# locate the prompts package
if args.prompts:
    D = Path(args.prompts)
else:
    found = glob.glob("**/lectures.csv", recursive=True) + glob.glob("/kaggle/input/**/lectures.csv", recursive=True)
    if not found:
        sys.exit("lectures.csv not found; pass --prompts <folder>")
    D = Path(found[0]).parent
lectures = list(csv.DictReader(open(D / "lectures.csv", encoding="utf-8")))
if args.only:
    lectures = [r for r in lectures if r["lecture_id"] in args.only]

OUT = Path(args.out); OUT.mkdir(parents=True, exist_ok=True)
extra_body = json.loads(args.extra_body) if args.extra_body else None

(OUT / "_run_log.json").write_text(json.dumps({
    "started": time.strftime("%Y-%m-%d %H:%M:%S"), "model": args.model,
    "endpoint": "https://integrate.api.nvidia.com/v1", "temperature": args.temperature,
    "seed": args.seed, "max_tokens": args.max_tokens, "extra_body": extra_body,
    "thinking_setting": args.thinking_label}, indent=2))


def call(prompt):
    """One call per lecture. Retry only if NO answer came back."""
    for attempt in range(3):
        try:
            t = time.time()
            kw = dict(model=args.model, messages=[{"role": "user", "content": prompt}],
                      temperature=args.temperature, max_tokens=args.max_tokens, seed=args.seed,
                      stream=False)
            if extra_body:
                kw["extra_body"] = extra_body
            r = client.chat.completions.create(**kw)
            return r, time.time() - t
        except Exception as e:
            print(f"  no answer ({e}) - retrying in 20s")
            time.sleep(20)
    return None, None


for row in lectures:
    lid = row["lecture_id"]
    txt, usage = OUT / f"{lid}.txt", OUT / f"{lid}_usage.json"
    if txt.exists() and usage.exists():
        print("skip", lid); continue
    prompt = open(D / "prompts" / f"{lid}.txt", encoding="utf-8").read()
    r, secs = call(prompt)
    if r is None:
        print(lid, "NO ANSWER - nothing saved, rerun to retry"); continue

    ch = r.choices[0]
    reply = ch.message.content or ""
    reasoning = getattr(ch.message, "reasoning_content", None) or getattr(ch.message, "reasoning", None)
    txt.write_text(reply, encoding="utf-8")            # raw reply, unchanged
    if reasoning:
        (OUT / f"{lid}_reasoning.txt").write_text(reasoning, encoding="utf-8")
    u = r.usage
    usage.write_text(json.dumps({
        "lecture_id": lid, "model": args.model, "model_returned": r.model,
        "temperature": args.temperature, "seed": args.seed, "max_tokens": args.max_tokens,
        "thinking_setting": args.thinking_label, "extra_body": extra_body,
        "prompt_tokens": getattr(u, "prompt_tokens", None),
        "completion_tokens": getattr(u, "completion_tokens", None),
        "seconds": round(secs, 1), "finish_reason": ch.finish_reason,
        "prompt_chars": len(prompt), "empty_reply": not reply.strip(),
        "has_reasoning_text": bool(reasoning),
    }, indent=2), encoding="utf-8")
    flag = "  <-- finish_reason=length: output cut off, consider raising --max-tokens" if ch.finish_reason == "length" else ""
    print(f"{lid}: prompt_tokens={getattr(u,'prompt_tokens',None)} out={getattr(u,'completion_tokens',None)} "
          f"{secs:.0f}s finish={ch.finish_reason} empty={not reply.strip()}{flag}")
    time.sleep(args.sleep)

print("done")
