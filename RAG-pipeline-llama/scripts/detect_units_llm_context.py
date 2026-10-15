#!/usr/bin/env python3
"""
GPT-4o-mini with discourse context: the LLM's fair fight against carry.

detect_units.py --layer 4 shows the model one unit in isolation, so it loses to
carry for the same reason every per-unit method does: a paraphrase unit on its
own does not say which verse it belongs to. Here the model gets what carry gets
and more:

  - a list of all 72 verses with a translation
  - the previous --context units of the same lecture
  - its own answer for the previous unit
  - the current unit

Only EARLIER units are shown, never later ones, so the model is causal exactly
like carry. The previous answer is the model's own, never the gold label.

There is one prompt, written before any result was seen; nothing is tuned.

Lectures run in parallel; within a lecture units run in order (each needs the
previous answer). Progress is checkpointed per unit to <out>.partial, and token
usage is recorded so the cost is known.

Usage:
    python scripts/detect_units_llm_context.py
    python scripts/detect_units_llm_context.py --video video_01          # one lecture
    python scripts/detect_units_llm_context.py --resume
"""
import argparse
import csv
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

from evalkit import UNITS, UNITS_HELDOUT, load_csv  # noqa: E402

OUT = os.path.join(HERE, "results", "llmctx_all.csv")
PRICE = {"input": 0.15, "cached": 0.075, "output": 0.60}     # USD per 1M tokens, gpt-4o-mini

HEADER = """You are an expert on the Bhagavad Gita Chapter 2. You will read consecutive passages from one lecture and decide which verse(s) of Chapter 2 the speaker is explaining or discussing in the CURRENT passage.

The 72 verses of Chapter 2:
{verses}

Rules:
- Reply with ONLY the verse number(s), e.g. "2.17" or "2.22, 2.23", or "none".
- A speaker often keeps explaining one verse across several passages without quoting it again. Use the earlier passages to tell which verse is still being explained.
- If the current passage is a digression that is not about any verse of Chapter 2, reply "none".
- Do NOT explain your reasoning.
"""

TURN = """Earlier passages from the same lecture (context only):
{context}

Your answer for the previous passage: {previous}

CURRENT passage:
\"\"\"
{current}
\"\"\"

Verse(s):"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--context", type=int, default=3, help="earlier units shown (default 3)")
    ap.add_argument("--model", default="gpt-4o-mini")
    ap.add_argument("--video", help="run a single lecture")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    from dotenv import find_dotenv, load_dotenv
    load_dotenv(find_dotenv(usecwd=True))
    from openai import OpenAI
    from layer4_llm_detector import parse_verse_response
    from verse_translations import VERSE_TRANSLATIONS

    client = OpenAI()
    header = HEADER.format(verses="\n".join(
        f"{ref.replace('BG ', '')}: {trs[0]}" for ref, trs in VERSE_TRANSLATIONS.items()))

    lectures = {}
    for u in load_csv(UNITS) + load_csv(UNITS_HELDOUT):
        lectures.setdefault(u["video_file"], []).append(u)
    for us in lectures.values():
        us.sort(key=lambda u: int(u["unit_index"]))
    if args.video:
        lectures = {args.video: lectures[args.video]}
    n_units = sum(len(us) for us in lectures.values())

    partial = args.out + ".partial"
    done = {}
    if args.resume and os.path.exists(partial):
        with open(partial, newline="", encoding="utf-8") as f:
            done = {r["unit_id"]: r["detected_verses"] for r in csv.DictReader(f)}
        print(f"  resuming: {len(done)}/{n_units} units already done")
    elif os.path.exists(partial):
        os.remove(partial)

    lock = threading.Lock()
    usage = {"input": 0, "cached": 0, "output": 0}
    pf = open(partial, "a", newline="", encoding="utf-8")
    pw = csv.writer(pf)
    if not done:
        pw.writerow(["unit_id", "video_file", "detected_verses"])

    def run_lecture(video):
        us = lectures[video]
        for i, u in enumerate(us):
            if u["unit_id"] in done:
                continue
            prev = done[us[i - 1]["unit_id"]].replace("BG ", "").replace(";", ", ") if i else ""
            ctx = "\n\n".join(x["text"] for x in us[max(0, i - args.context):i])
            prompt = TURN.format(context=ctx or "(this is the first passage of the lecture)",
                                 previous=prev or ("none" if i else "(no previous passage)"),
                                 current=u["text"])
            # Errors are deliberately not caught: a failed call must stop this
            # lecture, not be recorded as "no verse". --resume picks it up again.
            r = client.chat.completions.create(
                model=args.model, temperature=0.0, max_tokens=50,
                messages=[{"role": "system", "content": header},
                          {"role": "user", "content": prompt}])
            refs = ";".join(f"BG 2.{n}" for n in parse_verse_response(r.choices[0].message.content))
            cached = getattr(r.usage.prompt_tokens_details, "cached_tokens", 0) or 0
            with lock:
                done[u["unit_id"]] = refs
                pw.writerow([u["unit_id"], video, refs])
                pf.flush()
                usage["input"] += r.usage.prompt_tokens - cached
                usage["cached"] += cached
                usage["output"] += r.usage.completion_tokens
                if len(done) % 100 == 0:
                    print(f"  ... {len(done)}/{n_units}", flush=True)

    errors = []
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(run_lecture, v): v for v in lectures}
        for fut, v in futures.items():
            try:
                fut.result()
            except Exception as e:
                errors.append(f"{v}: {type(e).__name__}: {str(e)[:120]}")
    pf.close()

    cost = sum(usage[k] * PRICE[k] for k in usage) / 1e6
    print(f"\n  tokens this run: {usage['input']:,} input + {usage['cached']:,} cached input "
          f"+ {usage['output']:,} output   ~ ${cost:.2f} at the listed {args.model} prices")
    if errors or len(done) < n_units:
        for e in errors[:5]:
            print(f"  !! {e}")
        sys.exit(f"\nINCOMPLETE: {len(done)}/{n_units} units. Progress kept in "
                 f"{os.path.basename(partial)} -- re-run with --resume.")

    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["unit_id", "video_file", "detected_verses"])
        for video, us in lectures.items():
            for u in us:
                w.writerow([u["unit_id"], video, done[u["unit_id"]]])
    os.remove(partial)
    n_hit = sum(1 for v in done.values() if v)
    print(f"  {n_units} units, {n_hit} with a detection ({n_hit / n_units:.1%})")
    print(f"  wrote -> {os.path.relpath(args.out, HERE)}")


if __name__ == "__main__":
    main()
