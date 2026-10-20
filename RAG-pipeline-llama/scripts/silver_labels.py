#!/usr/bin/env python3
"""
Silver labels for the off-verse (digression) detector (work plan, Phase 1).

For each passage, one question to an LLM about the MIDDLE passage, shown with up
to 2 neighbouring passages on each side of the same lecture: is the speaker
explaining a verse of Bhagavad Gita Chapter 2, or telling a story / aside /
digression? The model returns a label and a probability that it is a digression.

  run    query every passage of the chosen split that has no saved answer yet.
         Answers are appended to results/silver/<model>.jsonl (one JSON per line,
         raw response kept). A saved passage is never queried again.
  eval   on the TUNING lectures only: AUC, accuracy and confusion of the silver
         labels against gold (gold off-verse = passage with no Chapter 2 verse).

Usage:
    python scripts/silver_labels.py run  --model gpt-5-mini --split dev
    python scripts/silver_labels.py eval --model gpt-5-mini
    python scripts/silver_labels.py run  --model gpt-5-mini --split all
"""
import argparse
import json
import os
import re
import sys
import threading
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

from evalkit import UNITS, UNITS_HELDOUT, load_csv, load_gold, read_split, verse_number  # noqa: E402

OUT_DIR = os.path.join(HERE, "results", "silver")
CONTEXT = 2
PROMPT = """Below are consecutive passages from a recorded lecture on the Bhagavad Gita, Chapter 2 (transcribed speech, in Kannada or English). The passage marked TARGET is in the middle; the others are context.

Decide what the speaker is doing in the TARGET passage only:
- VERSE: reciting, quoting, translating, explaining, paraphrasing or discussing the meaning of a verse of Bhagavad Gita Chapter 2.
- DIGRESSION: anything else - a story or anecdote, a modern example that has moved away from the verse, an introduction or closing, logistics, an audience question, or a verse from another chapter or text.

Reply with one line of JSON and nothing else:
{{"label": "VERSE" or "DIGRESSION", "p_digression": a number from 0 to 1}}

{passages}"""


def lectures_in_order():
    units = load_csv(UNITS) + load_csv(UNITS_HELDOUT)
    by = {}
    for u in sorted(units, key=lambda u: (u["video_file"], int(u["unit_index"]))):
        by.setdefault(u["video_file"], []).append(u)
    return by


def build(us, k):
    lo, hi = max(0, k - CONTEXT), min(len(us), k + CONTEXT + 1)
    parts = []
    for j in range(lo, hi):
        tag = "TARGET" if j == k else ("BEFORE" if j < k else "AFTER")
        parts.append(f"[{tag}]\n{us[j]['text']}")
    return PROMPT.format(passages="\n\n".join(parts))


def parse(text):
    m = re.search(r"\{.*\}", text or "", re.S)
    if not m:
        return None, None
    try:
        d = json.loads(m.group(0))
        return d.get("label"), float(d.get("p_digression"))
    except (ValueError, TypeError):
        return None, None


def path_for(model):
    return os.path.join(OUT_DIR, f"{model}.jsonl")


def load_saved(model):
    p = path_for(model)
    if not os.path.exists(p):
        return {}
    return {d["unit_id"]: d for d in map(json.loads, open(p, encoding="utf-8"))}


def run(model, split, effort, workers):
    from dotenv import load_dotenv
    from openai import OpenAI
    load_dotenv(os.path.join(os.path.dirname(HERE), ".env"), override=True)
    client = OpenAI()
    by = lectures_in_order()
    vids = [v for L in ("kn", "en") for s in (["dev", "test", "heldout"] if split == "all" else [split])
            for v in read_split(L, s)]
    saved = load_saved(model)
    todo = [(v, k) for v in vids for k, u in enumerate(by[v]) if u["unit_id"] not in saved]
    print(f"  {len(todo)} passage(s) to label ({len(saved)} already saved) with {model}, effort {effort}")
    os.makedirs(OUT_DIR, exist_ok=True)
    lock = threading.Lock()
    usage = {"in": 0, "out": 0, "n": 0}

    def one(item):
        v, k = item
        u = by[v][k]
        r = client.chat.completions.create(model=model, reasoning_effort=effort,
                                           messages=[{"role": "user", "content": build(by[v], k)}])
        text = r.choices[0].message.content
        label, p = parse(text)
        rec = dict(unit_id=u["unit_id"], video_file=v, model=r.model, effort=effort,
                   label=label, p_digression=p, raw=text,
                   prompt_tokens=r.usage.prompt_tokens, completion_tokens=r.usage.completion_tokens)
        with lock:
            with open(path_for(model), "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            usage["in"] += r.usage.prompt_tokens
            usage["out"] += r.usage.completion_tokens
            usage["n"] += 1

    with ThreadPoolExecutor(workers) as ex:
        list(ex.map(one, todo))
    print(f"  done: {usage['n']} calls, {usage['in']} input + {usage['out']} output tokens")


def evaluate(model):
    from sklearn.metrics import confusion_matrix, roc_auc_score
    saved = load_saved(model)
    lectures, gold, _ = load_gold()
    print(f"  silver labels from {model} vs gold, TUNING lectures only")
    for L in ("kn", "en", "both"):
        vids = [v for lang in (("kn", "en") if L == "both" else (L,)) for v in read_split(lang, "dev")]
        us = [u for v in vids for u in lectures[v] if u in saved and saved[u]["p_digression"] is not None]
        missing = sum(1 for v in vids for u in lectures[v]) - len(us)
        y = [int(not any(verse_number(r) for r in gold[u])) for u in us]
        p = [saved[u]["p_digression"] for u in us]
        pred = [int(saved[u]["label"] == "DIGRESSION") for u in us]
        auc = roc_auc_score(y, p) if 0 < sum(y) < len(y) else float("nan")
        acc = sum(a == b for a, b in zip(y, pred)) / len(y)
        tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
        print(f"  {L:4} n={len(us)} (unparsed/missing {missing})  gold off-verse {sum(y)}  "
              f"AUC {auc:.3f}  accuracy {acc:.3f}  | verse->verse {tn}  verse->digr {fp}  "
              f"digr->verse {fn}  digr->digr {tp}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["run", "eval"])
    ap.add_argument("--model", default="gpt-5-mini")
    ap.add_argument("--split", choices=["dev", "all"], default="dev")
    ap.add_argument("--effort", default="low", help="reasoning effort: minimal | low | medium | high")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    if args.step == "run":
        run(args.model, args.split, args.effort, args.workers)
    else:
        evaluate(args.model)


if __name__ == "__main__":
    main()
