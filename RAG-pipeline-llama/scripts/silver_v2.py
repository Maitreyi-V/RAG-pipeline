#!/usr/bin/env python3
"""
Silver labels, version 2: the 2 x 2 teacher pilot (work plan, Phase 1).

Changes from silver_labels.py (v1, kept for the record):
  - the off-verse definition is quoted word for word from
    docs/annotation_guideline_mention_types.md (decision procedure + T5 rule)
  - OpenAI Structured Outputs with a strict JSON schema: label is VERSE or
    DIGRESSION, confidence 1-5 for that label; score = +confidence for
    DIGRESSION, -confidence for VERSE (consistent with the label by construction)
  - two context modes:
      window   the target passage with up to 2 neighbours each side (one call per passage)
      lecture  the whole lecture in one call, passages numbered P001..; the answer
               must contain every passage exactly once, otherwise the call is
               repeated (max 3 attempts) and the lecture is skipped if it still fails

Configurations of the pilot (tuning lectures only):
  A window + minimal   B window + low   C lecture + minimal   D lecture + low

Answers: results/silver/v2_<mode>_<effort>_<model>.jsonl (one record per
passage; raw text of the call is kept). Saved passages are never re-queried.

Usage:
    python scripts/silver_v2.py run  --mode window  --effort minimal
    python scripts/silver_v2.py eval
"""
import argparse
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

from evalkit import load_gold, read_split, verse_number  # noqa: E402
from silver_labels import lectures_in_order  # noqa: E402

OUT_DIR = os.path.join(HERE, "results", "silver")
CONTEXT = 2

DEFINITION = """The annotators labelled each passage with the verse it grounds, using this decision procedure (quoted from the annotation guideline):

```
1. Is the FULL verse spoken in Sanskrit (ASR errors OK)? ............ YES → T1
2. Is a Sanskrit FRAGMENT of the verse spoken? ..................... YES → T2
3. Is there a faithful translation, framed as the verse? .......... YES → T3
4. Is the verse explained in the speaker's own words,
   with the specific verse still identifiable? .................... YES → T4
5. Only the theme/idea is present, no signal of a specific verse,
   BUT you can still name ONE specific verse it explicates
   (from content or discourse context — see ADDENDUM) ............. YES → T5
6. Otherwise — narration, story, logistics, a digression, or a
   general theme you cannot pin to a specific verse ............... → ∅ No verse here
```

and this rule (quoted from the same guideline):

> **Tag T5 only if you can name ONE specific verse** (or a tight 2–3 verse set) whose content this
> passage is explicating. Your confidence must come from either:
> - **(a) content** — the passage matches that verse's specific content closely, or
> - **(b) discourse context** — the speaker has been working through that verse and is still
>   elaborating it without re-citing.
>
> If the most you can honestly say is *"this is generally about Chapter 2's themes"* without landing
> on a specific verse → **∅ No verse here**, **not** T5.

VERSE means the passage would get T1-T5 for some verse of Bhagavad Gita Chapter 2.
DIGRESSION means step 6: no verse here (including verses from other chapters or texts)."""

WINDOW_PROMPT = """Below are consecutive passages from a recorded lecture on the Bhagavad Gita, Chapter 2 (transcribed speech, Kannada or English). Label ONLY the passage marked TARGET; the others are context.

{definition}

Give the label for the TARGET passage and your confidence in that label from 1 (unsure) to 5 (certain).

{passages}"""

LECTURE_PROMPT = """Below is a whole recorded lecture on the Bhagavad Gita, Chapter 2 (transcribed speech, Kannada or English), split into numbered passages P001 to {last} in the order they were spoken.

{definition}

Label EVERY passage from P001 to {last}, once each, in order. For each, give the label and your confidence in that label from 1 (unsure) to 5 (certain).

{passages}"""

LABEL = {"type": "string", "enum": ["VERSE", "DIGRESSION"]}
CONF = {"type": "integer", "enum": [1, 2, 3, 4, 5]}
WINDOW_SCHEMA = {"type": "object", "additionalProperties": False, "required": ["label", "confidence"],
                 "properties": {"label": LABEL, "confidence": CONF}}
LECTURE_SCHEMA = {"type": "object", "additionalProperties": False, "required": ["labels"], "properties": {
    "labels": {"type": "array", "items": {
        "type": "object", "additionalProperties": False, "required": ["id", "label", "confidence"],
        "properties": {"id": {"type": "string"}, "label": LABEL, "confidence": CONF}}}}}


def fmt(schema, name):
    return {"type": "json_schema", "json_schema": {"name": name, "strict": True, "schema": schema}}


def path_for(mode, effort, model):
    return os.path.join(OUT_DIR, f"v2_{mode}_{effort}_{model}.jsonl")


def load_saved(mode, effort, model):
    p = path_for(mode, effort, model)
    return {d["unit_id"]: d for d in map(json.loads, open(p, encoding="utf-8"))} if os.path.exists(p) else {}


def call(client, model, effort, prompt, schema, name, tries=6):
    """one API call; retries the intermittent 403 / rate limits with backoff"""
    for i in range(tries):
        try:
            return client.chat.completions.create(model=model, reasoning_effort=effort,
                                                  response_format=fmt(schema, name),
                                                  messages=[{"role": "user", "content": prompt}])
        except Exception as e:
            if i == tries - 1:
                raise
            time.sleep(2 * (i + 1))
            last = e
    raise last


def run(mode, effort, model, workers, split="dev", sample=None):
    from dotenv import load_dotenv
    from openai import OpenAI
    load_dotenv(os.path.join(os.path.dirname(HERE), ".env"), override=True)
    client = OpenAI()
    by = lectures_in_order()
    vids = [v for L in ("kn", "en") for s in (["dev", "test", "heldout"] if split == "all" else [split])
            for v in read_split(L, s)]
    saved = load_saved(mode, effort, model)
    os.makedirs(OUT_DIR, exist_ok=True)
    lock = threading.Lock()
    tot = {"in": 0, "out": 0, "calls": 0, "failed": 0, "err": ""}

    def save(recs, r):
        with lock:
            with open(path_for(mode, effort, model), "a", encoding="utf-8") as f:
                for rec in recs:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            tot["in"] += r.usage.prompt_tokens
            tot["out"] += r.usage.completion_tokens
            tot["calls"] += 1

    def fail(e):
        with lock:
            tot["failed"] += 1
            tot["err"] = f"{type(e).__name__}: {str(e)[:120]}"

    if mode == "window":
        todo = [(v, k) for v in vids for k, u in enumerate(by[v]) if u["unit_id"] not in saved]
        if sample:                              # a random subset (fixed seed) of what is left
            import random
            todo = random.Random(2026).sample(todo, min(sample, len(todo)))

        def one(item):
            v, k = item
            us = by[v]
            lo, hi = max(0, k - CONTEXT), min(len(us), k + CONTEXT + 1)
            parts = [f"[{'TARGET' if j == k else ('BEFORE' if j < k else 'AFTER')}]\n{us[j]['text']}"
                     for j in range(lo, hi)]
            prompt = WINDOW_PROMPT.format(definition=DEFINITION, passages="\n\n".join(parts))
            try:
                r = call(client, model, effort, prompt, WINDOW_SCHEMA, "passage_label")
            except Exception as e:
                return fail(e)
            d = json.loads(r.choices[0].message.content)
            save([dict(unit_id=us[k]["unit_id"], video_file=v, model=r.model, mode=mode, effort=effort,
                       label=d["label"], confidence=d["confidence"],
                       score=d["confidence"] if d["label"] == "DIGRESSION" else -d["confidence"],
                       raw=r.choices[0].message.content)], r)
    else:
        todo = [v for v in vids if any(u["unit_id"] not in saved for u in by[v])]

        def one(v):
            us = by[v]
            ids = [f"P{k + 1:03d}" for k in range(len(us))]
            prompt = LECTURE_PROMPT.format(definition=DEFINITION, last=ids[-1], passages="\n\n".join(
                f"{i}: {u['text']}" for i, u in zip(ids, us)))
            for attempt in range(3):
                try:
                    r = call(client, model, effort, prompt, LECTURE_SCHEMA, "lecture_labels")
                except Exception as e:
                    return fail(e)
                labels = json.loads(r.choices[0].message.content)["labels"]
                got = [x["id"] for x in labels]
                if sorted(got) == ids and len(got) == len(set(got)):
                    m = {x["id"]: x for x in labels}
                    save([dict(unit_id=u["unit_id"], video_file=v, model=r.model, mode=mode, effort=effort,
                               label=m[i]["label"], confidence=m[i]["confidence"],
                               score=m[i]["confidence"] if m[i]["label"] == "DIGRESSION" else -m[i]["confidence"],
                               attempt=attempt + 1) for i, u in zip(ids, us)], r)
                    return
                with lock:
                    tot["in"] += r.usage.prompt_tokens
                    tot["out"] += r.usage.completion_tokens
            fail(ValueError(f"{v}: wrong passage ids in 3 attempts"))

    print(f"  {mode}/{effort}/{model}: {len(todo)} {'passage' if mode == 'window' else 'lecture'}(s) to label "
          f"({len(saved)} passages already saved)")
    with ThreadPoolExecutor(workers) as ex:
        list(ex.map(one, todo))
    print(f"  done: {tot['calls']} saved call(s), {tot['in']} input + {tot['out']} output tokens; "
          f"{tot['failed']} failed{'; last error: ' + tot['err'] if tot['failed'] else ''}")


def evaluate(model):
    from sklearn.metrics import roc_auc_score
    lectures, gold, _ = load_gold()
    us = [u for v in read_split("en", "dev") for u in lectures[v]]
    y = [int(not any(verse_number(r) for r in gold[u])) for u in us]
    print(f"  ENGLISH tuning lectures: {len(us)} passages, {sum(y)} gold off-verse "
          f"(Kannada tuning has 1 off-verse passage and is not used for decisions)")
    print(f"  {'config':22} {'n':>4} {'AUC':>6} {'accuracy':>9} {'off recall':>11} {'verse->digr':>12}")
    for name, mode, effort in (("A window + minimal", "window", "minimal"), ("B window + low", "window", "low"),
                               ("C lecture + minimal", "lecture", "minimal"), ("D lecture + low", "lecture", "low")):
        saved = load_saved(mode, effort, model)
        have = [(u, t) for u, t in zip(us, y) if u in saved]
        if not have:
            print(f"  {name:22} not run")
            continue
        yy = [t for _, t in have]
        sc = [saved[u]["score"] for u, _ in have]
        pred = [int(saved[u]["label"] == "DIGRESSION") for u, _ in have]
        auc = roc_auc_score(yy, sc)
        acc = sum(p == t for p, t in zip(pred, yy)) / len(yy)
        rec = sum(p for p, t in zip(pred, yy) if t) / sum(yy)
        fp = sum(p for p, t in zip(pred, yy) if not t)
        print(f"  {name:22} {len(have):>4} {auc:>6.3f} {acc:>9.3f} {rec:>11.3f} "
              f"{fp:>5}/{len(yy) - sum(yy):<6}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["run", "eval"])
    ap.add_argument("--mode", choices=["window", "lecture"])
    ap.add_argument("--effort", choices=["minimal", "low"], default="minimal")
    ap.add_argument("--model", default="gpt-5-mini")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--split", choices=["dev", "all"], default="dev")
    ap.add_argument("--sample", type=int, help="window mode: label a random subset of this size")
    args = ap.parse_args()
    if args.step == "run":
        run(args.mode, args.effort, args.model, args.workers, args.split, args.sample)
    else:
        evaluate(args.model)


if __name__ == "__main__":
    main()
