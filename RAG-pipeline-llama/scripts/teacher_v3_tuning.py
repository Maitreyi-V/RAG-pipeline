#!/usr/bin/env python3
"""
Teacher v3, TUNING lectures only: the off-verse teacher is shown the verse and asked a 3-way question.

Changes from silver_v2.py config B (everything else kept: gpt-5-mini, reasoning effort low, the
TARGET passage with +-2 neighbours, the guideline definition, confidence 1-5):
  verse    the verse most recently recited in the lecture (the latest string anchor at or before
           the TARGET; text + roman + translation). If no anchor has fired yet: the three verses the
           verse scorer rates highest for the TARGET (step4_viterbi LOLO Pv, so no gold of the
           lecture itself is used)
  question EXPLAINS the verse / ILLUSTRATES it with a story or example / UNRELATED; the first two
           count as on-verse. score = +confidence for UNRELATED, -confidence otherwise;
           soft_target = clip((score + 5) / 10, 0.02, 0.98), as v2

Answers: results/silver/v3_verse_window_low_gpt-5-mini.jsonl (one record per passage, raw kept;
saved passages are never re-queried).

Score (no test gold): gamma, beta_off, tau re-chosen by leave-one-lecture-out exactly as for v2
(step4_viterbi.run, 145-state resume); Cue confidence and the confident set (conf >= 0.9) as in
cascade_tuning.load; the 3-layer cascade (anchor > Cue conf >= 0.9 > LLM, keep rule of 65dbb40)
redone on tuning for Gemini, Sol and Opus. v2 is recomputed alongside as the baseline.

Targets (written before the run; English tuning, v2 baseline from results/cascade_tuning.txt):
  1 paraphrase (T4) passages in the confident set decided wrong: fewer than 14 (v2: 14 of 16)
  2 digressions in the confident set decided right: at least 94 (v2: 96 of 98), share >= 0.95
  3 decoder strict F1 on English tuning (LOLO): at least 0.731 (v2)
  v3 passes only if all three are met; tuning only, no test lecture is labelled

-> results/teacher_v3_tuning.txt

Usage:
    python scripts/teacher_v3_tuning.py run
    python scripts/teacher_v3_tuning.py score
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
sys.path.insert(0, os.path.join(HERE, "scripts"))

import numpy as np  # noqa: E402
from cascade_tuning import CONF, NO_ANCHOR_LAYER, tier  # noqa: E402
from evalkit import score  # noqa: E402
from silver_labels import lectures_in_order  # noqa: E402
from silver_v2 import CONTEXT, DEFINITION, OUT_DIR, call  # noqa: E402
from step4_viterbi import FEATS, near_f1, run as lolo, theme_f1  # noqa: E402

MODEL, EFFORT = "gpt-5-mini", "low"
OUT = os.path.join(OUT_DIR, f"v3_verse_window_{EFFORT}_{MODEL}.jsonl")
LABELS = ("EXPLAINS", "ILLUSTRATES", "UNRELATED")
SCHEMA = {"type": "object", "additionalProperties": False, "required": ["label", "confidence"],
          "properties": {"label": {"type": "string", "enum": list(LABELS)},
                         "confidence": {"type": "integer", "enum": [1, 2, 3, 4, 5]}}}
assert DEFINITION.count("\n\nVERSE means") == 1
GUIDE = DEFINITION.split("\n\nVERSE means")[0]          # v2's VERSE/DIGRESSION lines replaced below

PROMPT = """Below are consecutive passages from a recorded lecture on the Bhagavad Gita, Chapter 2 (transcribed speech, Kannada or English). Label ONLY the passage marked TARGET; the others are context.

{guide}

{verses}

Choose one label for the TARGET passage:
- EXPLAINS: it recites, translates, explains or paraphrases a verse of Chapter 2 (most likely the verse shown above)
- ILLUSTRATES: it illustrates such a verse with a story, an example or an analogy
- UNRELATED: neither; step 6 of the procedure (narration, logistics, a digression, a verse from another chapter or text, or a general theme you cannot pin to a specific verse)

Give the label for the TARGET passage and your confidence in that label from 1 (unsure) to 5 (certain).

{passages}"""
HEAD_ANCHOR = ("The verse most recently recited in this lecture, at or before the TARGET passage "
               "(detected automatically from the Sanskrit):")
HEAD_CAND = ("No verse has been recited yet in this lecture. The three verses an automatic scorer rates "
             "most likely for the TARGET passage (it may be wrong):")


def verse_text(ref):
    from roman_detector import romanize
    from verse_database import VERSES_BG_CH2
    from verse_translations import VERSE_TRANSLATIONS
    v = next(x for x in VERSES_BG_CH2 if x["ref"] == ref)
    roman = re.sub(r"\s+", " ", romanize(v["devanagari"])).strip()
    return (f"{ref.replace('BG ', '')}\n  Devanagari: {v['devanagari']}\n  Roman: {roman}\n"
            f"  Translation: {VERSE_TRANSLATIONS[ref][0]}")


def teacher(path):
    """unit -> soft_target, from a jsonl (v3) or the v2 silver csv"""
    if path.endswith(".csv"):
        import csv
        return {r["unit_id"]: min(0.98, max(0.02, float(r["soft_target"]))) for r in csv.DictReader(open(path))}
    return {u: min(0.98, max(0.02, (d["score"] + 5) / 10)) for u, d in load_saved_v3().items()}


def load_saved_v3():
    return {d["unit_id"]: d for d in map(json.loads, open(OUT, encoding="utf-8"))} if os.path.exists(OUT) else {}


def cue_of(soft):
    """LOLO Cue on tuning with this P_off -> (gold, lectures, tune, cue, conf, anchor, probs, chosen)"""
    info = {}
    order, gold, lectures, tune = lolo(alphas=("a05",), verbose=False, resume=True, off_override=soft, info=info)
    cue = dict(order)["Viterbi (alpha 0.5)"]
    probs = info["probs_a05"]
    z = np.load(FEATS)
    refs = list(z["refs"])
    row = {u: i for i, u in enumerate(z["unit_ids"])}
    conf, anchor = {}, {}
    for v in tune["kn"] + tune["en"]:
        pv, po = probs[v]
        for k, u in enumerate(lectures[v]):
            conf[u] = (1 - po[k]) * pv[k, refs.index(cue[u][0])] / pv[k].sum() if cue[u] else po[k]
            anchor[u] = [refs[j] for j in np.flatnonzero(z["anchor"][row[u]])]
    return gold, lectures, tune, cue, conf, anchor, probs, sorted({c[1:] for c in info["chosen"]})


def run_(workers):
    from dotenv import load_dotenv
    from openai import OpenAI
    load_dotenv(os.path.join(os.path.dirname(HERE), ".env"), override=True)
    client = OpenAI()
    _, lectures, tune, _, _, anchor, probs, _ = cue_of(teacher(os.path.join(OUT_DIR, "silver_labels.csv")))
    refs = list(np.load(FEATS)["refs"])
    by = lectures_in_order()
    saved = load_saved_v3()
    todo = [(v, k) for v in tune["kn"] + tune["en"] for k, u in enumerate(by[v]) if u["unit_id"] not in saved]
    lock, tot = threading.Lock(), {"in": 0, "out": 0, "calls": 0, "failed": 0, "anchor": 0, "cand": 0}

    def one(item):
        v, k = item
        us = by[v]
        assert [u["unit_id"] for u in us] == lectures[v]
        last = next((anchor[us[j]["unit_id"]] for j in range(k, -1, -1) if anchor[us[j]["unit_id"]]), None)
        if last:
            block, kind = HEAD_ANCHOR + "\n\n" + "\n\n".join(verse_text(r) for r in last), "anchor"
        else:
            top = [refs[j] for j in np.argsort(-probs[v][0][k])[:3]]
            block, kind = HEAD_CAND + "\n\n" + "\n\n".join(verse_text(r) for r in top), "cand"
        lo, hi = max(0, k - CONTEXT), min(len(us), k + CONTEXT + 1)
        parts = [f"[{'TARGET' if j == k else ('BEFORE' if j < k else 'AFTER')}]\n{us[j]['text']}" for j in range(lo, hi)]
        prompt = PROMPT.format(guide=GUIDE, verses=block, passages="\n\n".join(parts))
        try:
            r = call(client, MODEL, EFFORT, prompt, SCHEMA, "passage_label")
        except Exception as e:
            with lock:
                tot["failed"] += 1
                print(f"  {us[k]['unit_id']}: {type(e).__name__}: {str(e)[:120]}", flush=True)
            return
        d = json.loads(r.choices[0].message.content)
        rec = dict(unit_id=us[k]["unit_id"], video_file=v, model=r.model, effort=EFFORT, verse_shown=kind,
                   label=d["label"], confidence=d["confidence"],
                   score=d["confidence"] if d["label"] == "UNRELATED" else -d["confidence"],
                   raw=r.choices[0].message.content)
        with lock:
            with open(OUT, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            tot["in"] += r.usage.prompt_tokens
            tot["out"] += r.usage.completion_tokens
            tot["calls"] += 1
            tot[kind] += 1

    print(f"  {len(todo)} passage(s) to label ({len(saved)} already saved)", flush=True)
    with ThreadPoolExecutor(workers) as ex:
        list(ex.map(one, todo))
    cost = tot["in"] / 1e6 * 0.25 + tot["out"] / 1e6 * 2.0
    print(f"  done: {tot['calls']} saved ({tot['anchor']} shown the last anchor, {tot['cand']} the top 3), "
          f"{tot['failed']} failed; {tot['in']} in + {tot['out']} out tokens = ${cost:.3f}")


def score_():
    from sklearn.metrics import roc_auc_score
    from cascade4_tuning import labels
    from cascade_frontier_tuning import LLMS
    v2 = teacher(os.path.join(OUT_DIR, "silver_labels.csv"))
    v3 = teacher(OUT)
    saved = load_saved_v3()
    res = {name: cue_of(t) for name, t in (("v2", v2), ("v3", v3))}
    gold, lectures, tune = res["v2"][:3]
    us_all = [u for L in ("kn", "en") for v in tune[L] for u in lectures[v]]
    missing = [u for u in us_all if u not in saved]
    lines = [f"Teacher v3 (verse shown, EXPLAINS / ILLUSTRATES / UNRELATED), TUNING lectures only; "
             f"{len(us_all) - len(missing)}/{len(us_all)} passages answered"
             + (f"; MISSING {len(missing)} (v2 value used)" if missing else "")]
    for u in missing:
        v3[u] = v2[u]
    en = [u for v in tune["en"] for u in lectures[v]]
    y = [int(not gold[u]) for u in en]
    lines += ["", "1. TEACHER vs gold, English tuning (off-verse = positive)",
              f"  {'':4}{'AUC':>7}{'accuracy':>10}{'off recall':>12}{'verse->off':>12}"]
    for name, t in (("v2", v2), ("v3", v3)):
        pred = [int(t[u] > 0.5) for u in en]
        lines.append(f"  {name:4}{roc_auc_score(y, [t[u] for u in en]):>7.3f}"
                     f"{sum(p == g for p, g in zip(pred, y)) / len(y):>10.3f}"
                     f"{sum(p for p, g in zip(pred, y) if g) / sum(y):>12.3f}"
                     f"{sum(p for p, g in zip(pred, y) if not g):>6}/{len(y) - sum(y)}")
    lab = {}
    for u in en:
        if u in saved:
            lab.setdefault(saved[u]["label"], []).append(u)
    lines.append("  v3 labels on English tuning (gold off / gold verse): " + ", ".join(
        f"{k} {sum(not gold[u] for u in lab.get(k, []))}/{sum(bool(gold[u]) for u in lab.get(k, []))}" for k in LABELS))
    lines.append(f"  verse shown: last anchor {sum(d['verse_shown'] == 'anchor' for d in saved.values())}, "
                 f"scorer top 3 {sum(d['verse_shown'] == 'cand' for d in saved.values())}")

    lines += ["", "2. DECODER, leave-one-lecture-out on tuning (145-state resume, alpha 0.5)",
              f"  {'':4}{'KN strict':>10}{'EN strict':>10}{'EN +-1':>8}{'EN theme':>9}   settings chosen (gamma, beta_off, delta, tau)"]
    for name, (g, lec, tn, cue, *_r, chosen) in res.items():
        lines.append(f"  {name:4}{score(g, cue, lec, tn['kn'])['F1']:>10.3f}{score(g, cue, lec, tn['en'])['F1']:>10.3f}"
                     f"{near_f1(g, cue, lec, tn['en'])[2]:>8.3f}{theme_f1(g, cue, lec, tn['en']):>9.3f}   {chosen}")

    lines += ["", f"3. CONFIDENT SET, English tuning (Cue conf >= {CONF}): accepted / right by passage type"]
    conf_set = {}
    for name, (g, lec, tn, cue, conf, *_r) in res.items():
        acc = [u for u in en if conf[u] >= CONF]
        conf_set[name] = acc
        by = {}
        for u in acc:
            by.setdefault(tier(g, u), []).append(set(cue[u]) == set(g[u]))
        lines.append(f"  {name}: {len(acc)} accepted, right {sum(set(cue[u]) == set(g[u]) for u in acc)}; " + ", ".join(
            f"{t} {len(by[t])}/{sum(by[t])}" for t in ("T1", "T2", "T3", "T4", "T5", "OFF") if t in by))

    lines += ["", "4. 3-LAYER CASCADE on tuning (anchor > Cue conf >= 0.9 > LLM; keep rule of 65dbb40, per language)"]
    summary = []
    for llm_name, folder in LLMS:
        llm, _ = labels(folder, tune, lectures)
        for name, (g, lec, tn, cue, conf, anchor, *_r) in res.items():
            take = {"1 anchor": (lambda u: bool(anchor[u]), anchor), "2 Cue conf": (lambda u: conf[u] >= CONF, cue)}
            for L in ("kn", "en"):
                us = [u for v in tn[L] for u in lec[v]]
                left, out, notes = list(us), {u: llm[u] for u in us}, []
                for la, (ok_, src) in take.items():
                    if la == "1 anchor" and L in NO_ANCHOR_LAYER:
                        continue
                    a = [u for u in left if ok_(u)]
                    r_l, r_m = sum(set(src[u]) == set(g[u]) for u in a), sum(set(llm[u]) == set(g[u]) for u in a)
                    keep = r_l >= r_m
                    notes.append(f"{la} {len(a)}: {r_l} vs {r_m} {'KEEP' if keep else 'drop'}")
                    if keep:
                        out.update({u: src[u] for u in a})
                        left = [u for u in left if u not in a]
                fa, fc = score(g, llm, lec, tn[L])["F1"], score(g, out, lec, tn[L])["F1"]
                lines.append(f"  {llm_name:7} {L} Cue {name}: alone {fa:.3f}  3-layer {fc:.3f} ({fc - fa:+.3f})   "
                             + "; ".join(notes))
                summary.append((llm_name, L, name, fc - fa))

    f_en = {n: score(r[0], r[3], r[1], r[2]["en"])["F1"] for n, r in res.items()}
    t4 = {n: sum(1 for u in conf_set[n] if tier(gold, u) == "T4" and set(res[n][3][u]) != set(gold[u])) for n in res}
    n4 = {n: sum(1 for u in conf_set[n] if tier(gold, u) == "T4") for n in res}
    off = {n: [set(res[n][3][u]) == set(gold[u]) for u in conf_set[n] if tier(gold, u) == "OFF"] for n in res}
    share = sum(off["v3"]) / len(off["v3"]) if off["v3"] else 0
    lines += ["", "TARGETS (English tuning, written before the run)",
              f"  1 paraphrase wrong in confident set < 14:  v2 {t4['v2']} of {n4['v2']}, v3 {t4['v3']} of {n4['v3']} -> "
              f"{'MET' if t4['v3'] < 14 else 'NOT MET'}",
              f"  2 digressions right >= 94 and share >= 0.95:  v2 {sum(off['v2'])} of {len(off['v2'])}, "
              f"v3 {sum(off['v3'])} of {len(off['v3'])} ({share:.2f}) -> "
              f"{'MET' if sum(off['v3']) >= 94 and share >= 0.95 else 'NOT MET'}",
              f"  3 decoder EN strict >= 0.731:  v2 {f_en['v2']:.3f}, v3 {f_en['v3']:.3f} -> "
              f"{'MET' if f_en['v3'] >= 0.731 else 'NOT MET'}"]
    text = "\n".join(lines)
    print(text)
    with open(os.path.join(HERE, "results", "teacher_v3_tuning.txt"), "w", encoding="utf-8") as f:
        f.write(text + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["run", "score"])
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    run_(a.workers) if a.step == "run" else score_()


if __name__ == "__main__":
    main()
