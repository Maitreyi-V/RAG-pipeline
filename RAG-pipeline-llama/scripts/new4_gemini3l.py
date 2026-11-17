#!/usr/bin/env python3
"""
Gemini 3.1 Pro alone vs GitaGround-Cue alone vs the 3-layer cascade with Gemini, on the 4 newly annotated
English lectures (en_chinmaya_day06-09, Swami Aparajitananda, the E3 Chinmaya speaker) and pooled with the
3 earlier E3 English lectures (en_chinmaya_day03, day04, en_iskcon). Nothing is re-tuned.

  prepare   paste files for the 4 lectures, built as llm_frontier.prepare (same instructions, catalogue and
            passage format), into review/llm_frontier/new4/; first rebuilds en_chinmaya_day03 the same way
            and stops unless it equals the saved 40_en_chinmaya_day03_PASTE_THIS.txt byte for byte
  send      llm_frontier_gemini_api.send on those 4 files (gemini-3.1-pro-preview, thinking low, as the 42),
            answers into review/llm_frontier/answers_api_gemini-3.1-pro-preview_low/
  score     Cue: v2_teacher145.pkl as frozen; the new passages' P_off is the same gpt-5-mini teacher (config B,
            results/silver/v2_window_low_gpt-5-mini.jsonl), soft_target = clip((score+5)/10, 0.02, 0.98)
            as results/silver/silver_labels.csv (checked on its 1,884 rows); the frozen pkl is not changed.
            3-layer: cascade4_test.THREE for Gemini, English = Cue conf >= 0.9 > Gemini.
            Gemini (3 earlier lectures): results/llm_frontier_gemini31pro_api_low.csv.
            Cost: Gemini from its usage files ($2 / $12 per 1M); Cue teacher on the 4 new lectures from the
            logged tokens of that run (372,167 in / 76,195 out over 299 passages, $0.25 / $2 per 1M), on the
            3 earlier lectures as all51_gemini3l.py (shared out by passage count).
            Paired lecture bootstrap (10,000) and the claim table (margin -0.02) on groups of 5+ lectures.

-> results/new4_gemini3l.txt

Usage:
    python scripts/new4_gemini3l.py prepare
    python scripts/new4_gemini3l.py send
    python scripts/new4_gemini3l.py score
"""
import argparse
import copy
import csv
import json
import os
import pickle
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "scripts"))

import numpy as np  # noqa: E402
from all51_gemini3l import GEMINI_TEST, TEACHER_PASSAGES, TEACHER_TOKENS, gemini_cost  # noqa: E402
from cascade4_test import THREE  # noqa: E402
from cascade4_tuning import L2, L4  # noqa: E402
from cascade_tuning import CONF  # noqa: E402
from evalkit import UNITS, UNITS_HELDOUT, load_csv, load_gold  # noqa: E402
from freeze_grounding import f1_near, f1_strict, near_counts, paired, strict_counts  # noqa: E402
from llm_frontier import DIR, INSTRUCTIONS, parse  # noqa: E402
from llm_pilot import catalogue  # noqa: E402
from recall_by_type import read_llm  # noqa: E402
from step4_viterbi import FEATS, OFF, predict_locked  # noqa: E402

NEW = ["en_chinmaya_day06", "en_chinmaya_day07", "en_chinmaya_day08", "en_chinmaya_day09"]
OLD = ["en_chinmaya_day03", "en_chinmaya_day04", "en_iskcon"]
NEW_DIR = os.path.join(DIR, "new4")
PKL = os.path.join(HERE, "results", "v2_teacher145.pkl")
SILVER = os.path.join(HERE, "results", "silver")
NEW_TOKENS = (372167, 76195)
N_BOOT = 10000
MARGIN = -0.02


def paste_text(v, lectures, units, cat):
    us = lectures[v]
    ids = [f"P{k + 1:03d}" for k in range(len(us))]
    lang = units[us[0]]["language"].upper()
    body = "\n\n".join(f"{i}: {units[u]['text']}" for i, u in zip(ids, us))
    return (INSTRUCTIONS + "\n" + cat + f"\n\nLECTURE: {v}, {len(us)} passages, language {lang}\n\n" + body + "\n",
            list(zip(ids, us)))


def prepare():
    lectures, _, _ = load_gold()
    units = {u["unit_id"]: u for u in load_csv(UNITS) + load_csv(UNITS_HELDOUT)}
    cat = catalogue()
    text, _ = paste_text("en_chinmaya_day03", lectures, units, cat)
    if text != open(os.path.join(DIR, "40_en_chinmaya_day03_PASTE_THIS.txt"), encoding="utf-8").read():
        sys.exit("rebuilt en_chinmaya_day03 paste file differs from the saved one; nothing written")
    os.makedirs(NEW_DIR, exist_ok=True)
    lines = []
    for n, v in enumerate(NEW, start=43):
        text, ids = paste_text(v, lectures, units, cat)
        with open(os.path.join(NEW_DIR, f"{n}_{v}_PASTE_THIS.txt"), "w", encoding="utf-8") as f:
            f.write(text)
        with open(os.path.join(NEW_DIR, f"{v}_ids.csv"), "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerows([("passage_id", "unit_id")] + ids)
        lines.append(f"{n}\t{v}\tEN\t{len(ids)} passages")
    with open(os.path.join(NEW_DIR, "ORDER.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"  day03 rebuilt identical; wrote {len(NEW)} paste files to {NEW_DIR}")


def send():
    import llm_frontier_gemini_api as g
    files = [(v, os.path.join(NEW_DIR, f"{n}_{v}_PASTE_THIS.txt")) for n, v in enumerate(NEW, start=43)]
    assert all(os.path.exists(p) for _, p in files), "run prepare first"
    g.paste_files = lambda split=None: files              # send() otherwise reads the 42 of ORDER.txt
    g.send("gemini-3.1-pro-preview", "low", "heldout")


def teacher_soft():
    """{unit_id: soft_target} from the v2 config B answers, checked against silver_labels.csv"""
    soft = {}
    for line in open(os.path.join(SILVER, "v2_window_low_gpt-5-mini.jsonl"), encoding="utf-8"):
        r = json.loads(line)
        soft.setdefault(r["unit_id"], min(0.98, max(0.02, (r["score"] + 5) / 10)))
    saved = {r["unit_id"]: float(r["soft_target"]) for r in load_csv(os.path.join(SILVER, "silver_labels.csv"))}
    assert all(abs(soft[u] - p) < 1e-9 for u, p in saved.items()), "formula does not reproduce silver_labels.csv"
    return soft


def claim(d, ci):
    if ci[0] > 0:
        return "better"
    if ci[1] < 0:
        return "worse"
    if ci[0] > MARGIN:
        return "matches (within 0.02)"
    return f"no significant difference; a loss up to {-ci[0]:.3f} cannot be ruled out"


def score():
    lectures, gold, _ = load_gold()
    vs_all = OLD + NEW
    soft = teacher_soft()
    m = pickle.load(open(PKL, "rb"))["a05"]
    m2 = copy.copy(m)
    m2["p_off"] = dict(m["p_off"])
    m2["p_off"].update({u: soft[u] for v in NEW for u in lectures[v]})
    dec = predict_locked({v: lectures[v] for v in vs_all}, "a05", model=m2)
    frozen = predict_locked({v: lectures[v] for v in OLD}, "a05", path=PKL)
    assert all([(s, e, p) for s, e, p, _ in dec[v]] == [(s, e, p) for s, e, p, _ in frozen[v]] for v in OLD)
    refs = list(np.load(FEATS)["refs"])
    cue, conf = {}, {}
    for v in vs_all:
        for u, (st, extra, p_state, _) in zip(lectures[v], dec[v]):
            cue[u], conf[u] = ([] if st == OFF else [refs[st]]) + [refs[j] for j in extra], p_state
    gem = read_llm("llm_frontier_gemini31pro_api_low.csv")
    gem = {u: gem[u] for v in OLD for u in lectures[v]}
    problems = {}
    for v in NEW:
        ids = [(r["passage_id"], r["unit_id"]) for r in load_csv(os.path.join(NEW_DIR, f"{v}_ids.csv"))]
        p, problems[v] = parse(os.path.join(GEMINI_TEST, f"{v}.txt"), ids)
        gem.update({u: p.get(u, []) for u in lectures[v]})       # missing passage = NONE, as the 42
    design = THREE[("Gemini", "en")]
    assert design == (L2,), design
    three, layer = {}, {}
    for v in vs_all:
        for u in lectures[v]:
            la = L2 if conf[u] >= CONF else L4
            layer[u], three[u] = la, (cue[u] if la == L2 else gem[u])
    S = {"Gemini": gem, "Cue": cue, "3L Gemini": three}
    cols = list(S)

    old_psg = (TEACHER_TOKENS[0] / 1e6 * 0.25 + TEACHER_TOKENS[1] / 1e6 * 2) / TEACHER_PASSAGES
    new_n = sum(len(lectures[v]) for v in NEW)
    new_psg = (NEW_TOKENS[0] / 1e6 * 0.25 + NEW_TOKENS[1] / 1e6 * 2) / new_n
    cost = {}
    for v in vs_all:
        g = gemini_cost(os.path.join(GEMINI_TEST, f"{v}_usage.json"))
        c = (new_psg if v in NEW else old_psg) * len(lectures[v])
        cost[v] = {"Gemini": g, "Cue": c, "3L Gemini": g + c}

    def pooled(f, counts, p, vs):
        return f([sum(x) for x in zip(*(counts(gold, p, lectures[v]) for v in vs))])
    f1 = lambda p, vs: pooled(f1_strict, strict_counts, p, vs)  # noqa: E731
    near = lambda p, vs: pooled(f1_near, near_counts, p, vs)  # noqa: E731
    dig = lambda vs: [u for v in vs for u in lectures[v] if not gold[u]]  # noqa: E731
    ver = lambda vs: [u for v in vs for u in lectures[v] if gold[u]]  # noqa: E731
    fv = lambda p, us: sum(bool(p.get(u)) for u in us)  # noqa: E731
    mv = lambda p, us: sum(not p.get(u) for u in us)  # noqa: E731
    ok = lambda p, u: set(p.get(u, ())) == set(gold[u])  # noqa: E731

    lines = ["Gemini 3.1 Pro (API, low) alone vs GitaGround-Cue alone vs 3-layer cascade with Gemini "
             "(English: Cue conf >= 0.9 > Gemini), nothing re-tuned",
             "4 new English lectures (en_chinmaya_day06-09, same speaker as en_chinmaya_day03/04) + the 3 earlier "
             "E3 English lectures", "",
             "Gemini answer check (new 4): " + "; ".join(
                 f"{v} missing {len(p['missing'])} unparsed {len(p['unparsed'])} outside Ch2 {p['outside_ch2']}"
                 for v, p in problems.items()), "",
             "1. LECTURE BY LECTURE  (strict F1, best marked *; dig = gold no-verse passages; FV = false verse on "
             "them; $ per lecture)", "",
             f"  {'lecture':20}{'psg':>4}" + "".join(f"{c:>11}" for c in cols) + f"{'dig':>5}"
             + "".join(f"{'FV ' + c:>14}" for c in cols) + "".join(f"{'$ ' + c:>13}" for c in cols)]
    for g, vs in (("earlier E3 English", OLD), ("NEW", NEW)):
        lines.append(f"  -- {g} --")
        for v in vs:
            vals = [f1(S[c], [v]) for c in cols]
            d = dig([v])
            lines.append(f"  {v:20}{len(lectures[v]):>4}" + "".join(
                f"{x:>10.3f}{'*' if abs(x - max(vals)) < 1e-9 else ' '}" for x in vals) + f"{len(d):>5}"
                + "".join(f"{fv(S[c], d):>14}" for c in cols) + "".join(f"{cost[v][c]:>13.3f}" for c in cols))

    sets = [("New 4", NEW), ("Earlier 3 E3 EN", OLD), ("All 7 E3 English", vs_all)]
    lines += ["", "2. OVERVIEW", "", "  2a. strict F1, +-1 F1; paired vs Gemini alone: diff [95% CI] W/T/L "
              "(CI and claim only for 5+ lectures)", ""]
    for g, vs in sets:
        lines.append(f"  {g:17}{len(vs):>2} lect  strict " + "  ".join(f"{c} {f1(S[c], vs):.3f}" for c in cols)
                     + "   +-1 " + "  ".join(f"{c} {near(S[c], vs):.3f}" for c in cols))
        for c in ("Cue", "3L Gemini"):
            d, ci, _, wtl, _ = paired(gold, lectures, S[c], gem, vs, N_BOOT)
            if len(vs) >= 5:
                lines.append(f"      {c} - Gemini {d:+.3f} [{ci[0]:+.3f},{ci[1]:+.3f}]  W/T/L "
                             f"{wtl[0]}/{wtl[1]}/{wtl[2]}  -> {claim(d, ci)}")
            else:
                lines.append(f"      {c} - Gemini {d:+.3f}  W/T/L {wtl[0]}/{wtl[1]}/{wtl[2]}  (no CI: <5 lectures)")
    lines += ["", "  2b. digressions (gold: no verse) and verse passages left with no verse", ""]
    for g, vs in sets:
        d, w = dig(vs), ver(vs)
        lines.append(f"  {g:17} digressions {len(d):>3}: false verse " + "  ".join(
            f"{c} {fv(S[c], d):>3} ({fv(S[c], d) / len(d):.0%})" for c in cols)
            + f"   | verse passages {len(w):>3}: missed " + "  ".join(f"{c} {mv(S[c], w):>3}" for c in cols))
    lines += ["", "  2c. cost (USD)", ""]
    for g, vs in sets:
        lines.append(f"  {g:17}" + "  ".join(f"{c} ${sum(cost[v][c] for v in vs):.3f}" for c in cols))
    lines += ["", "  2d. 3-layer: who decided (passages; right = set equals gold; Gemini alone right on the same)", ""]
    for g, vs in sets:
        us = [u for v in vs for u in lectures[v]]
        lines.append(f"  {g:17}" + "   ".join(
            f"{la}: {len(a)} right {sum(ok(three, u) for u in a)} (Gemini {sum(ok(gem, u) for u in a)})"
            for la in (L2, L4) for a in [[u for u in us if layer[u] == la]] if a))
    text = "\n".join(lines)
    print(text)
    with open(os.path.join(HERE, "results", "new4_gemini3l.txt"), "w", encoding="utf-8") as f:
        f.write(text + "\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["prepare", "send", "score"])
    {"prepare": prepare, "send": send, "score": score}[ap.parse_args().step]()
