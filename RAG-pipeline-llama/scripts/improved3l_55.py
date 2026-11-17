#!/usr/bin/env python3
"""
Improved 3-layer system (GitaGround-Cue + an LLM) vs the old 3-layer cascade vs the LLM alone vs Cue alone,
from stored labels only (no API call). LLM = Gemini 3.1 Pro (55 lectures: 9 tuning + 42 test + 4 new
Chinmaya) or GPT-6.1 Sol (51 lectures: no Sol answers for the 4 new ones).

  improved 3L   1 Cue digression veto (English only): Cue labels the passage off-verse with P_off >= VETO
                  -> NONE
                2 the LLM: its whole-lecture label for every other passage
                3 Cue fill: the LLM says NONE and Cue names a verse with confidence >= FILL -> Cue's verse
                (veto is applied after fill, so a vetoed passage is NONE). The string-anchor layer and the
                Cue conf >= 0.9 layer of the old 3L are not used. FILL and VETO were chosen post-hoc, after
                seeing the 42 test lectures with Gemini (17 Nov 2026); the 9 tuning + 4 new lectures and Sol
                were scored after.
  keep rule     per LLM and language, on the 9 tuning lectures only (as 65dbb40): a layer is kept if it is
                right (set equals gold) on at least as many of the tuning passages it decides as the LLM
                alone; veto before fill
  old 3L        Kannada anchor > Cue conf >= 0.9 > LLM; English: cascade4_test.THREE for that LLM
                (Gemini: Cue conf >= 0.9 > Gemini; Sol: Sol alone)
  Cue           tuning: leave-one-lecture-out (as cascade_tuning.load); test: predict_locked v2_teacher145.pkl;
                new 4: the same pkl with those passages' teacher P_off (as new4_gemini3l.py)
  Gemini        labels per passage from --gemini (default results/gemini_runs/gita_data.csv: Gemini 3.1 Pro
                API thinking low, one answer per lecture; per lecture this is the 7 Nov run or the 16 Nov
                re-run, the original answer folders having been deleted)
  Sol           results/llm_frontier_gpt61sol.csv (42 test) + its tuning answers (cascade_frontier_tuning.LLMS)
  report        strict / +-1 / theme F1 (micro); paired lecture bootstrap (10,000) of improved 3L - LLM,
                improved 3L - old 3L and Cue - LLM on groups of 5+ lectures; false verse on digressions;
                which layer decided

-> results/improved3l_55.txt + .csv (Gemini) | results/improved3l_sol51.txt + .csv (Sol)

Usage:
    python scripts/improved3l_55.py [--llm gemini | sol] [--gemini results/gemini_runs/gita_data.csv]
"""
import argparse
import copy
import csv
import os
import pickle
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "scripts"))

import numpy as np  # noqa: E402
from cascade4_test import THREE  # noqa: E402
from cascade4_tuning import L2  # noqa: E402
from cascade_frontier_tuning import LLMS as TUNING_LLMS  # noqa: E402
from cascade_tuning import CONF, ids_of  # noqa: E402
from evalkit import load_gold  # noqa: E402
from freeze_grounding import f1_near, f1_strict, near_counts, paired, strict_counts, theme_counts  # noqa: E402
from gemini_runs_55 import NEW4, TUNE, groups  # noqa: E402
from llm_frontier import parse  # noqa: E402
from new4_gemini3l import PKL, teacher_soft  # noqa: E402
from recall_by_type import read_llm  # noqa: E402
from step4_viterbi import FEATS, OFF, predict_locked, run  # noqa: E402

FILL, VETO = 0.5, 0.95
N_BOOT = 10000
SILVER_LABELS = os.path.join(HERE, "results", "silver", "silver_labels.csv")


def cue_all(lectures):
    """-> cue labels, confidence (decoded state's probability), P_off, anchor verses, for the 55 lectures"""
    z = np.load(FEATS)
    refs = list(z["refs"])
    row = {u: i for i, u in enumerate(z["unit_ids"])}
    cue, conf, poff = {}, {}, {}
    teacher = {r["unit_id"]: min(0.98, max(0.02, float(r["soft_target"])))
               for r in csv.DictReader(open(SILVER_LABELS, encoding="utf-8"))}
    info = {}
    order, _, _, tune = run(alphas=("a05",), verbose=False, resume=True, off_override=teacher, info=info)
    lolo = dict(dict(order)["Viterbi (alpha 0.5)"])
    for v in TUNE["kn"] + TUNE["en"]:                          # as cascade_tuning.load
        pv, po = info["probs_a05"][v]
        for k, u in enumerate(lectures[v]):
            cue[u], poff[u] = lolo[u], float(po[k])
            conf[u] = (1 - po[k]) * pv[k, refs.index(cue[u][0])] / pv[k].sum() if cue[u] else po[k]
    soft = teacher_soft()
    m = pickle.load(open(PKL, "rb"))["a05"]
    m2 = copy.copy(m)
    m2["p_off"] = dict(m["p_off"])
    m2["p_off"].update({u: soft[u] for v in NEW4 for u in lectures[v]})
    rest = [v for _, vs in groups() for v in vs if v not in TUNE["kn"] + TUNE["en"]]
    dec = predict_locked({v: lectures[v] for v in rest}, "a05", model=m2)
    for v in rest:
        for u, (st, extra, p_state, p_off) in zip(lectures[v], dec[v]):
            cue[u] = ([] if st == OFF else [refs[st]]) + [refs[j] for j in extra]   # as freeze_grounding
            conf[u], poff[u] = p_state, float(p_off)
    anchor = {u: [refs[j] for j in np.flatnonzero(z["anchor"][row[u]])] for v in lectures for u in lectures[v] if u in cue}
    return cue, conf, poff, anchor


def llm_labels(which, gemini_csv, lectures, G):
    if which == "gemini":
        with open(gemini_csv, newline="", encoding="utf-8") as f:
            return {r["unit_id"]: [x for x in r["refs"].split(";") if x] for r in csv.DictReader(f)}
    sol = read_llm("llm_frontier_gpt61sol.csv")
    for v in TUNE["kn"] + TUNE["en"]:
        p, _ = parse(os.path.join(dict(TUNING_LLMS)["Sol"], f"{v}.txt"), ids_of(v))
        sol.update({u: p.get(u, []) for u in lectures[v]})
    return sol


def main(which, gemini_csv):
    lectures, gold, _ = load_gold()
    name = {"gemini": "Gemini", "sol": "Sol"}[which]
    G = [(g, vs) for g, vs in groups() if not (which == "sol" and vs == NEW4)]
    allv = [v for _, vs in G for v in vs]
    assert len(allv) == (55 if which == "gemini" else 51), len(allv)
    lang = {v: ("kn" if "Kannada" in g else "en") for g, vs in G for v in vs}
    llm = llm_labels(which, gemini_csv, lectures, G)
    us = [u for v in allv for u in lectures[v]]
    assert all(u in llm for u in us), f"{name} labels missing for some passages"
    llm = {u: llm[u] for u in us}
    cue, conf, poff, anchor = cue_all(lectures)
    L = {u: lang[v] for v in allv for u in lectures[v]}
    ok = lambda p, u: set(p[u]) == set(gold[u])  # noqa: E731

    veto = lambda u: L[u] == "en" and not cue[u] and poff[u] >= VETO  # noqa: E731
    fill = lambda u: not llm[u] and bool(cue[u]) and conf[u] >= FILL  # noqa: E731
    tune = TUNE["kn"] + TUNE["en"]
    kept, keep_lines = {}, []
    for lg in ("kn", "en"):                                        # keep rule, tuning lectures only
        tu = [u for v in tune if lang[v] == lg for u in lectures[v]]
        kept[lg] = []
        for la, fires, out in (("1 Cue veto", veto, lambda u: []), ("3 Cue fill", fill, lambda u: cue[u])):
            acc = [u for u in tu if fires(u)]
            if not acc and la == "1 Cue veto" and lg == "kn":
                keep_lines.append(f"  {lg} {la:12} English only (not used in Kannada)")
                continue
            r_l, r_x = sum(set(out(u)) == set(gold[u]) for u in acc), sum(ok(llm, u) for u in acc)
            keep_lines.append(f"  {lg} {la:12} decides {len(acc):>3} tuning passages: right {r_l} vs {name} {r_x} "
                              f"-> {'KEEP' if r_l >= r_x else 'drop'}")
            if r_l >= r_x:
                kept[lg].append(la)
    old_en = THREE[(name, "en")]
    old, new, layer = {}, {}, {}
    for u in us:
        if L[u] == "kn":
            old[u] = anchor[u] if anchor[u] else cue[u] if conf[u] >= CONF else llm[u]
        else:
            old[u] = cue[u] if L2 in old_en and conf[u] >= CONF else llm[u]
        p, la = llm[u], f"2 {name}"
        if "3 Cue fill" in kept[L[u]] and fill(u):
            p, la = cue[u], "3 Cue fill"
        if "1 Cue veto" in kept[L[u]] and veto(u):
            p, la = [], "1 Cue veto"
        new[u], layer[u] = p, la
    S = {name: llm, "Cue": cue, "old 3L": old, "improved 3L": new}

    def f(fn, cnt, p, vs):
        return fn([sum(x) for x in zip(*(cnt(gold, p, lectures[v]) for v in vs))])

    test = [v for g, vs in G[2:6] for v in vs]
    en = [v for v in allv if lang[v] == "en"]
    sets = G + [("All 9 tuning", tune), ("All 42 test", test), ("All Kannada (38)", [v for v in allv if lang[v] == "kn"]),
                (f"All English ({len(en)})", en), (f"All {len(allv)}", allv)]
    src = os.path.relpath(gemini_csv, HERE) if which == "gemini" else "results/llm_frontier_gpt61sol.csv + tuning answers"
    lines = [f"Improved 3-layer system vs old 3L vs {name} vs Cue, {len(allv)} lectures. {name} labels: {src}",
             f"improved 3L: 1 Cue veto (English, Cue off-verse with P_off >= {VETO}) > 2 {name} > 3 Cue fill "
             f"({name} NONE, Cue conf >= {FILL}); thresholds chosen post-hoc on the 42 test lectures with Gemini",
             f"old 3L: Kannada anchor > Cue conf >= 0.9 > {name}; English "
             + ("Cue conf >= 0.9 > " + name if L2 in old_en else name + " alone") + ". Cue on tuning = leave-one-lecture-out",
             "", "keep rule on the 9 tuning lectures (layers kept for the improved 3L):"] + keep_lines + [
             f"  -> Kannada: {' > '.join(kept['kn'] + [f'2 {name}']) if kept['kn'] else name}; "
             f"English: {' > '.join(kept['en'][:1] + [f'2 {name}'] + kept['en'][1:]) if kept['en'] else name}", "",
             "1. STRICT / +-1 / THEME F1", "",
             f"  {'group':24}{'lect':>5}" + "".join(f"{k:>22}" for k in S)]
    for g, vs in sets:
        lines.append(f"  {g:24}{len(vs):>5}" + "".join(
            f"{f(f1_strict, strict_counts, p, vs):>10.3f}{f(f1_near, near_counts, p, vs):>6.3f}{f(f1_strict, theme_counts, p, vs):>6.3f}"
            for p in S.values()))
    lines += ["", "2. PAIRED, STRICT F1 (diff [95% CI], W/T/L; 10,000 lecture resamples; groups of 5+ lectures)", "",
              f"  {'group':24}{'improved 3L - ' + name:>36}{'improved 3L - old 3L':>36}{'Cue - ' + name:>36}"]
    for g, vs in sets:
        if len(vs) < 5:
            continue
        cells = []
        for a, b in (("improved 3L", name), ("improved 3L", "old 3L"), ("Cue", name)):
            d, ci, _, wtl, _ = paired(gold, lectures, S[a], S[b], vs, N_BOOT)
            cells.append(f"{d:+.3f} [{ci[0]:+.3f},{ci[1]:+.3f}] {wtl[0]}/{wtl[1]}/{wtl[2]}")
        lines.append(f"  {g:24}" + "".join(f"{c:>36}" for c in cells))
    lines += ["", "3. DIGRESSIONS (gold no verse): false verse; VERSE PASSAGES: missed (no verse given)", ""]
    for g, vs in sets:
        dig = [u for v in vs for u in lectures[v] if not gold[u]]
        ver = [u for v in vs for u in lectures[v] if gold[u]]
        lines.append(f"  {g:24} digressions {len(dig):>4}: " + "  ".join(
            f"{k} {sum(bool(p[u]) for u in dig):>3}" for k, p in S.items())
            + f"   | verse psg {len(ver):>4}: missed " + "  ".join(f"{k} {sum(not p[u] for u in ver):>3}" for k, p in S.items()))
    lines += ["", f"4. IMPROVED 3L: WHICH LAYER DECIDED (passages; right = set equals gold; {name} alone right on the same)", ""]
    for g, vs in sets:
        uu = [u for v in vs for u in lectures[v]]
        lines.append(f"  {g:24}" + "   ".join(
            f"{la}: {len(a)} right {sum(ok(new, u) for u in a)} ({name} {sum(ok(llm, u) for u in a)})"
            for la in ("1 Cue veto", f"2 {name}", "3 Cue fill") for a in [[u for u in uu if layer[u] == la]]))
    lines += ["", "5. LECTURE BY LECTURE (strict F1, best marked *)", "",
              f"  {'lecture':26}{'psg':>5}" + "".join(f"{k:>13}" for k in S)]
    for g, vs in G:
        lines.append(f"  -- {g} --")
        for v in vs:
            vals = [f(f1_strict, strict_counts, p, [v]) for p in S.values()]
            lines.append(f"  {v:26}{len(lectures[v]):>5}" + "".join(
                f"{x:>12.3f}{'*' if abs(x - max(vals)) < 1e-9 else ' '}" for x in vals))
    text = "\n".join(lines)
    print(text)
    stem = "improved3l_55" if which == "gemini" else "improved3l_sol51"
    with open(os.path.join(HERE, "results", f"{stem}.txt"), "w", encoding="utf-8") as fh:
        fh.write(text + "\n")
    with open(os.path.join(HERE, "results", f"{stem}.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["unit_id", "video_file", which, "cue", "old_3l", "improved_3l", "improved_layer"])
        for v in allv:
            for u in lectures[v]:
                w.writerow([u, v] + [";".join(S[k][u]) for k in S] + [layer[u]])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--llm", choices=["gemini", "sol"], default="gemini")
    ap.add_argument("--gemini", default=os.path.join(HERE, "results", "gemini_runs", "gita_data.csv"))
    a = ap.parse_args()
    main(a.llm, a.gemini)
