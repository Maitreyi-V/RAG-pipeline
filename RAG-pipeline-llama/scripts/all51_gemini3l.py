#!/usr/bin/env python3
"""
Gemini 3.1 Pro alone vs GitaGround-Cue alone vs the 3-layer cascade with Gemini (anchor > Cue conf >= 0.9
> Gemini; no gpt-5-mini layer), on the 9 tuning + 42 test lectures, from stored labels only (no API call):
strict F1, digressions, cost; lecture by lecture and overview.

  labels    Gemini: tuning answers (cascade_frontier_tuning.LLMS) and results/llm_frontier_gemini31pro_api_low.csv;
            Cue: tuning leave-one-lecture-out (cascade_tuning.load), test predict_locked v2_teacher145.pkl;
            3-layer: cascade4_test.THREE for Gemini (Kannada: anchor > Cue conf; English: Cue conf),
            chosen on tuning, so in sample on the tuning lectures
  digression  gold has no verse; "false verse" = the system names a verse there; also verse passages
            the system leaves with no verse ("missed verse")
  cost      Gemini: its usage files, $2 / 1M input, $12 / 1M output + thinking tokens;
            Cue: the gpt-5-mini teacher (config B), logged total 2,729,615 in / 423,229 out tokens over
            the 1,884 passages of these 51 lectures at $0.25 / $2 per 1M (= $1.529); per lecture it is
            shared out by passage count (an estimate; the per-call counts were not saved);
            3-layer = Gemini whole-lecture call + Cue (the override cascade needs both)
  tests     paired lecture bootstrap of strict F1 (10,000 resamples), W/T/L, groups of 5+ lectures

  --structure local   GitaGround-Local (step4_locked.pkl: old off-verse scorer, no API) in place of Cue,
            so no gpt-5-mini teacher: tuning LOLO = step4_viterbi.run defaults (as all51_table.py), test
            predict_locked step4_locked.pkl; confidence as Cue's; the 3-layer layers are re-chosen on
            the 9 tuning lectures by the keep rule of 65dbb40 (for Cue the same rule gives THREE, checked)

-> results/all51_gemini3l.txt (--structure local: all51_gemini3l_local.txt)

Usage:
    python scripts/all51_gemini3l.py [--structure cue | local]
"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "scripts"))

import numpy as np  # noqa: E402
from cascade4_test import THREE  # noqa: E402
from cascade4_tuning import L1, L2, L4  # noqa: E402
from cascade_frontier_tuning import LLMS as TUNING_LLMS  # noqa: E402
from cascade_tuning import CONF, NO_ANCHOR_LAYER, ids_of, load  # noqa: E402
from evalkit import read_split  # noqa: E402
from freeze_grounding import f1_strict, paired, strict_counts  # noqa: E402
from llm_frontier import DIR, parse  # noqa: E402
from recall_by_type import read_llm  # noqa: E402
from step4_viterbi import FEATS, OFF, OUT, predict_locked, run  # noqa: E402

N_BOOT = 10000
GEMINI_TEST = os.path.join(DIR, "answers_api_gemini-3.1-pro-preview_low")
TEACHER_TOKENS, TEACHER_PASSAGES = (2729615, 423229), 1884


def gemini_cost(path):
    u = json.load(open(path))["usage"]
    return u["promptTokenCount"] / 1e6 * 2 + (u.get("candidatesTokenCount", 0) + u.get("thoughtsTokenCount", 0)) / 1e6 * 12


def main(structure="cue"):
    gold, lectures, tune, cue, _, conf, anchor = load()
    name = "Cue" if structure == "cue" else "Local"
    pkl = os.path.join(HERE, "results", "v2_teacher145.pkl") if structure == "cue" else OUT
    if structure == "local":                                    # tuning: Local LOLO, confidence as Cue's
        info = {}
        order, *_ = run(alphas=("a05",), verbose=False, info=info)
        cue = dict(dict(order)["Viterbi (alpha 0.5)"])
        refs0 = list(np.load(FEATS)["refs"])
        for v in tune["kn"] + tune["en"]:
            pv, po = info["probs_a05"][v]
            for k, u in enumerate(lectures[v]):
                conf[u] = (1 - po[k]) * pv[k, refs0.index(cue[u][0])] / pv[k].sum() if cue[u] else po[k]
    groups = [("Tuning Kannada", "kn", tune["kn"]), ("Tuning English", "en", tune["en"]),
              ("E2 Kannada test", "kn", read_split("kn", "test")), ("E2 English test", "en", read_split("en", "test")),
              ("E3 Kannada", "kn", read_split("kn", "heldout")), ("E3 English", "en", read_split("en", "heldout"))]
    test = [v for _, _, vs in groups[2:] for v in vs]
    lang = {v: L for _, L, vs in groups for v in vs}
    z = np.load(FEATS)
    refs = list(z["refs"])
    row = {u: i for i, u in enumerate(z["unit_ids"])}
    dec = predict_locked({v: lectures[v] for v in test}, "a05", path=pkl)
    for v in test:
        for u, (st, extra, p_state, _) in zip(lectures[v], dec[v]):
            cue[u], conf[u] = ([] if st == OFF else [refs[st]]) + [refs[j] for j in extra], p_state   # as freeze_grounding
            anchor[u] = [refs[j] for j in np.flatnonzero(z["anchor"][row[u]])]
    gem = read_llm("llm_frontier_gemini31pro_api_low.csv")
    folder = dict(TUNING_LLMS)["Gemini"]
    for v in tune["kn"] + tune["en"]:
        p, _ = parse(os.path.join(folder, f"{v}.txt"), ids_of(v))
        gem.update({u: p.get(u, []) for u in lectures[v]})
    take = {L1: (lambda u: bool(anchor[u]), anchor), L2: (lambda u: conf[u] >= CONF, cue)}
    design, keep_lines = {}, []
    for L in ("kn", "en"):                                      # keep rule of 65dbb40, tuning only
        left, kept = [u for v in tune[L] for u in lectures[v]], []
        for la in (L1, L2):
            if la == L1 and L in NO_ANCHOR_LAYER:
                continue
            acc = [u for u in left if take[la][0](u)]
            r_s = sum(set(take[la][1][u]) == set(gold[u]) for u in acc)
            r_g = sum(set(gem[u]) == set(gold[u]) for u in acc)
            keep_lines.append(f"  {L} {la:16} accepts {len(acc):>3} tuning passages: right {r_s} vs Gemini {r_g} "
                              f"-> {'KEEP' if r_s >= r_g else 'drop'}")
            if r_s >= r_g:
                kept.append(la)
                left = [u for u in left if not take[la][0](u)]
        design[L] = tuple(kept)
    if structure == "cue":
        assert design == {L: THREE[("Gemini", L)] for L in ("kn", "en")}, design
    three, layer = {}, {}
    for v, L in lang.items():
        for u in lectures[v]:
            la = next((x for x in design[L] if take[x][0](u)), L4)
            layer[u], three[u] = la, (take[la][1][u] if la != L4 else gem.get(u, []))
    S = {"Gemini": gem, name: cue, "3L Gemini": three}

    per_psg = (TEACHER_TOKENS[0] / 1e6 * 0.25 + TEACHER_TOKENS[1] / 1e6 * 2) / TEACHER_PASSAGES if name == "Cue" else 0
    assert sum(len(lectures[v]) for v in lang) == TEACHER_PASSAGES
    cost = {}
    for v in lang:
        g = gemini_cost(os.path.join(folder if v in tune[lang[v]] else GEMINI_TEST, f"{v}_usage.json"))
        c = per_psg * len(lectures[v])
        cost[v] = {"Gemini": g, name: c, "3L Gemini": g + c}

    f1 = lambda p, vs: f1_strict([sum(x) for x in zip(*(strict_counts(gold, p, lectures[v]) for v in vs))])  # noqa: E731
    dig = lambda vs: [u for v in vs for u in lectures[v] if not gold[u]]  # noqa: E731
    ver = lambda vs: [u for v in vs for u in lectures[v] if gold[u]]  # noqa: E731
    fv = lambda p, us: sum(bool(p.get(u)) for u in us)  # noqa: E731     false verse on a digression
    mv = lambda p, us: sum(not p.get(u) for u in us)  # noqa: E731       verse passage left with no verse
    cols = list(S)

    lines = [f"Gemini 3.1 Pro alone vs GitaGround-{name} alone vs 3-layer cascade with Gemini (anchor > {name} conf >= 0.9 "
             "> Gemini), 9 tuning + 42 test lectures",
             f"tuning: {name} leave-one-lecture-out; 3-layer in sample (its layers were chosen on tuning). "
             + ("Cue cost is an estimate (teacher tokens shared out by passage count)" if name == "Cue" else
                "Local uses no API (cost 0); no gpt-5-mini teacher anywhere"), ""]
    if name != "Cue":
        lines += ["keep rule on the 9 tuning lectures (layers chosen for Local):"] + keep_lines + [""]
    lines += [
             "1. LECTURE BY LECTURE  (strict F1, best marked *; dig = gold digression passages; "
             "false verse on digressions; $ per lecture)", "",
             f"  {'lecture':24}{'psg':>4}" + "".join(f"{c:>11}" for c in cols)
             + f"{'dig':>6}" + "".join(f"{'FV ' + c:>14}" for c in cols) + "".join(f"{'$ ' + c:>13}" for c in cols)]
    for g, _, vs in groups:
        lines.append(f"  -- {g} --")
        for v in vs:
            vals = [f1(S[c], [v]) for c in cols]
            d = dig([v])
            lines.append(f"  {v:24}{len(lectures[v]):>4}" + "".join(
                f"{x:>10.3f}{'*' if abs(x - max(vals)) < 1e-9 else ' '}" for x in vals)
                + f"{len(d):>6}" + "".join(f"{fv(S[c], d):>14}" for c in cols)
                + "".join(f"{cost[v][c]:>13.3f}" for c in cols))

    sets = groups + [("All 9 tuning", None, [v for _, _, vs in groups[:2] for v in vs]),
                     ("All 42 test", None, test), ("All 51", None, list(lang))]
    lines += ["", "2. OVERVIEW", "", "  2a. strict F1 (paired vs Gemini alone: diff [95% CI], W/T/L; 5+ lectures)", ""]
    for g, _, vs in sets:
        out = f"  {g:17}{len(vs):>4} lect " + "  ".join(f"{c} {f1(S[c], vs):.3f}" for c in cols)
        if len(vs) >= 5:
            for c in (name, "3L Gemini"):
                d, ci, _, wtl, _ = paired(gold, lectures, S[c], gem, vs, N_BOOT)
                out += f"   | {c} - Gemini {d:+.3f} [{ci[0]:+.3f},{ci[1]:+.3f}] {wtl[0]}/{wtl[1]}/{wtl[2]}"
        lines.append(out)
    lines += ["", "  2b. digressions (gold: no verse) and verse passages left with no verse", ""]
    for g, _, vs in sets:
        d, w = dig(vs), ver(vs)
        lines.append(f"  {g:17} digressions {len(d):>4}: false verse " + "  ".join(
            f"{c} {fv(S[c], d):>3} ({fv(S[c], d) / len(d):.0%})" if d else f"{c}   -" for c in cols)
            + f"   | verse passages {len(w):>4}: missed " + "  ".join(f"{c} {mv(S[c], w):>3}" for c in cols))
    lines += ["", "  2c. cost (USD)", ""]
    for g, _, vs in sets:
        lines.append(f"  {g:17}" + "  ".join(f"{c} ${sum(cost[v][c] for v in vs):.3f}" for c in cols))
    lines += ["", "  2d. 3-layer: who decided (passages; right = set equals gold; Gemini alone right on the same)", ""]
    ok = lambda p, u: set(p.get(u, ())) == set(gold[u])  # noqa: E731
    for g, _, vs in sets:
        us = [u for v in vs for u in lectures[v]]
        lines.append(f"  {g:17}" + "   ".join(
            f"{la}: {len(a)} right {sum(ok(three, u) for u in a)} (Gemini {sum(ok(gem, u) for u in a)})"
            for la in (L1, L2, L4) for a in [[u for u in us if layer[u] == la]] if a))
    text = "\n".join(lines)
    if name != "Cue":
        text = text.replace(L2, L2.replace("Cue", name))      # layer 2 is the Local confidence here
    print(text)
    fname = "all51_gemini3l.txt" if name == "Cue" else "all51_gemini3l_local.txt"
    with open(os.path.join(HERE, "results", fname), "w", encoding="utf-8") as f:
        f.write(text + "\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--structure", choices=["cue", "local"], default="cue")
    main(ap.parse_args().structure)
