#!/usr/bin/env python3
"""
Gemini 3.1 Pro (thinking medium) alone vs GitaGround-Cue alone vs the 3-layer cascade with Gemini medium
(1 string-match anchor (Kannada only) > 2 Cue conf >= 0.9 > Gemini medium on the passages left), on all 55
annotated lectures: 9 tuning + 42 test + the 4 new English lectures (en_chinmaya_day06-09). Stored labels
only, no API call.

  labels    Gemini medium: results/gemini_runs/gita_data.csv (Maitreyi's run, all 2,183 passages; identical
            to data.txt apart from a final newline); Cue: tuning leave-one-lecture-out (cascade_tuning.load), other lectures v2_teacher145.pkl with the
            new passages' teacher P_off (new4_gemini3l.py)
  3-layer   layers chosen on the 9 tuning lectures with Gemini medium by the keep rule of 65dbb40 (as
            all51_gemini3l.py); in sample on tuning
  cost      Cue: teacher tokens (all51_gemini3l.py, new4_gemini3l.py); Gemini medium: no usage files with
            the run, so not reported
  tests     paired lecture bootstrap of strict F1 (10,000), W/T/L, claim table (margin -0.02, set after the
            earlier results) on groups of 5+ lectures

-> results/all55_gemini_medium.txt

Usage:
    python scripts/all55_gemini_medium.py
"""
import copy
import os
import pickle
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "scripts"))

import numpy as np  # noqa: E402
from all51_gemini3l import TEACHER_PASSAGES, TEACHER_TOKENS  # noqa: E402
from cascade4_tuning import L1, L2, L4  # noqa: E402
from cascade_tuning import CONF, NO_ANCHOR_LAYER, load  # noqa: E402
from evalkit import read_split  # noqa: E402
from freeze_grounding import f1_strict, paired, strict_counts  # noqa: E402
from new4_gemini3l import NEW, NEW_TOKENS, PKL, claim, teacher_soft  # noqa: E402
from recall_by_type import read_llm  # noqa: E402
from step4_viterbi import FEATS, OFF, predict_locked  # noqa: E402

N_BOOT = 10000
MEDIUM = "gemini_runs/gita_data.csv"


def main():
    gold, lectures, tune, cue, _, conf, anchor = load()
    groups = [("Tuning Kannada", "kn", tune["kn"]), ("Tuning English", "en", tune["en"]),
              ("E2 Kannada test", "kn", read_split("kn", "test")), ("E2 English test", "en", read_split("en", "test")),
              ("E3 Kannada", "kn", read_split("kn", "heldout")),
              ("E3 English, earlier 3", "en", [v for v in read_split("en", "heldout") if v not in NEW]),
              ("E3 English, new 4", "en", NEW)]
    assert sorted(read_split("en", "heldout")) == sorted(groups[5][2] + NEW)
    rest = [v for _, _, vs in groups[2:] for v in vs]
    lang = {v: L for _, L, vs in groups for v in vs}
    z = np.load(FEATS)
    refs = list(z["refs"])
    row = {u: i for i, u in enumerate(z["unit_ids"])}
    m = copy.copy(pickle.load(open(PKL, "rb"))["a05"])
    soft = teacher_soft()
    m["p_off"] = dict(m["p_off"])
    m["p_off"].update({u: soft[u] for v in NEW for u in lectures[v]})
    dec = predict_locked({v: lectures[v] for v in rest}, "a05", model=m)
    for v in rest:
        for u, (st, extra, p_state, _) in zip(lectures[v], dec[v]):
            cue[u], conf[u] = ([] if st == OFF else [refs[st]]) + [refs[j] for j in extra], p_state
            anchor[u] = [refs[j] for j in np.flatnonzero(z["anchor"][row[u]])]

    med = read_llm(MEDIUM)
    assert set(med) == {u for v in lang for u in lectures[v]}

    take = {L1: (lambda u: bool(anchor[u]), anchor), L2: (lambda u: conf[u] >= CONF, cue)}
    design, keep_lines = {}, []
    for L in ("kn", "en"):                                      # keep rule of 65dbb40, tuning only
        left, kept = [u for v in tune[L] for u in lectures[v]], []
        for la in (L1, L2):
            if la == L1 and L in NO_ANCHOR_LAYER:
                continue
            acc = [u for u in left if take[la][0](u)]
            r_s = sum(set(take[la][1][u]) == set(gold[u]) for u in acc)
            r_g = sum(set(med[u]) == set(gold[u]) for u in acc)
            keep_lines.append(f"  {L} {la:16} accepts {len(acc):>3} tuning passages: right {r_s} vs Gemini medium "
                              f"{r_g} -> {'KEEP' if r_s >= r_g else 'drop'}")
            if r_s >= r_g:
                kept.append(la)
                left = [u for u in left if not take[la][0](u)]
        design[L] = tuple(kept)
    three, layer = {}, {}
    for v, L in lang.items():
        for u in lectures[v]:
            la = next((x for x in design[L] if take[x][0](u)), L4)
            layer[u], three[u] = la, (take[la][1][u] if la != L4 else med[u])
    S = {"Gemini med": med, "Cue": cue, "3L med": three}
    cols = list(S)

    old_psg = (TEACHER_TOKENS[0] / 1e6 * 0.25 + TEACHER_TOKENS[1] / 1e6 * 2) / TEACHER_PASSAGES
    new_psg = (NEW_TOKENS[0] / 1e6 * 0.25 + NEW_TOKENS[1] / 1e6 * 2) / sum(len(lectures[v]) for v in NEW)
    cue_cost = {v: (new_psg if v in NEW else old_psg) * len(lectures[v]) for v in lang}

    f1 = lambda p, vs: f1_strict([sum(x) for x in zip(*(strict_counts(gold, p, lectures[v]) for v in vs))])  # noqa: E731
    dig = lambda vs: [u for v in vs for u in lectures[v] if not gold[u]]  # noqa: E731
    ver = lambda vs: [u for v in vs for u in lectures[v] if gold[u]]  # noqa: E731
    fv = lambda p, us: sum(bool(p.get(u)) for u in us)  # noqa: E731
    mv = lambda p, us: sum(not p.get(u) for u in us)  # noqa: E731
    ok = lambda p, u: set(p.get(u, ())) == set(gold[u])  # noqa: E731

    lines = ["Gemini 3.1 Pro medium alone vs GitaGround-Cue alone vs 3-layer cascade with Gemini medium "
             "(string anchor > Cue conf >= 0.9 > Gemini medium), 55 lectures",
             "tuning: Cue leave-one-lecture-out; 3-layer in sample (its layers were chosen on tuning)", "",
             "keep rule on the 9 tuning lectures (layers chosen for Gemini medium):"] + keep_lines + [
             f"  -> Kannada: {' > '.join(design['kn']) or 'none'} > Gemini;  English: "
             f"{' > '.join(design['en']) or 'none'} > Gemini", "",
             "1. LECTURE BY LECTURE  (strict F1, best marked *; dig = gold no-verse passages; "
             "FV = false verse on them)", "",
             f"  {'lecture':24}{'psg':>4}" + "".join(f"{c:>12}" for c in cols) + f"{'dig':>5}"
             + "".join(f"{'FV ' + c:>15}" for c in cols)]
    for g, _, vs in groups:
        lines.append(f"  -- {g} --")
        for v in vs:
            vals = [f1(S[c], [v]) for c in cols]
            d = dig([v])
            lines.append(f"  {v:24}{len(lectures[v]):>4}" + "".join(
                f"{x:>11.3f}{'*' if abs(x - max(vals)) < 1e-9 else ' '}" for x in vals)
                + f"{len(d):>5}" + "".join(f"{fv(S[c], d):>15}" for c in cols))

    e3en = [v for v in read_split("en", "heldout")]
    sets = groups + [("E3 English, all 7", None, e3en), ("All 9 tuning", None, tune["kn"] + tune["en"]),
                     ("All 42 test", None, [v for _, _, vs in groups[2:6] for v in vs]),
                     ("All 46 test (+new 4)", None, rest), ("All 55", None, list(lang))]
    lines += ["", "2. OVERVIEW", "", "  2a. strict F1; paired vs Gemini medium alone: diff [95% CI] W/T/L -> claim "
              "(5+ lectures)", ""]
    for g, _, vs in sets:
        lines.append(f"  {g:22}{len(vs):>3} lect  " + "  ".join(f"{c} {f1(S[c], vs):.3f}" for c in cols))
        if len(vs) >= 5:
            for c in ("Cue", "3L med"):
                d, ci, _, wtl, _ = paired(gold, lectures, S[c], med, vs, N_BOOT)
                lines.append(f"      {c:10} - Gemini med {d:+.3f} [{ci[0]:+.3f},{ci[1]:+.3f}]  W/T/L "
                             f"{wtl[0]}/{wtl[1]}/{wtl[2]}  -> {claim(d, ci)}")
    lines += ["", "  2b. digressions (gold: no verse) and verse passages left with no verse", ""]
    for g, _, vs in sets:
        d, w = dig(vs), ver(vs)
        lines.append(f"  {g:22} digressions {len(d):>4}: false verse " + "  ".join(
            f"{c} {fv(S[c], d):>3} ({fv(S[c], d) / len(d):.0%})" if d else f"{c}   -" for c in cols)
            + f"   | verse passages {len(w):>4}: missed " + "  ".join(f"{c} {mv(S[c], w):>3}" for c in cols))
    lines += ["", "  2c. cost (USD; Gemini medium: no usage files, not known)", ""]
    for g, _, vs in sets:
        lines.append(f"  {g:22} Cue ${sum(cue_cost[v] for v in vs):.3f}   3L med = Gemini medium + Cue")
    lines += ["", "  2d. 3-layer: who decided (passages; right = set equals gold; Gemini medium alone right on the same)", ""]
    for g, _, vs in sets:
        us = [u for v in vs for u in lectures[v]]
        lines.append(f"  {g:22}" + "   ".join(
            f"{la}: {len(a)} right {sum(ok(three, u) for u in a)} (Gemini med {sum(ok(med, u) for u in a)})"
            for la in (L1, L2, L4) for a in [[u for u in us if layer[u] == la]] if a))
    text = "\n".join(lines)
    print(text)
    with open(os.path.join(HERE, "results", "all55_gemini_medium.txt"), "w", encoding="utf-8") as f:
        f.write(text + "\n")


if __name__ == "__main__":
    main()
