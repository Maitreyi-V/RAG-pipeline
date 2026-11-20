#!/usr/bin/env python3
"""
Gemini 3.1 Pro (thinking high, API) alone vs GitaGround-Cue alone vs the 3-layer cascade with Gemini high
(1 string-match anchor > 2 Cue conf >= 0.9 > Gemini high on the passages left), on all 55 annotated lectures:
9 tuning + 42 test + the 4 new English lectures (en_chinmaya_day06-09). Stored answers only, no API call.

  labels    Gemini high: review/llm_frontier/answers_api_gemini-3.1-pro-preview_high/ (tuning/ for the 9
            tuning lectures; the new 4 parsed with the ids of review/llm_frontier/new4/);
            Cue: tuning leave-one-lecture-out (cascade_tuning.load), other lectures v2_teacher145.pkl with
            the new passages' teacher P_off (new4_gemini3l.py)
  3-layer   layers chosen on the 9 tuning lectures with Gemini high by the keep rule of 65dbb40 (as
            all51_gemini3l.py); in sample on tuning
  cost      Gemini high: its usage files, $2 / 1M input, $12 / 1M output + thinking tokens; Cue: teacher
            tokens shared out by passage count (all51_gemini3l.py, new4_gemini3l.py); 3L = Gemini + Cue
  tests     paired lecture bootstrap of strict F1 (10,000), W/T/L, claim table (margin -0.02, set after the
            earlier results) on groups of 5+ lectures; the same bootstrap on false verses on digressions

-> results/all55_gemini_high.txt

Usage:
    python scripts/all55_gemini_high.py
"""
import os
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "scripts"))

import numpy as np  # noqa: E402
from all51_gemini3l import TEACHER_PASSAGES, TEACHER_TOKENS, gemini_cost  # noqa: E402
from cascade4_tuning import L1, L2, L4  # noqa: E402
from cascade_tuning import CONF, NO_ANCHOR_LAYER, ids_of  # noqa: E402
from evalkit import load_csv, load_gold, read_split  # noqa: E402
from freeze_grounding import f1_strict, paired, strict_counts  # noqa: E402
from gemini_runs_55 import TUNE  # noqa: E402
from improved3l_55 import cue_all  # noqa: E402
from llm_frontier import DIR, parse  # noqa: E402
from new4_gemini3l import NEW, NEW_DIR, NEW_TOKENS, claim  # noqa: E402

N_BOOT = 10000
HIGH = os.path.join(DIR, "answers_api_gemini-3.1-pro-preview_high")


def main():
    lectures, gold, _ = load_gold()
    tune = TUNE
    cue, conf, _, anchor = cue_all(lectures)                    # Cue as improved3l_55.py (tuning LOLO)
    groups = [("Tuning Kannada", "kn", tune["kn"]), ("Tuning English", "en", tune["en"]),
              ("E2 Kannada test", "kn", read_split("kn", "test")), ("E2 English test", "en", read_split("en", "test")),
              ("E3 Kannada", "kn", read_split("kn", "heldout")),
              ("E3 English, earlier 3", "en", [v for v in read_split("en", "heldout") if v not in NEW]),
              ("E3 English, new 4", "en", NEW)]
    assert sorted(read_split("en", "heldout")) == sorted(groups[5][2] + NEW)
    tuning = tune["kn"] + tune["en"]
    rest = [v for _, _, vs in groups[2:] for v in vs]
    lang = {v: L for _, L, vs in groups for v in vs}
    assert set(cue) >= {u for v in lang for u in lectures[v]}

    def where(v):                                               # (answer folder, [(passage_id, unit_id)])
        if v in tuning:
            return os.path.join(HIGH, "tuning"), ids_of(v)
        d = NEW_DIR if v in NEW else DIR
        return HIGH, [(r["passage_id"], r["unit_id"]) for r in load_csv(os.path.join(d, f"{v}_ids.csv"))]

    high, missing, gcost = {}, {}, {}
    for v in lang:
        folder, ids = where(v)
        p, pr = parse(os.path.join(folder, f"{v}.txt"), ids)
        missing[v] = len(pr["missing"])
        high.update({u: p.get(u, []) for u in lectures[v]})
        gcost[v] = gemini_cost(os.path.join(folder, f"{v}_usage.json"))

    take = {L1: (lambda u: bool(anchor[u]), anchor), L2: (lambda u: conf[u] >= CONF, cue)}
    design, keep_lines = {}, []
    for L in ("kn", "en"):                                      # keep rule of 65dbb40, tuning only
        left, kept = [u for v in tune[L] for u in lectures[v]], []
        for la in (L1, L2):
            if la == L1 and L in NO_ANCHOR_LAYER:
                continue
            acc = [u for u in left if take[la][0](u)]
            r_s = sum(set(take[la][1][u]) == set(gold[u]) for u in acc)
            r_g = sum(set(high[u]) == set(gold[u]) for u in acc)
            keep_lines.append(f"  {L} {la:16} accepts {len(acc):>3} tuning passages: right {r_s} vs Gemini high "
                              f"{r_g} -> {'KEEP' if r_s >= r_g else 'drop'}")
            if r_s >= r_g:
                kept.append(la)
                left = [u for u in left if not take[la][0](u)]
        design[L] = tuple(kept)
    three, layer = {}, {}
    for v, L in lang.items():
        for u in lectures[v]:
            la = next((x for x in design[L] if take[x][0](u)), L4)
            layer[u], three[u] = la, (take[la][1][u] if la != L4 else high[u])
    S = {"Gemini high": high, "Cue": cue, "3L high": three}
    cols = list(S)

    old_psg = (TEACHER_TOKENS[0] / 1e6 * 0.25 + TEACHER_TOKENS[1] / 1e6 * 2) / TEACHER_PASSAGES
    new_psg = (NEW_TOKENS[0] / 1e6 * 0.25 + NEW_TOKENS[1] / 1e6 * 2) / sum(len(lectures[v]) for v in NEW)
    cost = {}
    for v in lang:
        c = (new_psg if v in NEW else old_psg) * len(lectures[v])
        cost[v] = {"Gemini high": gcost[v], "Cue": c, "3L high": gcost[v] + c}

    f1 = lambda p, vs: f1_strict([sum(x) for x in zip(*(strict_counts(gold, p, lectures[v]) for v in vs))])  # noqa: E731
    dig = lambda vs: [u for v in vs for u in lectures[v] if not gold[u]]  # noqa: E731
    ver = lambda vs: [u for v in vs for u in lectures[v] if gold[u]]  # noqa: E731
    fv = lambda p, us: sum(bool(p.get(u)) for u in us)  # noqa: E731     false verse on a digression
    mv = lambda p, us: sum(not p.get(u) for u in us)  # noqa: E731       verse passage left with no verse
    ok = lambda p, u: set(p.get(u, ())) == set(gold[u])  # noqa: E731

    def fv_boot(p, vs):
        """paired lecture bootstrap of false-verse rate on digressions, p minus Gemini high:
        (diff in points, 95% CI, ratio p/Gemini, 95% CI); lectures with no digression drop out"""
        vs = [v for v in vs if dig([v])]
        a = np.array([[fv(p, dig([v])), fv(high, dig([v])), len(dig([v]))] for v in vs], float)
        rng = np.random.default_rng(0)
        idx = rng.integers(0, len(vs), (N_BOOT, len(vs)))
        s = a[idx].sum(axis=1)
        diff, ratio = (s[:, 0] - s[:, 1]) / s[:, 2], s[:, 0] / np.maximum(s[:, 1], 1)
        t = a.sum(axis=0)
        return ((t[0] - t[1]) / t[2], np.percentile(diff, [2.5, 97.5]),
                t[0] / t[1], np.percentile(ratio, [2.5, 97.5]))

    lines = ["Gemini 3.1 Pro high alone vs GitaGround-Cue alone vs 3-layer cascade with Gemini high "
             "(string anchor > Cue conf >= 0.9 > Gemini high), 55 lectures",
             "tuning: Cue leave-one-lecture-out; 3-layer in sample (its layers were chosen on tuning). "
             "Cue cost is an estimate (teacher tokens shared out by passage count)",
             f"Gemini high answers: passages not found in the reply {sum(missing.values())} "
             + str({v: n for v, n in missing.items() if n}), "",
             "keep rule on the 9 tuning lectures (layers chosen for Gemini high):"] + keep_lines + [
             f"  -> Kannada: {' > '.join(design['kn']) or 'none'} > Gemini;  English: "
             f"{' > '.join(design['en']) or 'none'} > Gemini", "",
             "1. LECTURE BY LECTURE  (strict F1, best marked *; dig = gold no-verse passages; "
             "FV = false verse on them; $ per lecture)", "",
             f"  {'lecture':24}{'psg':>4}" + "".join(f"{c:>12}" for c in cols) + f"{'dig':>5}"
             + "".join(f"{'FV ' + c:>15}" for c in cols) + "".join(f"{'$ ' + c:>14}" for c in cols)]
    for g, _, vs in groups:
        lines.append(f"  -- {g} --")
        for v in vs:
            vals = [f1(S[c], [v]) for c in cols]
            d = dig([v])
            lines.append(f"  {v:24}{len(lectures[v]):>4}" + "".join(
                f"{x:>11.3f}{'*' if abs(x - max(vals)) < 1e-9 else ' '}" for x in vals)
                + f"{len(d):>5}" + "".join(f"{fv(S[c], d):>15}" for c in cols)
                + "".join(f"{cost[v][c]:>14.3f}" for c in cols))

    sets = groups + [("E3 English, all 7", None, read_split("en", "heldout")), ("All 9 tuning", None, tuning),
                     ("All 42 test", None, [v for _, _, vs in groups[2:6] for v in vs]),
                     ("All 46 test (+new 4)", None, rest), ("All 55", None, list(lang))]
    lines += ["", "2. OVERVIEW", "", "  2a. strict F1; paired vs Gemini high alone: diff [95% CI] W/T/L -> claim "
              "(5+ lectures)", ""]
    for g, _, vs in sets:
        lines.append(f"  {g:22}{len(vs):>3} lect  " + "  ".join(f"{c} {f1(S[c], vs):.3f}" for c in cols))
        if len(vs) >= 5:
            for c in ("Cue", "3L high"):
                d, ci, _, wtl, _ = paired(gold, lectures, S[c], high, vs, N_BOOT)
                lines.append(f"      {c:10} - Gemini high {d:+.3f} [{ci[0]:+.3f},{ci[1]:+.3f}]  W/T/L "
                             f"{wtl[0]}/{wtl[1]}/{wtl[2]}  -> {claim(d, ci)}")
    lines += ["", "  2b. digressions (gold: no verse) and verse passages left with no verse", ""]
    for g, _, vs in sets:
        d, w = dig(vs), ver(vs)
        lines.append(f"  {g:22} digressions {len(d):>4}: false verse " + "  ".join(
            f"{c} {fv(S[c], d):>3} ({fv(S[c], d) / len(d):.0%})" if d else f"{c}   -" for c in cols)
            + f"   | verse passages {len(w):>4}: missed " + "  ".join(f"{c} {mv(S[c], w):>3}" for c in cols))
    lines += ["", "  2c. false verse on digressions, paired lecture bootstrap vs Gemini high (5+ lectures with a "
              "digression): rate diff in points [95% CI]; ratio to Gemini high [95% CI]", ""]
    for g, _, vs in sets:
        if sum(bool(dig([v])) for v in vs) < 5:
            continue
        for c in ("Cue", "3L high"):
            dd, dci, r, rci = fv_boot(S[c], vs)
            lines.append(f"  {g:22} {c:8} {100 * dd:+6.1f} [{100 * dci[0]:+6.1f},{100 * dci[1]:+6.1f}]   "
                         f"x{r:.2f} [{rci[0]:.2f},{rci[1]:.2f}]")
    lines += ["", "  2d. cost (USD)", ""]
    for g, _, vs in sets:
        lines.append(f"  {g:22}" + "  ".join(f"{c} ${sum(cost[v][c] for v in vs):.3f}" for c in cols))
    lines += ["", "  2e. 3-layer: who decided (passages; right = set equals gold; Gemini high alone right on the same)",
              ""]
    for g, _, vs in sets:
        us = [u for v in vs for u in lectures[v]]
        lines.append(f"  {g:22}" + "   ".join(
            f"{la}: {len(a)} right {sum(ok(three, u) for u in a)} (Gemini high {sum(ok(high, u) for u in a)})"
            for la in (L1, L2, L4) for a in [[u for u in us if layer[u] == la]] if a))
    text = "\n".join(lines)
    print(text)
    with open(os.path.join(HERE, "results", "all55_gemini_high.txt"), "w", encoding="utf-8") as f:
        f.write(text + "\n")


if __name__ == "__main__":
    main()
