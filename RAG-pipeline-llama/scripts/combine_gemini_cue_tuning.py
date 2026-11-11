#!/usr/bin/env python3
"""
Gemini + structure: a Dawid-Skene style combination decoded with the existing Step 4 Viterbi,
TUNING lectures only, leave-one-lecture-out (no test gold). Spec in docs/experiment_log.md.

  states       73 (72 verses + off-verse), step4_viterbi.transitions: the five tied buckets
               counted on the gold state sequences of the other 8 tuning lectures (+1), as run()
  annotators   Gemini 3.1 Pro (API, low), every passage
               Cue (v2 primary, LOLO) only where the string anchor fired or Cue's scorer
               probability for its own label >= 0.9 (as diag_disagreement_tuning.py); else abstains
  reliability  per annotator and language, tied buckets, +1 smoothing:
                 true verse s: correct (s in the set) / off by one (s+-1 in the set) /
                               other verse / NONE (said digression on a verse)
                 true off    : NONE / a verse (said a verse on a digression)
               a label's probability under state s = bucket probability / number of single-verse
               labels in that bucket (1; neighbours of s; 71 - neighbours; 72 for a verse on off)
  emission     sum over voting annotators of log P(label | state); path score = emission +
               1 * log transition (gamma 1; nothing is tuned)
  output       decoded off -> no verse; decoded verse s -> Gemini's set if it contains s, else
               Cue's set if Cue voted and it contains s, else [s]
  variants     base       reliability counted on the gold of the other tuning lectures of the
                          same language
               em         no gold: reliability from EM (forward-backward over the same chain,
                          transitions fixed), started from Gemini's labels as truth, on all
                          tuning lectures of the language; up to 200 iterations
               resume145  as base, decoded with the 145-state resume chain
                          (transitions_resume, viterbi_resume)
  tables       confident row: English, Cue conf >= 0.9 and no anchor, Cue vs Gemini disagree
               cascade: anchor fired -> anchor verses; else Cue conf >= 0.9 -> Cue; else Gemini

-> results/combine_gemini_cue_tuning.txt

Usage:
    python scripts/combine_gemini_cue_tuning.py
"""
import csv
import os
import sys
from collections import Counter

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "scripts"))

import numpy as np  # noqa: E402
from evalkit import load_csv, score  # noqa: E402
from llm_frontier import DIR, parse  # noqa: E402
from score_by_tier_units import FIDELITY, TIERS  # noqa: E402
from step4_viterbi import (FEATS, OFF, V, run, transitions, transitions_resume,  # noqa: E402
                           viterbi, viterbi_resume)

CONF = 0.9
GEMINI = os.path.join(DIR, "answers_api_gemini-3.1-pro-preview_low", "tuning")
ON_B, OFF_B = ("correct", "off by one", "other verse", "NONE"), ("NONE", "a verse")
EM_ITERS = 200


def buckets(A):
    """bucket index per state (73) for label set A (verse indices)"""
    b = np.empty(V + 1, dtype=int)
    for s in range(V):
        b[s] = 3 if not A else 0 if s in A else 1 if (s - 1 in A or s + 1 in A) else 2
    b[OFF] = 0 if not A else 1
    return b


NBR = np.array([(s > 0) + (s < V - 1) for s in range(V)])
N_ON = np.column_stack([np.ones(V), NBR, V - 1 - NBR, np.ones(V)])     # labels per bucket, verse s


def loglik(A, p_on, p_off):
    """log P(label A | state), 73 states"""
    b = buckets(A)
    out = np.empty(V + 1)
    out[:V] = np.log(p_on[b[:V]] / N_ON[np.arange(V), b[:V]])
    out[OFF] = np.log(p_off[0]) if not A else np.log(p_off[1] / V)
    return out


def table(weights):
    """weights: list of (A, q) with q a (73,) state weight -> smoothed (p_on, p_off)"""
    c_on, c_off = np.ones(4), np.ones(2)
    for A, q in weights:
        b = buckets(A)
        np.add.at(c_on, b[:V], q[:V])
        c_off[b[OFF]] += q[OFF]
    return c_on / c_on.sum(), c_off / c_off.sum()


def forward_backward(emit, logA):
    """state posteriors (T, 73), uniform start, gamma 1"""
    T = len(emit)
    la = np.empty_like(emit)
    lb = np.zeros_like(emit)
    la[0] = emit[0] - np.log(V + 1)
    for t in range(1, T):
        m = la[t - 1][:, None] + logA
        la[t] = np.logaddexp.reduce(m, axis=0) + emit[t]
    for t in range(T - 2, -1, -1):
        lb[t] = np.logaddexp.reduce(logA + (emit[t + 1] + lb[t + 1])[None, :], axis=1)
    g = la + lb
    g -= np.logaddexp.reduce(g, axis=1, keepdims=True)
    return np.exp(g)


def main():
    teacher = {r["unit_id"]: min(0.98, max(0.02, float(r["soft_target"]))) for r in
               csv.DictReader(open(os.path.join(HERE, "results/silver/silver_labels.csv"), encoding="utf-8"))}
    info = {}
    order, gold, lectures, tune = run(alphas=("a05",), verbose=False, resume=True, off_override=teacher, info=info)
    cue = dict(order)["Viterbi (alpha 0.5)"]
    probs = info["probs_a05"]
    z = np.load(FEATS)
    refs = list(z["refs"])
    idx = {r: i for i, r in enumerate(refs)}
    row = {u: i for i, u in enumerate(z["unit_ids"])}
    allv = [v for L in ("kn", "en") for v in tune[L]]
    lang = {v: L for L in tune for v in tune[L]}

    gem, missing = {}, 0
    for v in allv:
        ids = [(r["passage_id"], r["unit_id"]) for r in load_csv(os.path.join(DIR, "tuning", f"{v}_ids.csv"))]
        p, prob = parse(os.path.join(GEMINI, f"{v}.txt"), ids)
        missing += len(prob["missing"])
        for u in lectures[v]:
            gem[u] = p.get(u, [])

    conf, anchor, votes = {}, {}, {}
    for v in allv:
        pv, po = probs[v]
        for k, u in enumerate(lectures[v]):
            if cue[u]:
                s = idx[cue[u][0]]
                conf[u] = (1 - po[k]) * pv[k, s] / pv[k].sum()
            else:
                conf[u] = po[k]
            anchor[u] = [refs[j] for j in np.flatnonzero(z["anchor"][row[u]])]
            votes[u] = bool(anchor[u]) or conf[u] >= CONF

    def S(labels):
        return {idx[r] for r in labels if r in idx}

    def gstate(u):
        g = S(gold[u])
        return min(g) if g else OFF

    seq = {v: [gstate(u) for u in lectures[v]] for v in allv}

    def onehot(s):
        q = np.zeros(V + 1)
        q[s] = 1
        return q

    def gold_tables(vs):
        tg = table([(S(gem[u]), onehot(gstate(u))) for v in vs for u in lectures[v]])
        tc = table([(S(cue[u]), onehot(gstate(u))) for v in vs for u in lectures[v] if votes[u]])
        return tg, tc

    def emission(v, tg, tc):
        return np.array([loglik(S(gem[u]), *tg) + (loglik(S(cue[u]), *tc) if votes[u] else 0)
                         for u in lectures[v]])

    def output(v, path):
        out = {}
        for u, s in zip(lectures[v], path):
            if s == OFF:
                out[u] = []
            elif s in S(gem[u]):
                out[u] = list(gem[u])
            elif votes[u] and s in S(cue[u]):
                out[u] = list(cue[u])
            else:
                out[u] = [refs[s]]
        return out

    def em_tables(vs, logA):
        """EM without gold, started from Gemini's labels as truth"""
        q = {u: onehot(min(S(gem[u])) if S(gem[u]) else OFF) for v in vs for u in lectures[v]}
        prev, it = None, 0
        for it in range(1, EM_ITERS + 1):
            tg = table([(S(gem[u]), q[u]) for v in vs for u in lectures[v]])
            tc = table([(S(cue[u]), q[u]) for v in vs for u in lectures[v] if votes[u]])
            flat = np.concatenate([*tg, *tc])
            if prev is not None and np.abs(flat - prev).max() < 1e-6:
                break
            prev = flat
            for v in vs:
                for u, g in zip(lectures[v], forward_backward(emission(v, tg, tc), logA)):
                    q[u] = g
        return tg, tc, it

    preds = {k: {} for k in ("base", "em", "resume145")}
    rel, em_iters = {}, {}
    for h in allv:
        train = [w for w in allv if w != h]
        same = [w for w in tune[lang[h]] if w != h]
        logA, _ = transitions([seq[w] for w in train])
        logR, _ = transitions_resume([seq[w] for w in train])
        tg, tc = gold_tables(same)
        rel[h] = (tg, tc)
        emit = emission(h, tg, tc)
        preds["base"].update(output(h, viterbi(emit, logA, 1)))
        preds["resume145"].update(output(h, viterbi_resume(emit, logR, 1)))
        tg2, tc2, em_iters[h] = em_tables(tune[lang[h]], logA)
        preds["em"].update(output(h, viterbi(emission(h, tg2, tc2), logA, 1)))

    lines = [f"Gemini + structure, TUNING lectures only, leave-one-lecture-out; Gemini passages missing: {missing}; "
             f"Cue votes on {sum(votes[u] for v in allv for u in lectures[v])} of "
             f"{sum(len(lectures[v]) for v in allv)} passages"]
    systems = [("Gemini alone", gem), ("Cue (LOLO)", cue), ("base (gold reliability, 73)", preds["base"]),
               ("EM, no gold (73)", preds["em"]), ("resume 145 (gold reliability)", preds["resume145"])]
    for L, lname in (("kn", "Kannada"), ("en", "English")):
        vs = tune[L]
        lines += ["", f"{lname}: strict F1 pooled (tp/fp/fn) and per lecture",
                  f"  {'system':32}{'pooled':>24}" + "".join(f"{v:>16}" for v in vs)]
        for name, p in systems:
            f = score(gold, p, lectures, vs)
            pooled = f"{f['F1']:.3f} ({f['tp']}/{f['fp']}/{f['fn']})"
            lines.append(f"  {name:32}{pooled:>24}"
                         + "".join(f"{score(gold, p, lectures, [v])['F1']:>16.3f}" for v in vs))
        g = score(gold, gem, lectures, vs)["F1"]
        lines.append("  difference vs Gemini alone: " + ", ".join(
            f"{n} {score(gold, preds[k], lectures, vs)['F1'] - g:+.3f}" for n, k in
            (("base", "base"), ("EM", "em"), ("resume145", "resume145"))))
        h = vs[0]
        tg, tc = rel[h]
        lines.append(f"  reliability (fold held out {h}; from {len(vs) - 1} lectures):")
        for name, (p_on, p_off) in (("Gemini", tg), ("Cue (voting)", tc)):
            lines.append(f"    {name:13} true verse: " + ", ".join(f"{b} {x:.3f}" for b, x in zip(ON_B, p_on))
                         + "  | true off: " + ", ".join(f"{b} {x:.3f}" for b, x in zip(OFF_B, p_off)))
        lines.append("  EM iterations per fold: " + ", ".join(f"{v} {em_iters[v]}" for v in vs))

    # ---- table 1: the English confident row (Cue conf >= 0.9, no anchor, disagree with Gemini)
    en = [u for v in tune["en"] for u in lectures[v]]
    rowu = [u for u in en if not anchor[u] and conf[u] >= CONF and set(cue[u]) != set(gem[u])]

    def kind(u):
        return "Cue off / Gemini verse" if not cue[u] else "Cue verse / Gemini NONE" if not gem[u] \
            else "both verses, different"

    def who(u):
        c, g = set(cue[u]) == set(gold[u]), set(gem[u]) == set(gold[u])
        return "Cue right" if c else "Gemini right" if g else "both wrong"

    def tier(u):
        return min(gold[u].values(), key=lambda t: FIDELITY.get(t, 9)) if gold[u] else "OFF"

    lines += ["", f"TABLE 1: English, Cue conf >= {CONF}, no anchor, Cue and Gemini disagree: {len(rowu)} passages",
              "  by type x who is right:"]
    for (k, w), n in sorted(Counter((kind(u), who(u)) for u in rowu).items()):
        lines.append(f"    {k:26} {w:13} {n:>4}")
    lines.append("  by gold (passage's highest-fidelity type, OFF = digression) x who is right:")
    for (t, w), n in sorted(Counter((tier(u), who(u)) for u in rowu).items()):
        lines.append(f"    {t:5} {w:13} {n:>4}")
    lines.append("  by lecture x who is right:")
    for (v, w), n in sorted(Counter((u.rsplit("_", 1)[0], who(u)) for u in rowu).items()):
        lines.append(f"    {v:20} {w:13} {n:>4}")

    # ---- table 2: cascade
    def layer(u):
        return "1 anchor" if anchor[u] else f"2 Cue conf>={CONF}" if conf[u] >= CONF else "3 Gemini"

    LAYERS = ("1 anchor", f"2 Cue conf>={CONF}", "3 Gemini")
    casc = {u: anchor[u] if layer(u) == LAYERS[0] else cue[u] if layer(u) == LAYERS[1] else gem[u]
            for v in allv for u in lectures[v]}
    lines += ["", "TABLE 2: cascade  layer 1 anchor fired -> the anchor's verses; layer 2 Cue conf >= 0.9 -> Cue; "
                  "layer 3 -> Gemini.  correct = set equals gold"]
    for L, lname in (("kn", "Kannada"), ("en", "English")):
        us = [u for v in tune[L] for u in lectures[v]]
        lines += ["", f"  {lname}: {len(us)} passages",
                  f"    {'layer':18}{'accepted':>14}{'layer label ok':>16}{'Cue ok':>8}{'Gemini ok':>11}"]
        for la in LAYERS:
            lu = [u for u in us if layer(u) == la]
            ok = lambda p: sum(1 for u in lu if set(p[u]) == set(gold[u]))  # noqa: E731
            lines.append(f"    {la:18}{f'{len(lu)} ({len(lu) / len(us):.0%})':>14}{ok(casc):>16}{ok(cue):>8}{ok(gem):>11}")
        lines.append(f"    by passage type, accepted / layer label ok:  " + "   ".join(f"{la}" for la in LAYERS))
        for t in TIERS + ["OFF"]:
            tu = [u for u in us if tier(u) == t]
            if not tu:
                continue
            cells = []
            for la in LAYERS:
                lu = [u for u in tu if layer(u) == la]
                cells.append(f"{len(lu):>4} / {sum(1 for u in lu if set(casc[u]) == set(gold[u])):<4}")
            lines.append(f"    {t:4} (n={len(tu):>3})" + "      ".join(cells))
        f_c, f_g = score(gold, casc, lectures, tune[L]), score(gold, gem, lectures, tune[L])
        reach = sum(1 for u in us if layer(u) == LAYERS[2])
        lines.append(f"    cascade strict F1 {f_c['F1']:.3f} (tp {f_c['tp']} fp {f_c['fp']} fn {f_c['fn']}) vs "
                     f"Gemini alone {f_g['F1']:.3f}; passages reaching Gemini {reach}/{len(us)} ({reach / len(us):.0%})")
    text = "\n".join(lines)
    print(text)
    with open(os.path.join(HERE, "results", "combine_gemini_cue_tuning.txt"), "w", encoding="utf-8") as f:
        f.write(text + "\n")


if __name__ == "__main__":
    main()
