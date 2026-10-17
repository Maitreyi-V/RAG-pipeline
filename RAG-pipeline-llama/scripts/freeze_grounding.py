#!/usr/bin/env python3
"""
Freeze run for the grounding experiments E2 (test lectures), E3 (new speakers)
and E5 (alpha 0 vs 0.5), with the LOCKED system (results/step4_locked.pkl).

  --split dev       DRY RUN on the 9 tuning lectures. Our rows are IN-SAMPLE (the
                    locked models were fit on these lectures): this checks that
                    the script works, the numbers are not results.
  --split test      E2: 23 Kannada + 7 English test lectures. Run ONCE.
  --split heldout   E3: the 4 new speakers, each and pooled. Run ONCE.

Rows (every row scored on the same passages):
  string matching           Step 2 detections (current verse database)
  embeddings (margin)       results/embed_scores*.npz, expanded verse documents,
                            top-1 if the margin >= t; t = best F1 on the dev lectures
  GPT-4o-mini               results/llm_*.csv, heldout_llm.csv (only if every unit is covered)
  GPT-4o-mini + context     results/llmctx_all.csv (same rule)
  plain carry               Step 2 detections, carried until the next detection
  carry, best anchor        the same, keeping only the best-matching detection
  gated carry (resume)      negative baseline; embedding z gate, tau = best dev F1
  oracle stopping           carry that stops / resumes exactly where gold does (ceiling)
  ours, alpha 0.5           the locked system
  ours, alpha 0 (E5)        the locked system without verse translations

Per row: P, R, strict / within-1 / theme F1, each with a 95% lecture-bootstrap
CI (only for groups with >= 5 lectures), and recall per mention type T1..T5.
Then wrong labels by cause for plain carry vs ours (false positives):
  A  carried into a no-verse passage    B  carried past an unrecited switch
  C  a string detection fired the wrong verse on this passage

Writes results/freeze_grounding_<split>.txt.

Usage:
    python scripts/freeze_grounding.py --split dev
"""
import argparse
import os
import random
import sys
from collections import Counter

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

from baselines import carry  # noqa: E402
from catalogue import load_themes  # noqa: E402
from evalkit import (LECTURES_NEW, TIERS, _counts, _prf, load_csv, load_gold,  # noqa: E402
                     load_pred, read_split, verse_number)
from gated_propagate import TAUS  # noqa: E402
from heldout_eval import load_scores  # noqa: E402
from step4_viterbi import FEATS, OFF, predict_locked  # noqa: E402

RES = os.path.join(HERE, "results")
MIN_LECTURES_FOR_CI = 5


# ---- per-lecture counts for the three granularities ---------------------------
def strict_counts(gold, pred, us):
    return _counts(gold, pred, us)[:3]                        # tp, fp, fn


def near_counts(gold, pred, us, k=1):
    """(predicted hit, predicted, gold hit, gold): a verse within +-k counts."""
    hp = n_p = hg = n_g = 0
    for u in us:
        g, p = gold[u], list(pred.get(u, ()))
        gn = [x for x in map(verse_number, g) if x]
        pn = [x for x in map(verse_number, p) if x]
        for r in p:
            x = verse_number(r)
            n_p += 1
            hp += r in g or (x is not None and any(abs(x - y) <= k for y in gn))
        for r in g:
            x = verse_number(r)
            n_g += 1
            hg += r in p or (x is not None and any(abs(x - y) <= k for y in pn))
    return hp, n_p, hg, n_g


THEMES = load_themes()


def theme_counts(gold, pred, us):
    def m(refs):
        return {THEMES[n] for n in map(verse_number, refs) if n in THEMES}
    tg = {u: {t: "" for t in m(gold[u])} for u in us}
    tp = {u: sorted(m(pred.get(u, ()))) for u in us}
    return _counts(tg, tp, us)[:3]


def f1_strict(c):
    return _prf(*c)[2]


def f1_near(c):
    hp, n_p, hg, n_g = c
    P, R = (hp / n_p if n_p else 0.0), (hg / n_g if n_g else 0.0)
    return 2 * P * R / (P + R) if P + R else 0.0


def with_ci(per_lecture, f1, n_boot, seed=0):
    """point F1 over the pooled counts and a 95% CI resampling lectures"""
    tot = [sum(x) for x in zip(*per_lecture)]
    point = f1(tot)
    if len(per_lecture) < MIN_LECTURES_FOR_CI:
        return point, None
    rng = random.Random(seed)
    draws = sorted(f1([sum(x) for x in zip(*(per_lecture[rng.randrange(len(per_lecture))]
                                                for _ in per_lecture))]) for _ in range(n_boot))
    return point, (draws[int(0.025 * n_boot)], draws[min(int(0.975 * n_boot), n_boot - 1)])


def causes(gold, pred, anchors, us):
    """false positives by cause A / B / C"""
    c = Counter()
    for u in us:
        g, fired = set(gold[u]), set(anchors.get(u, ()))
        for v in set(pred.get(u, ())) - g:
            c["C" if v in fired else ("A" if not g else "B")] += 1
    return c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["dev", "test", "heldout"], required=True)
    ap.add_argument("--n-boot", type=int, default=1000)
    args = ap.parse_args()

    lectures, gold, _ = load_gold()
    z = np.load(FEATS)
    refs = list(z["refs"])
    row = {u: i for i, u in enumerate(z["unit_ids"])}
    anchors = {u: [refs[j] for j in np.flatnonzero(z["anchor"][row[u]])] for u in row}
    best_anchor = {u: ([max(a, key=lambda r: z["string"][row[u], refs.index(r)])] if a else [])
                   for u, a in anchors.items()}

    # groups of lectures to report
    speaker = {r["video_file"]: f"{r['speaker']} ({r['tradition']})" for r in load_csv(LECTURES_NEW)}
    groups = []
    for L, lname in (("kn", "Kannada"), ("en", "English")):
        vids = [v for v in read_split(L, args.split) if v in lectures]
        if args.split == "heldout":
            for sp in sorted({speaker[v] for v in vids}):
                groups.append((L, sp, [v for v in vids if speaker[v] == sp]))
            groups.append((L, f"New {lname} (pooled)", vids))
        else:
            groups.append((L, f"{lname} {'tuning' if args.split == 'dev' else 'test'}", vids))
    scope = sorted({v for _, _, vs in groups for v in vs})

    # ---- baselines whose setting is chosen on the dev lectures ------------------
    emb_row, mats, erefs = load_scores("expanded")
    col = {r: i for i, r in enumerate(erefs)}

    def sims(u):
        m, i = emb_row[u]
        return mats[m][i]

    def zgate(u, v):
        s = sims(u)
        return (s[col[v]] - s.mean()) / s.std() if v in col else float("-inf")

    def margin(u):
        s = np.sort(sims(u))
        return s[-1] - s[-2]

    def carry_all(vids, anc, **kw):
        out = {}
        for v in vids:
            out.update(carry(lectures[v], anc, **kw))
        return out

    def embed_pred(vids, t):
        return {u: ([erefs[int(sims(u).argmax())]] if margin(u) >= t else [])
                for v in vids for u in lectures[v]}

    def strict_f1(pred, vids):
        return f1_strict([sum(x) for x in zip(*(strict_counts(gold, pred, lectures[v]) for v in vids))])

    settings = {}
    for L in ("kn", "en"):
        dev = [v for v in read_split(L, "dev") if v in lectures]
        dm = np.array([margin(u) for v in dev for u in lectures[v]])
        cands = np.percentile(dm, np.arange(0, 100, 2))
        t_emb = max(cands, key=lambda t: (strict_f1(embed_pred(dev, t), dev), -t))
        tau = max(TAUS, key=lambda t: (strict_f1(carry_all(dev, anchors, keep=lambda u, v: zgate(u, v) >= t,
                                                           resume=True), dev), -t))
        settings[L] = (t_emb, tau)

    llm, ctx = {}, {}
    for name in ("llm_kn.csv", "llm_en.csv", "heldout_llm.csv"):
        if os.path.exists(os.path.join(RES, name)):
            llm.update(load_pred(os.path.join(RES, name)))
    if os.path.exists(os.path.join(RES, "llmctx_all.csv")):
        ctx = load_pred(os.path.join(RES, "llmctx_all.csv"))

    lecture_units = {v: lectures[v] for v in scope}
    ours = {}
    for a in ("a05", "a0"):
        dec = predict_locked(lecture_units, a)
        ours[a] = {u: ([] if st == OFF else [refs[st]]) + [refs[j] for j in extra]
                   for v in scope for u, (st, extra, _, _) in zip(lectures[v], dec[v])}

    lines = [f"FREEZE RUN — grounding, split = {args.split}"
             + ("   (DRY RUN: our rows are IN-SAMPLE, not results)" if args.split == "dev" else ""),
             "Settings chosen on the dev lectures: " + ", ".join(
                 f"{L}: embedding margin t = {settings[L][0]:.4f}, gate tau = {settings[L][1]}"
                 for L in settings),
             f"CI: 95% lecture bootstrap, {args.n_boot} resamples, only for groups with "
             f">= {MIN_LECTURES_FOR_CI} lectures. Tier cells are hit/n (recall)."]

    for L, gname, vids in groups:
        units = [u for v in vids for u in lectures[v]]
        t_emb, tau = settings[L]
        rows = [("string matching", {u: anchors[u] for u in units}),
                ("embeddings (margin)", embed_pred(vids, t_emb))]
        for name, src in (("GPT-4o-mini", llm), ("GPT-4o-mini + context", ctx)):
            if all(u in src for u in units):
                rows.append((name, src))
            else:
                lines.append(f"  !! {gname}: {name} row skipped, "
                             f"{sum(u not in src for u in units)} unit(s) have no prediction")
        rows += [("plain carry", carry_all(vids, anchors)),
                 ("carry, best anchor", carry_all(vids, best_anchor)),
                 (f"gated carry, resume (tau {tau})",
                  carry_all(vids, anchors, keep=lambda u, v: zgate(u, v) >= tau, resume=True)),
                 ("oracle stopping (ceiling)",
                  carry_all(vids, anchors, keep=lambda u, v: v in gold[u], resume=True)),
                 ("ours, alpha 0.5", ours["a05"]),
                 ("ours, alpha 0 (E5)", ours["a0"])]

        n_empty = sum(1 for u in units if not gold[u])
        lines += ["", "=" * 132,
                  f"{gname}   lectures {len(vids)}   passages {len(units)}   no-verse {n_empty}",
                  "=" * 132,
                  f"  {'system':30} {'P':>6} {'R':>6}  {'strict F1 [CI]':22} {'+-1 F1 [CI]':22} "
                  f"{'theme F1 [CI]':22} " + " ".join(f"{t:>8}" for t in TIERS)]
        for name, pred in rows:
            tp, fp, fn = [sum(x) for x in zip(*(strict_counts(gold, pred, lectures[v]) for v in vids))]
            P, R, _ = _prf(tp, fp, fn)
            cells = []
            for counts, f1 in ((strict_counts, f1_strict), (near_counts, f1_near), (theme_counts, f1_strict)):
                point, ci = with_ci([counts(gold, pred, lectures[v]) for v in vids], f1, args.n_boot)
                cells.append(f"{point:.3f}" + (f" [{ci[0]:.3f},{ci[1]:.3f}]" if ci else "") )
            tiers = _counts(gold, pred, units)[4]
            lines.append(f"  {name:30} {P:>6.3f} {R:>6.3f}  {cells[0]:22} {cells[1]:22} {cells[2]:22} "
                         + " ".join(f"{h}/{n}".rjust(8) for h, n in (tiers[t] for t in TIERS)))
        lines.append("  wrong labels (false positives) by cause:   A carried into no-verse   "
                     "B carried past an unrecited switch   C wrong string detection")
        for name, pred in rows:
            if name in ("plain carry", "ours, alpha 0.5"):
                c = causes(gold, pred, anchors, units)
                lines.append(f"    {name:28} A {c['A']:4}   B {c['B']:4}   C {c['C']:4}   "
                             f"total {sum(c.values())}")

    text = "\n".join(lines)
    print(text)
    out = os.path.join(RES, f"freeze_grounding_{args.split}.txt")
    with open(out, "w", encoding="utf-8") as f:
        f.write(text + "\n")
    print(f"\n  wrote -> {os.path.relpath(out, HERE)}")


if __name__ == "__main__":
    main()
