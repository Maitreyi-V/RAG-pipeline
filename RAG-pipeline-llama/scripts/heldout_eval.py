#!/usr/bin/env python3
"""
New-speaker generalisation: every system, settings frozen, scored per speaker.

Nothing here is tuned on the held-out lectures. Each setting is either the
results_v3 value or is chosen on the ORIGINAL dev lectures inside this script,
then applied unchanged to the new speakers (data/lectures_new.csv):

  string matching   match thresholds as in results_v3 (run detect_units.py first)
  carry max_gap     the results_v3 values (kn 3, en 50)
  embeddings        expanded verse documents, top-1 with a margin; threshold = max dev F1
  GPT-4o-mini       no settings (detect_units.py --layer 4); shown once every unit is covered
  gated             string anchors + embedding gate; tau = max dev F1
  oracle gate       the ceiling for any gate, given the same anchors

Confidence intervals resample lectures, so they are only printed for a speaker
with at least 5 lectures. One long lecture gives a point estimate, not an interval.

Usage:
    python scripts/detect_units.py --layer 1 --units data/units_heldout_v1.csv \\
        --language kn --match-threshold 45 --out results/heldout_l1_kn.csv
    python scripts/detect_units.py --layer 1 --units data/units_heldout_v1.csv \\
        --language en --script roman --match-threshold 55 --out results/heldout_l1_en.csv
    python scripts/embed_scores.py --units data/units_heldout_v1.csv \\
        --out results/embed_scores_heldout.npz
    python scripts/heldout_eval.py --language en
"""
import argparse
import csv
import os

import numpy as np

from baselines import carry
from evalkit import (HERE, LECTURES_NEW, TIERS, bootstrap, load_csv, load_gold,
                     load_pred, read_split, score)
from gated_propagate import TAUS

FROZEN_GAP = {"kn": 3, "en": 50}            # results_v3
MIN_LECTURES_FOR_CI = 5
RES = os.path.join(HERE, "results")


def load_scores(rep):
    """Unit x verse similarities for the original AND held-out grids."""
    row, mats, refs = {}, [], None
    for name in ("embed_scores.npz", "embed_scores_heldout.npz"):
        z = np.load(os.path.join(RES, name), allow_pickle=False)
        refs = list(z["refs"])
        for i, u in enumerate(z["unit_ids"]):
            row[u] = (len(mats), i)
        mats.append(z[rep])
    return row, mats, refs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--language", choices=["kn", "en"], required=True)
    ap.add_argument("--rep", choices=["expanded", "trans"], default="expanded")
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--out-dir", default=RES, help="where to write heldout_<lang>.csv")
    args = ap.parse_args()
    L = args.language

    lectures, gold, _ = load_gold()
    dev = read_split(L, "dev")
    held = [v for v in read_split(L, "heldout") if v in lectures]
    speaker = {r["video_file"]: f"{r['speaker']} ({r['tradition']})"
               for r in load_csv(LECTURES_NEW)}

    anchors = load_pred(os.path.join(RES, "results_v3", f"final_l1_{L}.csv"))
    anchors.update(load_pred(os.path.join(RES, f"heldout_l1_{L}.csv")))

    row, mats, refs = load_scores(args.rep)
    col = {r: i for i, r in enumerate(refs)}

    def sims(u):
        m, i = row[u]
        return mats[m][i]

    def z(u, v):
        s = sims(u)
        return (s[col[v]] - s.mean()) / s.std() if v in col else float("-inf")

    def margin(u):
        s = np.sort(sims(u))
        return s[-1] - s[-2]

    test = read_split(L, "test")
    everything = dev + test + held          # test is scored only as a reference row

    def run_carry(**kw):
        pred = {}
        for v in everything:
            pred.update(carry(lectures[v], anchors, **kw))
        return pred

    def run_embed(t):
        return {u: ([refs[int(sims(u).argmax())]] if margin(u) >= t else [])
                for v in everything for u in lectures[v]}

    def best_on_dev(candidates, make):
        """First candidate with the highest dev F1 (same rule as the tuning scripts)."""
        best = None
        for c in candidates:
            f1 = score(gold, make(c), lectures, dev)["F1"]
            if best is None or f1 > best[1]:
                best = (c, f1)
        return best[0]

    dev_margins = np.array([margin(u) for v in dev for u in lectures[v]])
    t_embed = best_on_dev(np.percentile(dev_margins, np.arange(0, 100, 2)), run_embed)
    tau_drop = best_on_dev(TAUS, lambda t: run_carry(keep=lambda u, v: z(u, v) >= t))
    tau_res = best_on_dev(TAUS, lambda t: run_carry(keep=lambda u, v: z(u, v) >= t, resume=True))

    # GPT-4o-mini needs no tuning. It is only reported if it covers every unit:
    # a unit missing from the file would silently score as "no verse".
    llm = {}
    for name in (f"llm_{L}.csv", "heldout_llm.csv"):
        if os.path.exists(os.path.join(RES, name)):
            llm.update(load_pred(os.path.join(RES, name)))
    need = [u for v in everything for u in lectures[v]]
    missing = sum(1 for u in need if u not in llm)

    recited = {u: [v for v, t in g.items() if t in ("T1", "T2")] for u, g in gold.items()}

    def run_oracle_anchors():
        pred = {}
        for v in everything:
            pred.update(carry(lectures[v], recited))
        return pred

    systems = [
        ("string matching", run_carry(max_gap=0)),
        ("embeddings (margin)", run_embed(t_embed)),
    ]
    if llm and not missing:
        systems.append(("GPT-4o-mini", llm))
    elif llm:
        print(f"  !! GPT-4o-mini row skipped: {missing} unit(s) have no prediction")
    # the same model given the verse list, earlier units and its previous answer
    ctx_path = os.path.join(RES, "llmctx_all.csv")
    if os.path.exists(ctx_path):
        ctx = load_pred(ctx_path)
        if all(u in ctx for u in need):
            systems.append(("GPT-4o-mini + context", ctx))
    systems += [
        (f"carry max_gap={FROZEN_GAP[L]}", run_carry(max_gap=FROZEN_GAP[L])),
        ("carry unbounded", run_carry()),
        (f"gated, drop (tau={tau_drop})", run_carry(keep=lambda u, v: z(u, v) >= tau_drop)),
        (f"gated, resume (tau={tau_res})",
         run_carry(keep=lambda u, v: z(u, v) >= tau_res, resume=True)),
        ("monotonic unbounded", run_carry(monotonic=True)),
        ("oracle gate (resume)", run_carry(keep=lambda u, v: v in gold[u], resume=True)),
        ("oracle anchors (T1+T2)", run_oracle_anchors()),
    ]

    groups = [("ORIGINAL TEST SET (for reference)", test)]
    for sp in sorted({speaker[v] for v in held}):
        groups.append((sp, [v for v in held if speaker[v] == sp]))

    out_rows = []
    print("=" * 112)
    print(f"NEW-SPEAKER GENERALISATION — {L}   settings chosen on {len(dev)} original dev lectures")
    print("=" * 112)
    for name, videos in groups:
        n_units = sum(len(lectures[v]) for v in videos)
        n_empty = sum(1 for v in videos for u in lectures[v] if not gold[u])
        print(f"\n  {name}   lectures {len(videos)}   units {n_units}   empty {n_empty}")
        print(f"  {'system':28} {'P':>6} {'R':>6} {'F1':>6}  {'F1 95% CI':13} {'fire':>5}  "
              + "  ".join(f"{t:>8}" for t in TIERS))
        print("  " + "-" * 108)
        for sname, pred in systems:
            s = score(gold, pred, lectures, videos)
            if len(videos) >= MIN_LECTURES_FOR_CI:
                lo, hi = bootstrap(gold, pred, lectures, videos, n=args.n_boot)["F1"]
                ci = f"[{lo:.3f},{hi:.3f}]"
            else:
                ci = "n/a"
            tiers = [f"{h}/{n}" for h, n in (s["tiers"][t] for t in TIERS)]
            print(f"  {sname:28} {s['P']:>6.3f} {s['R']:>6.3f} {s['F1']:>6.3f}  {ci:13} "
                  f"{s['fired']:>5}  " + "  ".join(f"{t:>8}" for t in tiers))
            out_rows.append([name, len(videos), n_units, sname, f"{s['P']:.4f}",
                             f"{s['R']:.4f}", f"{s['F1']:.4f}", ci, s["fired"]] + tiers)

    out = os.path.join(args.out_dir, f"heldout_{L}.csv")
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["group", "lectures", "units", "system", "P", "R", "F1", "F1_CI", "fired"] + TIERS)
        w.writerows(out_rows)
    print(f"\n  CI only for groups with >= {MIN_LECTURES_FOR_CI} lectures. Tier cells are hit/n.")
    print(f"  wrote -> {os.path.relpath(out, HERE)}")


if __name__ == "__main__":
    main()
