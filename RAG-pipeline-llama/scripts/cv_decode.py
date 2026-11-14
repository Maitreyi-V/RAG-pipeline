#!/usr/bin/env python3
"""
Secondary protocol (gg_finetune/README sections 4-5): decode with the fine-tuned scores.

Per run in gg_finetune/folds.json (cv0..cv4, e3), same decoder as GitaGround-Cue
(145-state resume, decoupled emissions, alpha 0.5):
  transitions   gold sequences of that run's TRAIN + CALIB lectures
  verse scorer  the 8 locked features + verse_feature (fine-tuned), LR fit on CALIB gold
  P_off         A cue (silver soft_target) | B p_off_ft | C mean of A and B
  selection     variant, gamma, beta_off, tau on CALIB, max (F1_kn + F1_en) / 2 strict.
                For selection each calib lecture is scored leave-one-lecture-out
                (LR and transition counts without it), as in step4_viterbi.run.
  test          decoded ONCE with the chosen settings.

Needs gg_finetune/scores/<run>/<lecture>_cal.npz (from calibrate_eval.py on Kaggle).

Usage:
    python scripts/cv_decode.py --smoke     # code check: random scores, scored on CALIB lectures, no test gold
    python scripts/cv_decode.py             # the one run -> results/cv_ft/
    python scripts/cv_decode.py --no_ft     # ablation: same protocol, 8 features, cue only -> results/cv_noft/
"""
import argparse
import csv
import itertools
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

from evalkit import LECTURES_NEW, load_csv, load_gold, read_split, score  # noqa: E402
from freeze_grounding import f1_strict, paired, strict_counts, with_ci  # noqa: E402
from phase2_decoder import column  # noqa: E402
from step3_scorer import FEATS, VERSE_COLS, model, verse_X  # noqa: E402
from step4_viterbi import (BETAS, GAMMAS, OFF, TAUS, V, gold_states, predict_locked,  # noqa: E402
                           transitions_resume, viterbi_resume)

FT = os.path.join(HERE, "gg_finetune")
COLS = VERSE_COLS + ("last",)
VARIANTS = ("cue", "ft", "mean")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--no_ft", action="store_true")
    args = ap.parse_args()
    out_dir = os.path.join(HERE, "results", "cv_smoke" if args.smoke else "cv_noft" if args.no_ft else "cv_ft")
    plan = json.load(open(os.path.join(FT, "folds.json"), encoding="utf-8"))
    z = np.load(FEATS)
    F = {k: z[k] for k in z.files}
    refs = list(F["refs"])
    row = {u: i for i, u in enumerate(F["unit_ids"])}
    lectures, gold, lang = load_gold()
    teacher = column("results/silver/silver_labels.csv", "soft_target")
    rows = {v: [row[u] for u in us] for v, us in lectures.items()}
    Y = {}
    for v, us in lectures.items():
        Y[v] = np.zeros((len(us), V), dtype=int)
        for k, u in enumerate(us):
            for r in gold[u]:
                Y[v][k, refs.index(r)] = 1
    seq = {v: gold_states(Y[v]) for v in lectures}
    rng = np.random.default_rng(0)

    def ft_scores(run, v):
        if args.smoke:
            return rng.normal(size=(len(rows[v]), V)), rng.uniform(0.02, 0.98, len(rows[v]))
        d = np.load(os.path.join(FT, "scores", run, f"{v}_cal.npz"))
        assert d["verse_feature"].shape == (len(rows[v]), V), f"{run}/{v}: shape {d['verse_feature'].shape}"
        return d["verse_feature"], np.clip(d["p_off_ft"], 0.02, 0.98)

    def decode(pv, po, logA, gamma, beta):
        e = (1 - po)[:, None] * pv / pv.sum(1, keepdims=True)
        emit = np.log(np.clip(np.column_stack([e, po]), 1e-12, 1))
        emit[:, OFF] += beta
        return viterbi_resume(emit, logA, gamma)

    def to_pred(v, pv, path, tau):
        return {u: [] if s == OFF else [refs[s]] + [refs[j] for j in np.flatnonzero(pv[k] >= tau) if j != s]
                for k, (u, s) in enumerate(zip(lectures[v], path))}

    pred, chosen = {}, {}
    for run in plan["runs"]:
        name, train, calib = run["name"], run["train"], run["calib"]
        test = calib if args.smoke else run["test"]
        assert not (set(train) | set(calib)) & set(run["test"]), "leak"
        ft = {} if args.no_ft else {v: ft_scores(name, v) for v in set(calib) | set(test)}

        def X(v):
            x = verse_X(F, rows[v], "a05", COLS)
            return x if args.no_ft else np.column_stack([x, ft[v][0].ravel()])

        def p_off(v, variant):
            cue = np.array([teacher[u] for u in lectures[v]])
            return cue if variant == "cue" else ft[v][1] if variant == "ft" else (cue + ft[v][1]) / 2

        Xc = {v: X(v) for v in calib}
        lolo, mats = {}, {}
        for w in calib:                                   # selection view: w left out
            rest = [v for v in calib if v != w]
            m = model().fit(np.concatenate([Xc[v] for v in rest]), np.concatenate([Y[v].ravel() for v in rest]))
            lolo[w] = m.predict_proba(Xc[w])[:, 1].reshape(-1, V)
            mats[w] = transitions_resume([seq[v] for v in train + rest])[0]
        best = None
        for variant in (("cue",) if args.no_ft else VARIANTS):
            po = {w: p_off(w, variant) for w in calib}
            for gamma, beta in itertools.product(GAMMAS, BETAS):
                paths = {w: decode(lolo[w], po[w], mats[w], gamma, beta) for w in calib}
                for tau in TAUS:
                    p = {}
                    for w in calib:
                        p.update(to_pred(w, lolo[w], paths[w], tau))
                    f = [score(gold, p, lectures, [w for w in calib if lang[w] == L])["F1"] for L in ("kn", "en")]
                    if best is None or sum(f) / 2 > best[0]:
                        best = (sum(f) / 2, variant, gamma, beta, tau)
        _, variant, gamma, beta, tau = best
        lr = model().fit(np.concatenate([Xc[v] for v in calib]), np.concatenate([Y[v].ravel() for v in calib]))
        logA, counts = transitions_resume([seq[v] for v in train + calib])
        for v in test:
            pv = lr.predict_proba(X(v))[:, 1].reshape(-1, V)
            pred.update(to_pred(v, pv, decode(pv, p_off(v, variant), logA, gamma, beta), tau))
        chosen[name] = dict(p_off=variant, gamma=gamma, beta_off=beta, tau=tau, calib_macro_f1=round(best[0], 4),
                            lr_weights=dict(zip(COLS + (() if args.no_ft else ("verse_feature",)),
                                                [round(float(x), 3) for x in lr[-1].coef_[0]])), counts=counts)
        print(f"  {name}: P_off={variant} gamma={gamma} beta_off={beta} tau={tau}  calib macro F1 {best[0]:.3f}")

    os.makedirs(out_dir, exist_ok=True)
    json.dump(chosen, open(os.path.join(out_dir, "chosen.json"), "w"), indent=1)
    with open(os.path.join(out_dir, "predictions.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["unit_id", "video_file", "refs"])
        for v in lectures:
            w.writerows([u, v, ";".join(pred[u])] for u in lectures[v] if u in pred)
    if args.smoke:
        print("  SMOKE run finished (random scores, calib lectures only). No test lecture was decoded.")
        return

    me = "CV, 8 features (no FT)" if args.no_ft else "CV system (FT)"
    vids = [v for s in ("test", "heldout") for L in ("kn", "en") for v in read_split(L, s)]
    dec = predict_locked({v: lectures[v] for v in vids}, "a05", path=os.path.join(HERE, "results", "v2_teacher145.pkl"))
    systems = {me: pred,
               "Frozen GitaGround-Cue": {u: ([] if st == OFF else [refs[st]]) + [refs[j] for j in extra]
                                         for v in vids for u, (st, extra, _, _) in zip(lectures[v], dec[v])}}
    for label, stem in (("GPT-6.1 Sol", "gpt61sol"), ("Opus 5.5", "opus55")):
        systems[label] = {r["unit_id"]: [x for x in r["refs"].split(";") if x]
                          for r in load_csv(os.path.join(HERE, "results", f"llm_frontier_{stem}.csv"))}
    groups = [("E2 Kannada test", read_split("kn", "test")), ("E2 English test", read_split("en", "test")),
              ("E3 new Kannada", read_split("kn", "heldout")), ("E3 new English", read_split("en", "heldout"))]
    lines = [f"Secondary protocol: {me}, 5-fold lecture CV + e3 run. Strict F1 [95% lecture bootstrap CI]",
             "P_off chosen per run: " + ", ".join(f"{k} {c['p_off']}" for k, c in chosen.items())]
    go = False
    for g, vs in groups:
        lines += ["", f"{g} ({len(vs)} lectures, {sum(len(lectures[v]) for v in vs)} passages)"]
        for sname, p in systems.items():
            point, ci = with_ci([strict_counts(gold, p, lectures[v]) for v in vs], f1_strict, 10000)
            lines.append(f"  {sname:24} {point:.3f}" + (f" [{ci[0]:.3f}, {ci[1]:.3f}]" if ci else ""))
        for other in ("Frozen GitaGround-Cue", "GPT-6.1 Sol"):
            d, ci, le0, (w, t, lo), pv = paired(gold, lectures, pred, systems[other], vs, 10000)
            lines.append(f"  paired CV - {other:22} {d:+.3f} [{ci[0]:+.3f}, {ci[1]:+.3f}]  P(<=0) {le0:.3f}  W/T/L {w}/{t}/{lo}")
            go |= g.startswith("E2") and other.endswith("Cue") and d >= 0.03
    if not args.no_ft:
        lines += ["", f"GO/NO-GO (CV >= Cue + 0.03 strict F1 on an E2 test language): {'GO' if go else 'NO-GO'}"]
    out = "\n".join(lines)
    print("\n" + out)
    open(os.path.join(out_dir, "report.txt"), "w", encoding="utf-8").write(out + "\n")


if __name__ == "__main__":
    main()
