#!/usr/bin/env python3
"""
Strict F1 lecture by lecture and pooled, on the 9 tuning + 42 test lectures, from stored labels only:
Opus 5.5, GPT-6.1 Sol, Gemini 3.1 Pro (API, low), GitaGround-Cue, GitaGround-Local, and the 3-layer and
4-layer override cascades with the designs frozen on tuning (eb5b08b; cascade4_test.FROZEN).

  tuning lectures  LLMs: their tuning answers; Cue and Local: leave-one-lecture-out (step4_viterbi.run);
                   cascades: in sample (their layers were chosen on these lectures)
  test lectures    LLMs: results/llm_frontier_*.csv; Cue: predict_locked v2_teacher145.pkl; Local:
                   predict_locked step4_locked.pkl; cascades exactly as cascade4_test.py
  3-layer          layers 1 anchor (KN) and 2 Cue conf >= 0.9 that each LLM kept on tuning, then the LLM
  4-layer          as cascade4_test.FROZEN (adds 3 gpt-5-mini equals Cue)

-> results/all51_table.txt

Usage:
    python scripts/all51_table.py
"""
import os
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "scripts"))

import numpy as np  # noqa: E402
from cascade4_test import FROZEN  # noqa: E402
from cascade4_tuning import L1, L2, L3, L4, MINI  # noqa: E402
from cascade_frontier_tuning import LLMS as TUNING_LLMS  # noqa: E402
from cascade_tuning import CONF, ids_of, load  # noqa: E402
from evalkit import load_csv, read_split  # noqa: E402
from freeze_grounding import f1_strict, strict_counts  # noqa: E402
from llm_frontier import parse  # noqa: E402
from recall_by_type import read_llm  # noqa: E402
from step4_viterbi import FEATS, OFF, OUT, predict_locked, run  # noqa: E402

CSV = {"Sol": "llm_frontier_gpt61sol.csv", "Opus": "llm_frontier_opus55.csv",
       "Gemini": "llm_frontier_gemini31pro_api_low.csv"}
THREE = {k: tuple(x for x in v if x != L3) for k, v in FROZEN.items()}   # = cascade_frontier_tuning keeps


def main():
    gold, lectures, tune, cue, _, conf, anchor = load()
    order, *_ = run(alphas=("a05",), verbose=False)
    local = dict(dict(order)["Viterbi (alpha 0.5)"])
    groups = [("Tuning Kannada", "kn", tune["kn"]), ("Tuning English", "en", tune["en"]),
              ("E2 Kannada test", "kn", read_split("kn", "test")), ("E2 English test", "en", read_split("en", "test")),
              ("E3 Kannada", "kn", read_split("kn", "heldout")), ("E3 English", "en", read_split("en", "heldout"))]
    test = [v for _, _, vs in groups[2:] for v in vs]
    lang = {v: L for _, L, vs in groups for v in vs}
    z = np.load(FEATS)
    refs = list(z["refs"])
    row = {u: i for i, u in enumerate(z["unit_ids"])}
    for path, target in ((os.path.join(HERE, "results", "v2_teacher145.pkl"), "cue"), (OUT, "local")):
        dec = predict_locked({v: lectures[v] for v in test}, "a05", path=path)
        for v in test:
            for u, (st, extra, p_state, _) in zip(lectures[v], dec[v]):
                lab = ([] if st == OFF else [refs[st]]) + [refs[j] for j in extra]   # as freeze_grounding
                if target == "cue":
                    cue[u], conf[u] = lab, p_state
                    anchor[u] = [refs[j] for j in np.flatnonzero(z["anchor"][row[u]])]
                else:
                    local[u] = lab
    mini = {}
    for _, _, vs in groups:
        for v in vs:
            ids = [(r["passage_id"], r["unit_id"]) for r in load_csv(os.path.join(MINI, f"{v}_ids.csv"))]
            p, _ = parse(os.path.join(MINI, f"{v}.txt"), ids)
            mini.update({u: p.get(u, []) for u in lectures[v]})       # missing = NONE, as frozen
    llm = {}
    for name, folder in TUNING_LLMS:
        llm[name] = read_llm(CSV[name])
        for v in tune["kn"] + tune["en"]:
            p, _ = parse(os.path.join(folder, f"{v}.txt"), ids_of(v))
            llm[name].update({u: p.get(u, []) for u in lectures[v]})
    take = {L1: (lambda u: bool(anchor[u]), anchor), L2: (lambda u: conf[u] >= CONF, cue),
            L3: (lambda u: set(mini[u]) == set(cue[u]), cue)}

    def cascade(name, layers_of):
        out = {}
        for v, L in lang.items():
            for u in lectures[v]:
                la = next((x for x in layers_of[(name, L)] if take[x][0](u)), L4)
                out[u] = take[la][1][u] if la != L4 else llm[name].get(u, [])
        return out

    S = {"Opus": llm["Opus"], "Sol": llm["Sol"], "Gemini": llm["Gemini"], "Cue": cue, "Local": local}
    for name in ("Sol", "Opus", "Gemini"):
        S[f"3L {name}"] = cascade(name, THREE)
        S[f"4L {name}"] = cascade(name, FROZEN)
    f1 = lambda p, vs: f1_strict([sum(x) for x in zip(*(strict_counts(gold, p, lectures[v]) for v in vs))])  # noqa: E731

    per = ["Opus", "Sol", "Gemini", "Cue", "Local", "3L Sol", "4L Sol"]
    lines = ["Strict F1, 9 tuning + 42 test lectures (tuning: Cue/Local leave-one-lecture-out, cascades in sample)",
             "", "1. LECTURE BY LECTURE (cascades with Sol at the last layer; best per row marked *)", "",
             f"  {'lecture':26}{'psg':>4}" + "".join(f"{c:>8}" for c in per)]
    for g, _, vs in groups:
        lines.append(f"  -- {g} --")
        for v in vs:
            vals = [f1(S[c], [v]) for c in per]
            best = max(vals)
            lines.append(f"  {v:26}{len(lectures[v]):>4}" + "".join(
                f"{x:>7.3f}{'*' if abs(x - best) < 1e-9 else ' '}" for x in vals))
    cols = list(S)
    lines += ["", "2. OVERVIEW, pooled strict F1 per group", "",
              f"  {'group':18}{'lect':>5}" + "".join(f"{c:>10}" for c in cols)]
    for g, _, vs in groups:
        lines.append(f"  {g:18}{len(vs):>5}" + "".join(f"{f1(S[c], vs):>10.3f}" for c in cols))
    t42 = [v for _, _, vs in groups[2:] for v in vs]
    lines.append(f"  {'all 42 test':18}{len(t42):>5}" + "".join(f"{f1(S[c], t42):>10.3f}" for c in cols))
    text = "\n".join(lines)
    print(text)
    with open(os.path.join(HERE, "results", "all51_table.txt"), "w", encoding="utf-8") as f:
        f.write(text + "\n")


if __name__ == "__main__":
    main()
