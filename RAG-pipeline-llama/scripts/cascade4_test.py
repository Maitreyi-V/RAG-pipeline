#!/usr/bin/env python3
"""
Four-layer override cascade on the 42 lectures, design frozen from the tuning lectures
(cascade4_tuning.py, eb5b08b): no API call, stored labels only.

  layers   1 string anchor fired -> the anchor's verses
           2 Cue (v2 primary, predict_locked with v2_teacher145.pkl) probability for its own label
             >= 0.9 -> Cue's label (the decoded state's emission probability, as on tuning)
           3 gpt-5-mini's whole-lecture label (Phase 3 run) equals Cue's label -> that label
           4 the frontier LLM's whole-lecture label (results/llm_frontier_*.csv)
  frozen   the layers each LLM kept on tuning, per language (FROZEN below); nothing is re-chosen
  splits   E2 Kannada test, E2 English test, E3 Kannada, E3 English
  report   strict F1 [95% lecture CI] of the LLM alone and the cascade; paired lecture bootstrap of
           cascade - LLM alone (10,000 resamples), W/T/L, sign-test p; passages decided and right per
           layer; passages reaching the LLM; per-lecture F1

  --split  test42 (default) | tuning_b (the 5 tuning-B lectures, pick_tuning_b.py) | rest37 (the
           42 without tuning B)
  --layers 4 (default) | 3: FROZEN without layer 3 (gpt-5-mini), i.e. the 3-layer cascade (THREE)

-> results/cascade4_test.txt | cascade4_tuning_b.txt | cascade4_rest37.txt (--layers 3: cascade3_*)

Usage:
    python scripts/cascade4_test.py [--split test42 | tuning_b | rest37] [--layers 4 | 3]
"""
import argparse
import os
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "scripts"))

import numpy as np  # noqa: E402
from cascade4_tuning import L1, L2, L3, L4, MINI  # noqa: E402
from cascade_tuning import CONF  # noqa: E402
from evalkit import load_csv, load_gold, read_split  # noqa: E402
from freeze_grounding import f1_strict, paired, strict_counts, with_ci  # noqa: E402
from llm_frontier import parse  # noqa: E402
from pick_tuning_b import tuning_b  # noqa: E402
from recall_by_type import read_llm  # noqa: E402
from step4_viterbi import FEATS, OFF, predict_locked  # noqa: E402

N_BOOT = 10000
LLMS = (("Sol", "llm_frontier_gpt61sol.csv"), ("Opus", "llm_frontier_opus55.csv"),
        ("Gemini", "llm_frontier_gemini31pro_api_low.csv"))
FROZEN = {("Sol", "kn"): (L1, L2), ("Sol", "en"): (L3,),          # from results/cascade4_tuning.txt
          ("Opus", "kn"): (L1, L2), ("Opus", "en"): (L2, L3),
          ("Gemini", "kn"): (L1, L2), ("Gemini", "en"): (L2, L3)}
THREE = {k: tuple(x for x in v if x != L3) for k, v in FROZEN.items()}   # = cascade_frontier_tuning keeps


OUTFILE = {"test42": "cascade4_test.txt", "tuning_b": "cascade4_tuning_b.txt", "rest37": "cascade4_rest37.txt"}


def splits_for(which):
    full = [("E2 Kannada test", "kn", read_split("kn", "test")), ("E2 English test", "en", read_split("en", "test")),
            ("E3 Kannada", "kn", read_split("kn", "heldout")), ("E3 English", "en", read_split("en", "heldout"))]
    if which == "test42":
        return full
    tb = tuning_b()
    if which == "tuning_b":
        return [("tuning B Kannada", "kn", tb["kn"]), ("tuning B English", "en", tb["en"])]
    return [(n, L, [v for v in vs if v not in tb[L]]) for n, L, vs in full]


def main(which="test42", n_layers=4):
    design = FROZEN if n_layers == 4 else THREE
    lectures, gold, lang = load_gold()
    splits = splits_for(which)
    vids = [v for _, _, vs in splits for v in vs]
    z = np.load(FEATS)
    refs = list(z["refs"])
    row = {u: i for i, u in enumerate(z["unit_ids"])}
    anchor = {u: [refs[j] for j in np.flatnonzero(z["anchor"][row[u]])] for v in vids for u in lectures[v]}
    dec = predict_locked({v: lectures[v] for v in vids}, "a05", path=os.path.join(HERE, "results", "v2_teacher145.pkl"))
    cue, conf = {}, {}
    for v in vids:
        for u, (st, extra, p_state, _) in zip(lectures[v], dec[v]):
            cue[u] = ([] if st == OFF else [refs[st]]) + [refs[j] for j in extra]   # as freeze_grounding
            conf[u] = p_state
    mini, mini_missing = {}, 0
    for v in vids:
        ids = [(r["passage_id"], r["unit_id"]) for r in load_csv(os.path.join(MINI, f"{v}_ids.csv"))]
        p, prob = parse(os.path.join(MINI, f"{v}.txt"), ids)
        mini_missing += len(prob["missing"])
        mini.update({u: p.get(u, []) for u in lectures[v]})
    take = {L1: (lambda u: bool(anchor[u]), anchor), L2: (lambda u: conf[u] >= CONF, cue),
            L3: (lambda u: set(mini[u]) == set(cue[u]), cue)}
    ok = lambda p, u: set(p.get(u, ())) == set(gold[u])  # noqa: E731

    lines = [f"{'Four' if n_layers == 4 else 'Three'}-layer override cascade on the 42 lectures, design frozen "
             f"from tuning (eb5b08b{'' if n_layers == 4 else '; layer 3 gpt-5-mini removed'}); "
             f"gpt-5-mini passages missing {mini_missing} (= NONE); strict F1 [95% lecture CI], "
             f"paired lecture bootstrap {N_BOOT}"]
    summary = []
    for name, csvname in LLMS:
        llm = read_llm(csvname)
        lines += ["", f"===== layer 4 = {name} ====="]
        for sname, L, vs in splits:
            us = [u for v in vs for u in lectures[v]]
            layer, out = {}, {}
            for u in us:
                la = next((x for x in design[(name, L)] if take[x][0](u)), L4)
                layer[u], out[u] = la, (take[la][1][u] if la != L4 else llm.get(u, []))
            fa, ca = with_ci([strict_counts(gold, llm, lectures[v]) for v in vs], f1_strict, N_BOOT)
            fc, cc = with_ci([strict_counts(gold, out, lectures[v]) for v in vs], f1_strict, N_BOOT)
            d, ci, _, wtl, p = paired(gold, lectures, out, llm, vs, N_BOOT)
            civ = lambda c: f" [{c[0]:.3f},{c[1]:.3f}]" if c else ""  # noqa: E731
            reach = sum(layer[u] == L4 for u in us)
            lines += ["", f"  {sname}: {len(vs)} lectures, {len(us)} passages; layers {' > '.join(design[(name, L)])} > {L4}",
                      f"    {name} alone   strict F1 {fa:.3f}{civ(ca)}   passages right {sum(ok(llm, u) for u in us)}/{len(us)}",
                      f"    cascade      strict F1 {fc:.3f}{civ(cc)}   passages right {sum(ok(out, u) for u in us)}/{len(us)}",
                      f"    cascade - {name}: {d:+.3f} [{ci[0]:+.3f},{ci[1]:+.3f}]  W/T/L {wtl[0]}/{wtl[1]}/{wtl[2]}  sign p {p:.3f}",
                      f"    reaching {name}: {reach}/{len(us)} ({reach / len(us):.0%})"]
            for la in design[(name, L)] + (L4,):
                a = [u for u in us if layer[u] == la]
                lines.append(f"      {la:22} decided {len(a):>4}  right {sum(ok(out, u) for u in a):>4}  "
                             f"({name} alone right on these {sum(ok(llm, u) for u in a)})")
            lines.append("    per lecture, alone / cascade: " + ", ".join(
                f"{v} {f1_strict(strict_counts(gold, llm, lectures[v])):.3f}/{f1_strict(strict_counts(gold, out, lectures[v])):.3f}"
                for v in vs))
            summary.append(f"  {name:7} {sname:16} alone {fa:.3f}  cascade {fc:.3f}  diff {d:+.3f} "
                           f"[{ci[0]:+.3f},{ci[1]:+.3f}]  W/T/L {wtl[0]}/{wtl[1]}/{wtl[2]}")
    lines += ["", "SUMMARY"] + summary
    text = "\n".join(lines)
    print(text)
    name_out = OUTFILE[which] if n_layers == 4 else OUTFILE[which].replace("cascade4", "cascade3")
    with open(os.path.join(HERE, "results", name_out), "w", encoding="utf-8") as f:
        f.write(text + "\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=list(OUTFILE), default="test42")
    ap.add_argument("--layers", type=int, choices=[4, 3], default=4)
    args = ap.parse_args()
    main(args.split, args.layers)
