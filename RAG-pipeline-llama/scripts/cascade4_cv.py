#!/usr/bin/env python3
"""
Four-layer override cascade with the keep rule decided by leave-one-lecture-out cross-validation
over the same-speaker lectures (9 tuning + 30 E2), instead of the 9 tuning lectures alone.
No API call, stored labels only.

  pool     per language: Kannada 6 tuning + 23 E2 = 29 lectures; English 3 tuning + 7 E2 = 10
  layers   as cascade4_tuning.py: 1 anchor -> anchor's verses (Kannada only); 2 Cue conf >= 0.9 ->
           Cue; 3 gpt-5-mini equals Cue -> that label; 4 the frontier LLM's whole-lecture label
  change   a passage gpt-5-mini did not answer abstains at layer 3 (before: counted as NONE)
  CV       for each held-out lecture, each layer is kept iff on the OTHER pool lectures of its
           language it is right (set equals gold) on at least as many of the passages it would
           accept as the LLM (layers in order, as 65dbb40); then applied to the held-out lecture
  Cue      tuning lectures: LOLO labels (as cascade_tuning.load); other lectures: predict_locked
           with v2_teacher145.pkl; both out of sample for the lecture scored
  E3       layers decided on the whole pool of its language (secondary: already seen once)
  report   strict F1 [95% lecture CI] LLM alone vs cascade on the pool (and its E2 part), paired
           lecture bootstrap, W/T/L; how often each layer was kept across folds; E3

-> results/cascade4_cv.txt

Usage:
    python scripts/cascade4_cv.py
"""
import os
import sys
from collections import Counter

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "scripts"))

import numpy as np  # noqa: E402
from cascade4_tuning import L1, L2, L3, L4, MINI  # noqa: E402
from cascade_frontier_tuning import LLMS as TUNING_LLMS  # noqa: E402
from cascade_tuning import CONF, NO_ANCHOR_LAYER, ids_of, load  # noqa: E402
from evalkit import load_csv, read_split  # noqa: E402
from freeze_grounding import f1_strict, paired, strict_counts, with_ci  # noqa: E402
from llm_frontier import parse  # noqa: E402
from recall_by_type import read_llm  # noqa: E402
from step4_viterbi import FEATS, OFF, predict_locked  # noqa: E402

N_BOOT = 10000
TEST_CSV = {"Sol": "llm_frontier_gpt61sol.csv", "Opus": "llm_frontier_opus55.csv",
            "Gemini": "llm_frontier_gemini31pro_api_low.csv"}


def main():
    gold, lectures, tune, cue, _, conf, anchor = load()          # tuning: LOLO Cue, its confidence, anchors
    e2 = {L: read_split(L, "test") for L in ("kn", "en")}
    e3 = {L: read_split(L, "heldout") for L in ("kn", "en")}
    pool = {L: tune[L] + e2[L] for L in ("kn", "en")}
    others = [v for L in ("kn", "en") for v in e2[L] + e3[L]]
    z = np.load(FEATS)
    refs = list(z["refs"])
    row = {u: i for i, u in enumerate(z["unit_ids"])}
    dec = predict_locked({v: lectures[v] for v in others}, "a05", path=os.path.join(HERE, "results", "v2_teacher145.pkl"))
    for v in others:
        for u, (st, extra, p_state, _) in zip(lectures[v], dec[v]):
            cue[u] = ([] if st == OFF else [refs[st]]) + [refs[j] for j in extra]   # as freeze_grounding
            conf[u] = p_state
            anchor[u] = [refs[j] for j in np.flatnonzero(z["anchor"][row[u]])]
    mini = {}                                                    # missing passage -> absent (abstains)
    for v in [v for L in ("kn", "en") for v in pool[L] + e3[L]]:
        ids = [(r["passage_id"], r["unit_id"]) for r in load_csv(os.path.join(MINI, f"{v}_ids.csv"))]
        p, _ = parse(os.path.join(MINI, f"{v}.txt"), ids)
        mini.update({u: p[u] for u in lectures[v] if u in p})
    take = {L1: (lambda u: bool(anchor[u]), anchor), L2: (lambda u: conf[u] >= CONF, cue),
            L3: (lambda u: u in mini and set(mini[u]) == set(cue[u]), cue)}
    ok = lambda p, u: set(p.get(u, ())) == set(gold[u])  # noqa: E731

    def decide(L, vs, llm):
        """layers kept, deciding on lectures vs (keep rule of 65dbb40, in order)"""
        kept, left = [], [u for v in vs for u in lectures[v]]
        for la in (L1, L2, L3):
            if la == L1 and L in NO_ANCHOR_LAYER:
                continue
            acc = [u for u in left if take[la][0](u)]
            if sum(ok(take[la][1], u) for u in acc) >= sum(ok(llm, u) for u in acc):
                kept.append(la)
                left = [u for u in left if not take[la][0](u)]
        return tuple(kept)

    def apply(kept, vs, llm, out, layer):
        for v in vs:
            for u in lectures[v]:
                la = next((x for x in kept if take[x][0](u)), L4)
                layer[u], out[u] = la, (take[la][1][u] if la != L4 else llm.get(u, []))

    lines = ["Four-layer cascade, keep rule by leave-one-lecture-out CV over the same-speaker lectures "
             "(9 tuning + 30 E2); gpt-5-mini missing passages abstain at layer 3",
             f"strict F1 [95% lecture CI], paired lecture bootstrap {N_BOOT}"]
    summary = []
    civ = lambda c: f" [{c[0]:.3f},{c[1]:.3f}]" if c else ""  # noqa: E731
    for name, folder in TUNING_LLMS:
        llm = read_llm(TEST_CSV[name])
        for L in ("kn", "en"):
            for v in tune[L]:
                p, _ = parse(os.path.join(folder, f"{v}.txt"), ids_of(v))
                llm.update({u: p.get(u, []) for u in lectures[v]})
        lines += ["", f"===== layer 4 = {name} ====="]
        for L, lname in (("kn", "Kannada"), ("en", "English")):
            out, layer, folds = {}, {}, Counter()
            for h in pool[L]:
                kept = decide(L, [w for w in pool[L] if w != h], llm)
                folds[kept] += 1
                apply(kept, [h], llm, out, layer)
            full = decide(L, pool[L], llm)
            apply(full, e3[L], llm, out, layer)
            lines += ["", f"  {lname}: layers kept per fold: " + ", ".join(
                f"{' > '.join(k) or 'none'} ({n}/{len(pool[L])})" for k, n in folds.most_common())
                + f"; on the whole pool (used for E3): {' > '.join(full) or 'none'}"]
            for part, vs in ((f"CV, all {len(pool[L])} same-speaker", pool[L]), (f"CV, E2 part ({len(e2[L])})", e2[L]),
                             (f"CV, tuning part ({len(tune[L])})", tune[L]), (f"E3 ({len(e3[L])}, secondary)", e3[L])):
                us = [u for v in vs for u in lectures[v]]
                fa, ca = with_ci([strict_counts(gold, llm, lectures[v]) for v in vs], f1_strict, N_BOOT)
                fc, cc = with_ci([strict_counts(gold, out, lectures[v]) for v in vs], f1_strict, N_BOOT)
                d, ci, _, wtl, p = paired(gold, lectures, out, llm, vs, N_BOOT)
                reach = sum(layer[u] == L4 for u in us)
                lines.append(f"    {part:26} {name} {fa:.3f}{civ(ca)}  cascade {fc:.3f}{civ(cc)}  "
                             f"diff {d:+.3f} [{ci[0]:+.3f},{ci[1]:+.3f}]  W/T/L {wtl[0]}/{wtl[1]}/{wtl[2]}  "
                             f"right {sum(ok(llm, u) for u in us)}->{sum(ok(out, u) for u in us)}/{len(us)}  "
                             f"to {name} {reach / len(us):.0%}")
                if part.startswith("CV, all"):
                    summary.append(f"  {name:7} {lname:8} CV {len(vs)} lectures: alone {fa:.3f}  cascade {fc:.3f}  "
                                   f"diff {d:+.3f} [{ci[0]:+.3f},{ci[1]:+.3f}]  W/T/L {wtl[0]}/{wtl[1]}/{wtl[2]}")
            for la in (L1, L2, L3):
                a = [u for v in pool[L] for u in lectures[v] if layer[u] == la]
                if a:
                    lines.append(f"      CV pool, {la:22} decided {len(a):>4}  right {sum(ok(out, u) for u in a):>4}  "
                                 f"({name} alone {sum(ok(llm, u) for u in a)})")
    lines += ["", "SUMMARY (CV over the same-speaker pool)"] + summary
    text = "\n".join(lines)
    print(text)
    with open(os.path.join(HERE, "results", "cascade4_cv.txt"), "w", encoding="utf-8") as f:
        f.write(text + "\n")


if __name__ == "__main__":
    main()
