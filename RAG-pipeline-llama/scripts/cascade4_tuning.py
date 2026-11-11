#!/usr/bin/env python3
"""
Four-layer override cascade, TUNING lectures only (no test gold, no API call).

  layer 1  string anchor fired -> the anchor's verses           (not used in English, as 65dbb40)
  layer 2  Cue (v2 primary, LOLO) scorer probability for its own label >= 0.9 -> Cue's label
  layer 3  gpt-5-mini's whole-lecture label (review/llm_e2e/gpt-5-mini, Phase 3 run) equals Cue's
           label -> that label
  layer 4  the frontier LLM's whole-lecture label (Gemini 3.1 Pro API low, GPT-6.1 Sol, Opus 5.5)
  keep rule as 65dbb40, per language and per LLM: a layer is kept only if it is right (set equals
  gold) on at least as many of the passages it would accept as the frontier LLM; layers checked in
  order, each on what the kept layers before it left. Also reported: layers 1-2 only (3-layer).

-> results/cascade4_tuning.txt

Usage:
    python scripts/cascade4_tuning.py
"""
import os
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "scripts"))

from cascade_frontier_tuning import LLMS  # noqa: E402
from cascade_tuning import CONF, NO_ANCHOR_LAYER, ids_of, load  # noqa: E402
from evalkit import score  # noqa: E402
from llm_frontier import parse  # noqa: E402

MINI = "/Users/maitreyi/Documents/Capstone/review/llm_e2e/gpt-5-mini"
L1, L2, L3, L4 = "1 anchor", f"2 Cue conf>={CONF}", "3 mini agrees w/ Cue", "4 frontier"


def labels(folder, tune, lectures):
    out, missing = {}, 0
    for L in tune:
        for v in tune[L]:
            p, prob = parse(os.path.join(folder, f"{v}.txt"), ids_of(v))
            missing += len(prob["missing"])
            out.update({u: p.get(u, []) for u in lectures[v]})
    return out, missing


def candidates(anchor, cue, conf, mini):
    """layer -> (accepts passage?, label source)"""
    return {L1: (lambda u: bool(anchor[u]), anchor), L2: (lambda u: conf[u] >= CONF, cue),
            L3: (lambda u: set(mini[u]) == set(cue[u]), cue)}


def assign(L, us, llm, name, gold, cand):
    """keep rule per layer, in order -> (layer per passage, label per passage, keep-rule lines)"""
    ok = lambda p, u: set(p[u]) == set(gold[u])  # noqa: E731
    left, layer, out, lines = list(us), {}, {}, []
    for la in (L1, L2, L3):
        if la == L1 and L in NO_ANCHOR_LAYER:
            lines.append(f"    {la:22} not used in English")
            continue
        take, src = cand[la]
        acc = [u for u in left if take(u)]
        a, g = sum(ok(src, u) for u in acc), sum(ok(llm, u) for u in acc)
        keep = a >= g
        lines.append(f"    {la:22} would accept {len(acc):>3}: right {a:>3} vs {name} {g:>3} -> "
                     f"{'KEEP' if keep else 'drop'}")
        if keep:
            for u in acc:
                layer[u], out[u] = la, src[u]
            left = [u for u in left if u not in layer]
    for u in left:
        layer[u], out[u] = L4, llm[u]
    return layer, out, lines


def main():
    gold, lectures, tune, cue, _, conf, anchor = load()
    mini, mini_missing = labels(MINI, tune, lectures)
    cand = candidates(anchor, cue, conf, mini)
    lines = [f"Four-layer override cascade, TUNING lectures only; gpt-5-mini passages missing {mini_missing} (= NONE)"]
    summary = []
    for name, folder in LLMS:
        llm, _ = labels(folder, tune, lectures)
        lines += ["", f"=== layer 4 = {name} (whole-lecture labels) ==="]
        for L, lname in (("kn", "Kannada"), ("en", "English")):
            us = [u for v in tune[L] for u in lectures[v]]
            lines.append(f"  {lname}: {len(us)} passages")
            layer, out, keep = assign(L, us, llm, name, gold, cand)
            lines += keep
            three = {u: out[u] if layer[u] in (L1, L2) else llm[u] for u in us}
            reach = sum(layer[u] == L4 for u in us)
            f4, f3, fl = (score(gold, p, lectures, tune[L]) for p in (out, three, llm))
            lines.append(f"    reaching {name}: {reach}/{len(us)} ({reach / len(us):.0%})")
            for lab, f in ((f"{name} alone", fl), ("3-layer (1-2 + frontier)", f3), ("4-layer (1-3 + frontier)", f4)):
                lines.append(f"    {lab:26} strict F1 {f['F1']:.3f}  (tp {f['tp']} fp {f['fp']} fn {f['fn']})")
            lines.append(f"    per lecture, {name} alone / 3-layer / 4-layer: " + ", ".join(
                f"{v} {score(gold, llm, lectures, [v])['F1']:.3f} / {score(gold, three, lectures, [v])['F1']:.3f} / "
                f"{score(gold, out, lectures, [v])['F1']:.3f}" for v in tune[L]))
            summary.append(f"  {name:7} {L}: alone {fl['F1']:.3f}  3-layer {f3['F1']:.3f} ({f3['F1'] - fl['F1']:+.3f})  "
                           f"4-layer {f4['F1']:.3f} ({f4['F1'] - fl['F1']:+.3f})")
    lines += ["", "SUMMARY, strict F1 on tuning"] + summary
    text = "\n".join(lines)
    print(text)
    with open(os.path.join(HERE, "results", "cascade4_tuning.txt"), "w", encoding="utf-8") as f:
        f.write(text + "\n")


if __name__ == "__main__":
    main()
