#!/usr/bin/env python3
"""
Override cascade with each frontier model, TUNING lectures only (no test gold, no API call).
Spec in docs/experiment_log.md.

  the LLM labels the whole lecture (its existing tuning answers); a cheap layer overrides it on the
  passages it accepts, with the layers and keep rule of cascade_tuning.py (65dbb40): KN anchor, then
  Cue conf >= 0.9; EN Cue conf >= 0.9 only (no anchor layer); a layer is kept only if, on that
  language's tuning lectures, it is right on at least as many of its passages as THIS LLM
  LLMs  Gemini 3.1 Pro (API, low), GPT-6.1 Sol (API, high), Claude Opus 5.5 (chat, high)

-> results/cascade_frontier_tuning.txt

Usage:
    python scripts/cascade_frontier_tuning.py
"""
import os
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "scripts"))

from cascade_tuning import cascade, ids_of, load, report  # noqa: E402
from llm_frontier import DIR, parse  # noqa: E402

LLMS = (("Gemini", os.path.join(DIR, "answers_api_gemini-3.1-pro-preview_low", "tuning")),
        ("Sol", os.path.join(DIR, "answers_gpt-6.1-sol", "tuning")),
        ("Opus", os.path.join(DIR, "tuning", "answers")))


def main():
    gold, lectures, tune, cue, _, conf, anchor = load()
    lines = ["Override cascade per frontier model, TUNING lectures only: the LLM's whole-lecture labels, "
             "overridden where a kept cheap layer accepts the passage (layers and keep rule of 65dbb40)"]
    for name, folder in LLMS:
        llm = {}
        for L in tune:
            for v in tune[L]:
                p, _ = parse(os.path.join(folder, f"{v}.txt"), ids_of(v))
                llm.update({u: p.get(u, []) for u in lectures[v]})
        layer, label, keep = cascade(gold, lectures, tune, cue, llm, conf, anchor)
        out, _ = report(gold, lectures, tune, layer, label, llm, llm,
                        f"=== {name}: layer 3 = {name}'s whole-lecture labels ('Gemini' columns = {name}) ===")
        lines += out[:2] + ["  keep rule:"] + keep + out[2:]
    text = "\n".join(lines)
    print(text)
    with open(os.path.join(HERE, "results", "cascade_frontier_tuning.txt"), "w", encoding="utf-8") as f:
        f.write(text + "\n")


if __name__ == "__main__":
    main()
