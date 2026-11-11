#!/usr/bin/env python3
"""
Free check, ENGLISH TUNING lectures only (no test gold, no API call): does disagreement between two
whole-lecture Gemini runs predict Gemini's errors, and is Cue right where they disagree?
Spec in docs/experiment_log.md.

  run A   Gemini 3.1 Pro (API, low), whole lecture, original prompt (answers_api_..._low/tuning)
  run B   the same model, whole lecture, every passage [LABEL], v2 prompt wording
          (answers_api_..._low/tuning_window_control); the wording differs from run A, so this is
          two runs on near-identical input, not two identical calls
  Cue     v2 primary, LOLO; "confident" = scorer probability for its own label >= 0.9 (c55e0ff)
  right   set equals gold
  rule    (simulation) A and B agree -> A; disagree and Cue confident -> Cue; else -> A

-> results/selfconsistency_check_tuning.txt

Usage:
    python scripts/selfconsistency_check_tuning.py
"""
import os
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "scripts"))

from cascade_tuning import CONF, RUNS, answers_of, load  # noqa: E402
from evalkit import score  # noqa: E402


def main():
    gold, lectures, tune, cue, gem, conf, anchor = load()
    vs = tune["en"]
    us = [u for v in vs for u in lectures[v]]
    b, missing = answers_of(RUNS["control"][1], vs, lectures, set(us))
    ok = lambda p, u: set(p[u]) == set(gold[u])  # noqa: E731
    agree = [u for u in us if set(gem[u]) == set(b[u])]
    dis = [u for u in us if set(gem[u]) != set(b[u])]
    rule = {u: gem[u] if u in agree else cue[u] if conf[u] >= CONF else gem[u] for u in us}
    lines = [f"Self-consistency check, English tuning ({len(us)} passages, {len(vs)} lectures); "
             f"run B missing {missing}",
             "",
             f"  runs agree     {len(agree):>4} ({len(agree) / len(us):.0%})   run A right {sum(ok(gem, u) for u in agree)}",
             f"  runs disagree  {len(dis):>4} ({len(dis) / len(us):.0%})   run A right {sum(ok(gem, u) for u in dis)}, "
             f"run B right {sum(ok(b, u) for u in dis)}, Cue right {sum(ok(cue, u) for u in dis)}",
             f"  disagree, Cue confident: {sum(conf[u] >= CONF for u in dis)}   Cue right "
             f"{sum(ok(cue, u) for u in dis if conf[u] >= CONF)}, run A right "
             f"{sum(ok(gem, u) for u in dis if conf[u] >= CONF)}",
             "",
             "  share right: agree " + f"{sum(ok(gem, u) for u in agree) / len(agree):.0%}" +
             f", disagree {sum(ok(gem, u) for u in dis) / len(dis):.0%} (run A)",
             "",
             f"  strict F1 per lecture{'':6}{'run A':>8}{'run B':>8}{'rule':>8}"]
    for v in vs + ["POOLED"]:
        vv = vs if v == "POOLED" else [v]
        lines.append(f"    {v:25}" + "".join(f"{score(gold, p, lectures, vv)['F1']:>8.3f}" for p in (gem, b, rule)))
    text = "\n".join(lines)
    print(text)
    with open(os.path.join(HERE, "results", "selfconsistency_check_tuning.txt"), "w", encoding="utf-8") as f:
        f.write(text + "\n")


if __name__ == "__main__":
    main()
