#!/usr/bin/env python3
"""
E7: retrieval and abstention with the LOCKED GitaGround (gitaground.py).

Two steps, always in this order:

  1. calibrate  (DEVELOPMENT questions only)
       answerable   the 74 dev questions that have a relevant passage in the index
       unanswerable data/e7_dev_unanswerable.csv (12, team-written)
     Two candidate signals (best passage cosine in the index; best meaning-profile
     cosine). For each, the threshold with the best balanced accuracy
     ((answered answerable + abstained unanswerable) / 2); the better signal wins.
     Saved to results/e7_abstain.json together with the index it was set on.
     LOCKED 17 Oct on the tuning index: signal "profile", which does not depend
     on the index, so it is NOT recalibrated at the freeze. (Had "passage" won, it
     would have to be re-set on the full index before the test questions.)

  2. evaluate
       --questions dev    DRY RUN: the dev questions (the threshold was set on
                          them, so abstention here is in-sample)
       --questions test   data/e7_questions_150.csv, ONCE, at the freeze

Reported:
  (a) verse Hit@1 / Hit@3 (verse number, shloka match, then meaning profile)
  (b) passage Hit@5 and P@5: locked GitaGround vs plain vector search over the
      same passages ("fixed-chunk vector RAG"; our passages are the chunks)
  (c) primary vs passing precision: of the retrieved passages the index marks as
      explaining the answered verse (primary) or only mentioning it (secondary),
      the share whose gold labels contain that verse
  (d) unsupported topics: every verse shown comes from the 72-verse catalogue and
      every passage from the index, so 0 by construction (the optional LLM
      summary text is not evaluated)
  (e) abstention: share of unanswerable questions abstained; share of answerable
      questions wrongly abstained. Plain vector search never abstains.
  per input mode (question / shloka / explanation) where the file has one.
The "LLM-extracted topics" baseline from the plan is not built.

Usage:
    python scripts/freeze_e7.py --index dev --calibrate      # done 17 Oct
    python scripts/freeze_e7.py --index dev --questions dev  # dry run
    python scripts/step5_index.py --split all                 # at the freeze
    python scripts/freeze_e7.py --index all --questions test # at the freeze, once
"""
import argparse
import csv
import json
import os
import sys
from collections import defaultdict

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

from evalkit import load_gold  # noqa: E402

RES = os.path.join(HERE, "results")
DEV_Q = os.path.join(HERE, "data", "test_questions.xlsx")
DEV_UNANS = os.path.join(HERE, "data", "e7_dev_unanswerable.csv")
TEST_Q = os.path.join(HERE, "data", "e7_questions_150.csv")
ABSTAIN = os.path.join(RES, "e7_abstain.json")
K = 5


def dev_questions():
    q = pd.read_excel(DEV_Q)
    out = [dict(id=r.question_id, text=r.question_en_no_ref, mode="question",
                expected={e.strip() for e in str(r.expected_verse).split(",")})
           for r in q.itertuples()]
    out += [dict(id=r["question_id"], text=r["question_en_no_ref"], mode="unanswerable",
                 expected=set()) for r in csv.DictReader(open(DEV_UNANS, encoding="utf-8"))]
    return out


def test_questions():
    out = []
    for r in csv.DictReader(open(TEST_Q, encoding="utf-8")):
        exp = set() if r["expected_verse"].strip() in ("", "NONE") else \
            {e.strip() for e in r["expected_verse"].split(",")}
        out.append(dict(id=r["question_id"], text=r["question_en_no_ref"],
                        mode=r["input_mode"], expected=exp))
    return out


def balanced(scores_ans, scores_un, t):
    return ((np.array(scores_ans) >= t).mean() + (np.array(scores_un) < t).mean()) / 2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", choices=["dev", "all"], default="dev")
    ap.add_argument("--calibrate", action="store_true")
    ap.add_argument("--questions", choices=["dev", "test"])
    args = ap.parse_args()
    from gitaground import GitaGround

    g = GitaGround(args.index)
    _, gold, _ = load_gold()
    relevant_of = lambda exp: {k for k, r in enumerate(g.index)          # noqa: E731
                               if set(gold[r["unit_id"]]) & exp}

    if args.calibrate:
        qs = dev_questions()
        ans = [q for q in qs if q["expected"] and relevant_of(q["expected"])]
        un = [q for q in qs if not q["expected"]]
        sc = {q["id"]: g.abstain_scores(q["text"]) for q in ans + un}
        best = None
        for sig in ("passage", "profile"):
            a = [sc[q["id"]][sig] for q in ans]
            u = [sc[q["id"]][sig] for q in un]
            vals = sorted(set(a + u))
            cands = [(x + y) / 2 for x, y in zip(vals, vals[1:])]
            accs = [balanced(a, u, t) for t in cands]
            top = max(accs)
            tied = [t for t, acc in zip(cands, accs) if acc == top]
            t = tied[len(tied) // 2]                       # middle of the best range
            print(f"  {sig:8} answerable mean {np.mean(a):.3f} (min {min(a):.3f})   "
                  f"unanswerable mean {np.mean(u):.3f} (max {max(u):.3f})   "
                  f"best balanced accuracy {top:.3f} at threshold {t:.4f}")
            if best is None or top > best[2]:
                best = (sig, t, top)
        rule = dict(signal=best[0], threshold=round(best[1], 6), index=args.index,
                    dev_balanced_accuracy=round(best[2], 4),
                    calibrated_on=f"{len(ans)} answerable dev questions, {len(un)} dev unanswerable")
        json.dump(rule, open(ABSTAIN, "w"), indent=2)
        print(f"  LOCKED abstention rule -> {os.path.relpath(ABSTAIN, HERE)}: {rule}")
        g = GitaGround(args.index)                         # reload with the rule

    if not args.questions:
        return
    if args.questions == "test" and input("Run the 150 TEST questions now (once)? [yes/no] ") != "yes":
        sys.exit("not run")
    qs = dev_questions() if args.questions == "dev" else test_questions()

    by_mode = defaultdict(lambda: defaultdict(float))
    claims = {"primary": [0, 0], "passing": [0, 0]}
    lines = [f"E7 — index {args.index} ({len(g.index)} passages), questions {args.questions}"
             + ("   (DRY RUN: abstention threshold set on these questions)"
                if args.questions == "dev" else ""),
             f"abstention rule: {g.abstain_rule}"]
    for q in qs:
        m = by_mode[q["mode"]]
        m["n"] += 1
        abst = g.abstains(q["text"])
        if not q["expected"]:
            m["abstained"] += abst
            continue
        m["wrongly abstained"] += abst
        ranked = [r for r, _ in g.rank_verses(q["text"])]
        m["hit1"] += ranked[0] in q["expected"]
        m["hit3"] += bool(set(ranked[:3]) & q["expected"])
        rel = relevant_of(q["expected"])
        if not rel:
            continue
        m["scored"] += 1
        ks, top = g.retrieve(q["text"])
        vec = list(np.argsort(-(g.P @ g.embed([q["text"]])[0]))[:K])
        for name, got in (("ours", ks), ("vector", vec)):
            m[f"{name} hit5"] += bool(set(got) & rel)
            m[f"{name} p5"] += len(set(got) & rel) / K
        for k in ks:
            r = g.index[k]
            kind = "primary" if r["primary_verse"] == top else \
                   "passing" if top in r["secondary_verses"].split(";") else None
            if kind:
                claims[kind][1] += 1
                claims[kind][0] += top in gold[r["unit_id"]]

    lines.append(f"\n  {'mode':13} {'n':>4} {'Hit@1':>6} {'Hit@3':>6} {'scored':>7} "
                 f"{'ours Hit@5':>10} {'ours P@5':>9} {'vec Hit@5':>10} {'vec P@5':>8} "
                 f"{'abstained':>10} {'wrongly abst.':>14}")
    for mode, m in by_mode.items():
        na = m["n"] if mode != "unanswerable" else 0
        sc = m["scored"]
        f = lambda x, d: f"{x / d:.3f}" if d else "-"          # noqa: E731
        lines.append(f"  {mode:13} {int(m['n']):>4} {f(m['hit1'], na):>6} {f(m['hit3'], na):>6} "
                     f"{int(sc):>7} {f(m['ours hit5'], sc):>10} {f(m['ours p5'], sc):>9} "
                     f"{f(m['vector hit5'], sc):>10} {f(m['vector p5'], sc):>8} "
                     f"{f(m['abstained'], m['n'] if mode == 'unanswerable' else 0):>10} "
                     f"{f(m['wrongly abstained'], na):>14}")
    for kind, (hit, n) in claims.items():
        lines.append(f"  {kind} claims among retrieved passages: {hit}/{n} correct"
                     + (f" ({hit / n:.3f})" if n else ""))
    lines.append("  unsupported / hallucinated topics: 0 by construction (catalogue verses and "
                 "indexed passages only; LLM summary text not evaluated)")
    lines.append("  plain vector search never abstains (abstention on unanswerable = 0).")
    text = "\n".join(lines)
    print(text)
    out = os.path.join(RES, f"freeze_e7_{args.index}_{args.questions}.txt")
    open(out, "w", encoding="utf-8").write(text + "\n")
    print(f"\n  wrote -> {os.path.relpath(out, HERE)}")


if __name__ == "__main__":
    main()
