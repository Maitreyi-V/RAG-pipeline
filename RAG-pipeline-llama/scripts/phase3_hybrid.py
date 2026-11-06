#!/usr/bin/env python3
"""
Phase 3 v3 hybrid: the LLM proposes the verse, the decoder keeps the structure.

The v2 primary system (teacher P_off, 145-state resume decoder) with one more
piece of evidence: gpt-5-mini's end-to-end labels (llm_e2e.py, same prompt and
effort as before) as a verse vote. bonus_share multiplies the voted verses'
verse-scorer value by e^lambda before decoupling (total verse mass stays 1 - P_off,
but the voted verse becomes more peaked, so the path can also leave off-verse). lambda is chosen with gamma, beta_off and tau by leave-one-lecture-out
on the 9 TUNING lectures (lambda = 0 = ignore the vote). No test gold.

  (no flag)  LOLO report on the tuning lectures, lambda fixed 0 vs chosen
  --save     also fit on all 9 tuning lectures and write results/v3_hybrid145.pkl
  --tuning-rule  pre-registered language decision per vote source (tuning lectures only)
  --test     the ONE pre-registered run on the 42 -> results/v3_hybrid_test.txt

Usage:
    python scripts/phase3_hybrid.py
    python scripts/phase3_hybrid.py --save
"""
import argparse
import csv
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

from evalkit import load_csv, score  # noqa: E402
from llm_e2e import REVIEW  # noqa: E402
from llm_pilot import parse  # noqa: E402
from step4_viterbi import FEATS, near_f1, run, theme_f1  # noqa: E402

LLM = "gpt-5-mini"
OUT = os.path.join(HERE, "results", "v3_hybrid145.pkl")


def llm_votes():
    """{unit_id: [verse index]} from every lecture gpt-5-mini answered (tuning + test + held-out)"""
    refs = list(np.load(FEATS)["refs"])
    folder = os.path.join(REVIEW, LLM)
    votes = {}
    for f in sorted(os.listdir(folder)):
        if not f.endswith("_ids.csv"):
            continue
        v = f[:-len("_ids.csv")]
        if not os.path.exists(os.path.join(folder, f"{v}.txt")):
            continue
        ids = [(r["passage_id"], r["unit_id"]) for r in load_csv(os.path.join(folder, f))]
        p = parse(os.path.join(folder, f"{v}.txt"), ids)
        p = p[0] if isinstance(p, tuple) else p
        for _, u in ids:
            votes[u] = [refs.index(r) for r in p.get(u, []) if r in refs]
    return votes


def opus_votes():
    """{unit_id: [verse index]} from the Opus 5.5 chat answers: 9 tuning (llm_frontier/tuning) + the 42"""
    import llm_frontier as LF
    refs = list(np.load(FEATS)["refs"])
    votes = {}
    for d, ans in ((LF.TUNING_DIR, os.path.join(LF.TUNING_DIR, "answers")), (LF.DIR, LF.ANSWERS)):
        for f in sorted(os.listdir(d)):
            v = f[:-len("_ids.csv")] if f.endswith("_ids.csv") else None
            if not v or not os.path.exists(os.path.join(ans, f"{v}.txt")):
                continue
            ids = [(r["passage_id"], r["unit_id"]) for r in load_csv(os.path.join(d, f))]
            p, _ = LF.parse(os.path.join(ans, f"{v}.txt"), ids)
            for _, u in ids:
                votes[u] = [refs.index(r) for r in p.get(u, []) if r in refs]
    return votes


def hard_rule(v2, votes, refs, langs, lang_of_unit):
    """pre-registered rule: v2 off -> off; v2 on and vote names verses -> the vote's set;
    v2 on and vote NONE/missing -> v2's verses. Applied only in `langs`; elsewhere v2."""
    out = {}
    for u, p in v2.items():
        if lang_of_unit[u] not in langs or not p:
            out[u] = p
        else:
            out[u] = [refs[j] for j in votes.get(u, [])] or p
    return out


def tuning_rule():
    """the pre-registered language decision, per vote source, on the tuning lectures (LOLO v2)"""
    refs = list(np.load(FEATS)["refs"])
    teacher = {r["unit_id"]: min(0.98, max(0.02, float(r["soft_target"]))) for r in
               csv.DictReader(open(os.path.join(HERE, "results/silver/silver_labels.csv"), encoding="utf-8"))}
    order, gold, lectures, tune = run(alphas=("a05",), verbose=False, resume=True, off_override=teacher)
    v2 = dict(order)["Viterbi (alpha 0.5)"]
    lang_of = {u: L for L, vs in tune.items() for v in vs for u in lectures[v]}
    lines = ["v3 language decision on the TUNING lectures (pre-registered rule: use the vote in a language iff "
             "hard-rule strict >= v2 AND worse on at most half of that language's lectures)"]
    decision = {}
    for src, votes in (("gpt-5-mini", llm_votes()), ("Opus 5.5", opus_votes())):
        alone = {u: [refs[j] for j in votes.get(u, [])] for u in v2}
        hyb = hard_rule(v2, votes, refs, ("kn", "en"), lang_of)
        use = []
        for L in ("kn", "en"):
            f2, fh, fa = (score(gold, p, lectures, tune[L])["F1"] for p in (v2, hyb, alone))
            per = [(v, score(gold, v2, lectures, [v])["F1"], score(gold, hyb, lectures, [v])["F1"]) for v in tune[L]]
            worse = sum(1 for _, a, b in per if b < a)
            ok = fh >= f2 and worse <= len(per) / 2
            if ok:
                use.append(L)
            lines.append(f"  {src:11} {L}: v2 {f2:.3f}  hard rule {fh:.3f}  vote alone {fa:.3f}  worse on {worse}/{len(per)}"
                         f"  -> {'USE vote' if ok else 'keep v2'}   " + ", ".join(f"{v} {a:.3f}->{b:.3f}" for v, a, b in per))
        decision[src] = use
    lines.append(f"  decision: {decision}")
    out = "\n".join(lines)
    print(out)
    open(os.path.join(HERE, "results", "v3_language_decision.txt"), "w").write(out + "\n")


LANGS = {"gpt-5-mini": ("en",), "Opus 5.5": ("en",)}     # from tuning_rule(), results/v3_language_decision.txt


def test():
    """the one pre-registered run on the 42: hybrids vs their vote source alone, Cue, Plain Carry"""
    from baselines import carry
    from freeze_grounding import f1_near, f1_strict, near_counts, paired, strict_counts, theme_counts, with_ci
    from evalkit import LECTURES_NEW, read_split
    from llm_e2e import all_lectures
    from step4_viterbi import OFF, predict_locked
    lectures, gold, _ = load_gold_()
    z = np.load(FEATS)
    refs = list(z["refs"])
    row = {u: i for i, u in enumerate(z["unit_ids"])}
    vids = all_lectures()
    lang = {v: L for s_ in ("test", "heldout") for L in ("kn", "en") for v in read_split(L, s_)}
    lang_of = {u: lang[v] for v in vids for u in lectures[v]}
    dec = predict_locked({v: lectures[v] for v in vids}, "a05", path=os.path.join(HERE, "results", "v2_teacher145.pkl"))
    cue = {u: ([] if st == OFF else [refs[st]]) + [refs[j] for j in extra]
           for v in vids for u, (st, extra, _, _) in zip(lectures[v], dec[v])}
    mini, opus = llm_votes(), opus_votes()
    alone = lambda vt: {u: [refs[j] for j in vt.get(u, [])] for u in cue}
    anchors = {u: [refs[j] for j in np.flatnonzero(z["anchor"][row[u]])] for u in cue}
    plain = {}
    for v in vids:
        plain.update(carry(lectures[v], anchors))
    systems = {"Opus-hybrid (v3)": hard_rule(cue, opus, refs, LANGS["Opus 5.5"], lang_of),
               "Opus 5.5 alone": alone(opus),
               "mini-hybrid (v3)": hard_rule(cue, mini, refs, LANGS["gpt-5-mini"], lang_of),
               "gpt-5-mini alone": alone(mini),
               "GitaGround-Cue (v2)": cue, "Plain Carry": plain}
    pairs = [("Opus-hybrid (v3)", "Opus 5.5 alone")] + [(a_, b_) for a_, bs in (
        ("Opus-hybrid (v3)", ("GitaGround-Cue (v2)", "Plain Carry")),
        ("mini-hybrid (v3)", ("gpt-5-mini alone", "GitaGround-Cue (v2)", "Plain Carry"))) for b_ in bs]
    speaker = {r["video_file"]: r["speaker"] for r in load_csv(LECTURES_NEW)}
    groups = [(f"E2 {n} test", read_split(L, "test")) for L, n in (("kn", "Kannada"), ("en", "English"))]
    for L, n in (("kn", "Kannada"), ("en", "English")):
        vs = read_split(L, "heldout")
        groups += [(f"E3 {n}: {sp}", [v for v in vs if speaker[v] == sp]) for sp in sorted({speaker[v] for v in vs})]
        groups.append((f"E3 new {n} (pooled)", vs))
    lines = ["Phase 3 post-hoc v3 hybrid — ONE run on the 42 (pre-registered; test-informed design, see log)",
             f"rule: v2 off -> off; v2 on + vote -> vote's verses; v2 on + vote NONE -> v2. Vote used in: {LANGS}"]
    for gname, vs in groups:
        units = [u for v in vs for u in lectures[v]]
        off = [u for u in units if not gold[u]]
        lines += ["", f"{gname}   lectures {len(vs)}   passages {len(units)}   no-verse {len(off)}",
                  f"  {'system':22} {'strict F1 [CI]':22} {'+-1 F1 [CI]':22} {'theme F1 [CI]':22} off-verse given a verse"]
        for name, p in systems.items():
            cells = []
            for counts, f1 in ((strict_counts, f1_strict), (near_counts, f1_near), (theme_counts, f1_strict)):
                point, ci = with_ci([counts(gold, p, lectures[v]) for v in vs], f1, 1000)
                cells.append(f"{point:.3f}" + (f" [{ci[0]:.3f},{ci[1]:.3f}]" if ci else ""))
            lines.append(f"  {name:22} {cells[0]:22} {cells[1]:22} {cells[2]:22} {sum(1 for u in off if p[u])}/{len(off)}")
        lines.append(f"  paired strict F1 (A - B), 10000 resamples  {'diff':>7} {'95% CI':>17} {'P(<=0)':>7} {'W/T/L':>8} {'sign p':>7}")
        for k, (a_, b_) in enumerate(pairs):
            d, ci, le0, (w, t, lo), pv = paired(gold, lectures, systems[a_], systems[b_], vs, 10000)
            tag = "PRIMARY " if k == 0 and gname == "E2 English test" else ""
            lines.append(f"    {tag}{a_} - {b_:22} {d:>+7.3f} [{ci[0]:+.3f},{ci[1]:+.3f}] {le0:>7.3f} "
                         f"{f'{w}/{t}/{lo}':>8} {pv:>7.3f}")
    out = "\n".join(lines)
    print(out)
    open(os.path.join(HERE, "results", "v3_hybrid_test.txt"), "w").write(out + "\n")


def load_gold_():
    from evalkit import load_gold
    return load_gold()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--save", action="store_true")
    ap.add_argument("--tuning-rule", action="store_true", help="pre-registered language decision (tuning only)")
    ap.add_argument("--test", action="store_true", help="the ONE pre-registered run on the 42 (test gold)")
    args = ap.parse_args()
    if args.tuning_rule:
        return tuning_rule()
    if args.test:
        return test()
    teacher = {r["unit_id"]: min(0.98, max(0.02, float(r["soft_target"]))) for r in
               csv.DictReader(open(os.path.join(HERE, "results/silver/silver_labels.csv"), encoding="utf-8"))}
    votes = llm_votes()
    lines = [f"Phase 3 v3 hybrid — TUNING lectures, leave-one-lecture-out, alpha 0.5 (no test gold)",
             f"verse vote: {LLM} end-to-end labels on {len(votes)} passages; P_off = teacher; 145-state resume",
             f"  {'system':28} {'KN strict':>9} {'KN +-1':>7} {'EN strict':>9} {'EN +-1':>7} {'EN theme':>8}"
             f"   EN strict per lecture"]
    for name, kw in (("v2 teacher 145 (lambda 0)", {}), ("v3 hybrid (lambda chosen)", {"verse_bonus": votes})):
        info = {}
        order, gold, lectures, tune = run(alphas=("a05",), verbose=False, resume=True, off_override=teacher,
                                          info=info, save=OUT if (args.save and kw) else None, **kw)
        pred = dict(order)["Viterbi (alpha 0.5)"]
        kn = (score(gold, pred, lectures, tune["kn"])["F1"], near_f1(gold, pred, lectures, tune["kn"])[2])
        en = (score(gold, pred, lectures, tune["en"])["F1"], near_f1(gold, pred, lectures, tune["en"])[2],
              theme_f1(gold, pred, lectures, tune["en"]))
        per = ", ".join(f"{v} {score(gold, pred, lectures, [v])['F1']:.3f}" for v in tune["en"])
        lines.append(f"  {name:28} {kn[0]:>9.3f} {kn[1]:>7.3f} {en[0]:>9.3f} {en[1]:>7.3f} {en[2]:>8.3f}   {per}")
        if kw:
            lines.append("  lambda chosen per held-out lecture: "
                         + ", ".join(f"{c[0]} {c[-1]}" for c in info["chosen"]))
            kn_per = ", ".join(f"{v} {score(gold, pred, lectures, [v])['F1']:.3f}" for v in tune["kn"])
            lines.append(f"  KN strict per lecture: {kn_per}")
    out = "\n".join(lines)
    print(out)
    open(os.path.join(HERE, "results", "phase3_hybrid_tuning.txt"), "w").write(out + "\n")


if __name__ == "__main__":
    main()
