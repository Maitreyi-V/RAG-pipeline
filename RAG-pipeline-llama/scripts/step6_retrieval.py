#!/usr/bin/env python3
"""
Step 6: question -> verse -> passages, on the Step 5 index. Development harness
for E7.

QUESTION -> VERSE. Each of the 72 verses gets scores from:
  catalogue   cosine(question, verse document = Sanskrit + translations)
  profile     cosine(question, mu_v), the Step 2 meaning profile (alpha 0.5),
              built from detections in all 39 original lectures (no labels)
  bm25        BM25 of the question against the verse catalogue text
              (English translations + romanised verse)
  vote        the 5 indexed passages closest to the question each vote for
              their Step 4 primary verse with weight cosine * p_primary
  combined    z(profile) + z(bm25) + z(vote), each z-scored over the 72
              verses for that question; equal weights, nothing fitted

VERSE -> PASSAGES (5 returned):
  vector RAG  cosine(question, passage) over all indexed passages (baseline)
  n1 + n2     n1 passages from the best-ranked verse (by `combined`, or by
              `profile` alone) that has passages, n2 from the next; inside a
              verse, passages ranked by cosine * p_primary
  hybrid      plain search's top 2, then 3 passages from one verse: the best
              profile verse, or the Step 4 verse of plain search's best passage
  LOCKED      (16 Oct) hybrid with the profile verse, run through gitaground.py,
              which also puts string-matched verses first (pasted shlokas)

Questions: data/test_questions.xlsx `question_en_no_ref` (no verse number).
A passage is relevant if its GOLD labels contain an expected verse. These 74
questions are the DEVELOPMENT set (agreed 16 Oct); the team's new questions
are the E7 test set. Abstention is not measured yet.

Usage:
    python scripts/step6_retrieval.py                # tuning index
"""
import argparse
import csv
import math
import os
import re
import sys
from collections import Counter

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

from embed_scores import unit_cache  # noqa: E402
from evalkit import UNITS, UNITS_HELDOUT, load_csv, load_gold  # noqa: E402
from step2_features import anchor_profile_sums, normalise, verse_docs  # noqa: E402

QUESTIONS = os.path.join(HERE, "data", "test_questions.xlsx")
K = 5
ALPHA = 0.5


def tokens(text):
    from roman_detector import normalize_asr
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
    return [w for w in re.findall(r"[a-z]+", normalize_asr(text)) if w not in ENGLISH_STOP_WORDS]


def bm25(query_tokens, docs_tokens, k1=1.5, b=0.75):
    """Okapi BM25 of one query against every document."""
    n = len(docs_tokens)
    avg = sum(map(len, docs_tokens)) / n
    df = Counter(w for d in docs_tokens for w in set(d))
    out = np.zeros(n)
    for i, d in enumerate(docs_tokens):
        tf = Counter(d)
        for w in query_tokens:
            if tf[w]:
                idf = math.log(1 + (n - df[w] + 0.5) / (df[w] + 0.5))
                out[i] += idf * tf[w] * (k1 + 1) / (tf[w] + k1 * (1 - b + b * len(d) / avg))
    return out


def z(x):
    s = x.std()
    return (x - x.mean()) / s if s > 0 else np.zeros_like(x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="dev")
    args = ap.parse_args()
    from embeddings import embed_texts
    from roman_detector import RomanShlokaDetector
    from verse_database import VERSES_BG_CH2
    from verse_translations import VERSE_TRANSLATIONS

    refs = [v["ref"] for v in VERSES_BG_CH2]
    vi = {r: j for j, r in enumerate(refs)}

    # ---- unit grids, embeddings, Step 2 anchors (same order as step2) --------
    units, E = [], []
    for path in (UNITS, UNITS_HELDOUT):
        units += load_csv(path)
        E.append(np.load(unit_cache(path))["emb"])
    E = np.concatenate(E)
    f2 = np.load(os.path.join(HERE, "results", "step2_features.npz"))
    assert list(f2["unit_ids"]) == [u["unit_id"] for u in units]
    row = {u["unit_id"]: i for i, u in enumerate(units)}
    original = {u["video_file"] for u in load_csv(UNITS)}
    lectures = {}
    for u in sorted(units, key=lambda u: (u["video_file"], int(u["unit_index"]))):
        lectures.setdefault(u["video_file"], []).append(row[u["unit_id"]])

    # ---- verse representations ------------------------------------------------
    T = normalise(np.asarray(embed_texts(verse_docs()), dtype=np.float32))
    sums, cnts = anchor_profile_sums(E, lectures, original, f2["anchor"])
    corpus = normalise(sum(sums.values()).astype(np.float32))
    mu = normalise((1 - ALPHA) * corpus + ALPHA * T)
    roman = RomanShlokaDetector().verses
    cat_tokens = [tokens(" ".join(VERSE_TRANSLATIONS[r]) + " " + roman[r]) for r in refs]

    # ---- index -----------------------------------------------------------------
    index = list(csv.DictReader(open(os.path.join(HERE, "results", f"step5_index_{args.split}.csv"),
                                     encoding="utf-8")))
    P = np.stack([E[row[r["unit_id"]]] for r in index])
    pp = np.array([float(r["p_primary"]) for r in index])
    on = [k for k, r in enumerate(index) if r["primary_verse"] != "OFF"]
    by_verse = {}
    for k in on:
        by_verse.setdefault(index[k]["primary_verse"], []).append(k)
    _, gold, _ = load_gold()

    q = pd.read_excel(QUESTIONS)
    texts = list(q["question_en_no_ref"])
    expected = [{e.strip() for e in str(x).split(",")} for x in q["expected_verse"]]
    Q = normalise(np.asarray(embed_texts(texts), dtype=np.float32))
    sim = Q @ P.T                                         # question x passage

    def scores(i):
        vote = np.zeros(len(refs))
        for k in np.argsort(-sim[i])[:K]:
            if index[k]["primary_verse"] != "OFF":
                vote[vi[index[k]["primary_verse"]]] += sim[i, k] * pp[k]
        s = dict(catalogue=T @ Q[i], profile=mu @ Q[i],
                 bm25=bm25(tokens(texts[i]), cat_tokens), vote=vote)
        s["combined"] = z(s["profile"]) + z(s["bm25"]) + z(s["vote"])
        return s

    S = [scores(i) for i in range(len(texts))]

    print(f"  QUESTION -> VERSE, all {len(texts)} questions "
          f"(Hit@n = an expected verse among the top n of 72)")
    for name in ("catalogue", "profile", "bm25", "vote", "combined"):
        h1 = h3 = 0
        for i, exp in enumerate(expected):
            top = [refs[j] for j in np.argsort(-S[i][name])[:3]]
            h1 += top[0] in exp
            h3 += bool(set(top) & exp)
        print(f"    {name:10} Hit@1 {h1 / len(texts):.3f}   Hit@3 {h3 / len(texts):.3f}")

    relevant = [{k for k, r in enumerate(index) if set(gold[r["unit_id"]]) & exp}
                for exp in expected]
    scored = [i for i, rel in enumerate(relevant) if rel]

    def vector(i):
        return list(np.argsort(-sim[i])[:K])

    def split_pick(n1, n2, by="combined"):
        def fn(i):
            ranked = [refs[j] for j in np.argsort(-S[i][by]) if refs[j] in by_verse]
            out = []
            for v, n in zip(ranked[:2], (n1, n2)):
                ks = sorted(by_verse[v], key=lambda k: -sim[i, k] * pp[k])
                out += ks[:n]
            if len(out) < K:                                   # a verse had too few passages
                out += [k for k in vector(i) if k not in out][:K - len(out)]
            return out[:K]
        return fn

    def hybrid(verse_of):
        """plain search's top 2, then 3 more from one grounded verse"""
        def fn(i):
            out = vector(i)[:2]
            v = verse_of(i)
            ks = sorted(by_verse.get(v, []), key=lambda k: -sim[i, k] * pp[k])
            out += [k for k in ks if k not in out][:K - 2]
            if len(out) < K:
                out += [k for k in vector(i) if k not in out][:K - len(out)]
            return out[:K]
        return fn

    def top_profile_verse(i):
        return next(refs[j] for j in np.argsort(-S[i]["profile"]) if refs[j] in by_verse)

    def viterbi_verse_of_best(i):
        return next(index[k]["primary_verse"] for k in np.argsort(-sim[i])
                    if index[k]["primary_verse"] != "OFF")

    print(f"\n  VERSE -> PASSAGES, {len(scored)} questions with a relevant passage in the "
          f"{args.split} index ({len(index)} passages)")
    for name, fn in (("vector RAG", vector), ("combined, 4 + 1", split_pick(4, 1)),
                     ("combined, 3 + 2", split_pick(3, 2)),
                     ("profile, 4 + 1", split_pick(4, 1, "profile")),
                     ("profile, 3 + 2", split_pick(3, 2, "profile")),
                     ("hybrid: 2 + profile verse", hybrid(top_profile_verse)),
                     ("hybrid: 2 + Viterbi verse", hybrid(viterbi_verse_of_best))):
        hits = prec = 0
        for i in scored:
            top = fn(i)
            hits += bool(set(top) & relevant[i])
            prec += len(set(top) & relevant[i]) / K
        print(f"    {name:26} Hit@{K} {hits / len(scored):.3f}   P@{K} {prec / len(scored):.3f}")

    # the LOCKED system exactly as the app runs it (adds the shloka matcher)
    from gitaground import GitaGround
    g = GitaGround(args.split)
    assert [r["unit_id"] for r in g.index] == [r["unit_id"] for r in index]
    hits = prec = fired = right = 0
    for i in scored:
        top, verse = g.retrieve(texts[i])
        hits += bool(set(top) & relevant[i])
        prec += len(set(top) & relevant[i]) / K
    for i, t in enumerate(texts):
        m = g.shloka_matches(t)
        fired += bool(m)
        right += bool(m) and m[0] in expected[i]
    print(f"    {'LOCKED (gitaground.py)':26} Hit@{K} {hits / len(scored):.3f}   "
          f"P@{K} {prec / len(scored):.3f}")
    print(f"    string matcher fired on {fired} of {len(texts)} questions; "
          f"its first verse is an expected verse on {right}")


if __name__ == "__main__":
    main()
