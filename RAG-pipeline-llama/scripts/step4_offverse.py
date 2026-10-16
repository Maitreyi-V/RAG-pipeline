#!/usr/bin/env python3
"""
What separates OFF-VERSE passages from on-verse ones, on the ENGLISH tuning
lectures, among passages AFTER the first recitation (where carry holds a verse)?

Target: gold has no verse on the passage. For each signal, AUC with the
direction chosen so that > 0.5 means "higher value -> more likely off-verse"
(flipped where the signal should be LOW off-verse; the direction is printed).

  cos_active        cosine (alpha 0.5 profile) to the verse carry holds (max if several)
  cos_active_win    the same, averaged over passages t-1, t, t+1 of the lecture
  since_anchor      passages since the last anchor passage (0 on an anchor passage)
  max_cos           best cosine to any of the 72 verses (an off-verse scorer input)
  max_string        best string score to any verse   (an off-verse scorer input)
  domain_density    fraction of the passage's words (sklearn English stop words
                    removed) that occur in the verse catalogue: the 72 English
                    translations + romanised verses. Built automatically, no
                    hand-made word list.

Usage:
    python scripts/step4_offverse.py
"""
import os
import re
import sys

import numpy as np
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.metrics import roc_auc_score

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

from baselines import carry  # noqa: E402
from evalkit import UNITS, load_csv, load_gold, read_split  # noqa: E402
from step3_scorer import FEATS  # noqa: E402

WORD = re.compile(r"[a-z]+")


def words(text):
    from roman_detector import normalize_asr
    return [w for w in WORD.findall(normalize_asr(text)) if w not in ENGLISH_STOP_WORDS]


def main():
    from roman_detector import RomanShlokaDetector
    from verse_translations import VERSE_TRANSLATIONS

    z = np.load(FEATS)
    refs = list(z["refs"])
    row = {u: i for i, u in enumerate(z["unit_ids"])}
    cos, A, S = z["cos_a05"], z["anchor"], z["string"]
    text = {u["unit_id"]: u["text"] for u in load_csv(UNITS)}
    lectures, gold, _ = load_gold()

    vocab = set()
    for r in refs:
        for t in VERSE_TRANSLATIONS[r]:
            vocab.update(words(t))
    vocab.update(w for v in RomanShlokaDetector().verses.values() for w in words(v))
    print(f"  catalogue vocabulary: {len(vocab)} words")

    sig = {k: [] for k in ("cos_active", "cos_active_win", "since_anchor",
                           "max_cos", "max_string", "domain_density")}
    y = []
    for v in read_split("en", "dev"):
        us = lectures[v]
        anchors = {u: [refs[j] for j in np.flatnonzero(A[row[u]])] for u in us}
        held = carry(us, anchors)
        last_anchor = None
        for k, u in enumerate(us):
            if anchors[u]:
                last_anchor = k
            if not held[u]:
                continue
            act = [refs.index(r) for r in held[u]]
            i = row[u]
            win = [row[w] for w in us[max(0, k - 1):k + 2]]
            ws = words(text[u])
            sig["cos_active"].append(cos[i, act].max())
            sig["cos_active_win"].append(np.mean([cos[j, act].max() for j in win]))
            sig["since_anchor"].append(k - last_anchor)
            sig["max_cos"].append(cos[i].max())
            sig["max_string"].append(S[i].max())
            sig["domain_density"].append(sum(w in vocab for w in ws) / max(len(ws), 1))
            y.append(int(not gold[u]))
    y = np.array(y)
    print(f"  English tuning passages after the first recitation: {len(y)} "
          f"({int(y.sum())} off-verse, {int((1 - y).sum())} on-verse)\n")
    print(f"  {'signal':16} {'AUC':>6}  direction             mean on   mean off")
    for k, x in sig.items():
        x = np.array(x, dtype=float)
        auc = roc_auc_score(y, x)
        d = "higher -> off-verse" if auc >= 0.5 else "lower -> off-verse"
        print(f"  {k:16} {max(auc, 1 - auc):>6.3f}  {d:20} {x[y == 0].mean():>9.3f} "
              f"{x[y == 1].mean():>9.3f}")


if __name__ == "__main__":
    main()
