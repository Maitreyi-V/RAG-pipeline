#!/usr/bin/env python3
"""
Export the annotated lectures for gg_finetune/ (README section 1).

  gg_finetune/data/passages.jsonl      one line per passage, all 51 lectures
  gg_finetune/data/verses.json         72 verses: Kannada script, IAST, translation
  gg_finetune/data/prior_scores/*.npy  [T, 72] cos_a05 (unsupervised BGE-M3 profile
                                       cosine, Step 2; no gold) for hard negatives

lecture_id = video_file; idx = position in evalkit.load_gold()'s lecture order;
gold = Chapter 2 verse numbers, lowest first (the decoder's gold state), then the rest;
split: dev -> tune, test -> e2, heldout -> e3.

Usage:
    python scripts/export_finetune.py
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

from evalkit import UNITS, UNITS_HELDOUT, load_csv, load_gold, read_split, verse_number  # noqa: E402
from step3_scorer import FEATS  # noqa: E402

OUT = os.path.join(HERE, "gg_finetune", "data")


def main():
    from indic_transliteration.sanscript import DEVANAGARI, IAST, transliterate
    from verse_database import VERSES_BG_CH2
    from verse_translations import VERSE_TRANSLATIONS

    lectures, gold, lang = load_gold()
    text = {u["unit_id"]: u["text"] for u in load_csv(UNITS) + load_csv(UNITS_HELDOUT)}
    speaker = {r["video_file"]: r["speaker"] for r in load_csv(os.path.join(HERE, "data", "folds_v1.csv"))}
    split = {v: name for s, name in (("dev", "tune"), ("test", "e2"), ("heldout", "e3"))
             for L in ("kn", "en") for v in read_split(L, s)}
    assert set(split) == set(lectures) and len(split) == 51, "split files and gold disagree"

    z = np.load(FEATS)
    row = {u: i for i, u in enumerate(z["unit_ids"])}
    assert list(z["refs"]) == [f"BG 2.{i}" for i in range(1, 73)]
    os.makedirs(os.path.join(OUT, "prior_scores"), exist_ok=True)
    n = 0
    with open(os.path.join(OUT, "passages.jsonl"), "w", encoding="utf-8") as f:
        for v in sorted(lectures):
            for k, u in enumerate(lectures[v]):
                g = sorted(verse_number(r) for r in gold[u])
                assert None not in g, f"{u}: gold verse outside Chapter 2"
                f.write(json.dumps({"lecture_id": v, "idx": k, "lang": lang[v], "split": split[v],
                                    "speaker": speaker[v], "unit_id": u, "text": text[u], "gold": g},
                                   ensure_ascii=False) + "\n")
                n += 1
            np.save(os.path.join(OUT, "prior_scores", f"{v}.npy"),
                    z["cos_a05"][[row[u] for u in lectures[v]]])
    verses = [{"v": i + 1, "kn": x["kannada"],
               "roman": " ".join(transliterate(x["devanagari"].replace("|", " "), DEVANAGARI, IAST).split()),
               "trans": VERSE_TRANSLATIONS[x["ref"]][0]} for i, x in enumerate(VERSES_BG_CH2)]
    json.dump(verses, open(os.path.join(OUT, "verses.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"  wrote {n} passages, {len(lectures)} lectures, {len(verses)} verses -> {os.path.relpath(OUT, HERE)}")


if __name__ == "__main__":
    main()
