#!/usr/bin/env python3
"""
Score every unit against every verse with BGE-M3, once, and cache the matrix.

detect_units.py --layer 3 only ever saved the verses that cleared a threshold,
so every new thresholding idea meant re-embedding the corpus. This saves the
full unit x verse similarity matrix instead; the margin baseline and the
propagation gate both read it and run in seconds.

Two verse representations are scored:

  trans     each verse's English translations, embedded separately, max taken
            (what layer3_semantic.py does today)
  expanded  ONE document per verse: Sanskrit (Devanagari + Kannada script) and
            all translations, plus any commentary found in --commentary

--commentary is an optional JSON {"BG 2.47": ["paragraph", ...], ...}. Leave it
out until the team has sourced commentary text it is allowed to use; nothing
here invents any.

Unit embeddings are cached separately (results/unit_embeddings.npz), so adding
commentary later only re-embeds the 72 verse documents.

Usage:
    python scripts/embed_scores.py
    python scripts/embed_scores.py --commentary data/verse_commentary.json
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

from evalkit import UNITS, load_csv  # noqa: E402

UNIT_CACHE = os.path.join(HERE, "results", "unit_embeddings.npz")
OUT = os.path.join(HERE, "results", "embed_scores.npz")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--units", default=UNITS)
    ap.add_argument("--commentary", help="optional JSON of commentary paragraphs per verse")
    ap.add_argument("--rebuild-units", action="store_true", help="ignore the unit cache")
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()

    from embeddings import embed_texts
    from verse_database import VERSES_BG_CH2
    from verse_translations import VERSE_TRANSLATIONS

    units = load_csv(args.units)
    unit_ids = [u["unit_id"] for u in units]

    if os.path.exists(UNIT_CACHE) and not args.rebuild_units:
        z = np.load(UNIT_CACHE, allow_pickle=False)
        if list(z["unit_ids"]) != unit_ids:
            sys.exit("unit cache does not match --units; re-run with --rebuild-units")
        U = z["emb"]
        print(f"  loaded {len(unit_ids)} unit embeddings from cache")
    else:
        print(f"  embedding {len(unit_ids)} units (a few minutes)...")
        U = np.asarray(embed_texts([u["text"] for u in units]), dtype=np.float32)
        np.savez_compressed(UNIT_CACHE, unit_ids=np.array(unit_ids), emb=U)

    commentary = {}
    if args.commentary:
        with open(args.commentary, encoding="utf-8") as f:
            commentary = json.load(f)

    refs = [v["ref"] for v in VERSES_BG_CH2]
    missing = [r for r in refs if not VERSE_TRANSLATIONS.get(r)]
    if missing:
        sys.exit(f"no translation for: {missing}")

    # trans: one vector per translation, max over a verse's translations
    flat = [(i, t) for i, r in enumerate(refs) for t in VERSE_TRANSLATIONS[r]]
    T = np.asarray(embed_texts([t for _, t in flat]), dtype=np.float32)
    sims = U @ T.T                                   # embeddings are L2-normalised
    trans = np.full((len(unit_ids), len(refs)), -1.0, dtype=np.float32)
    for col, (i, _) in enumerate(flat):
        trans[:, i] = np.maximum(trans[:, i], sims[:, col])

    # expanded: one document per verse
    docs = []
    for v in VERSES_BG_CH2:
        parts = [v["devanagari"], v["kannada"]] + list(VERSE_TRANSLATIONS[v["ref"]])
        parts += commentary.get(v["ref"], [])
        docs.append("\n".join(parts))
    E = np.asarray(embed_texts(docs), dtype=np.float32)
    expanded = (U @ E.T).astype(np.float32)

    np.savez_compressed(args.out, unit_ids=np.array(unit_ids), refs=np.array(refs),
                        trans=trans, expanded=expanded)
    print(f"  {len(unit_ids)} units x {len(refs)} verses   "
          f"commentary for {sum(1 for r in refs if commentary.get(r))} verses")
    print(f"  wrote -> {os.path.relpath(args.out, HERE)}")


if __name__ == "__main__":
    main()
