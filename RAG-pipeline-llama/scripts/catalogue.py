#!/usr/bin/env python3
"""
The verse catalogue: everything the system knows about each verse, in one place.

One entry per verse of Chapter 2:
  ref           "BG 2.47"
  number        47
  devanagari    the verse in Devanagari          (verse_database.py)
  kannada       the verse in Kannada script      (verse_database.py)
  roman         the verse in plain Roman letters, derived from the Devanagari
                the same way roman_detector.py does it
  translations  English translations             (verse_translations.py)
  theme         the verse's theme, from data/verse_themes.csv, or None until
                that file is filled in

data/verse_themes.csv (made by the domain expert) has one row per verse:
verse_id,verse_index,theme_id,theme_name   e.g. 2.47,46,T06,Core Doctrine of ...
Once it has any rows, every verse must appear exactly once, or loading stops
with an error.

Usage:
    python scripts/catalogue.py          # prints a summary and checks the themes file
"""
import csv
import os
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

THEMES = os.path.join(HERE, "data", "verse_themes.csv")


def load_themes(path=THEMES, n_verses=72):
    """{verse_number: theme name}; {} if the file is missing or has no rows."""
    if not os.path.exists(path):
        return {}
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    themes = {}
    for r in rows:
        n = int(r["verse_id"].strip().split(".")[1])
        if n in themes:
            sys.exit(f"{path}: verse 2.{n} is listed twice")
        themes[n] = r["theme_name"].strip()
    if themes:
        missing = [n for n in range(1, n_verses + 1) if n not in themes]
        if missing:
            sys.exit(f"{path}: verses with no theme: {missing}")
    return themes


def load_catalogue():
    from roman_detector import romanize
    from verse_database import VERSES_BG_CH2
    from verse_translations import VERSE_TRANSLATIONS

    themes = load_themes(n_verses=len(VERSES_BG_CH2))
    catalogue = []
    for v in VERSES_BG_CH2:
        n = int(v["ref"].split(".")[1])
        catalogue.append({
            "ref": v["ref"],
            "number": n,
            "devanagari": v["devanagari"],
            "kannada": v["kannada"],
            "roman": " ".join(romanize(v["devanagari"]).split()),
            "translations": list(VERSE_TRANSLATIONS.get(v["ref"], [])),
            "theme": themes.get(n),
        })
    return catalogue


def main():
    cat = load_catalogue()
    n_themes = len({e["theme"] for e in cat if e["theme"]})
    print(f"  {len(cat)} verses   "
          f"{sum(1 for e in cat if e['translations'])} with a translation   "
          f"themes: {n_themes or 'not filled in yet (data/verse_themes.csv)'}")
    e = next(x for x in cat if x["number"] == 47)
    print(f"\n  example {e['ref']}")
    for k in ("devanagari", "kannada", "roman"):
        print(f"    {k:11} {e[k][:70]}")
    print(f"    {'translation':11} {e['translations'][0][:70]}")
    print(f"    {'theme':11} {e['theme']}")


if __name__ == "__main__":
    main()
