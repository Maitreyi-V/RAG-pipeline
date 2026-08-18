"""
build_units.py — freeze the annotation unit grid for the verse-grounding corpus.

Emits:
  data/normalized/<video_key>.txt  canonical text; ALL char offsets refer to these
  data/units_v1.csv                one row per annotation unit

IMMUTABLE once committed: every annotation row references unit_id.
Never re-run with a different TARGET_WORDS — write units_v2.csv instead.
"""
import csv, json, os, re, sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "scripts"))
from score_by_tier import norm_video

TARGET_WORDS = 100
MAX_SENTENCE_WORDS = 2 * TARGET_WORDS

BASE        = os.path.dirname(os.path.abspath(__file__))
TRANSCRIPTS = os.path.join(BASE, "data", "transcripts")
NORMALIZED  = os.path.join(BASE, "data", "normalized")
OUT_CSV     = os.path.join(BASE, "data", "units_v1.csv")

FIELDS = ["unit_id", "video_file", "language", "tradition", "unit_index",
          "text", "char_start", "char_end", "time_start", "time_end", "n_words"]


def _flush(buf, full):
    """buf: list of (char_start, char_end, time_start|None, time_end|None)"""
    cs, ce = buf[0][0], buf[-1][1]
    text = full[cs:ce]
    return {"text": text, "char_start": cs, "char_end": ce,
            "time_start": buf[0][2], "time_end": buf[-1][3],
            "n_words": len(text.split())}


def english_units(path):
    segs = json.load(open(path, encoding="utf-8"))["segments"]
    parts, pieces, cur = [], [], 0
    for s in segs:
        t = s["text"].strip()
        if not t:
            continue
        if parts:
            cur += 1                                   # the joining space
        pieces.append((cur, cur + len(t), float(s["start"]), float(s["end"]), t))
        parts.append(t)
        cur += len(t)
    full = " ".join(parts)

    units, buf, words = [], [], 0
    for cs, ce, ts, te, t in pieces:
        buf.append((cs, ce, ts, te))
        words += len(t.split())
        if words >= TARGET_WORDS:
            units.append(_flush(buf, full)); buf, words = [], 0
    if buf:
        units.append(_flush(buf, full))                # never drop the tail
    return full, units


def _hard_split(full, cs, ce, max_words):
    toks = [(m.start() + cs, m.end() + cs) for m in re.finditer(r"\S+", full[cs:ce])]
    return [(c[0][0], c[-1][1])
            for c in (toks[i:i + max_words] for i in range(0, len(toks), max_words)) if c]


def kannada_units(path):
    full = open(path, encoding="utf-8").read().strip()
    sents = []
    for m in re.finditer(r"[^.?]+[.?]?", full):
        seg = m.group()
        cs = m.start() + (len(seg) - len(seg.lstrip()))
        ce = m.end()   - (len(seg) - len(seg.rstrip()))
        if ce > cs:
            sents.append((cs, ce))

    expanded = []
    for cs, ce in sents:
        if len(full[cs:ce].split()) > MAX_SENTENCE_WORDS:
            expanded.extend(_hard_split(full, cs, ce, TARGET_WORDS))
        else:
            expanded.append((cs, ce))

    units, buf, words = [], [], 0
    for cs, ce in expanded:
        buf.append((cs, ce, None, None))
        words += len(full[cs:ce].split())
        if words >= TARGET_WORDS:
            units.append(_flush(buf, full)); buf, words = [], 0
    if buf:
        units.append(_flush(buf, full))
    return full, units


def main():
    os.makedirs(NORMALIZED, exist_ok=True)
    rows = []
    for fn in sorted(os.listdir(TRANSCRIPTS)):
        path = os.path.join(TRANSCRIPTS, fn)
        if fn.endswith(".json"):
            lang, tradition = "en", "Advaita"
            full, units = english_units(path)
        elif fn.endswith(".txt"):
            lang, tradition = "kn", "Dvaita"
            full, units = kannada_units(path)
        else:
            continue                                   # skips .DS_Store
        vk = norm_video(fn)
        with open(os.path.join(NORMALIZED, vk + ".txt"), "w", encoding="utf-8") as f:
            f.write(full)
        for i, u in enumerate(units):
            rows.append({"unit_id": f"{lang}_{vk}_{i:04d}", "video_file": vk,
                         "language": lang, "tradition": tradition,
                         "unit_index": i, **u})

    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)

    for lg in ("kn", "en"):
        sub = [r for r in rows if r["language"] == lg]
        if sub:
            print(f"  {lg}: {len(sub):5} units  "
                  f"{sum(r['n_words'] for r in sub):7} words  "
                  f"avg {sum(r['n_words'] for r in sub)/len(sub):5.1f} w/unit")
    print(f"  videos: {len({r['video_file'] for r in rows})}   total units: {len(rows)}")


if __name__ == "__main__":
    main()
