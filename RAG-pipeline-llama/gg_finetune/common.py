"""Shared helpers for GitaGround fine-tuning. Data formats are documented in README.md."""
import json, os, random

V = 72  # verses in Bhagavad Gita chapter 2
SEED = 42


def load_passages(path="data/passages.jsonl"):
    """Returns {lecture_id: [passage dicts sorted by idx]}; validates the schema."""
    lect = {}
    with open(path, encoding="utf-8") as f:
        for n, line in enumerate(f, 1):
            if not line.strip():
                continue
            p = json.loads(line)
            for k in ("lecture_id", "idx", "lang", "split", "text", "gold"):
                if k not in p:
                    raise ValueError(f"{path}:{n} missing field '{k}'")
            if p["lang"] not in ("kn", "en"):
                raise ValueError(f"{path}:{n} lang must be 'kn' or 'en'")
            if p["split"] not in ("tune", "e2", "e3", "silver"):
                raise ValueError(f"{path}:{n} split must be tune/e2/e3/silver")
            for v in p["gold"]:
                if not (isinstance(v, int) and 1 <= v <= V):
                    raise ValueError(f"{path}:{n} gold verse {v!r} not an int in 1..{V}")
            lect.setdefault(p["lecture_id"], []).append(p)
    for lid, ps in lect.items():
        ps.sort(key=lambda p: p["idx"])
        if [p["idx"] for p in ps] != list(range(len(ps))):
            raise ValueError(f"lecture {lid}: idx must be 0..T-1 with no gaps")
        if len({(p["lang"], p["split"]) for p in ps}) != 1:
            raise ValueError(f"lecture {lid}: lang/split must be constant within a lecture")
    return lect


def load_verses(path="data/verses.json"):
    """Returns {v: {"kn":..., "roman":..., "trans":...}} for v = 1..72."""
    vs = {int(x["v"]): x for x in json.load(open(path, encoding="utf-8"))}
    missing = [v for v in range(1, V + 1) if v not in vs]
    if missing:
        raise ValueError(f"verses.json missing verses {missing}")
    return vs


def verse_text(verses, v, lang):
    """Verse in the script the lecture uses, plus its translation."""
    x = verses[v]
    src = x["kn"] if lang == "kn" else x["roman"]
    return f"BG 2.{v}: {src} || {x['trans']}"


def off_context(ps, t, max_chars=700):
    """Neighbour context for the off-verse classifier (text_b); target goes in text_a."""
    prev = ps[t - 1]["text"][-max_chars:] if t > 0 else ""
    nxt = ps[t + 1]["text"][:max_chars] if t + 1 < len(ps) else ""
    return f"PREVIOUS: {prev} || NEXT: {nxt}"


def load_plan(path="folds.json"):
    return json.load(open(path, encoding="utf-8"))


def rng(seed=SEED):
    return random.Random(seed)
