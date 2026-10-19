"""
gitaground_ui.py: display-only helpers for the GitaGround app.

Nothing here retrieves, ranks, matches or abstains. All of that stays in
gitaground.py (build spec, rule 1). These functions only reshape what the engine
already returned (grouping, labels, links, neighbours, filters), so they need
no model and can be tested on the Step 5 CSVs alone.
"""
import csv
import os
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
META_PATH = os.path.join(HERE, "data", "lecture_meta.csv")
NEW_LECTURES_PATH = os.path.join(HERE, "data", "lectures_new.csv")

TRADITION_ORDER = ["Dvaita", "Advaita", "Gaudiya Vaishnava"]
LANGUAGE_NAMES = {"kn": "Kannada", "en": "English"}


# ---- lecture metadata (spec 6.4) ------------------------------------------------
def _read(path):
    if not os.path.exists(path):
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_lecture_meta(meta_path=META_PATH, new_path=NEW_LECTURES_PATH):
    """{video_file: {speaker, organisation, tradition, youtube_id, duration_sec}}.

    data/lectures_new.csv gives speaker / organisation / YouTube id for the 12
    new-speaker lectures. data/lecture_meta.csv (Maitreyi) overrides it and may
    add durations and names for the other lectures. Blank cells never override
    a known value; a blank duration means "no real time yet".
    """
    meta = {}
    for path in (new_path, meta_path):
        for r in _read(path):
            vf = (r.get("video_file") or "").strip()
            if not vf:
                continue
            cur = meta.setdefault(vf, {"speaker": "", "organisation": "", "tradition": "",
                                       "youtube_id": "", "duration_sec": None})
            for key in ("speaker", "organisation", "tradition", "youtube_id"):
                val = (r.get(key) or "").strip()
                if val:
                    cur[key] = val
            dur = (r.get("duration_sec") or "").strip()
            if dur:
                try:
                    cur["duration_sec"] = float(dur)
                except ValueError:
                    pass
    return meta


def mmss(seconds):
    s = int(round(seconds))
    h, rem = divmod(s, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


def position_label(pos_start, pos_end, video_file, meta):
    """'~MM:SS – MM:SS (approximate)' once a duration is known, else the share of
    the lecture. Always labelled approximate: constant speaking rate assumed."""
    a, b = float(pos_start), float(pos_end)
    dur = (meta.get(video_file) or {}).get("duration_sec")
    if dur:
        return f"~{mmss(a * dur)} – {mmss(b * dur)} (approximate)"
    return f"~{a:.0%}–{b:.0%} of the lecture (approximate)"


def youtube_url(video_file, pos_start, meta):
    """Link to the video; with &t= only when a duration lets us estimate the second."""
    m = meta.get(video_file) or {}
    vid = m.get("youtube_id")
    if not vid:
        return ""
    base = f"https://www.youtube.com/watch?v={vid}"
    if m.get("duration_sec"):
        return f"{base}&t={int(float(pos_start) * m['duration_sec'])}s"
    return base


def speaker_aliases(index, meta):
    """{index speaker label: real name} when every lecture of that label has the same
    real name in the metadata. Placeholders ('original Kannada speaker') stay
    until names are supplied."""
    names = defaultdict(set)
    for r in index:
        names[r["speaker"]].add((meta.get(r["video_file"]) or {}).get("speaker") or "")
    return {label: next(iter(n)) for label, n in names.items() if len(n) == 1 and next(iter(n))}


def lecture_language(index):
    return {r["video_file"]: r["language"] for r in index}


# ---- Level 2 --------------------------------------------------------------------
def group_coverage(cov, aliases=None):
    """Engine coverage {(speaker, tradition): {...}} -> [(tradition, [(speaker, group)])].

    Traditions in a fixed order, speakers within a tradition by their deepest
    segment first (then name). `group` is the engine's dict, untouched."""
    aliases = aliases or {}
    by_trad = defaultdict(list)
    for (speaker, tradition), grp in cov.items():
        by_trad[tradition].append((aliases.get(speaker, speaker), grp))

    def depth(item):
        segs = item[1]["segments"]
        return -max((float(s["effective_dwell"]) for s in segs), default=0.0)

    order = [t for t in TRADITION_ORDER if t in by_trad] + sorted(set(by_trad) - set(TRADITION_ORDER))
    return [(t, sorted(by_trad[t], key=lambda it: (depth(it), it[0]))) for t in order]


def filter_coverage(cov, traditions=None, languages=None, lang_of=None):
    """Display filter on engine coverage. Drops groups of other traditions and
    segments of other languages. Never touches answer()."""
    out = {}
    for (speaker, tradition), grp in cov.items():
        if traditions and tradition not in traditions:
            continue
        segs = grp["segments"]
        passing = grp["passing"]
        if languages and lang_of is not None:
            segs = [s for s in segs if lang_of.get(s["video_file"]) in languages]
            # passing mentions are counted per speaker, not per lecture; a speaker
            # whose segments are all filtered out is dropped, so the count is too
            if not segs and not passing:
                continue
        out[(speaker, tradition)] = {"segments": segs, "passing": passing}
    return out


def segment_rows(segments, meta):
    """Level 2 table rows for one speaker's segments (already deepest first)."""
    rows = []
    for s in segments:
        rows.append({
            "lecture": s["video_file"],
            "explained over": f"{int(s['dwell'])} passages",
            "effective dwell": float(s["effective_dwell"]),
            "where in the lecture": position_label(s["pos_start"], s["pos_end"], s["video_file"], meta),
            "watch": youtube_url(s["video_file"], s["pos_start"], meta) or None,
        })
    return rows


# ---- passages: neighbours and segments (spec 6.5, 6.6) ---------------------------
class LectureLookup:
    """unit_id -> k and (video_file, unit_index) -> k over the engine's index rows."""

    def __init__(self, index):
        self.index = index
        self.by_unit = {r["unit_id"]: k for k, r in enumerate(index)}
        self.by_pos = {(r["video_file"], int(r["unit_index"])): k for k, r in enumerate(index)}

    def neighbour(self, k, offset):
        r = self.index[k]
        return self.by_pos.get((r["video_file"], int(r["unit_index"]) + offset))

    def segment_ks(self, segment):
        """Passages of one segment in lecture order, first_unit ... last_unit."""
        a, b = self.by_unit.get(segment["first_unit"]), self.by_unit.get(segment["last_unit"])
        if a is None or b is None:
            return []
        va, vb = self.index[a]["video_file"], self.index[b]["video_file"]
        if va != vb:
            return []
        lo, hi = int(self.index[a]["unit_index"]), int(self.index[b]["unit_index"])
        ks = [self.by_pos.get((va, i)) for i in range(lo, hi + 1)]
        return [k for k in ks if k is not None]


def filter_passages(ks, index, traditions=None, languages=None):
    """Display filter on Level 3 passages."""
    keep = []
    for k in ks:
        r = index[k]
        if traditions and r["tradition"] not in traditions:
            continue
        if languages and r["language"] not in languages:
            continue
        keep.append(k)
    return keep


# ---- browse mode (spec 6.6) ------------------------------------------------------
def verse_number(ref):
    try:
        return int(str(ref).split(".")[1])
    except (IndexError, ValueError):
        return -1


def theme_table(themes):
    """{theme_id: (theme_name, [verse refs in order])} from g.themes."""
    out = {}
    for ref, (tid, tname) in themes.items():
        out.setdefault(tid, (tname, []))[1].append(ref)
    for tid in out:
        out[tid][1].sort(key=verse_number)
    return dict(sorted(out.items()))


def bare_number(ref):
    """'BG 2.30' -> '2.30', the text that re-runs the app on that verse."""
    return str(ref).replace("BG", "").strip()
