"""
annotate_app.py — span + mention-type annotation UI for the verse-grounding corpus.

Run:
    ../.venv/bin/python -m streamlit run annotate_app.py --server.fileWatcherType none

Reads  : data/units_v1.csv   (frozen unit grid — never regenerate)
Writes : data/mention_spans.csv  (append-only; safe for concurrent annotators)

Annotation guideline: docs/annotation_guideline_mention_types.md
"""
import csv
import datetime
import os

import pandas as pd
import streamlit as st

from verse_database import VERSES_BG_CH2
from verse_translations import VERSE_TRANSLATIONS

BASE = os.path.dirname(os.path.abspath(__file__))
UNITS = os.path.join(BASE, "data", "units_v1.csv")
OUT = os.path.join(BASE, "data", "mention_spans.csv")

COLS = ["unit_id", "video_file", "annotator", "verse_ref", "mention_type",
        "has_explicit_ref", "uncertain", "notes", "created_at"]

TYPES = {
    "T1": "Verbatim Sanskrit recitation",
    "T2": "Partial quote / key Sanskrit phrase",
    "T3": "Direct translation of the verse",
    "T4": "Paraphrase / explanation, no quoting",
    "T5": "Allusion — the idea, not the verse",
}

# The five thematic blocks of Chapter 2 — narrows 72 verses to ~8 in seconds.
BLOCKS = [
    (1, 10, "Arjuna's despair; he surrenders and asks Krishna to teach him"),
    (11, 30, "The self is eternal; the body perishes; grief is misplaced"),
    (31, 38, "Your duty as a warrior; fight, don't flee"),
    (39, 53, "Act without attachment to results (2.47)"),
    (54, 72, "The sthitaprajña — marks of the steady-minded sage"),
]


def verse_num(ref):
    try:
        return int(str(ref).split(".")[1])
    except (IndexError, ValueError):
        return -1

st.set_page_config(page_title="Verse Grounding Annotator", layout="wide")


@st.cache_data
def load_units():
    return pd.read_csv(UNITS, keep_default_na=False)


@st.cache_data
def verse_options():
    opts = {}
    for v in VERSES_BG_CH2:
        tr = VERSE_TRANSLATIONS.get(v["ref"], [""])[0]
        opts[v["ref"]] = f'{v["ref"]} — {tr[:70]}'
    return opts


def load_done():
    if not os.path.exists(OUT):
        return pd.DataFrame(columns=COLS)
    return pd.read_csv(OUT, keep_default_na=False)


def append(rows):
    is_new = not os.path.exists(OUT)
    with open(OUT, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLS)
        if is_new:
            w.writeheader()
        w.writerows(rows)


def now():
    return datetime.datetime.now().isoformat(timespec="seconds")


units = load_units()
vopts = verse_options()

# ── sidebar ───────────────────────────────────────────────
st.sidebar.header("Annotator")
who = st.sidebar.text_input("Your name").strip()
if not who:
    st.warning("Enter your name in the sidebar to begin.")
    st.stop()

done = load_done()
mine = done[done.annotator == who] if len(done) else done

vid = st.sidebar.selectbox("Video", sorted(units.video_file.unique()))
sub = units[units.video_file == vid].reset_index(drop=True)

reviewed = set(mine.unit_id) if len(mine) else set()
n_done = int(sub.unit_id.isin(reviewed).sum())
st.sidebar.progress(n_done / len(sub))
st.sidebar.caption(f"{n_done}/{len(sub)} units in this video")
st.sidebar.caption(f"{len(reviewed)}/{len(units)} units overall")
st.sidebar.divider()
st.sidebar.caption("**Mention types**")
for t, d in TYPES.items():
    st.sidebar.caption(f"**{t}** — {d}")

# ── current unit ──────────────────────────────────────────
key = f"idx::{who}::{vid}"
if key not in st.session_state:
    todo = [n for n, r in enumerate(sub.itertuples()) if r.unit_id not in reviewed]
    st.session_state[key] = todo[0] if todo else 0
i = min(st.session_state[key], len(sub) - 1)
st.session_state[key] = i
row = sub.iloc[i]

st.markdown(f"### {vid} — unit {i + 1} / {len(sub)}")
if row.time_start != "":
    st.caption(f"⏱ {float(row.time_start):.0f}s – {float(row.time_end):.0f}s")

if i > 0:
    st.caption(f"…{sub.iloc[i - 1].text[-220:]}")
st.markdown(
    "<div style='background:#1c2733;padding:1rem;border-radius:8px;"
    f"font-size:1.12rem;line-height:1.75'>{row.text}</div>",
    unsafe_allow_html=True,
)
if i < len(sub) - 1:
    st.caption(f"{sub.iloc[i + 1].text[:220]}…")

existing = done[done.unit_id == row.unit_id] if len(done) else done
if len(existing):
    st.info(" · ".join(f"{r.annotator}: {r.verse_ref}/{r.mention_type}"
                       for r in existing.itertuples()))

# ── verse reference (lookup aid — deliberately NOT model suggestions) ──
with st.expander("📖 Verse reference — search all 72", expanded=False):
    st.caption("**Where am I in Chapter 2?**")
    for lo, hi, desc in BLOCKS:
        st.caption(f"　**BG 2.{lo}–2.{hi}** — {desc}")
    st.divider()

    q = st.text_input(
        "Search by keyword or verse number",
        placeholder="e.g.  slain   ·   grief   ·   duty   ·   2.47",
        key=f"q{row.unit_id}",
    ).strip()

    if q:
        ql = q.lower()
        asnum = verse_num(q) if "." in q else (int(q) if q.isdigit() else -1)
        hits = []
        for v in VERSES_BG_CH2:
            ref = v["ref"]
            trs = VERSE_TRANSLATIONS.get(ref, [])
            if verse_num(ref) == asnum or any(ql in t.lower() for t in trs):
                hits.append((ref, trs))
        st.caption(f"**{len(hits)} match(es)**")
        for ref, trs in hits[:25]:
            blk = next((f"2.{lo}–2.{hi}" for lo, hi, _ in BLOCKS
                        if lo <= verse_num(ref) <= hi), "?")
            st.markdown(f"**{ref}**  ·  _block {blk}_")
            for t in trs:
                st.caption(t)
        if len(hits) > 25:
            st.caption(f"…and {len(hits) - 25} more — narrow your search.")
    else:
        st.caption("Type a keyword from the passage to find candidate verses.")

# ── labelling ─────────────────────────────────────────────
st.divider()
c1, c2 = st.columns([3, 2])
with c1:
    picked = st.multiselect("Verse(s) mentioned here", list(vopts),
                            format_func=lambda r: vopts[r], key=f"v{row.unit_id}")
with c2:
    mtype = st.radio("Mention type", list(TYPES),
                     format_func=lambda t: f"{t} — {TYPES[t]}", key=f"t{row.unit_id}")
    explicit = st.checkbox("Explicit reference ('verse 12')", key=f"e{row.unit_id}")
    uncertain = st.checkbox("Uncertain", key=f"u{row.unit_id}")
notes = st.text_input("Notes (optional)", key=f"n{row.unit_id}")


def advance():
    st.session_state[key] = min(i + 1, len(sub) - 1)
    st.rerun()


b1, b2, b3 = st.columns(3)

if b1.button("💾 Save label(s) →", type="primary", use_container_width=True):
    if not picked:
        st.error("Pick at least one verse, or use 'No verse here'.")
    else:
        append([{"unit_id": row.unit_id, "video_file": vid, "annotator": who,
                 "verse_ref": r, "mention_type": mtype,
                 "has_explicit_ref": int(explicit), "uncertain": int(uncertain),
                 "notes": notes, "created_at": now()} for r in picked])
        advance()

if b2.button("∅ No verse here →", use_container_width=True):
    append([{"unit_id": row.unit_id, "video_file": vid, "annotator": who,
             "verse_ref": "NONE", "mention_type": "NONE",
             "has_explicit_ref": 0, "uncertain": 0,
             "notes": notes, "created_at": now()}])
    advance()

if b3.button("← Previous", use_container_width=True):
    st.session_state[key] = max(i - 1, 0)
    st.rerun()
