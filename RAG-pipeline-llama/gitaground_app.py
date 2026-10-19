"""
GitaGround app: the three display levels over the Step 5 index (gitaground.py).

Run:
    ../.venv/bin/python -m streamlit run gitaground_app.py --server.fileWatcherType none

This file is a presentation layer (see the build specification). Retrieval,
verse matching and abstention are the engine's; the helpers in gitaground_ui.py
only reshape what the engine returned. No retrieval scores or confidence numbers
are shown, and lecture positions are always labelled approximate.
"""
import streamlit as st

from gitaground import ABSTAIN, GitaGround
from gitaground_ui import (LANGUAGE_NAMES, TRADITION_ORDER, LectureLookup, bare_number,
                           filter_coverage, filter_passages, group_coverage,
                           lecture_language, load_lecture_meta, position_label,
                           segment_rows, speaker_aliases, theme_table, youtube_url)

st.set_page_config(page_title="GitaGround", layout="wide")


@st.cache_resource
def load():
    g = GitaGround("all")
    meta = load_lecture_meta()
    seg_lectures = {}
    for s in g.segments:
        seg_lectures.setdefault(s["primary_verse"], set()).add(s["video_file"])
    return dict(g=g, meta=meta, lookup=LectureLookup(g.index),
                aliases=speaker_aliases(g.index, meta), lang_of=lecture_language(g.index),
                seg_lectures=seg_lectures)


@st.cache_resource
def _cache():
    return {"answer": {}, "llm": {}}


S = load()
g, meta, lookup = S["g"], S["meta"], S["lookup"]


def answer_for(text):
    c = _cache()["answer"]
    if text not in c:
        c[text] = g.answer(text)
    return c[text]


def go_verse(number):
    """Related-verse click: re-run the whole app on that verse number."""
    st.session_state["query"] = number
    st.session_state["submitted"] = number
    st.session_state["view"] = "Ask"


# ---- sidebar: view and display filters ----------------------------------------------
st.sidebar.title("GitaGround")
view = st.sidebar.radio("View", ["Ask", "Browse by theme", "About"], key="view")
st.sidebar.divider()
st.sidebar.caption("Display filters (they hide results; they never change what is found)")
sel_trad = st.sidebar.multiselect("Tradition", TRADITION_ORDER, default=TRADITION_ORDER)
sel_lang = st.sidebar.multiselect("Language", list(LANGUAGE_NAMES.values()),
                                  default=list(LANGUAGE_NAMES.values()))
show_kannada = st.sidebar.checkbox("Show Kannada script of verses", value=False)
trad_f = sel_trad or None
lang_codes = [c for c, n in LANGUAGE_NAMES.items() if n in sel_lang] or None

st.title("GitaGround")
n_lect = len({r["video_file"] for r in g.index})
st.caption(f"Bhagavad Gita chapter 2 · {n_lect} lectures indexed, covering {len(g.by_verse)} of "
           "72 verses · positions in lectures are approximate")


# ---- shared renderers ----------------------------------------------------------------
def speaker_label(speaker):
    return S["aliases"].get(speaker, speaker)


def verse_card(ref, how=None, expanded=False, related=True, uid=""):
    tid, tname = g.themes.get(ref, ("", ""))
    head = f"{ref} · {tid} {tname}".strip() + (f" · {how}" if how else "")
    with st.expander(head, expanded=expanded):
        st.write(g.verses[ref]["devanagari"])
        if show_kannada and g.verses[ref].get("kannada"):
            st.write(g.verses[ref]["kannada"])
        for t in g.translations[ref][:1]:
            st.write(f"*{t}*")
        if related:
            rel = g.related(ref)[:5]
            if rel:
                st.caption("Related verses (appear on the same passages), click to open:")
                cols = st.columns(len(rel))
                for col, (r, n) in zip(cols, rel):
                    col.button(f"{r} ({n})", key=f"rel_{uid}_{ref}_{r}", on_click=go_verse,
                               args=(bare_number(r),), use_container_width=True)
            else:
                st.caption("Related verses: none in the indexed lectures")


def level2(ref, uid=""):
    cov = filter_coverage(g.coverage(ref), trad_f, lang_codes, S["lang_of"])
    groups = group_coverage(cov, S["aliases"])
    if not groups:
        st.info("Nothing for this verse with the current filters.")
        return
    for tradition, speakers in groups:
        st.subheader(tradition)
        for speaker, grp in speakers:
            st.markdown(f"**{speaker}**")
            rows = segment_rows(grp["segments"], meta)
            if rows:
                cfg = {}
                if any(r["watch"] for r in rows):
                    cfg["watch"] = st.column_config.LinkColumn("watch", display_text="open video")
                else:
                    for r in rows:
                        r.pop("watch")
                st.dataframe(rows, hide_index=True, column_config=cfg, key=f"seg_{uid}_{speaker}_{tradition}")
            else:
                st.caption("No lecture of this speaker explains this verse as its main verse.")
            if grp["passing"]:
                st.caption(f"Mentioned in passing in {grp['passing']} other passages.")
    st.caption("Explained over N passages: how many consecutive 100-word passages stay on the verse. "
               "Effective dwell: the same count weighted by how sure the labelling is. "
               "Both are descriptive only; no depth threshold is applied.")


def passage_block(k, i, top, uid=""):
    r = g.index[k]
    where = position_label(r["pos_start"], r["pos_end"], r["video_file"], meta)
    label = (f"[{i}] {speaker_label(r['speaker'])} · {r['video_file']} · passage {r['unit_index']} · "
             f"{where} · {r['primary_verse']}"
             + ("" if r["primary_verse"] == top else " · found by text similarity"))
    with st.expander(label):
        st.write(g.text[r["unit_id"]])
        url = youtube_url(r["video_file"], r["pos_start"], meta)
        if url:
            st.markdown(f"[Open the lecture video]({url})")
        st.caption(f"theme {r['theme_id']} {r['theme_name']}"
                   + (f" · also mentions {r['secondary_verses'].replace(';', ', ')}"
                      if r["secondary_verses"] else ""))
        if st.toggle("Show surrounding passages", key=f"ctx_{uid}_{k}"):
            for off, name in ((-1, "Previous passage"), (1, "Next passage")):
                nk = lookup.neighbour(k, off)
                if nk is None:
                    st.caption(f"{name}: none (edge of the lecture)")
                else:
                    st.markdown(f"**{name}** (passage {g.index[nk]['unit_index']})")
                    st.write(g.text[g.index[nk]["unit_id"]])


# ---- Ask ----------------------------------------------------------------------------
def ask():
    st.text_area("Paste a shloka (any script), an explanation, or ask a question, "
                 "or type a verse number such as 2.47", height=100, key="query")
    mode = st.radio("Summary", ["Extractive (no AI)", "AI summary (GPT-4o-mini)"], horizontal=True)
    if st.button("Submit", type="primary"):
        st.session_state["submitted"] = st.session_state.get("query", "").strip()

    text = st.session_state.get("submitted", "")
    if not text:
        st.caption("Examples: `2.47` · `Why should a wise person not grieve when the body dies?` · "
                   "`nainam chindanti shastrani nainam dahati pavakah`")
        return

    a = answer_for(text)
    ks, top = a["passages"], a["verse"]
    explicit = [x for x in a["ranked"] if x[1] != "meaning profile"]

    if a["note"]:
        (st.error if a["note"] == ABSTAIN else st.warning)(a["note"])
    if top is None and not explicit:
        return                                   # abstention: the note and nothing else

    st.header("1 · Matched verses")
    shown = explicit or a["ranked"][:3]
    for i, (ref, how) in enumerate(shown):
        verse_card(ref, how, expanded=(i == 0), uid="ask")
    if top is None:
        return                                   # a named verse no lecture covers

    st.header(f"2 · Who explains {top}")
    level2(top, uid="ask")

    st.header("3 · Evidence")
    ks_view = filter_passages(ks, g.index, trad_f, lang_codes)
    if not ks_view:
        st.info("No retrieved passage matches the current filters.")
        return
    if len(ks_view) < len(ks):
        st.caption(f"Filters hide {len(ks) - len(ks_view)} of the {len(ks)} retrieved passages.")
    query = a["query_for_summary"]
    if mode.startswith("Extractive"):
        st.caption("Sentences from the retrieved passages closest to your input (no AI).")
        for k, s in g.extractive(query, ks_view):
            r = g.index[k]
            st.markdown(f"> {s}  \n— {speaker_label(r['speaker'])}, {r['video_file']}, "
                        f"{position_label(r['pos_start'], r['pos_end'], r['video_file'], meta)}")
    else:
        key = (query, tuple(ks_view))
        c = _cache()["llm"]
        if key not in c:
            with st.spinner(f"Summarising the {len(ks_view)} passages..."):
                try:
                    c[key] = g.llm_summary(query, ks_view)
                except Exception as e:           # missing API key, no network, ...
                    c[key] = None
                    st.error(f"The AI summary is unavailable ({type(e).__name__}). "
                             "Switch to the extractive summary.")
        if c[key] is not None:
            st.caption(f"AI summary of the {len(ks_view)} passages below")
            (st.info if c[key].strip() == ABSTAIN else st.write)(c[key])
    st.subheader("Passages")
    for i, k in enumerate(ks_view, 1):
        passage_block(k, i, top, uid="ask")


# ---- Browse by theme ---------------------------------------------------------------
def browse():
    themes = theme_table(g.themes)
    tid = st.selectbox("Theme", list(themes), format_func=lambda t: f"{t} · {themes[t][0]}")
    refs = themes[tid][1]

    def verse_label(ref):
        n = len(S["seg_lectures"].get(ref, ()))
        tail = f"explained in {n} lecture{'s' if n != 1 else ''}" if n else \
            "not explained in the indexed lectures"
        return f"{ref} · {tail}"

    st.write(f"{len(refs)} verses · " +
             f"{sum(1 for r in refs if S['seg_lectures'].get(r))} explained in the indexed lectures")
    ref = st.selectbox("Verse", refs, format_func=verse_label)
    verse_card(ref, expanded=True, related=False, uid="browse")
    if not S["seg_lectures"].get(ref):
        st.info(f"{ref} is not explained in the indexed lectures.")
        return
    st.header(f"Who explains {ref}")
    level2(ref, uid="browse")

    cov = filter_coverage(g.coverage(ref), trad_f, lang_codes, S["lang_of"])
    segs = [s for grp in cov.values() for s in grp["segments"]]
    if not segs:
        return
    segs.sort(key=lambda s: (s["video_file"], float(s["pos_start"])))
    st.header("Read a segment")
    seg = st.selectbox(
        "Segment", segs,
        format_func=lambda s: f"{s['video_file']} · {int(s['dwell'])} passages · "
                              f"{position_label(s['pos_start'], s['pos_end'], s['video_file'], meta)}")
    for i, k in enumerate(lookup.segment_ks(seg), 1):
        passage_block(k, i, ref, uid="seg")


# ---- About --------------------------------------------------------------------------
def about():
    st.header("About GitaGround")
    st.markdown("""
**What it does.** You paste a shloka (Devanagari, Kannada script or Roman letters), paste an
explanation, ask a question, or type a verse number. GitaGround returns the verse of
Bhagavad Gita chapter 2 it is about, the verse's theme, which speakers and traditions explain
it and over how many passages, and the lecture passages themselves.

**How it decides.** Verses are found by verse number, by string matching against the verified
verse text, or by similarity to a meaning profile of each verse. An optional AI summary only
rewrites the five passages already retrieved; the AI never chooses a verse or a topic. If the
input is off-topic the app says so instead of answering.

**What is not shown, on purpose.** No similarity scores and no High/Medium/Low confidence bands
(they were never calibrated). Lecture positions are estimates.
""")
    st.subheader("Known limitations")
    st.markdown(f"""
- {n_lect} lectures, 6 speakers, chapter 2 only.
- Verse labels on passages are automatic (strict F1 on held-out lectures: 0.82 Kannada,
  0.60 English); stories inside long explanations are often still labelled with the verse.
- Positions and times are estimates from word counts and assume a constant speaking rate.
- Abstention catches most off-topic questions (21 of 25 in the test set), not all.
- Speakers of the original lectures are shown by placeholder until names are supplied.
""")


{"Ask": ask, "Browse by theme": browse, "About": about}[view]()
