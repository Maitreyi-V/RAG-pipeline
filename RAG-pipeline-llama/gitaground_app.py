"""
GitaGround app: the three display levels over the Step 5 index (gitaground.py).

Run:
    ../.venv/bin/python -m streamlit run gitaground_app.py --server.fileWatcherType none

Currently indexes the 9 TUNING lectures only; the other lectures are added
after the 29 Oct results freeze.
"""
import streamlit as st

from gitaground import GitaGround

st.set_page_config(page_title="GitaGround", layout="wide")


@st.cache_resource
def load():
    return GitaGround("dev")


g = load()
st.title("GitaGround")
st.caption(f"Bhagavad Gita chapter 2 · {len({r['video_file'] for r in g.index})} lectures indexed "
           f"(tuning set), covering {len(g.by_verse)} of 72 verses · positions are approximate "
           "(share of the lecture's words)")

text = st.text_area("Paste a shloka (any script), an explanation, or ask a question", height=100)
mode = st.radio("Summary", ["Extractive (no LLM)", "LLM summary (gpt-4o-mini, passages only)"],
                horizontal=True)

if text.strip():
    a = g.answer(text)
    ks, top = a["passages"], a["verse"]
    explicit = [x for x in a["ranked"] if x[1] != "meaning profile"]
    ranked = explicit or a["ranked"][:3]      # a named verse or pasted shloka: show only those
    if a["note"]:
        st.warning(a["note"])

    # ---- Level 1 ------------------------------------------------------------
    st.header("1 · Matched verses")
    for ref, how in ranked:
        tid, tname = g.themes.get(ref, ("", ""))
        with st.expander(f"{ref} · {tid} {tname} · {how}", expanded=(ref == ranked[0][0])):
            st.write(g.verses[ref]["devanagari"])
            for t in g.translations[ref][:1]:
                st.write(f"*{t}*")
            rel = g.related(ref)[:5]
            st.write("Related verses: " + (", ".join(f"{r} ({n})" for r, n in rel) if rel
                                           else "none in the indexed lectures"))

    # ---- Level 2 ------------------------------------------------------------
    if top is None:
        st.stop()
    st.header(f"2 · Who explains {top}")
    for (speaker, tradition), c in g.coverage(top).items():
        st.subheader(f"{speaker} · {tradition}")
        rows = [{"lecture": s["video_file"], "passages (dwell)": int(s["dwell"]),
                 "effective dwell": float(s["effective_dwell"]),
                 "position": f"{float(s['pos_start']):.0%}–{float(s['pos_end']):.0%}"}
                for s in c["segments"]]
        if rows:
            st.dataframe(rows, hide_index=True)
        if c["passing"]:
            st.write(f"Passing mentions on other verses' passages: {c['passing']}")

    # ---- Level 3 ------------------------------------------------------------
    st.header("3 · Evidence")
    if mode.startswith("Extractive"):
        for k, s in g.extractive(a["query_for_summary"], ks):
            r = g.index[k]
            st.markdown(f"> {s}  \n— {r['speaker']}, {r['video_file']}, "
                        f"~{float(r['pos_start']):.0%} into the lecture")
    else:
        with st.spinner("Summarising the 5 passages..."):
            st.write(g.llm_summary(text if a["query_for_summary"] == text
                                   else f"What do the lectures say about {top}?", ks))
    for i, k in enumerate(ks, 1):
        r = g.index[k]
        label = (f"[{i}] {r['speaker']} · {r['video_file']} · passage {r['unit_index']} · "
                 f"~{float(r['pos_start']):.0%}–{float(r['pos_end']):.0%} · "
                 f"{r['primary_verse']}"
                 + ("" if r["primary_verse"] == top else " · found by text similarity"))
        with st.expander(label):
            st.write(g.text[r["unit_id"]])
            st.caption(f"theme {r['theme_id']} · confidence in verse {float(r['p_primary']):.2f}"
                       + (f" · also mentions {r['secondary_verses']}" if r["secondary_verses"] else ""))
