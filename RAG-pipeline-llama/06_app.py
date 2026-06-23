"""
06_app.py — GitaGround: Cross-Tradition Bhagavad Gita Q&A
"""
import os
import re
import sys
import tempfile
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
from retrieve_lib import (
    detect_language,
    retrieve_for_eval,
    retrieve_cross_tradition,
    generate_cross_tradition_answer,
    retrieve_chunks,
    build_context,
    generate_answer_with_prompt,
    SUMMARY_PROMPT,
    TRADITION_PROMPT,
    DVAITA_DETAILS,
    ADVAITA_DETAILS,
)

# ─────────────────────────────────────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────────────────────────────────────

DARK_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

/* ── Base ── */
html, body, [data-testid="stApp"] {
    background-color: #0e0e0e !important;
    color: #f9fafb !important;
    font-family: 'Inter', 'Segoe UI', system-ui, sans-serif !important;
}
.block-container {
    padding-top: 1.5rem !important;
    max-width: 1100px !important;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background-color: #111111 !important;
    border-right: 1px solid #1f1f1f !important;
}
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] .stMarkdown { color: #d1d5db !important; }
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 { color: #f9fafb !important; }

/* ── Headings ── */
h1, h2, h3, h4 { color: #f9fafb !important; font-family: 'Inter', sans-serif !important; }
p, li { color: #e5e7eb !important; }

/* ── Dividers ── */
hr { border: none !important; border-top: 1px solid #1f1f1f !important; margin: 1.5rem 0 !important; }

/* ── Text Input ── */
.stTextInput > div > div > input {
    background-color: #161616 !important;
    color: #f9fafb !important;
    border: 2px solid #2d2d2d !important;
    border-radius: 14px !important;
    padding: 14px 20px !important;
    font-size: 1rem !important;
    font-family: 'Inter', sans-serif !important;
    transition: border-color .2s, box-shadow .2s !important;
}
.stTextInput > div > div > input:focus {
    border-color: #7c3aed !important;
    box-shadow: 0 0 0 4px rgba(124,58,237,.18) !important;
    outline: none !important;
}
.stTextInput > label { color: #6b7280 !important; font-size: .85rem !important; }

/* ── Primary Button ── */
.stButton > button[kind="primary"],
button[data-testid="baseButton-primary"] {
    background: linear-gradient(135deg,#7c3aed,#6d28d9) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 12px 28px !important;
    font-weight: 700 !important;
    font-size: 1rem !important;
    letter-spacing: .02em !important;
    box-shadow: 0 4px 16px rgba(124,58,237,.35) !important;
    transition: all .2s !important;
}
.stButton > button[kind="primary"]:hover,
button[data-testid="baseButton-primary"]:hover {
    background: linear-gradient(135deg,#8b5cf6,#7c3aed) !important;
    box-shadow: 0 6px 24px rgba(124,58,237,.55) !important;
    transform: translateY(-1px) !important;
}

/* ── Secondary / example buttons ── */
.stButton > button:not([kind="primary"]) {
    background: #1a1a1a !important;
    color: #9ca3af !important;
    border: 1px solid #2a2a2a !important;
    border-radius: 8px !important;
    font-size: .8rem !important;
    transition: all .2s !important;
}
.stButton > button:not([kind="primary"]):hover {
    border-color: #7c3aed !important;
    color: #c4b5fd !important;
    background: #1e1426 !important;
}

/* ── Radio → pill buttons ── */
[data-testid="stRadio"] > div {
    display: flex !important;
    flex-direction: row !important;
    gap: 10px !important;
    flex-wrap: wrap !important;
    margin-top: 6px !important;
}
[data-testid="stRadio"] > div > label {
    background: #1a1a1a !important;
    border: 2px solid #2d2d2d !important;
    border-radius: 100px !important;
    padding: 8px 24px !important;
    cursor: pointer !important;
    font-size: .875rem !important;
    font-weight: 500 !important;
    color: #9ca3af !important;
    transition: all .2s !important;
}
[data-testid="stRadio"] > div > label:hover {
    border-color: #7c3aed !important;
    color: #c4b5fd !important;
}
[data-testid="stRadio"] > div [data-testid="stMarkdownContainer"] p { color: inherit !important; }

/* ── Expanders (source chunks) ── */
details {
    background: #141414 !important;
    border: 1px solid #222 !important;
    border-radius: 10px !important;
    margin-bottom: 8px !important;
    overflow: hidden !important;
    transition: border-color .2s !important;
}
details:hover { border-color: rgba(124,58,237,.35) !important; }
details > summary {
    background: #1a1a1a !important;
    padding: 11px 16px !important;
    cursor: pointer !important;
    color: #9ca3af !important;
    font-size: .82rem !important;
    font-weight: 500 !important;
}
details > summary:hover { color: #c4b5fd !important; }
details[open] { border-color: rgba(124,58,237,.3) !important; }
details[open] > summary { color: #a78bfa !important; }

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    background: #161616 !important;
    border-radius: 12px !important;
    padding: 4px !important;
    gap: 4px !important;
    border: 1px solid #222 !important;
}
.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    color: #9ca3af !important;
    border-radius: 8px !important;
    padding: 8px 22px !important;
    font-weight: 500 !important;
    border: none !important;
    transition: all .2s !important;
}
.stTabs [data-baseweb="tab"]:hover { color: #c4b5fd !important; background: #1f1f1f !important; }
.stTabs [aria-selected="true"] {
    background: #7c3aed !important;
    color: #fff !important;
    box-shadow: 0 2px 10px rgba(124,58,237,.4) !important;
}
.stTabs [data-baseweb="tab-panel"] { background: transparent !important; padding-top: 16px !important; }

/* ── Metrics ── */
[data-testid="metric-container"] {
    background: #161616 !important;
    border: 1px solid #222 !important;
    border-radius: 12px !important;
    padding: 14px !important;
}
[data-testid="stMetricValue"] { color: #7c3aed !important; font-weight: 700 !important; }
[data-testid="stMetricLabel"] { color: #6b7280 !important; }

/* ── Alerts ── */
[data-testid="stInfo"] {
    background: rgba(124,58,237,.1) !important;
    border-left: 4px solid #7c3aed !important;
    border-radius: 0 8px 8px 0 !important;
}
[data-testid="stWarning"] {
    background: rgba(245,158,11,.1) !important;
    border-left: 4px solid #f59e0b !important;
    border-radius: 0 8px 8px 0 !important;
}
[data-testid="stError"] {
    background: rgba(239,68,68,.1) !important;
    border-left: 4px solid #ef4444 !important;
    border-radius: 0 8px 8px 0 !important;
}
[data-testid="stSuccess"] {
    background: rgba(16,185,129,.1) !important;
    border-left: 4px solid #10b981 !important;
    border-radius: 0 8px 8px 0 !important;
}

/* ── Select, checkbox, file upload ── */
[data-testid="stSelectbox"] > div > div {
    background: #1a1a1a !important;
    border-color: #2d2d2d !important;
    border-radius: 8px !important;
    color: #f9fafb !important;
}
[data-testid="stSelectbox"] span { color: #f9fafb !important; }
[data-testid="stCheckbox"] label { color: #d1d5db !important; }
[data-testid="stSlider"] > label { color: #9ca3af !important; }
[data-testid="stFileUploader"] {
    background: #1a1a1a !important;
    border: 2px dashed #2d2d2d !important;
    border-radius: 10px !important;
}
[data-testid="stFileUploader"]:hover { border-color: #7c3aed !important; }

/* ── Captions ── */
small, [data-testid="stCaptionContainer"] { color: #6b7280 !important; }

/* ── Code inline ── */
code {
    background: #1e1530 !important;
    color: #c4b5fd !important;
    border-radius: 4px !important;
    padding: 2px 6px !important;
    font-size: .875em !important;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: #0e0e0e; }
::-webkit-scrollbar-thumb { background: #2d2d2d; border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: #7c3aed; }

/* ── Hide Streamlit chrome ── */
#MainMenu { visibility: hidden; }
footer { visibility: hidden; }
header { visibility: hidden; }

/* ── Custom card components ── */
.result-card {
    background: #141414;
    border-radius: 16px;
    padding: 24px 28px 20px;
    margin: 14px 0;
    border: 1px solid rgba(124,58,237,.2);
    box-shadow: 0 0 28px rgba(124,58,237,.07);
}
.summary-card {
    border-color: rgba(124,58,237,.35);
    box-shadow: 0 0 32px rgba(124,58,237,.13);
}
.dvaita-card {
    border-color: rgba(245,158,11,.3);
    box-shadow: 0 0 24px rgba(245,158,11,.08);
}
.advaita-card {
    border-color: rgba(6,182,212,.3);
    box-shadow: 0 0 24px rgba(6,182,212,.08);
}
.card-header {
    display: flex;
    align-items: center;
    gap: 12px;
    padding-bottom: 14px;
    margin-bottom: 6px;
    border-bottom: 1px solid #1f1f1f;
}
.card-icon { font-size: 1.4rem; }
.card-title { font-size: 1.15rem; font-weight: 700; margin: 0; line-height: 1.2; }
.card-subtitle { font-size: .78rem; color: #6b7280; margin-top: 3px; }
.summary-title { color: #a78bfa; }
.dvaita-title  { color: #f59e0b; }
.advaita-title { color: #22d3ee; }
.card-body { color: #e5e7eb; font-size: .97rem; line-height: 1.8; }
.card-body p  { margin: 0 0 .75em; }
.card-body ul { margin: 0 0 .75em 1.2em; }
.card-body li { margin-bottom: .3em; }
.card-body strong { color: #f9fafb; }
.card-body em    { color: #c4b5fd; }
.card-body h3, .card-body h4 { color: #f9fafb; margin: .8em 0 .4em; }
.card-body code { background:#1e1530; color:#c4b5fd; border-radius:4px; padding:2px 6px; font-size:.875em; }

.verse-badges { margin-top: 14px; display: flex; flex-wrap: wrap; gap: 6px; align-items: center; }
.badge-label { color: #6b7280; font-size: .78rem; }
.verse-badge {
    display: inline-block;
    background: rgba(124,58,237,.14);
    border: 1px solid rgba(124,58,237,.3);
    color: #c4b5fd;
    padding: 3px 10px;
    border-radius: 100px;
    font-size: .78rem;
    font-weight: 500;
    font-family: monospace;
}
.verse-badge-d {
    background: rgba(245,158,11,.12);
    border-color: rgba(245,158,11,.3);
    color: #fbbf24;
}
.verse-badge-a {
    background: rgba(6,182,212,.12);
    border-color: rgba(6,182,212,.3);
    color: #22d3ee;
}
.ts-badge {
    display: inline-block;
    background: rgba(16,185,129,.12);
    border: 1px solid rgba(16,185,129,.3);
    color: #34d399;
    padding: 2px 9px;
    border-radius: 100px;
    font-size: .75rem;
    font-weight: 500;
    text-decoration: none;
}
a.ts-badge:hover { background: rgba(16,185,129,.22); }

.section-label {
    color: #6b7280;
    font-size: .78rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: .1em;
    margin: 6px 0 14px;
}
.hero-section {
    text-align: center;
    padding: 1.5rem 1rem 1rem;
    margin-bottom: 1rem;
}
.hero-title {
    font-size: 2.8rem !important;
    font-weight: 800 !important;
    background: linear-gradient(135deg,#7c3aed,#a78bfa,#c4b5fd) !important;
    -webkit-background-clip: text !important;
    -webkit-text-fill-color: transparent !important;
    background-clip: text !important;
    margin: 0.3rem 0 0 !important;
    letter-spacing: -.02em !important;
    line-height: 1.1 !important;
}
.hero-tagline {
    color: #6b7280 !important;
    -webkit-text-fill-color: #6b7280 !important;
    font-size: 1.05rem !important;
    margin-top: .6rem !important;
    font-weight: 400 !important;
}
.app-footer {
    text-align: center;
    padding: 2rem 0 1rem;
    color: #374151;
    font-size: .78rem;
    border-top: 1px solid #1a1a1a;
    margin-top: 3rem;
}
</style>
"""

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _md_to_html(text: str) -> str:
    """Convert basic markdown patterns to HTML for embedding inside card divs."""
    # Headers
    text = re.sub(r'^#### (.+)$', r'<h4>\1</h4>', text, flags=re.MULTILINE)
    text = re.sub(r'^### (.+)$',  r'<h4>\1</h4>', text, flags=re.MULTILINE)
    text = re.sub(r'^## (.+)$',   r'<h3>\1</h3>', text, flags=re.MULTILINE)
    # Bold / italic / code
    text = re.sub(r'\*\*\*(.+?)\*\*\*', r'<strong><em>\1</em></strong>', text)
    text = re.sub(r'\*\*(.+?)\*\*',     r'<strong>\1</strong>', text)
    text = re.sub(r'\*(.+?)\*',         r'<em>\1</em>', text)
    text = re.sub(r'`(.+?)`',           r'<code>\1</code>', text)
    # Bullet lists
    lines = text.split('\n')
    out, in_list = [], False
    for line in lines:
        stripped = line.strip()
        if stripped.startswith('- ') or stripped.startswith('* '):
            if not in_list:
                out.append('<ul>')
                in_list = True
            out.append(f'<li>{stripped[2:]}</li>')
        elif stripped.startswith('• '):
            if not in_list:
                out.append('<ul>')
                in_list = True
            out.append(f'<li>{stripped[2:]}</li>')
        else:
            if in_list:
                out.append('</ul>')
                in_list = False
            out.append(line)
    if in_list:
        out.append('</ul>')
    text = '\n'.join(out)
    # Paragraphs
    paras = re.split(r'\n{2,}', text.strip())
    result = []
    for p in paras:
        p = p.strip()
        if not p:
            continue
        if re.match(r'^<(h[2-4]|ul|li|ol)', p):
            result.append(p)
        else:
            result.append(f'<p>{p}</p>')
    return '\n'.join(result)


def _verse_html(verses, badge_cls='verse-badge'):
    if not verses:
        return ''
    badges = ''.join(f'<span class="{badge_cls}">{v}</span>' for v in verses)
    return (
        f'<div class="verse-badges">'
        f'<span class="badge-label">Verses:</span>{badges}</div>'
    )


def init_db():
    import chromadb
    client = chromadb.PersistentClient(path=config.CHROMA_DIR)
    return client.get_collection(config.COLLECTION_NAME)


def get_tradition_counts(collection):
    try:
        dvaita  = collection.get(where={"tradition": "Dvaita"})
        advaita = collection.get(where={"tradition": "Advaita"})
        return len(dvaita["ids"]), len(advaita["ids"])
    except Exception:
        return 0, 0


def extract_verses(chunks):
    verses = set()
    for c in chunks:
        vr = c.get("verse_ref", "") or c.get("metadata", {}).get("verse_ref", "")
        if vr:
            for v in vr.split(","):
                v = v.strip()
                if v:
                    verses.add(v)
    return sorted(verses)


def _fmt_ts(seconds) -> str:
    try:
        s = int(float(seconds))
        return f"{s // 60}:{s % 60:02d}"
    except (TypeError, ValueError):
        return ""


def _timestamp_html(meta: dict) -> str:
    ts_start   = meta.get("start_time_sec")
    ts_end     = meta.get("end_time_sec")
    youtube_url = meta.get("youtube_url", "")
    if ts_start is None:
        return ""
    label = f"▶ {_fmt_ts(ts_start)}–{_fmt_ts(ts_end)}"
    if youtube_url:
        t       = int(float(ts_start))
        vid_id  = meta.get("video_id", "")
        base    = youtube_url.split("&")[0].split("?")[0]
        url     = (f"https://www.youtube.com/watch?v={vid_id}&t={t}s"
                   if vid_id else f"{base}?t={t}s")
        return f'<a href="{url}" target="_blank" class="ts-badge">{label}</a>'
    return f'<span class="ts-badge">{label}</span>'


def display_source_chunks(chunks, tradition_label):
    if not chunks:
        st.caption(f"No {tradition_label} sources retrieved.")
        return
    for i, chunk in enumerate(chunks):
        meta    = chunk.get("metadata", {})
        verse   = meta.get("verse_ref", "N/A")
        sim     = chunk.get("similarity", 0)
        speaker = meta.get("speaker", "Unknown")
        section = meta.get("section_type", "")
        ts_html = _timestamp_html(meta)

        header = f"Source {i+1}: {verse}  ·  {speaker}  ·  {section}  ·  score {sim:.3f}"
        with st.expander(header):
            st.markdown(chunk["document"][:1000])
            if len(chunk["document"]) > 1000:
                st.caption("(truncated — full text was sent to LLM)")
            footer = f"Video: {meta.get('video_file', '')}  ·  {tradition_label}"
            if ts_html:
                st.markdown(
                    f'<div style="display:flex;justify-content:space-between;align-items:center;margin-top:10px">'
                    f'<span style="color:#6b7280;font-size:.78rem">{footer}</span>{ts_html}</div>',
                    unsafe_allow_html=True,
                )
            else:
                st.caption(footer)


_SCOPE_NOTICE = (
    "This question appears to be outside the scope of the Bhagavad Gita Chapter 2 "
    "discourse corpus. No sufficiently relevant segments were found. Please ask a "
    "question related to Chapter 2 teachings, verses, or philosophy."
)


def get_indexed_sources(collection) -> list[dict]:
    """Return distinct video sources (video_id + title) already in ChromaDB."""
    try:
        results = collection.get(include=["metadatas"])
        seen, sources = set(), []
        for meta in results.get("metadatas", []):
            vid = meta.get("video_id", "")
            if vid and vid not in seen:
                seen.add(vid)
                sources.append({
                    "video_id": vid,
                    "tradition": meta.get("tradition", ""),
                    "youtube_url": meta.get("youtube_url", ""),
                })
        return sorted(sources, key=lambda s: s["video_id"])
    except Exception:
        return []


def _render_remove_source_panel(collection):
    sources = get_indexed_sources(collection)
    if not sources:
        st.caption("No indexed sources yet.")
        return

    options = {
        f"{s['video_id']} ({s['tradition']})": s["video_id"]
        for s in sources
    }
    selected_label = st.selectbox(
        "Select source to remove",
        list(options.keys()),
        key="remove_source_select",
    )
    selected_video_id = options[selected_label]

    if st.button("🗑 Remove source", key="remove_source_btn", use_container_width=True):
        try:
            collection.delete(where={"video_id": selected_video_id})
            st.success(f"Removed all chunks for **{selected_video_id}**. Refresh to update counts.")
        except Exception as e:
            st.error(f"Failed to remove source: {e}")


def _render_ingest_panel():
    ingest_tradition = st.selectbox(
        "Tradition of new source",
        options=["Advaita (English)", "Dvaita (Kannada)"],
        key="ingest_tradition",
    )
    tradition_key = "Advaita" if "Advaita" in ingest_tradition else "Dvaita"

    source_type = st.radio(
        "Source type", ["YouTube URL", "Upload audio file"],
        key="ingest_source_type", horizontal=True,
    )
    youtube_url   = ""
    uploaded_file = None

    if source_type == "YouTube URL":
        youtube_url = st.text_input(
            "YouTube URL", key="ingest_url",
            placeholder="https://www.youtube.com/watch?v=...",
        )
    else:
        uploaded_file = st.file_uploader(
            "Audio file (mp3 / wav / m4a)",
            type=["mp3", "wav", "m4a", "ogg", "webm"],
            key="ingest_file",
        )
        youtube_url_opt = st.text_input(
            "YouTube URL (optional — for timestamp links)",
            key="ingest_file_url",
            placeholder="https://www.youtube.com/watch?v=...",
        )
        if youtube_url_opt:
            youtube_url = youtube_url_opt

    if st.button("⬆ Transcribe & Index", key="ingest_btn", use_container_width=True):
        if source_type == "YouTube URL" and not youtube_url.strip():
            st.error("Please enter a YouTube URL.")
            return
        if source_type == "Upload audio file" and uploaded_file is None:
            st.error("Please upload an audio file.")
            return
        try:
            collection = init_db()
        except Exception as e:
            st.error(f"ChromaDB not available: {e}")
            return

        from ingest_pipeline import ingest_from_url, ingest_from_file

        status_area = st.empty()
        log_lines   = []

        def _cb(step, detail):
            icon = {"download": "⬇", "transcribe": "🎙", "validate": "✅",
                    "detect": "🔍", "chunk": "✂", "index": "💾"}.get(step, "•")
            log_lines.append(f"{icon} {detail}")
            status_area.markdown("\n\n".join(log_lines))

        try:
            if source_type == "YouTube URL":
                result = ingest_from_url(
                    youtube_url.strip(), tradition_key, collection,
                    progress_callback=_cb,
                )
            else:
                suffix = os.path.splitext(uploaded_file.name)[1]
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                    tmp.write(uploaded_file.read())
                    tmp_path = tmp.name
                result = ingest_from_file(
                    tmp_path, tradition_key, collection,
                    video_label=os.path.splitext(uploaded_file.name)[0],
                    youtube_url=youtube_url,
                    progress_callback=_cb,
                )
                os.unlink(tmp_path)

            st.success(
                f"✓ Indexed **{result['num_chunks']}** chunks from "
                f"_{result['title']}_ ({result['duration_sec'] / 60:.1f} min). "
                f"Refresh the page to update corpus counts."
            )
        except Exception as e:
            st.error(f"Ingestion failed: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# Card renderers
# ─────────────────────────────────────────────────────────────────────────────

def _card(title, subtitle, icon, title_cls, card_cls, body_html, verse_html=''):
    st.markdown(f"""
<div class="result-card {card_cls}">
  <div class="card-header">
    <span class="card-icon">{icon}</span>
    <div>
      <div class="card-title {title_cls}">{title}</div>
      <div class="card-subtitle">{subtitle}</div>
    </div>
  </div>
  <div class="card-body">{body_html}</div>
  {verse_html}
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    st.set_page_config(
        page_title="GitaGround",
        page_icon="🕉️",
        layout="wide",
    )
    st.markdown(DARK_CSS, unsafe_allow_html=True)

    # ── Hero ──
    st.markdown("""
<div class="hero-section">
  <div style="font-size:2.4rem">🕉️</div>
  <h1 class="hero-title">GitaGround</h1>
  <p class="hero-tagline">Explore Bhagavad Gita Chapter&nbsp;2 across Dvaita &amp; Advaita traditions</p>
</div>
""", unsafe_allow_html=True)

    # ── Sidebar ──
    with st.sidebar:
        st.markdown("### ⚙️ Settings")
        show_sources = st.checkbox("Show source segments", value=True)
        top_k = st.slider(
            "Sources per tradition", min_value=1, max_value=10, value=5,
            help="Discourse segments retrieved from each tradition",
        )

        st.markdown("---")
        st.markdown("### 📊 Corpus")
        try:
            collection  = init_db()
            d_count, a_count = get_tradition_counts(collection)
            c1, c2 = st.columns(2)
            with c1: st.metric("Dvaita",  d_count)
            with c2: st.metric("Advaita", a_count)
            if a_count == 0:
                st.warning("No Advaita chunks yet. Run `python index_advaita.py`.")
        except Exception:
            st.error("ChromaDB not initialized. Run the pipeline first.")
            d_count, a_count = 0, 0

        st.markdown("---")
        st.markdown("### ➕ Add New Source")
        _render_ingest_panel()

        st.markdown("---")
        st.markdown("### 🗑 Remove Indexed Source")
        try:
            _collection_for_remove = init_db()
            _render_remove_source_panel(_collection_for_remove)
        except Exception:
            st.caption("Database unavailable.")

        st.markdown("---")
        st.markdown(
            '<div style="color:#4b5563;font-size:.75rem;line-height:1.7">'
            '<strong style="color:#6b7280">Dvaita</strong> · Madhva · Kannada<br>'
            'Eternal distinction of soul &amp; God<br><br>'
            '<strong style="color:#6b7280">Advaita</strong> · Shankara · English<br>'
            'Swami Sarvapriyananda, Vedanta Society NY<br>'
            'Atman IS Brahman</div>',
            unsafe_allow_html=True,
        )

    # ── Search bar + tradition selector (centred) ──
    _, center, _ = st.columns([1, 5, 1])
    with center:
        query = st.text_input(
            "question",
            placeholder=(
                "e.g., What is the nature of the soul according to Krishna?  /  "
                "ಆತ್ಮದ ಬಗ್ಗೆ ಕೃಷ್ಣ ಏನು ಹೇಳುತ್ತಾನೆ?"
            ),
            label_visibility="collapsed",
        )

        tradition_mode = st.radio(
            "Tradition",
            options=["Both Traditions", "Dvaita Only", "Advaita Only"],
            index=0,
            horizontal=True,
            label_visibility="collapsed",
        )

        with st.expander("💡 Example questions"):
            examples = [
                "What does Krishna teach about the eternal nature of the soul?",
                "How should we deal with pleasure and pain according to BG 2.14?",
                "Why does Arjuna refuse to fight?",
                "What is the meaning of Sthitaprajna (person of steady wisdom)?",
                "Explain the concept of Karma Yoga from Chapter 2.",
                "What happens to the soul after death?",
                "ಆತ್ಮದ ಬಗ್ಗೆ ಕೃಷ್ಣ ಏನು ಹೇಳುತ್ತಾನೆ?",
            ]
            ec1, ec2 = st.columns(2)
            for i, ex in enumerate(examples):
                with (ec1 if i % 2 == 0 else ec2):
                    if st.button(ex, key=f"ex_{i}", use_container_width=True):
                        st.session_state["query_input"] = ex
                        query = ex

        ask = st.button("Search & Answer", type="primary", use_container_width=True)

    if not (ask and query):
        st.markdown(
            '<div class="app-footer">'
            'PES University Capstone · Multilingual Audio-Grounded Q&A · '
            'Bhagavad Gita Chapter 2 · Cross-Tradition RAG Pipeline'
            '</div>',
            unsafe_allow_html=True,
        )
        return

    # ── DB ──
    try:
        collection = init_db()
    except Exception as e:
        st.error(f"Database error: {e}")
        return

    # ── Scope guard ──
    with st.spinner("Checking corpus relevance…"):
        probe_tradition = (
            None if "Both" in tradition_mode
            else ("Dvaita" if "Dvaita" in tradition_mode else "Advaita")
        )
        probe_chunks = retrieve_chunks(query, collection, top_k=3, tradition=probe_tradition)

    if not probe_chunks:
        st.warning(_SCOPE_NOTICE)
        st.stop()

    st.markdown("---")

    # ════════════════════════════════════════════════════════════════
    # CROSS-TRADITION MODE
    # ════════════════════════════════════════════════════════════════
    if "Both" in tradition_mode:
        with st.spinner("Retrieving from both traditions and generating answers…"):
            result = generate_cross_tradition_answer(query, collection, top_k=top_k)

        if result.get("out_of_scope"):
            st.warning(
                "This question is outside the scope of our Chapter 2 corpus. "
                "Please ask a question related to Bhagavad Gita Chapter 2 teachings, "
                "verses, or philosophy."
            )
            st.stop()

        all_chunks  = result["dvaita_chunks"] + result["advaita_chunks"]
        all_verses  = extract_verses(all_chunks)

        # Summary card
        _card(
            title     = "Summary",
            subtitle  = "Synthesised from both Dvaita and Advaita discourse sources",
            icon      = "📖",
            title_cls = "summary-title",
            card_cls  = "summary-card",
            body_html = _md_to_html(result["summary"]),
            verse_html= _verse_html(all_verses, "verse-badge"),
        )

        st.markdown("---")

        # Tradition columns
        col_d, col_a = st.columns(2)

        with col_d:
            if result["dvaita_chunks"]:
                _card(
                    title     = "Dvaita Perspective",
                    subtitle  = "Madhva tradition · Kannada discourses",
                    icon      = "🔶",
                    title_cls = "dvaita-title",
                    card_cls  = "dvaita-card",
                    body_html = _md_to_html(result["dvaita_answer"]),
                    verse_html= _verse_html(extract_verses(result["dvaita_chunks"]), "verse-badge-d"),
                )
            else:
                st.markdown("""
<div class="result-card dvaita-card">
  <div class="card-header">
    <span class="card-icon">🔶</span>
    <div><div class="card-title dvaita-title">Dvaita Perspective</div>
         <div class="card-subtitle">Madhva tradition · Kannada discourses</div></div>
  </div>
</div>""", unsafe_allow_html=True)
                st.warning("No relevant Dvaita content found for this question.")

        with col_a:
            if result["advaita_chunks"]:
                _card(
                    title     = "Advaita Perspective",
                    subtitle  = "Shankara tradition · Swami Sarvapriyananda lectures",
                    icon      = "🔷",
                    title_cls = "advaita-title",
                    card_cls  = "advaita-card",
                    body_html = _md_to_html(result["advaita_answer"]),
                    verse_html= _verse_html(extract_verses(result["advaita_chunks"]), "verse-badge-a"),
                )
            else:
                st.markdown("""
<div class="result-card advaita-card">
  <div class="card-header">
    <span class="card-icon">🔷</span>
    <div><div class="card-title advaita-title">Advaita Perspective</div>
         <div class="card-subtitle">Shankara tradition · Swami Sarvapriyananda lectures</div></div>
  </div>
</div>""", unsafe_allow_html=True)
                st.warning(
                    "No Advaita content found. "
                    "Have you run `python index_advaita.py`?"
                )

        # Source segments
        if show_sources:
            st.markdown("---")
            st.markdown('<div class="section-label">📚 Retrieved Discourse Segments</div>',
                        unsafe_allow_html=True)
            tab_d, tab_a = st.tabs(["🔶 Dvaita Sources", "🔷 Advaita Sources"])
            with tab_d:
                display_source_chunks(result["dvaita_chunks"], "Dvaita")
            with tab_a:
                display_source_chunks(result["advaita_chunks"], "Advaita")

    # ════════════════════════════════════════════════════════════════
    # SINGLE TRADITION MODE
    # ════════════════════════════════════════════════════════════════
    else:
        tradition  = "Dvaita" if "Dvaita" in tradition_mode else "Advaita"
        is_dvaita  = tradition == "Dvaita"
        card_cls   = "dvaita-card"   if is_dvaita else "advaita-card"
        title_cls  = "dvaita-title"  if is_dvaita else "advaita-title"
        badge_cls  = "verse-badge-d" if is_dvaita else "verse-badge-a"
        icon       = "🔶" if is_dvaita else "🔷"
        subtitle   = ("Madhva tradition · Kannada discourses"
                      if is_dvaita else
                      "Shankara tradition · Swami Sarvapriyananda lectures")

        with st.spinner(f"Searching {tradition} discourses…"):
            chunks  = retrieve_chunks(query, collection, top_k=top_k, tradition=tradition)
            context = build_context(chunks, label=tradition)
            prompt  = TRADITION_PROMPT.format(
                tradition=tradition,
                tradition_details=(DVAITA_DETAILS if is_dvaita else ADVAITA_DETAILS),
            )
            lang    = detect_language(query)
            answer  = generate_answer_with_prompt(query, context, prompt, lang)

        if chunks:
            _card(
                title     = f"{tradition} Perspective",
                subtitle  = subtitle,
                icon      = icon,
                title_cls = title_cls,
                card_cls  = card_cls,
                body_html = _md_to_html(answer),
                verse_html= _verse_html(extract_verses(chunks), badge_cls),
            )
        else:
            st.warning(f"No relevant {tradition} content found for this question.")

        if show_sources:
            st.markdown("---")
            st.markdown('<div class="section-label">📚 Retrieved Discourse Segments</div>',
                        unsafe_allow_html=True)
            display_source_chunks(chunks, tradition)

    # ── Footer ──
    st.markdown(
        '<div class="app-footer">'
        'PES University Capstone · Multilingual Audio-Grounded Q&A · '
        'Bhagavad Gita Chapter 2 · Cross-Tradition RAG Pipeline'
        '</div>',
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()