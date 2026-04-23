"""
06_app.py — Cross-Tradition Bhagavad Gita Q&A
==============================================
Impressive UI showing:
  1. Summary (synthesized from both traditions)
  2. Dvaita (Madhva) perspective
  3. Advaita (Shankara) perspective

Supports: tradition toggle, English/Kannada, verse citations, source segments.
"""
import json
import os
import sys
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


def init_db():
    import chromadb
    client = chromadb.PersistentClient(path=config.CHROMA_DIR)
    return client.get_collection(config.COLLECTION_NAME)


def get_tradition_counts(collection):
    """Check how many chunks exist per tradition."""
    try:
        dvaita = collection.get(where={"tradition": "Dvaita"})
        advaita = collection.get(where={"tradition": "Advaita"})
        return len(dvaita["ids"]), len(advaita["ids"])
    except Exception:
        return 0, 0


def extract_verses(chunks):
    """Extract unique verse refs from chunks."""
    verses = set()
    for c in chunks:
        vr = c.get("verse_ref", "") or c.get("metadata", {}).get("verse_ref", "")
        if vr:
            for v in vr.split(","):
                v = v.strip()
                if v:
                    verses.add(v)
    return sorted(verses)


def display_source_chunks(chunks, tradition_label):
    """Display retrieved source chunks in expanders."""
    if not chunks:
        st.caption(f"No {tradition_label} sources retrieved.")
        return

    for i, chunk in enumerate(chunks):
        meta = chunk.get("metadata", {})
        verse = meta.get("verse_ref", "N/A")
        sim = chunk.get("similarity", 0)
        speaker = meta.get("speaker", "Unknown")
        section = meta.get("section_type", "")

        with st.expander(
            f"Source {i+1}: {verse} | {speaker} | "
            f"Type: {section} | Relevance: {sim:.3f}"
        ):
            st.write(chunk["document"][:1000])
            if len(chunk["document"]) > 1000:
                st.caption("(truncated — full text was sent to LLM)")
            st.caption(f"Video: {meta.get('video_file', '')} | Tradition: {tradition_label}")


def main():
    st.set_page_config(
        page_title="Gita Cross-Tradition Q&A",
        page_icon="🙏",
        layout="wide"
    )

    # ── Header ──
    st.title("🙏 Bhagavad Gita — Cross-Tradition Q&A")
    st.caption(
        "Chapter 2 · Dvaita (Madhva) + Advaita (Shankara) · "
        "Śloka-aware RAG · Abstractive answers"
    )

    # ── Sidebar ──
    with st.sidebar:
        st.header("⚙️ Settings")

        tradition_mode = st.radio(
            "Tradition",
            options=["Both (Cross-Tradition)", "Dvaita Only", "Advaita Only"],
            index=0,
            help="Choose which tradition's discourses to search"
        )

        top_k = st.slider(
            "Sources per tradition",
            min_value=1, max_value=10, value=5,
            help="Number of discourse segments to retrieve from each tradition"
        )

        show_sources = st.checkbox("Show source segments", value=True)

        st.markdown("---")
        st.header("📊 Corpus Info")

        try:
            collection = init_db()
            d_count, a_count = get_tradition_counts(collection)
            st.metric("Dvaita chunks", d_count)
            st.metric("Advaita chunks", a_count)
            if a_count == 0:
                st.warning(
                    "No Advaita chunks indexed yet! "
                    "Run `python index_advaita.py` first."
                )
        except Exception:
            st.error("ChromaDB not initialized. Run the pipeline first.")
            d_count, a_count = 0, 0

        st.markdown("---")
        st.header("About")
        st.markdown("""
        **Dvaita** (Madhva): Kannada discourses.
        Eternal distinction between soul and God.

        **Advaita** (Shankara): English lectures by
        Swami Sarvapriyananda, Vedanta Society of NY.
        Non-duality: Atman IS Brahman.

        The system retrieves relevant segments from
        each tradition and generates perspective-specific
        answers using Llama3.
        """)

    # ── Main Input ──
    query = st.text_input(
        "Ask a question about Bhagavad Gita Chapter 2",
        placeholder="e.g., What is the nature of the soul according to Krishna?  /  "
                    "ಆತ್ಮದ ಬಗ್ಗೆ ಕೃಷ್ಣ ಏನು ಹೇಳುತ್ತಾನೆ?"
    )

    # ── Example Questions ──
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
        cols = st.columns(2)
        for i, ex in enumerate(examples):
            with cols[i % 2]:
                if st.button(ex, key=f"ex_{i}", use_container_width=True):
                    st.session_state["query_input"] = ex
                    query = ex

    ask = st.button("🔍 Search & Answer", type="primary", use_container_width=True)

    if not (ask and query):
        return

    # ── Retrieve & Generate ──
    try:
        collection = init_db()
    except Exception as e:
        st.error(f"Database error: {e}")
        return

    st.markdown("---")

    # ────────────────────────────────────────
    # MODE: CROSS-TRADITION (BOTH)
    # ────────────────────────────────────────
    if tradition_mode == "Both (Cross-Tradition)":
        with st.spinner("🔎 Retrieving from both traditions and generating answers..."):
            result = generate_cross_tradition_answer(query, collection, top_k=top_k)

        # ── SUMMARY ──
        st.markdown("## 📖 Summary")
        st.info("Synthesized from both Dvaita and Advaita discourse sources.")
        st.markdown(result["summary"])

        # Cited verses
        all_chunks = result["dvaita_chunks"] + result["advaita_chunks"]
        verses = extract_verses(all_chunks)
        if verses:
            st.markdown(f"**Relevant verses:** {' · '.join(f'`{v}`' for v in verses)}")

        st.markdown("---")

        # ── TRADITION COLUMNS ──
        col_d, col_a = st.columns(2)

        with col_d:
            st.markdown("## 🔶 Dvaita Perspective")
            st.caption("Madhva tradition · Kannada discourses")
            if result["dvaita_chunks"]:
                st.markdown(result["dvaita_answer"])
                d_verses = extract_verses(result["dvaita_chunks"])
                if d_verses:
                    st.markdown(f"**Verses:** {' · '.join(f'`{v}`' for v in d_verses)}")
            else:
                st.warning("No relevant Dvaita content found for this question.")

        with col_a:
            st.markdown("## 🔷 Advaita Perspective")
            st.caption("Shankara tradition · Swami Sarvapriyananda lectures")
            if result["advaita_chunks"]:
                st.markdown(result["advaita_answer"])
                a_verses = extract_verses(result["advaita_chunks"])
                if a_verses:
                    st.markdown(f"**Verses:** {' · '.join(f'`{v}`' for v in a_verses)}")
            else:
                st.warning(
                    "No Advaita content found. "
                    "Have you run `python index_advaita.py`?"
                )

        # ── Source Segments ──
        if show_sources:
            st.markdown("---")
            st.markdown("## 📚 Retrieved Discourse Segments")
            tab_d, tab_a = st.tabs(["Dvaita Sources", "Advaita Sources"])
            with tab_d:
                display_source_chunks(result["dvaita_chunks"], "Dvaita")
            with tab_a:
                display_source_chunks(result["advaita_chunks"], "Advaita")

    # ────────────────────────────────────────
    # MODE: SINGLE TRADITION
    # ────────────────────────────────────────
    else:
        tradition = "Dvaita" if "Dvaita" in tradition_mode else "Advaita"
        emoji = "🔶" if tradition == "Dvaita" else "🔷"

        with st.spinner(f"🔎 Searching {tradition} discourses..."):
            chunks = retrieve_chunks(query, collection, top_k=top_k, tradition=tradition)
            context = build_context(chunks, label=tradition)

            if tradition == "Dvaita":
                prompt = TRADITION_PROMPT.format(
                    tradition="Dvaita", tradition_details=DVAITA_DETAILS)
            else:
                prompt = TRADITION_PROMPT.format(
                    tradition="Advaita", tradition_details=ADVAITA_DETAILS)

            lang = detect_language(query)
            answer = generate_answer_with_prompt(query, context, prompt, lang)

        st.markdown(f"## {emoji} {tradition} Perspective")
        if tradition == "Dvaita":
            st.caption("Madhva tradition · Kannada discourses")
        else:
            st.caption("Shankara tradition · Swami Sarvapriyananda lectures")

        if chunks:
            st.markdown(answer)
            verses = extract_verses(chunks)
            if verses:
                st.markdown(f"**Relevant verses:** {' · '.join(f'`{v}`' for v in verses)}")
        else:
            st.warning(f"No relevant {tradition} content found for this question.")

        if show_sources:
            st.markdown("---")
            st.markdown("## 📚 Retrieved Discourse Segments")
            display_source_chunks(chunks, tradition)

    # ── Footer ──
    st.markdown("---")
    st.caption(
        "PES University Capstone · Multilingual Audio-Grounded Q&A · "
        "Bhagavad Gita Chapter 2 · Cross-Tradition RAG Pipeline"
    )


if __name__ == "__main__":
    main()
