"""
Step 6: Streamlit Web UI
========================
Interactive Q&A interface for the Gita RAG pipeline.

Usage:
    streamlit run 06_app.py
"""
import json
import os
import sys
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
from retrieve_lib import retrieve_for_eval, detect_language


def init_db():
    import chromadb
    client = chromadb.PersistentClient(path=config.CHROMA_DIR)
    return client.get_collection(config.COLLECTION_NAME)


def main():
    st.set_page_config(
        page_title="Gita Discourse Q&A",
        page_icon="🙏",
        layout="wide"
    )

    st.title("🙏 Bhagavad Gita Discourse Q&A")
    st.caption("Śloka-aware, Dvaita tradition | Chapter 2 (BG 2.1–2.14) | Kannada discourses")

    # Sidebar
    with st.sidebar:
        st.header("About")
        st.markdown("""
        This system answers questions about **Bhagavad Gita Chapter 2** 
        based on Kannada discourses from the **Dvaita (Madhva) tradition**.
        
        **Features:**
        - Ask in English or Kannada
        - Verse-level citations
        - Grounded in actual discourse content
        - Sanskrit śloka + padavibhāga display
        """)

        st.header("Settings")
        top_k = st.slider("Number of sources to retrieve", 1, 10, config.TOP_K)
        show_sources = st.checkbox("Show source details", value=True)

        st.header("Coverage")
        st.markdown("**Verses:** BG 2.1 – 2.14")
        st.markdown("**Videos:** 6 Kannada discourses")
        st.markdown("**Tradition:** Dvaita (Madhva)")

    # Load chunks for metadata display
    chunks_data = {}
    if os.path.exists(config.CHUNKS_JSON):
        with open(config.CHUNKS_JSON, "r", encoding="utf-8") as f:
            for chunk in json.load(f):
                chunks_data[chunk["chunk_id"]] = chunk

    # Main interface
    col1, col2 = st.columns([3, 1])
    with col1:
        query = st.text_input(
            "Ask a question (English or Kannada / ಕನ್ನಡದಲ್ಲಿ ಪ್ರಶ್ನೆ ಕೇಳಿ)",
            placeholder="e.g., What does Krishna say about the soul? / ಆತ್ಮದ ಬಗ್ಗೆ ಕೃಷ್ಣ ಏನು ಹೇಳುತ್ತಾನೆ?"
        )
    with col2:
        st.write("")
        st.write("")
        ask_button = st.button("🔍 Ask", type="primary", use_container_width=True)

    # Example questions
    with st.expander("Example questions"):
        examples = [
            "What does Krishna say about grief and sorrow?",
            "Why does Arjuna refuse to fight?",
            "What is the meaning of BG 2.14?",
            "What is abhimana according to Madhvacharya?",
            "How does Krishna describe the soul?",
            "ಅರ್ಜುನನ ದುಃಖಕ್ಕೆ ಕಾರಣ ಏನು?",
        ]
        for ex in examples:
            if st.button(ex, key=f"ex_{ex[:20]}"):
                query = ex
                ask_button = True

    if ask_button and query:
        try:
            collection = init_db()
        except Exception as e:
            st.error(f"Database not initialized. Run the pipeline first (steps 01-03). Error: {e}")
            return

        with st.spinner("Searching discourses..."):
            lang = detect_language(query)
            if lang == "kn":
                st.info("ಕನ್ನಡ ಪ್ರಶ್ನೆ ಪತ್ತೆಯಾಗಿದೆ. ಇಂಗ್ಲಿಷ್‌ಗೆ ಅನುವಾದಿಸಿ ಹುಡುಕುತ್ತಿದ್ದೇನೆ...")

            retrieved, answer = retrieve_for_eval(query, collection, top_k=top_k)

        # Display answer
        st.markdown("### Answer")
        st.markdown(answer)

        # Display cited verses
        cited = list(set(c.get("verse_ref", "") for c in retrieved if c.get("verse_ref")))
        if cited:
            st.markdown(f"**Cited verses:** {', '.join(cited)}")

        # Display sources
        if show_sources and retrieved:
            st.markdown("### Sources")
            for i, chunk in enumerate(retrieved):
                verse = chunk.get("verse_ref", "N/A")
                sim = chunk.get("similarity", 0)
                meta = chunk.get("metadata", {})
                chunk_detail = chunks_data.get(chunk["chunk_id"], {})

                with st.expander(
                    f"Source {i+1}: {verse} | {meta.get('section_type', '')} | "
                    f"Similarity: {sim:.3f}"
                ):
                    # Sanskrit shloka
                    if chunk_detail.get("sanskrit_shloka"):
                        st.markdown("**Sanskrit Śloka:**")
                        st.code(chunk_detail["sanskrit_shloka"], language=None)

                    # Padavibhaga
                    if chunk_detail.get("padavibhaga"):
                        st.markdown("**Padavibhāga (word split):**")
                        st.code(chunk_detail["padavibhaga"], language=None)

                    # English explanation
                    st.markdown("**Explanation (English):**")
                    st.write(chunk_detail.get("explanation_summary_en", chunk["document"]))

                    # Kannada text
                    if chunk_detail.get("kannada_text"):
                        st.markdown("**Original Kannada discourse:**")
                        st.write(chunk_detail["kannada_text"][:500] + "...")

                    st.caption(f"Video: {meta.get('video_file', '')} | "
                             f"Speaker: {meta.get('speaker', '')} | "
                             f"Tradition: {meta.get('tradition', 'Dvaita')}")

    # Footer
    st.markdown("---")
    st.caption("Built for PES University Capstone Project | Dvaita Tradition Discourses on BG Chapter 2")


if __name__ == "__main__":
    main()
