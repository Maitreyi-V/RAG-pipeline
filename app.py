import streamlit as st
from backend.src.rag import build_index, query_index
from backend.src.llm_client import generate_answer
from backend.src.utils import ms_to_mmss

st.set_page_config(page_title="Capstone RAG – Kannada Gita", layout="wide")
st.title("📚 Capstone RAG – Kannada Gita Teaching Assistant (MVP)")
st.caption("Local pipeline: MP4 → MP3 → Whisper → Chunks → BGE-M3 → Chroma → LM Studio answer + timestamps")

st.divider()

st.subheader("1) Build / Refresh Index")
col1, col2 = st.columns(2)
with col1:
    reset = st.checkbox("Reset index (recommended on first run)", value=True)
with col2:
    top_k = st.number_input("Top-K results", min_value=1, max_value=10, value=3)

if st.button("Build Index"):
    with st.spinner("Indexing chunks into ChromaDB..."):
        result = build_index(reset=reset)
    if result.get("ok"):
        st.success(f"Indexed ✅ Files: {result['num_files']} | Chunks: {result['num_chunks']}")
    else:
        st.error(result.get("message", "Index build failed"))

st.divider()

st.subheader("2) Ask a Question")
q = st.text_input("Enter your question (Kannada / English / Hindi):")

if st.button("Search") and q.strip():
    with st.spinner("Retrieving relevant chunks..."):
        r = query_index(q.strip(), top_k=int(top_k))

    if not r.get("ok"):
        st.error(r.get("message", "Query failed"))
        st.stop()

    hits = r["hits"]
    if not hits:
        st.warning("No results found.")
        st.stop()

    with st.spinner("Generating answer (LM Studio)..."):
        ans = generate_answer(q.strip(), hits)

    if not ans.get("ok"):
        st.error(ans.get("message", "LLM call failed. Ensure LM Studio server is running."))
        st.stop()

    st.markdown("### ✅ Answer")
    st.write(ans["answer"])

    st.markdown("### 🔎 Top Retrieved Segments")
    for h in hits:
        meta = h["meta"]
        vid = meta.get("video_id", "video")
        stt = ms_to_mmss(meta.get("start", 0.0))
        ent = ms_to_mmss(meta.get("end", 0.0))
        st.markdown(f"**{vid} @ {stt}–{ent}**  _(distance: {h['distance']:.4f})_")
        st.write(h["text"])
        st.divider()
