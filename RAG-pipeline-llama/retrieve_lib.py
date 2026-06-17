"""
retrieve_lib.py  (UPDATED — cross-tradition retrieval)
======================================================
Now supports:
  - tradition="both"    → retrieve from Dvaita AND Advaita (default)
  - tradition="Dvaita"  → Dvaita only
  - tradition="Advaita" → Advaita only

The key change: retrieve_for_eval() now accepts a tradition parameter
and uses ChromaDB's `where` filter to scope the search.

For cross-tradition mode, it makes TWO separate queries (one per tradition)
so each perspective gets its own best chunks.
"""
import os
import sys
import requests

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config


def detect_language(text):
    for char in text:
        if "\u0C80" <= char <= "\u0CFF":
            return "kn"
    return "en"


def translate_query_to_english(query):
    from deep_translator import GoogleTranslator
    try:
        return GoogleTranslator(source="kn", target="en").translate(query)
    except Exception:
        return query


def get_embedding(text):
    resp = requests.post(
        f"{config.OLLAMA_BASE_URL}/api/embeddings",
        json={"model": config.EMBED_MODEL, "prompt": text},
        timeout=60
    )
    resp.raise_for_status()
    return resp.json()["embedding"]


def _fmt_timestamp(seconds) -> str:
    """Format seconds as MM:SS for display."""
    try:
        s = int(float(seconds))
        return f"{s // 60}:{s % 60:02d}"
    except (TypeError, ValueError):
        return ""


def build_context(retrieved_chunks, label=None):
    """Build context string from retrieved chunks, optionally with a tradition label."""
    if not retrieved_chunks:
        return ""

    parts = []
    if label:
        parts.append(f"=== {label} Tradition Sources ===\n")

    for j, chunk in enumerate(retrieved_chunks, 1):
        meta = chunk.get("metadata", {})
        verse = meta.get("verse_ref", "N/A")
        speaker = meta.get("speaker", "Unknown")
        section = meta.get("section_type", "")
        similarity = chunk.get("similarity", 0)
        tradition = meta.get("tradition", "")

        # Include timestamp range if available
        ts_start = meta.get("start_time_sec")
        ts_end = meta.get("end_time_sec")
        timestamp_str = ""
        if ts_start is not None:
            timestamp_str = f" | Timestamp: {_fmt_timestamp(ts_start)}–{_fmt_timestamp(ts_end)}"

        header = (f"[Source {j} | Verse: {verse} | Speaker: {speaker} | "
                  f"Tradition: {tradition} | Relevance: {similarity:.2f}{timestamp_str}]")
        parts.append(f"{header}\n{chunk['document']}")

    return "\n\n".join(parts)


# ─────────────────────────────────────────────
# GENERATION PROMPTS
# ─────────────────────────────────────────────

SUMMARY_PROMPT = """You are a scholarly teaching assistant for the Bhagavad Gita, Chapter 2.

Given the retrieved discourse segments from BOTH Dvaita and Advaita traditions,
write a clear, comprehensive SUMMARY that answers the user's question.

RULES:
1. Synthesize information from ALL provided segments.
2. Write 4-6 sentences minimum. Be thorough and educational.
3. Cite verse references (e.g., BG 2.13) when relevant.
4. Do NOT separate by tradition here — just give the best overall answer.
5. If the question is in Kannada, respond in Kannada.
6. IMPORTANT: If the question is NOT about the Bhagavad Gita, Vedanta, or Indian philosophy,
   say clearly: "This question is outside the scope of the Bhagavad Gita discourses in our corpus."
   Do NOT try to connect unrelated topics to the Gita. Stay honest and grounded.
"""

TRADITION_PROMPT = """You are a scholar of {tradition} Vedānta philosophy, specializing in the Bhagavad Gita.

Given the discourse segments from the {tradition} tradition, explain the answer to the user's
question FROM THE {tradition} PERSPECTIVE SPECIFICALLY.

{tradition_details}

RULES:
1. Focus on what makes the {tradition} interpretation DISTINCTIVE.
2. Use the specific terminology and concepts of this tradition.
3. Write 3-5 sentences. Be specific, not generic.
4. Cite verse references (BG X.Y) and the speaker/teacher where relevant.
5. If there is no relevant {tradition} content in the provided segments, say so honestly.
6. If the question is in Kannada, respond in Kannada.
7. IMPORTANT: If the question is NOT about the Bhagavad Gita, Vedanta, or Indian philosophy,
   say clearly: "This question is outside the scope of the Bhagavad Gita discourses in our corpus."
   Do NOT try to connect unrelated topics to the Gita. Stay honest and grounded.
"""

DVAITA_DETAILS = """The Dvaita tradition (founded by Madhvacharya) emphasizes:
- Eternal distinction between the individual soul (jiva) and God (Vishnu/Narayana)
- The soul is dependent on God but eternally separate
- Bhakti (devotion) as the primary path
- The discourse source is Kannada lectures on Chapter 2"""

ADVAITA_DETAILS = """The Advaita tradition (founded by Shankaracharya, taught here by Swami Sarvapriyananda) emphasizes:
- Non-duality: the individual self (Atman) IS Brahman (ultimate reality)
- The world of multiplicity is maya (apparent, not ultimately real)
- Jnana (knowledge/self-inquiry) as the primary path
- The discourse source is Swami Sarvapriyananda's English lectures on Chapter 2"""


def generate_answer_with_prompt(query, context, system_prompt, query_lang="en"):
    """Generate an answer using a specific system prompt."""
    if not context:
        return "No relevant content found for this perspective."

    lang_instruction = ""
    if query_lang == "kn":
        lang_instruction = ("\nIMPORTANT: The user asked in Kannada. "
                           "Respond fully in Kannada. Use verse refs in English (BG X.Y).")

    prompt = f"""{system_prompt}
{lang_instruction}

Retrieved discourse segments:
{context}

User question: {query}

Answer:"""

    try:
        resp = requests.post(
            f"{config.OLLAMA_BASE_URL}/api/generate",
            json={
                "model": config.LLM_MODEL,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": config.LLM_TEMPERATURE,
                    "num_predict": config.LLM_MAX_TOKENS,
                    "top_p": 0.9,
                    "repeat_penalty": 1.1,
                }
            },
            timeout=600
        )
        resp.raise_for_status()
        return resp.json().get("response", "").strip()
    except requests.exceptions.Timeout:
        return "⚠️ Answer generation timed out. Try reducing TOP_K or asking a simpler question."
    except Exception as e:
        return f"Error generating answer: {e}"


# ─────────────────────────────────────────────
# RETRIEVAL
# ─────────────────────────────────────────────

def retrieve_chunks(query, collection, top_k=5, tradition=None):
    """
    Retrieve chunks from ChromaDB, optionally filtered by tradition.

    Args:
        tradition: "Dvaita", "Advaita", or None (no filter)
    Returns:
        list of chunk dicts with document, metadata, similarity
    """
    lang = detect_language(query)
    search_query = query
    if lang == "kn":
        search_query = translate_query_to_english(query)

    query_embedding = get_embedding(search_query)

    query_params = {
        "query_embeddings": [query_embedding],
        "n_results": top_k,
        "include": ["documents", "metadatas", "distances"],
    }
    if tradition:
        query_params["where"] = {"tradition": tradition}

    results = collection.query(**query_params)

    retrieved = []
    for i in range(len(results["ids"][0])):
        distance = results["distances"][0][i]
        similarity = 1 - distance
        if similarity < config.SIMILARITY_THRESHOLD:
            continue
        retrieved.append({
            "chunk_id": results["ids"][0][i],
            "similarity": round(similarity, 4),
            "document": results["documents"][0][i],
            "verse_ref": results["metadatas"][0][i].get("verse_ref", ""),
            "metadata": results["metadatas"][0][i],
        })

    return retrieved


def retrieve_cross_tradition(query, collection, top_k=5):
    """
    Retrieve from BOTH traditions separately.
    Returns: (dvaita_chunks, advaita_chunks)
    """
    dvaita_chunks = retrieve_chunks(query, collection, top_k=top_k, tradition="Dvaita")
    advaita_chunks = retrieve_chunks(query, collection, top_k=top_k, tradition="Advaita")
    return dvaita_chunks, advaita_chunks


def generate_cross_tradition_answer(query, collection, top_k=5):
    """
    Full cross-tradition pipeline:
      1. Retrieve from both traditions
      2. Generate summary (using all chunks)
      3. Generate Dvaita perspective
      4. Generate Advaita perspective

    Returns dict with: summary, dvaita_answer, advaita_answer,
                       dvaita_chunks, advaita_chunks
    """
    lang = detect_language(query)
    dvaita_chunks, advaita_chunks = retrieve_cross_tradition(query, collection, top_k)

    # Combined context for summary
    all_context = build_context(dvaita_chunks + advaita_chunks)

    # Tradition-specific contexts
    dvaita_context = build_context(dvaita_chunks, label="Dvaita")
    advaita_context = build_context(advaita_chunks, label="Advaita")

    print(f"  Retrieved: {len(dvaita_chunks)} Dvaita + {len(advaita_chunks)} Advaita chunks")

    # Generate three answers
    summary = generate_answer_with_prompt(query, all_context, SUMMARY_PROMPT, lang)

    dvaita_prompt = TRADITION_PROMPT.format(
        tradition="Dvaita", tradition_details=DVAITA_DETAILS)
    dvaita_answer = generate_answer_with_prompt(query, dvaita_context, dvaita_prompt, lang)

    advaita_prompt = TRADITION_PROMPT.format(
        tradition="Advaita", tradition_details=ADVAITA_DETAILS)
    advaita_answer = generate_answer_with_prompt(query, advaita_context, advaita_prompt, lang)

    return {
        "summary": summary,
        "dvaita_answer": dvaita_answer,
        "advaita_answer": advaita_answer,
        "dvaita_chunks": dvaita_chunks,
        "advaita_chunks": advaita_chunks,
    }


# ─────────────────────────────────────────────
# BACKWARD-COMPATIBLE API
# ─────────────────────────────────────────────

def generate_answer(query, context, query_lang="en"):
    """Original single-answer generation (backward compatible)."""
    return generate_answer_with_prompt(query, context, SUMMARY_PROMPT, query_lang)


def retrieve_for_eval(query, collection, top_k=None, tradition=None):
    """
    Backward-compatible retrieval + generation.
    Now supports optional tradition filter.
    """
    if top_k is None:
        top_k = config.TOP_K

    retrieved = retrieve_chunks(query, collection, top_k, tradition)
    context = build_context(retrieved)
    lang = detect_language(query)
    answer = generate_answer(query, context, lang)

    return retrieved, answer
