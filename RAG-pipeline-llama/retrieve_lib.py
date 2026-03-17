"""
Shared retrieval library used by 04_retrieve.py and 05_evaluate.py
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
    except:
        return query


def get_embedding(text):
    resp = requests.post(
        f"{config.OLLAMA_BASE_URL}/api/embeddings",
        json={"model": config.EMBED_MODEL, "prompt": text},
        timeout=60
    )
    resp.raise_for_status()
    return resp.json()["embedding"]


def generate_answer(query, context, query_lang="en"):
    lang_instruction = ""
    if query_lang == "kn":
        lang_instruction = ("\nIMPORTANT: The user asked in Kannada. "
                          "Respond in Kannada with verse references in BG X.Y format.")

    prompt = f"""{config.SYSTEM_PROMPT}
{lang_instruction}

Retrieved discourse segments:
{context}

User question: {query}

Answer:"""

    resp = requests.post(
        f"{config.OLLAMA_BASE_URL}/api/generate",
        json={
            "model": config.LLM_MODEL,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.3, "num_predict": 800}
        },
        timeout=600
    )
    resp.raise_for_status()
    return resp.json().get("response", "").strip()


def retrieve_for_eval(query, collection, top_k=None):
    """
    Retrieve chunks and generate answer.
    Returns: (retrieved_chunks_with_metadata, answer_text)
    """
    if top_k is None:
        top_k = config.TOP_K

    lang = detect_language(query)
    search_query = query
    if lang == "kn":
        search_query = translate_query_to_english(query)

    query_embedding = get_embedding(search_query)

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"]
    )

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

    # Build context and generate answer
    context_parts = []
    for j, chunk in enumerate(retrieved, 1):
        meta = chunk["metadata"]
        verse = meta.get("verse_ref", "N/A")
        speaker = meta.get("speaker", "Unknown")
        header = f"--- Source {j} | {verse} | Speaker: {speaker} ---"
        context_parts.append(f"{header}\n{chunk['document']}")

    context = "\n\n".join(context_parts)
    answer = generate_answer(query, context, lang) if context else "No relevant context found."

    return retrieved, answer