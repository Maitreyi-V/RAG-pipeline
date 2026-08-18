"""
Step 4: Retrieval & Answer Generation
======================================
Given a user query (English or Kannada):
1. Translate query to English if needed
2. Embed query using BGE-M3
3. Retrieve top-k relevant chunks from ChromaDB
4. Generate answer using Llama3 with verse citations and Dvaita tradition context

Usage:
    python 04_retrieve.py "What does Krishna say about grief?"
    python 04_retrieve.py "ಅರ್ಜುನನ ದುಃಖಕ್ಕೆ ಕಾರಣ ಏನು?"
    python 04_retrieve.py --interactive
"""
import json
import os
import sys
import argparse
import requests

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
v


def detect_language(text):
    """Simple detection: if text contains Kannada Unicode, it's Kannada."""
    for char in text:
        if "\u0C80" <= char <= "\u0CFF":
            return "kn"
    return "en"


def translate_query_to_english(query):
    """Translate a Kannada query to English for embedding-based retrieval."""
    from deep_translator import GoogleTranslator
    try:
        return GoogleTranslator(source="kn", target="en").translate(query)
    except:
        return query



def retrieve(query, collection, top_k=None):
    """Retrieve top-k relevant chunks for a query."""
    if top_k is None:
        top_k = config.TOP_K

    lang = detect_language(query)
    search_query = query
    if lang == "kn":
        search_query = translate_query_to_english(query)
        print(f"  Translated query: {search_query}")

    query_embedding = embed_text(search_query)

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"]
    )

    retrieved = []
    for i in range(len(results["ids"][0])):
        chunk_id = results["ids"][0][i]
        distance = results["distances"][0][i]
        similarity = 1 - distance  # cosine distance -> similarity

        if similarity < config.SIMILARITY_THRESHOLD:
            continue

        retrieved.append({
            "chunk_id": chunk_id,
            "similarity": round(similarity, 4),
            "document": results["documents"][0][i],
            "metadata": results["metadatas"][0][i],
        })

    return retrieved, lang


def build_context(retrieved_chunks):
    """Build context string from retrieved chunks for LLM."""
    context_parts = []
    for i, chunk in enumerate(retrieved_chunks, 1):
        meta = chunk["metadata"]
        verse = meta.get("verse_ref", "N/A")
        speaker = meta.get("speaker", "Unknown")
        section = meta.get("section_type", "")

        header = f"--- Source {i} | {verse} | Speaker: {speaker} | Type: {section} ---"
        context_parts.append(f"{header}\n{chunk['document']}")

    return "\n\n".join(context_parts)


def generate_answer(query, context, query_lang="en"):
    """Generate answer using Llama3 via Ollama with grounding in retrieved context."""
    lang_instruction = ""
    if query_lang == "kn":
        lang_instruction = "\nIMPORTANT: The user asked in Kannada. Respond in Kannada, but include verse references in English format (BG X.Y)."

    prompt = f"""{config.SYSTEM_PROMPT}
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
                "options": {"temperature": 0.3, "num_predict": 800}
            },
            timeout=180
        )
        resp.raise_for_status()
        return resp.json().get("response", "").strip()
    except Exception as e:
        return f"Error generating answer: {e}"


def query_pipeline(query, collection):
    """Full pipeline: retrieve + generate."""
    print(f"\nQuery: {query}")
    print("-" * 50)

    # Retrieve
    retrieved, query_lang = retrieve(query, collection)
    print(f"\nRetrieved {len(retrieved)} relevant chunks:")
    for chunk in retrieved:
        meta = chunk["metadata"]
        print(f"  {chunk['chunk_id']} | {meta.get('verse_ref', 'N/A')} | "
              f"sim={chunk['similarity']:.3f} | {meta.get('section_type', '')}")

    if not retrieved:
        return {
            "query": query,
            "answer": "I could not find relevant information in the discourse corpus to answer this question.",
            "retrieved_verses": [],
            "num_retrieved": 0
        }

    # Generate
    context = build_context(retrieved)
    answer = generate_answer(query, context, query_lang)

    # Extract cited verses
    cited_verses = list(set(
        chunk["metadata"].get("verse_ref", "")
        for chunk in retrieved
        if chunk["metadata"].get("verse_ref")
    ))

    result = {
        "query": query,
        "query_language": query_lang,
        "answer": answer,
        "retrieved_verses": cited_verses,
        "num_retrieved": len(retrieved),
        "top_similarity": retrieved[0]["similarity"] if retrieved else 0,
        "retrieved_chunks": [
            {
                "chunk_id": c["chunk_id"],
                "verse_ref": c["metadata"].get("verse_ref", ""),
                "similarity": c["similarity"]
            }
            for c in retrieved
        ]
    }

    print(f"\n{'='*50}")
    print(f"ANSWER:\n{answer}")
    print(f"\nCited verses: {cited_verses}")

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("query", nargs="?", help="Question to ask")
    parser.add_argument("--interactive", "-i", action="store_true")
    parser.add_argument("--top_k", type=int, default=config.TOP_K)
    args = parser.parse_args()

    import chromadb

    client = chromadb.PersistentClient(path=config.CHROMA_DIR)
    try:
        collection = client.get_collection(config.COLLECTION_NAME)
    except:
        print("ERROR: ChromaDB collection not found. Run 03_index.py first.")
        sys.exit(1)

    print(f"Connected to ChromaDB: {collection.count()} documents")

    if args.interactive:
        print("\nInteractive mode. Type 'quit' to exit.")
        while True:
            query = input("\nYour question: ").strip()
            if query.lower() in ("quit", "exit", "q"):
                break
            if query:
                query_pipeline(query, collection)
    elif args.query:
        result = query_pipeline(args.query, collection)
        # Save result
        os.makedirs(config.RESULTS_DIR, exist_ok=True)
        with open(os.path.join(config.RESULTS_DIR, "last_query_result.json"), "w") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
    else:
        print("Provide a query or use --interactive mode.")
        print('  python 04_retrieve.py "What does Krishna say about grief?"')
        print('  python 04_retrieve.py --interactive')


if __name__ == "__main__":
    main()
