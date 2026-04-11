"""
Multi-Tradition Comparison Module
==================================
When a user asks about a verse, this module retrieves explanations
from BOTH Dvaita (Kannada) and Advaita (English) traditions and
presents them side-by-side WITHOUT blending.

Architecture:
  - Each tradition has its own ChromaDB collection
  - Retrieval queries both collections independently
  - Results are tagged and presented separately
  - The LLM is instructed to NOT merge interpretations

Usage:
    from multi_tradition import MultiTraditionRetriever
    retriever = MultiTraditionRetriever()
    result = retriever.compare("What does BG 2.12 mean?")
    # Returns { "dvaita": [...], "advaita": [...], "comparison_summary": "..." }
"""
import os
import json
import requests


class TraditionConfig:
    """Configuration for a single tradition's corpus."""
    def __init__(self, name, language, collection_name, description):
        self.name = name                     # "Dvaita" or "Advaita"
        self.language = language             # "Kannada" or "English"
        self.collection_name = collection_name
        self.description = description


# Define supported traditions
TRADITIONS = {
    "dvaita": TraditionConfig(
        name="Dvaita",
        language="Kannada",
        collection_name="gita_dvaita_kannada",
        description="Madhvāchārya's Dvaita Vedānta tradition, explained in Kannada"
    ),
    "advaita": TraditionConfig(
        name="Advaita",
        language="English",
        collection_name="gita_advaita_english",
        description="Śaṅkarāchārya's Advaita Vedānta tradition, explained in English"
    ),
}


class MultiTraditionRetriever:
    """Retrieves and compares verse explanations across traditions."""

    COMPARISON_PROMPT = """You are a scholarly assistant comparing different Vedānta interpretations
of the Bhagavad Gita. You have been given explanations of the same verse from two traditions.

CRITICAL RULES:
1. Present each tradition's view SEPARATELY and FAITHFULLY. Do NOT blend or merge.
2. Label each interpretation clearly with its tradition name.
3. Highlight where the traditions AGREE and where they DIFFER.
4. Do NOT declare one tradition "correct" — present both with equal respect.
5. Use the exact verse reference (BG X.Y) in your response.

Dvaita (Madhva) tradition explanation:
{dvaita_context}

Advaita (Śaṅkara) tradition explanation:
{advaita_context}

User question: {query}

Respond with:
**Verse:** [BG X.Y]

**Dvaita (Madhva) Perspective:**
[Their interpretation]

**Advaita (Śaṅkara) Perspective:**
[Their interpretation]

**Key Differences:**
[Where they diverge]

**Common Ground:**
[Where they agree, if any]
"""

    def __init__(self, chroma_dir="data/chroma_db",
                 ollama_url="http://localhost:11434",
                 embed_model="bge-m3", llm_model="llama3"):
        self.chroma_dir = chroma_dir
        self.ollama_url = ollama_url
        self.embed_model = embed_model
        self.llm_model = llm_model
        self.collections = {}

    def init_collections(self):
        """Initialize ChromaDB collections for each tradition."""
        import chromadb
        client = chromadb.PersistentClient(path=self.chroma_dir)

        for key, tradition in TRADITIONS.items():
            try:
                self.collections[key] = client.get_or_create_collection(
                    name=tradition.collection_name,
                    metadata={"hnsw:space": "cosine"}
                )
                count = self.collections[key].count()
                print(f"  {tradition.name}: {count} documents in {tradition.collection_name}")
            except Exception as e:
                print(f"  {tradition.name}: ERROR — {e}")

    def get_embedding(self, text):
        resp = requests.post(
            f"{self.ollama_url}/api/embeddings",
            json={"model": self.embed_model, "prompt": text},
            timeout=60
        )
        resp.raise_for_status()
        return resp.json()["embedding"]

    def retrieve_single_tradition(self, query, tradition_key, top_k=3):
        """Retrieve from a single tradition's collection."""
        if tradition_key not in self.collections:
            return []

        collection = self.collections[tradition_key]
        if collection.count() == 0:
            return []

        query_embedding = self.get_embedding(query)

        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"]
        )

        retrieved = []
        for i in range(len(results["ids"][0])):
            distance = results["distances"][0][i]
            similarity = 1 - distance
            if similarity < 0.3:
                continue
            retrieved.append({
                "chunk_id": results["ids"][0][i],
                "document": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "similarity": round(similarity, 4),
                "tradition": TRADITIONS[tradition_key].name,
            })

        return retrieved

    def compare(self, query, top_k=3):
        """
        Retrieve from all traditions and generate a comparison.
        Returns structured result with per-tradition explanations.
        """
        all_results = {}
        for key in TRADITIONS:
            all_results[key] = self.retrieve_single_tradition(query, key, top_k)

        # Build context for each tradition
        dvaita_context = self._build_context(all_results.get("dvaita", []))
        advaita_context = self._build_context(all_results.get("advaita", []))

        # Generate comparison only if both traditions have content
        comparison = None
        if dvaita_context and advaita_context:
            comparison = self._generate_comparison(query, dvaita_context, advaita_context)
        elif dvaita_context:
            comparison = f"Only Dvaita perspective available:\n\n{dvaita_context}"
        elif advaita_context:
            comparison = f"Only Advaita perspective available:\n\n{advaita_context}"

        # Collect all cited verses
        all_verses = set()
        for results in all_results.values():
            for r in results:
                v = r.get("metadata", {}).get("verse_ref", "")
                if v:
                    all_verses.add(v)

        return {
            "query": query,
            "dvaita_results": all_results.get("dvaita", []),
            "advaita_results": all_results.get("advaita", []),
            "comparison": comparison,
            "cited_verses": sorted(all_verses),
            "traditions_available": [
                k for k, v in all_results.items() if v
            ],
        }

    def _build_context(self, results):
        if not results:
            return ""
        parts = []
        for r in results:
            verse = r.get("metadata", {}).get("verse_ref", "N/A")
            parts.append(f"[{verse}] {r['document']}")
        return "\n\n".join(parts)

    def _generate_comparison(self, query, dvaita_ctx, advaita_ctx):
        prompt = self.COMPARISON_PROMPT.format(
            dvaita_context=dvaita_ctx,
            advaita_context=advaita_ctx,
            query=query,
        )
        try:
            resp = requests.post(
                f"{self.ollama_url}/api/generate",
                json={
                    "model": self.llm_model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"temperature": 0.3, "num_predict": 1000}
                },
                timeout=180
            )
            resp.raise_for_status()
            return resp.json().get("response", "").strip()
        except Exception as e:
            return f"Error generating comparison: {e}"

    def index_tradition(self, chunks, tradition_key):
        """
        Index a list of chunks into a tradition-specific collection.
        Each chunk must have: chunk_id, embedding_text, verse_ref, and metadata.
        """
        if tradition_key not in self.collections:
            print(f"ERROR: Collection for {tradition_key} not initialized")
            return

        collection = self.collections[tradition_key]

        for chunk in chunks:
            embedding = self.get_embedding(chunk["embedding_text"])
            collection.upsert(
                ids=[chunk["chunk_id"]],
                embeddings=[embedding],
                documents=[chunk.get("document", chunk["embedding_text"])],
                metadatas=[{
                    "verse_ref": chunk.get("verse_ref", ""),
                    "tradition": TRADITIONS[tradition_key].name,
                    "speaker": chunk.get("speaker", ""),
                    "section_type": chunk.get("section_type", ""),
                    "video_file": chunk.get("video_file", ""),
                }]
            )

        print(f"Indexed {len(chunks)} chunks into {TRADITIONS[tradition_key].name} collection")
