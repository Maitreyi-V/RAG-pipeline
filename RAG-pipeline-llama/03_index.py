"""
Step 3: Embedding & Indexing
============================
Embeds the ENGLISH translation/summary text using BGE-M3 via sentence-transformers,
then stores vectors + metadata in ChromaDB.

Key design: We embed English text (not raw Kannada) so that English queries
match semantically. The original Kannada + Sanskrit is stored as metadata
for bilingual answer generation.

Usage:
    python 03_index.py
    python 03_index.py --reset   # Wipe and rebuild the index
"""
import json
import os
import sys
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
from embeddings import embed_text



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", action="store_true", help="Reset ChromaDB collection")
    args = parser.parse_args()

    import chromadb

    if not os.path.exists(config.CHUNKS_JSON):
        print("ERROR: chunks.json not found. Run 01_chunk.py and 02_translate.py first.")
        sys.exit(1)

    with open(config.CHUNKS_JSON, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    # Verify chunks have embedding_text
    missing = [c for c in chunks if not c.get("embedding_text")]
    if missing:
        print(f"WARNING: {len(missing)} chunks missing embedding_text. Run 02_translate.py first.")

    # Initialize ChromaDB
    os.makedirs(config.CHROMA_DIR, exist_ok=True)
    client = chromadb.PersistentClient(path=config.CHROMA_DIR)

    if args.reset:
        try:
            client.delete_collection(config.COLLECTION_NAME)
            print("Reset: deleted existing collection")
        except:
            pass

    collection = client.get_or_create_collection(
        name=config.COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}  # cosine similarity
    )

    existing_count = collection.count()
    if existing_count > 0 and not args.reset:
        print(f"Collection already has {existing_count} documents.")
        print("Use --reset to rebuild, or this will skip existing IDs.")

    print(f"\nIndexing {len(chunks)} chunks into ChromaDB...")
    success_count = 0
    error_count = 0

    # Batch process
    batch_ids = []
    batch_embeddings = []
    batch_documents = []
    batch_metadatas = []

    for i, chunk in enumerate(chunks):
        chunk_id = chunk["chunk_id"]
        embedding_text = chunk.get("embedding_text", "")

        if not embedding_text:
            print(f"  [{i+1}] {chunk_id}: SKIP (no embedding text)")
            error_count += 1
            continue

        label = chunk.get("verse_ref") or chunk.get("section_type", "?")
        print(f"  [{i+1}/{len(chunks)}] {chunk_id}: {label} ...", end=" ", flush=True)

        embedding = embed_text(embedding_text)
        if embedding is None:
            print("FAILED")
            error_count += 1
            continue

        # Prepare metadata (ChromaDB only supports str/int/float/bool)
        metadata = {
            "video_file": chunk.get("video_file", ""),
            "verse_ref": chunk.get("verse_ref") or "",
            "speaker": chunk.get("speaker", ""),
            "section_type": chunk.get("section_type", ""),
            "tradition": chunk.get("tradition", "Dvaita"),
            "chapter": chunk.get("chapter", 2),
            "has_sanskrit": bool(chunk.get("sanskrit_shloka")),
            "has_kannada": bool(chunk.get("kannada_text")),
        }

        # Store the full combined text as the document
        # Include both English and Kannada for bilingual retrieval
        doc_parts = []
        if chunk.get("explanation_summary_en"):
            doc_parts.append(chunk["explanation_summary_en"])
        if chunk.get("kannada_text"):
            doc_parts.append(f"\n[Kannada]: {chunk['kannada_text']}")
        if chunk.get("sanskrit_shloka"):
            doc_parts.append(f"\n[Sanskrit]: {chunk['sanskrit_shloka']}")
        if chunk.get("padavibhaga"):
            doc_parts.append(f"\n[Padavibhāga]: {chunk['padavibhaga']}")

        document = "\n".join(doc_parts) if doc_parts else embedding_text

        batch_ids.append(chunk_id)
        batch_embeddings.append(embedding)
        batch_documents.append(document)
        batch_metadatas.append(metadata)
        success_count += 1
        print("✓")

        # Upsert in batches of 10
        if len(batch_ids) >= 10:
            collection.upsert(
                ids=batch_ids,
                embeddings=batch_embeddings,
                documents=batch_documents,
                metadatas=batch_metadatas
            )
            batch_ids, batch_embeddings, batch_documents, batch_metadatas = [], [], [], []

    # Final batch
    if batch_ids:
        collection.upsert(
            ids=batch_ids,
            embeddings=batch_embeddings,
            documents=batch_documents,
            metadatas=batch_metadatas
        )

    print(f"\n{'='*60}")
    print(f"Indexing complete:")
    print(f"  - Indexed: {success_count}")
    print(f"  - Errors: {error_count}")
    print(f"  - Total in collection: {collection.count()}")
    print(f"  - ChromaDB path: {config.CHROMA_DIR}")


if __name__ == "__main__":
    main()
