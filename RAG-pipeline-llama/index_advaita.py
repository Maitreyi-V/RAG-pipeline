"""
index_advaita.py
================
Index English Advaita transcripts into ChromaDB.

Adds Advaita chunks to the SAME collection as Dvaita chunks,
tagged with tradition="Advaita". This enables cross-tradition retrieval.

WHAT IT DOES:
  1. Reads each English transcript (video_01.json ... video_10.json)
  2. Splits into ~300-word overlapping chunks
  3. Tags each chunk with verse refs from advaita_ground_truth.py
  4. Embeds with BGE-M3 and stores in ChromaDB

USAGE:
  python index_advaita.py
  python index_advaita.py --dry-run          # preview without indexing
  python index_advaita.py --clear-advaita    # remove old Advaita chunks first

AFTER RUNNING:
  The ChromaDB collection will have both Dvaita and Advaita chunks.
  retrieve_lib.py can filter by tradition="Dvaita" or tradition="Advaita".
"""

import os
import sys
import json
import requests
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
from advaita_ground_truth import GROUND_TRUTH


def get_transcript_text(filepath: str) -> str:
    """Load transcript text, handling both 'text' and 'full_transcript' keys."""
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        return data.get("text", data.get("full_transcript", ""))
    return str(data)


def chunk_transcript(text: str, words_per_chunk: int = 300,
                     overlap_words: int = 75) -> list[str]:
    """Split transcript into overlapping chunks."""
    words = text.split()
    if not words:
        return []
    chunks = []
    step = max(words_per_chunk - overlap_words, 1)
    i = 0
    while i < len(words):
        chunk_text = " ".join(words[i: i + words_per_chunk])
        if len(chunk_text.split()) >= 50:  # skip tiny tail chunks
            chunks.append(chunk_text)
        i += step
    return chunks


def embed_text(text: str) -> list[float]:
    """Embed text using BGE-M3 via Ollama."""
    resp = requests.post(
        f"{config.OLLAMA_BASE_URL}/api/embeddings",
        json={"model": config.EMBED_MODEL, "prompt": text},
        timeout=120
    )
    resp.raise_for_status()
    return resp.json()["embedding"]


def main():
    parser = argparse.ArgumentParser(description="Index Advaita transcripts into ChromaDB")
    parser.add_argument("--transcripts", default="data/transcripts",
                        help="Path to transcript JSON files")
    parser.add_argument("--dry-run", action="store_true",
                        help="Preview chunks without indexing")
    parser.add_argument("--clear-advaita", action="store_true",
                        help="Remove existing Advaita chunks before indexing")
    parser.add_argument("--words-per-chunk", type=int, default=300)
    parser.add_argument("--overlap-words", type=int, default=75)
    args = parser.parse_args()

    print("=" * 70)
    print("INDEXING ADVAITA TRANSCRIPTS INTO CHROMADB")
    print("=" * 70)

    # Connect to ChromaDB
    import chromadb
    client = chromadb.PersistentClient(path=config.CHROMA_DIR)
    collection = client.get_or_create_collection(
        name=config.COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}
    )

    current_count = collection.count()
    print(f"Current collection: {config.COLLECTION_NAME} ({current_count} chunks)")

    # Optionally clear old Advaita chunks
    if args.clear_advaita:
        print("\nClearing existing Advaita chunks...")
        # Get all chunk IDs with tradition=Advaita
        all_data = collection.get(where={"tradition": "Advaita"})
        if all_data["ids"]:
            collection.delete(ids=all_data["ids"])
            print(f"  Removed {len(all_data['ids'])} Advaita chunks")
        else:
            print("  No existing Advaita chunks found")

    # Process each transcript
    all_chunks = []
    total_chunks = 0

    for video_id, gt_entry in sorted(GROUND_TRUTH.items()):
        filename = f"{video_id}.json"
        filepath = os.path.join(args.transcripts, filename)

        if not os.path.exists(filepath):
            print(f"\n  [SKIP] {filepath} not found")
            continue

        text = get_transcript_text(filepath)
        if not text:
            print(f"\n  [SKIP] {filename} — empty transcript")
            continue

        verses = gt_entry["verses"]
        verse_ref_str = ", ".join(verses)
        title = gt_entry.get("title", "")
        episode = gt_entry.get("episode", 0)

        chunks = chunk_transcript(text, args.words_per_chunk, args.overlap_words)

        print(f"\n  {filename}: {len(text.split())} words → {len(chunks)} chunks")
        print(f"    Verses: {verse_ref_str}")
        print(f"    Title: {title[:80]}")

        for i, chunk_text in enumerate(chunks):
            chunk_id = f"advaita_{video_id}_chunk_{i:03d}"
            metadata = {
                "tradition": "Advaita",
                "verse_ref": verse_ref_str,
                "video_file": filename,
                "speaker": "Swami Sarvapriyananda",
                "section_type": "discourse_explanation",
                "chapter": "2",
                "has_sanskrit": "false",
                "has_kannada": "false",
                "episode": str(episode),
            }

            all_chunks.append({
                "id": chunk_id,
                "text": chunk_text,
                "metadata": metadata,
            })
            total_chunks += 1

    print(f"\n{'='*70}")
    print(f"Total Advaita chunks prepared: {total_chunks}")

    if args.dry_run:
        print("\n[DRY RUN] No chunks were indexed. Remove --dry-run to index.")
        # Show a sample
        if all_chunks:
            sample = all_chunks[0]
            print(f"\nSample chunk:")
            print(f"  ID: {sample['id']}")
            print(f"  Metadata: {sample['metadata']}")
            print(f"  Text (first 200 chars): {sample['text'][:200]}...")
        return

    # Index into ChromaDB
    print(f"\nEmbedding and indexing {total_chunks} chunks...")
    print("(This will take a few minutes — one embedding call per chunk)\n")

    batch_size = 5  # process in small batches to show progress
    for batch_start in range(0, len(all_chunks), batch_size):
        batch = all_chunks[batch_start: batch_start + batch_size]

        ids = []
        documents = []
        metadatas = []
        embeddings = []

        for chunk in batch:
            print(f"  [{batch_start + len(ids) + 1}/{total_chunks}] {chunk['id']}", end="\r")
            try:
                emb = embed_text(chunk["text"])
                ids.append(chunk["id"])
                documents.append(chunk["text"])
                metadatas.append(chunk["metadata"])
                embeddings.append(emb)
            except Exception as e:
                print(f"\n  [ERROR] {chunk['id']}: {e}")

        if ids:
            collection.upsert(
                ids=ids,
                documents=documents,
                metadatas=metadatas,
                embeddings=embeddings,
            )

    final_count = collection.count()
    print(f"\n\n{'='*70}")
    print(f"DONE!")
    print(f"  Before: {current_count} chunks")
    print(f"  Added:  {total_chunks} Advaita chunks")
    print(f"  Total:  {final_count} chunks in collection")
    print(f"{'='*70}")

    # Verify
    print(f"\nVerification:")
    dvaita = collection.get(where={"tradition": "Dvaita"})
    advaita = collection.get(where={"tradition": "Advaita"})
    print(f"  Dvaita chunks:  {len(dvaita['ids'])}")
    print(f"  Advaita chunks: {len(advaita['ids'])}")


if __name__ == "__main__":
    main()
