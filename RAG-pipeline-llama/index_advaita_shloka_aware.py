"""
index_advaita_shloka_aware.py
=============================
Index English Advaita transcripts using ŚLOKA-AWARE chunking.

THIS IS THE CORE CONTRIBUTION — not fixed-size chunking.

HOW IT WORKS:
  1. Scans each transcript for explicit verse mentions
     ("verse 26", "22nd verse", "shloka 19", etc.)
  2. Uses these mentions as BOUNDARY MARKERS to segment the transcript
     into verse-level discussion chunks
  3. Each chunk = the portion of the lecture discussing a specific verse
  4. Chunks are tagged with the detected verse ref and indexed into ChromaDB

WHY THIS MATTERS:
  - Fixed chunking (300 words) arbitrarily cuts verse discussions in half
  - Śloka-aware chunking keeps each verse's full discussion together
  - When a user asks about BG 2.22, the retriever gets the COMPLETE
    explanation of that verse, not a random 300-word window

COMPARISON (for ablation table):
  - index_advaita.py         → fixed-size chunks (baseline)
  - index_advaita_shloka_aware.py → śloka-detected chunks (novel)

USAGE:
  python index_advaita_shloka_aware.py --dry-run              # preview
  python index_advaita_shloka_aware.py                        # index
  python index_advaita_shloka_aware.py --clear-advaita        # re-index fresh
"""

import os
import sys
import re
import json
import argparse
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
from embeddings import embed_text
from advaita_ground_truth import GROUND_TRUTH


# ─────────────────────────────────────────────
# VERSE MENTION SCANNER
# ─────────────────────────────────────────────

def find_verse_mention_positions(text: str, chapter: int = 2) -> list[dict]:
    """
    Find all positions where the speaker mentions a verse number.
    Returns list of {verse_ref, position, matched_text} sorted by position.
    """
    valid_range = set(range(1, 73))
    mentions = []

    text_lower = text.lower()

    patterns = [
        # "verse 11", "verse number 11", "shloka 19"
        (r'\b(?:verse|shloka|sloka|stanza)\s*(?:number\s*)?(\d{1,2})\b', 1),
        # "11th verse", "22nd shloka"
        (r'\b(\d{1,2})(?:st|nd|rd|th)\s+(?:verse|shloka|sloka|stanza)\b', 1),
        # "chapter 2 verse 11"
        (r'\b(?:chapter\s*2\s*(?:,?\s*)?(?:verse|shloka)?\s*(\d{1,2}))\b', 1),
    ]

    for pattern, group in patterns:
        for m in re.finditer(pattern, text_lower):
            num = int(m.group(group))
            if num in valid_range:
                mentions.append({
                    "verse_ref": f"BG {chapter}.{num}",
                    "verse_num": num,
                    "position": m.start(),
                    "matched_text": m.group(),
                })

    # Ordinal words
    ordinals = {
        'eleventh': 11, 'twelfth': 12, 'thirteenth': 13, 'fourteenth': 14,
        'fifteenth': 15, 'sixteenth': 16, 'seventeenth': 17, 'eighteenth': 18,
        'nineteenth': 19, 'twentieth': 20, 'twenty-first': 21, 'twenty-second': 22,
        'twenty-third': 23, 'twenty-fourth': 24, 'twenty-fifth': 25,
        'twenty-sixth': 26, 'twenty-seventh': 27, 'twenty-eighth': 28,
        'twenty-ninth': 29, 'thirtieth': 30,
    }
    for word, num in ordinals.items():
        pattern = rf'\b{re.escape(word)}\s+(?:verse|shloka|sloka)\b'
        for m in re.finditer(pattern, text_lower):
            mentions.append({
                "verse_ref": f"BG {chapter}.{num}",
                "verse_num": num,
                "position": m.start(),
                "matched_text": m.group(),
            })

    # Deduplicate by position (same mention caught by multiple patterns)
    seen_positions = set()
    unique = []
    for m in sorted(mentions, key=lambda x: x["position"]):
        # Don't add if there's already a mention within 20 chars
        if not any(abs(m["position"] - p) < 20 for p in seen_positions):
            unique.append(m)
            seen_positions.add(m["position"])

    return unique


# ─────────────────────────────────────────────
# ŚLOKA-AWARE CHUNKING
# ─────────────────────────────────────────────

def shloka_aware_chunk(text: str, mentions: list[dict],
                       min_chunk_words: int = 100,
                       max_chunk_words: int = 800,
                       fallback_chunk_words: int = 400) -> list[dict]:
    """
    Split transcript into chunks based on verse mention boundaries.

    Logic:
      1. Each verse mention marks the START of a new chunk
      2. The chunk runs from that mention until the next verse mention
      3. If a chunk is too large (>max_chunk_words), split it with overlap
      4. If a chunk is too small (<min_chunk_words), merge with the next
      5. The intro section (before first verse mention) becomes its own chunk
      6. If NO verse mentions found, fall back to fixed-size chunking

    Returns list of {text, verse_ref, start_pos, end_pos, chunk_type}
    """
    if not mentions:
        # Fallback: no verse mentions detected, use fixed chunking
        words = text.split()
        chunks = []
        for i in range(0, len(words), fallback_chunk_words):
            chunk_text = " ".join(words[i: i + fallback_chunk_words])
            if len(chunk_text.split()) >= 50:
                chunks.append({
                    "text": chunk_text,
                    "verse_ref": "unknown",
                    "chunk_type": "fixed_fallback",
                })
        return chunks

    chunks = []

    # Intro chunk (before first verse mention)
    if mentions[0]["position"] > 200:  # at least 200 chars of intro
        intro_text = text[:mentions[0]["position"]][:3000].strip()
        if len(intro_text.split()) >= min_chunk_words:
            chunks.append({
                "text": intro_text,
                "verse_ref": "introduction",
                "chunk_type": "intro",
            })

    # Verse-boundary chunks
    for i, mention in enumerate(mentions):
        start_pos = mention["position"]

        # End at next mention, or end of text
        if i + 1 < len(mentions):
            end_pos = mentions[i + 1]["position"]
        else:
            end_pos = len(text)

        chunk_text = text[start_pos:end_pos].strip()
        word_count = len(chunk_text.split())

        # Skip tiny chunks (recap mentions, passing references)
        if word_count < min_chunk_words:
            # Merge into previous chunk if possible
            if chunks and chunks[-1]["chunk_type"] == "verse_discussion":
                chunks[-1]["text"] += " " + chunk_text
                # Update verse_ref to include both
                prev_ref = chunks[-1]["verse_ref"]
                new_ref = mention["verse_ref"]
                if new_ref not in prev_ref:
                    chunks[-1]["verse_ref"] = f"{prev_ref}, {new_ref}"
            continue

        # Split oversized chunks
        if word_count > max_chunk_words:
            words = chunk_text.split()
            sub_start = 0
            while sub_start < len(words):
                sub_end = min(sub_start + max_chunk_words, len(words))
                sub_text = " ".join(words[sub_start:sub_end])
                chunks.append({
                    "text": sub_text,
                    "verse_ref": mention["verse_ref"],
                    "chunk_type": "verse_discussion",
                })
                sub_start += max_chunk_words - 100  # 100-word overlap
        else:
            chunks.append({
                "text": chunk_text,
                "verse_ref": mention["verse_ref"],
                "chunk_type": "verse_discussion",
            })

    return chunks


# ─────────────────────────────────────────────
# EMBEDDING
# ─────────────────────────────────────────────



def get_transcript_text(filepath: str) -> str:
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        return data.get("text", data.get("full_transcript", ""))
    return str(data)


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Śloka-aware Advaita indexing")
    parser.add_argument("--transcripts", default="data/transcripts")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--clear-advaita", action="store_true")
    parser.add_argument("--min-chunk-words", type=int, default=100)
    parser.add_argument("--max-chunk-words", type=int, default=800)
    args = parser.parse_args()

    print("=" * 70)
    print("ŚLOKA-AWARE ADVAITA INDEXING")
    print("(Novel contribution: verse-boundary chunking, not fixed-size)")
    print("=" * 70)

    import chromadb
    client = chromadb.PersistentClient(path=config.CHROMA_DIR)
    collection = client.get_or_create_collection(
        name=config.COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}
    )
    current_count = collection.count()
    print(f"Current collection: {current_count} chunks")

    if args.clear_advaita:
        print("\nClearing existing Advaita chunks...")
        existing = collection.get(where={"tradition": "Advaita"})
        if existing["ids"]:
            collection.delete(ids=existing["ids"])
            print(f"  Removed {len(existing['ids'])} Advaita chunks")

    # Process each transcript
    all_chunks = []
    total_verse_chunks = 0
    total_fallback_chunks = 0

    for video_id, gt_entry in sorted(GROUND_TRUTH.items()):
        filename = f"{video_id}.json"
        filepath = os.path.join(args.transcripts, filename)

        if not os.path.exists(filepath):
            print(f"\n  [SKIP] {filepath} not found")
            continue

        text = get_transcript_text(filepath)
        if not text:
            print(f"\n  [SKIP] {filename} — empty")
            continue

        title = gt_entry.get("title", "")
        episode = gt_entry.get("episode", 0)
        gt_verses = gt_entry["verses"]

        # Step 1: Find verse mentions
        mentions = find_verse_mention_positions(text)

        # Step 2: Śloka-aware chunking
        chunks = shloka_aware_chunk(
            text, mentions,
            min_chunk_words=args.min_chunk_words,
            max_chunk_words=args.max_chunk_words
        )

        verse_chunks = [c for c in chunks if c["chunk_type"] == "verse_discussion"]
        fallback_chunks = [c for c in chunks if c["chunk_type"] == "fixed_fallback"]

        print(f"\n  {filename}: {len(text.split())} words")
        print(f"    Verse mentions found: {len(mentions)}")
        if mentions:
            mentioned_verses = sorted(set(m["verse_ref"] for m in mentions))
            print(f"    Verses mentioned: {mentioned_verses}")
        print(f"    Chunks: {len(verse_chunks)} verse-aware + "
              f"{len(fallback_chunks)} fallback + "
              f"{len([c for c in chunks if c['chunk_type'] == 'intro'])} intro")
        print(f"    Ground truth verses: {gt_verses}")

        total_verse_chunks += len(verse_chunks)
        total_fallback_chunks += len(fallback_chunks)

        # Prepare for indexing
        for i, chunk in enumerate(chunks):
            chunk_id = f"advaita_shloka_{video_id}_chunk_{i:03d}"
            metadata = {
                "tradition": "Advaita",
                "verse_ref": chunk["verse_ref"],
                "video_file": filename,
                "speaker": "Swami Sarvapriyananda",
                "section_type": chunk["chunk_type"],
                "chapter": "2",
                "has_sanskrit": "false",
                "has_kannada": "false",
                "episode": str(episode),
                "chunking_method": "shloka_aware",  # KEY: marks this as novel
            }
            all_chunks.append({
                "id": chunk_id,
                "text": chunk["text"],
                "metadata": metadata,
            })

    # Summary
    print(f"\n{'='*70}")
    print(f"CHUNKING SUMMARY")
    print(f"{'='*70}")
    print(f"Total chunks:         {len(all_chunks)}")
    print(f"  Verse-aware:        {total_verse_chunks}  ← novel contribution")
    print(f"  Fallback (fixed):   {total_fallback_chunks}")
    print(f"  Intro sections:     {len(all_chunks) - total_verse_chunks - total_fallback_chunks}")

    if args.dry_run:
        print(f"\n[DRY RUN] Preview of first 3 chunks:")
        for c in all_chunks[:3]:
            print(f"\n  ID: {c['id']}")
            print(f"  Verse: {c['metadata']['verse_ref']}")
            print(f"  Type: {c['metadata']['section_type']}")
            print(f"  Words: {len(c['text'].split())}")
            print(f"  Text: {c['text'][:200]}...")
        print(f"\n[DRY RUN] Remove --dry-run to index into ChromaDB.")
        return

    # Index
    print(f"\nEmbedding and indexing {len(all_chunks)} chunks...")
    batch_size = 5
    for batch_start in range(0, len(all_chunks), batch_size):
        batch = all_chunks[batch_start: batch_start + batch_size]
        ids, docs, metas, embs = [], [], [], []

        for chunk in batch:
            print(f"  [{batch_start + len(ids) + 1}/{len(all_chunks)}] "
                  f"{chunk['id']}", end="\r")
            try:
                emb = embed_text(chunk["text"])
                ids.append(chunk["id"])
                docs.append(chunk["text"])
                metas.append(chunk["metadata"])
                embs.append(emb)
            except Exception as e:
                print(f"\n  [ERROR] {chunk['id']}: {e}")

        if ids:
            collection.upsert(ids=ids, documents=docs,
                             metadatas=metas, embeddings=embs)

    final = collection.count()
    print(f"\n\n{'='*70}")
    print(f"DONE! Collection now has {final} total chunks")

    # Verify
    dvaita = collection.get(where={"tradition": "Dvaita"})
    advaita = collection.get(where={"tradition": "Advaita"})
    print(f"  Dvaita:  {len(dvaita['ids'])} chunks")
    print(f"  Advaita: {len(advaita['ids'])} chunks")

    # Check chunking methods
    shloka_aware = [m for m in advaita["metadatas"]
                    if m.get("chunking_method") == "shloka_aware"]
    print(f"  Advaita śloka-aware: {len(shloka_aware)} chunks")
    print(f"{'='*70}")

    print(f"\nFor ABLATION (Step 7), compare against fixed-size baseline:")
    print(f"  python index_advaita.py --clear-advaita  # index with fixed chunks")
    print(f"  python 05_evaluate.py --no-ref            # evaluate")
    print(f"  # Then re-run this script and evaluate again")
    print(f"  # The difference in P@k = your chunking contribution")


if __name__ == "__main__":
    main()
