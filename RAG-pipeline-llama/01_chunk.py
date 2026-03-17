"""
Step 1: Śloka-Aware Chunking
============================
Uses the annotation spreadsheet to create verse-level chunks from transcripts.
Each chunk contains: verse_ref, video_file, speaker, sanskrit_shloka, padavibhaga,
kannada_text (from transcript), english_summary, section_type, and metadata.

Since transcripts are plain text (no timestamps), we use the annotation data
directly as our ground truth chunks. The Kannada transcript text for each verse
explanation is extracted using keyword matching on śloka markers.

Usage:
    python 01_chunk.py
"""
import csv
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config


def load_annotations(csv_path):
    """Load annotation CSV into list of dicts."""
    rows = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Skip empty rows
            if not row.get("video_file", "").strip():
                continue
            rows.append(row)
    print(f"Loaded {len(rows)} annotation rows")
    return rows


def load_transcript(video_file):
    """Load transcript text for a given video file name."""
    # Normalize: "gita 1st video" -> "gita-1st_video.txt"
    # The transcript filenames are: gita-1st_video.txt, gita-2nd_video.txt, etc.
    fname = video_file.strip()
    # Try exact match first
    txt_path = os.path.join(config.TRANSCRIPTS_DIR, fname + ".txt")
    if os.path.exists(txt_path):
        with open(txt_path, "r", encoding="utf-8") as f:
            return f.read()

    # Try with variations
    fname_normalized = fname.replace(" ", "-").replace("_", "_")
    for candidate in os.listdir(config.TRANSCRIPTS_DIR):
        if candidate.endswith(".txt"):
            base = candidate.replace(".txt", "")
            # Fuzzy match: compare without spaces/hyphens/underscores
            if base.replace("-", "").replace("_", "").lower() == \
               fname_normalized.replace("-", "").replace("_", "").lower():
                with open(os.path.join(config.TRANSCRIPTS_DIR, candidate), "r", encoding="utf-8") as f:
                    return f.read()

    print(f"  WARNING: Transcript not found for '{video_file}', using annotation text only")
    return None


def extract_kannada_segment(transcript_text, shloka_text, next_shloka_text=None):
    """
    Try to find the Kannada explanation segment in the transcript
    by looking for the Sanskrit śloka text (written in Kannada script).
    Returns the segment or None.
    """
    if not transcript_text or not shloka_text:
        return None

    # Look for first few words of the shloka in the transcript
    shloka_words = shloka_text.strip().split()[:3]
    search_pattern = r".*?".join(re.escape(w) for w in shloka_words)

    match = re.search(search_pattern, transcript_text)
    if not match:
        return None

    start_idx = match.start()

    # Find end: either next shloka or a large gap
    if next_shloka_text:
        next_words = next_shloka_text.strip().split()[:3]
        next_pattern = r".*?".join(re.escape(w) for w in next_words)
        next_match = re.search(next_pattern, transcript_text[start_idx + 100:])
        if next_match:
            end_idx = start_idx + 100 + next_match.start()
            return transcript_text[start_idx:end_idx].strip()

    # If no next shloka found, take ~2000 chars
    return transcript_text[start_idx:start_idx + 2000].strip()


def build_chunks(annotations):
    """
    Build verse-level chunks from annotations + transcripts.
    Each chunk = one row in the annotation sheet, enriched with transcript text.
    """
    chunks = []
    chunk_id = 0

    # Group annotations by video
    video_groups = {}
    for row in annotations:
        vf = row["video_file"].strip()
        if vf not in video_groups:
            video_groups[vf] = []
        video_groups[vf].append(row)

    for video_file, rows in video_groups.items():
        print(f"\nProcessing: {video_file} ({len(rows)} segments)")
        transcript = load_transcript(video_file)

        for i, row in enumerate(rows):
            chunk_id += 1
            verse_ref = row.get("verse_ref", "-").strip()
            section_type = row.get("section_type", "").strip().lower()

            # Try to extract Kannada text from transcript
            kannada_text = None
            if transcript and row.get("sanskrit_shloka", "").strip() not in ["-", ""]:
                next_shloka = rows[i + 1].get("sanskrit_shloka", "") if i + 1 < len(rows) else None
                kannada_text = extract_kannada_segment(
                    transcript, row["sanskrit_shloka"], next_shloka
                )

            # Use explanation_start/end fields if they have Kannada text
            expl_start = row.get("explanation_start", "").strip()
            expl_end = row.get("explanation_end", "").strip()
            if expl_start and expl_start != "-":
                if kannada_text:
                    kannada_text = expl_start + " ... " + kannada_text + " ... " + expl_end
                else:
                    kannada_text = expl_start + " ... " + expl_end

            chunk = {
                "chunk_id": f"chunk_{chunk_id:03d}",
                "video_file": video_file,
                "verse_ref": verse_ref if verse_ref != "-" else None,
                "speaker": row.get("speaker", "").strip(),
                "section_type": section_type,
                "sanskrit_shloka": row.get("sanskrit_shloka", "").strip()
                    if row.get("sanskrit_shloka", "").strip() not in ["-", ""] else None,
                "padavibhaga": row.get("padavibhaga_text", "").strip()
                    if row.get("padavibhaga_text", "").strip() not in ["-", ""] else None,
                "kannada_text": kannada_text,
                "explanation_summary_en": row.get("explanation_summary_en", "").strip(),
                "short_summary_en": row.get("short_summary", "").strip(),
                "tradition": "Dvaita",
                "chapter": 2,
            }
            chunks.append(chunk)

            status = "✓ with Kannada" if kannada_text else "○ English only"
            label = verse_ref if verse_ref else section_type
            print(f"  {chunk['chunk_id']}: {label} [{status}]")

    return chunks


def main():
    os.makedirs(config.DATA_DIR, exist_ok=True)
    os.makedirs(config.TRANSCRIPTS_DIR, exist_ok=True)

    if not os.path.exists(config.ANNOTATIONS_CSV):
        print(f"ERROR: Annotations file not found: {config.ANNOTATIONS_CSV}")
        print("Please place annotations.csv in the data/ directory.")
        sys.exit(1)

    annotations = load_annotations(config.ANNOTATIONS_CSV)
    chunks = build_chunks(annotations)

    with open(config.CHUNKS_JSON, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)

    print(f"\n{'='*60}")
    print(f"Created {len(chunks)} chunks")
    print(f"  - Shloka explanations: {sum(1 for c in chunks if c['section_type'] == 'shloka_explanation')}")
    print(f"  - Introductions: {sum(1 for c in chunks if c['section_type'] == 'introduction')}")
    print(f"  - Closings: {sum(1 for c in chunks if c['section_type'] == 'closing')}")
    print(f"  - With Kannada text: {sum(1 for c in chunks if c['kannada_text'])}")
    print(f"  - Unique verses: {len(set(c['verse_ref'] for c in chunks if c['verse_ref']))}")
    print(f"Saved to: {config.CHUNKS_JSON}")


if __name__ == "__main__":
    main()
