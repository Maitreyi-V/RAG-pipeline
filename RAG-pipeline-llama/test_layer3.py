"""
test_layer3.py
==============
Task C: Test Layer 3 semantic verse matching on English Advaita transcripts.

This script:
  1. Builds the verse embedding index (or loads from cache)
  2. Splits each English transcript into segments
  3. Runs Layer 3 detection on each segment
  4. Compares detected verse against ground truth from video title
  5. Reports Precision, Recall, F1 across all videos

GROUND TRUTH FORMAT:
  A simple CSV file: video_filename, verse_refs_covered
  e.g.:
    video_01.txt, BG 2.1, BG 2.2, BG 2.3
    video_02.txt, BG 2.4, BG 2.5
  
  OR: pass ground truth from video titles directly (see GROUND_TRUTH dict below)

USAGE:
  python test_layer3.py
  python test_layer3.py --transcripts path/to/english_transcripts/
  python test_layer3.py --threshold 0.5   # tune threshold
  python test_layer3.py --segment-words 300  # tune segment size
  python test_layer3.py --rebuild          # force re-embed (ignore cache)
  python test_layer3.py --debug            # show top-3 matches per segment
"""

import os
import sys
import json
import argparse
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from layer3_semantic import Layer3SemanticMatcher


# ─────────────────────────────────────────────
# GROUND TRUTH
# Edit this dict to match your actual English video files and their verse coverage.
# Key = transcript filename (just the filename, not full path)
# Value = list of verse refs covered in that video (from the video title)
# ─────────────────────────────────────────────

GROUND_TRUTH = {
    # Example — replace with your actual filenames and verse refs
    # "video_01.txt": ["BG 2.1", "BG 2.2", "BG 2.3"],
    # "video_02.txt": ["BG 2.4", "BG 2.5"],
    # "video_03.txt": ["BG 2.11", "BG 2.12", "BG 2.13"],
    # ...add all your videos here
}


# ─────────────────────────────────────────────
# TRANSCRIPT SEGMENTATION
# ─────────────────────────────────────────────

def split_into_segments(text: str, words_per_segment: int = 300,
                         overlap_words: int = 50) -> list[str]:
    """
    Split transcript into overlapping segments of ~N words.
    Overlap ensures verse boundary content isn't missed.
    
    Args:
        text: full transcript text
        words_per_segment: target segment size in words
        overlap_words: overlap between consecutive segments
    
    Returns:
        list of text segments
    """
    words = text.split()
    if not words:
        return []

    segments = []
    step = words_per_segment - overlap_words
    if step <= 0:
        step = words_per_segment

    i = 0
    while i < len(words):
        chunk = words[i: i + words_per_segment]
        segments.append(" ".join(chunk))
        i += step

    return segments


# ─────────────────────────────────────────────
# EVALUATION
# ─────────────────────────────────────────────

def evaluate_video(transcript_path: str, expected_verses: list[str],
                   matcher: Layer3SemanticMatcher,
                   words_per_segment: int, debug: bool) -> dict:
    import json as _json
    with open(transcript_path, "r", encoding="utf-8") as f:
        data = _json.load(f)
    if isinstance(data, dict) and "text" in data:
        text = data["text"]
    else:
        text = str(data)

    segments = split_into_segments(text, words_per_segment)
    filename = os.path.basename(transcript_path)

    print(f"\n  File: {filename}")
    print(f"  Segments: {len(segments)} | Expected verses: {expected_verses}")

    detected_verses = set()

    for i, segment in enumerate(segments):
        if debug:
            top3 = matcher.detect_top_k(segment, k=3)
            print(f"    Segment {i+1}: top3 = {top3}")
            verse, score = top3[0] if top3 else (None, 0.0)
            if score >= matcher.threshold:
                detected_verses.add(verse)
        else:
            verse, score = matcher.detect(segment)
            if verse:
                detected_verses.add(verse)

    expected_set = set(expected_verses)
    detected_set = detected_verses

    tp = len(expected_set & detected_set)
    fp = len(detected_set - expected_set)
    fn = len(expected_set - detected_set)

    print(f"  Detected: {sorted(detected_set)}")
    print(f"  Expected: {sorted(expected_set)}")
    print(f"  TP={tp} | FP={fp} | FN={fn}")

    return {
        "filename": filename,
        "expected": list(expected_set),
        "detected": list(detected_set),
        "tp": tp, "fp": fp, "fn": fn
    }


def compute_prf1(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    """Compute Precision, Recall, F1."""
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) > 0 else 0.0)
    return round(precision, 4), round(recall, 4), round(f1, 4)


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--transcripts", default="english_transcripts",
                        help="Folder containing English transcript .txt files")
    parser.add_argument("--threshold", type=float, default=0.45,
                        help="Cosine similarity threshold (default: 0.45)")
    parser.add_argument("--segment-words", type=int, default=300,
                        help="Words per transcript segment (default: 300)")
    parser.add_argument("--overlap-words", type=int, default=50,
                        help="Word overlap between segments (default: 50)")
    parser.add_argument("--rebuild", action="store_true",
                        help="Force rebuild embedding index (ignore cache)")
    parser.add_argument("--debug", action="store_true",
                        help="Show top-3 verse matches for every segment")
    parser.add_argument("--output", default="layer3_results.json",
                        help="Where to save results JSON")
    args = parser.parse_args()

    # ── Check ground truth ──────────────────────────────────────────────
    if not GROUND_TRUTH:
        print("ERROR: GROUND_TRUTH dict is empty.")
        print("Edit test_layer3.py and fill in your video filenames and verse refs.")
        print("\nExample:")
        print('  GROUND_TRUTH = {')
        print('      "video_01.txt": ["BG 2.1", "BG 2.2"],')
        print('      "video_02.txt": ["BG 2.11", "BG 2.12", "BG 2.13"],')
        print('  }')
        sys.exit(1)

    # ── Check transcripts folder ────────────────────────────────────────
    if not os.path.isdir(args.transcripts):
        print(f"ERROR: Transcripts folder not found: {args.transcripts}")
        print("Pass the correct path with --transcripts path/to/folder")
        sys.exit(1)

    # ── Build Layer 3 index ─────────────────────────────────────────────
    print("=" * 60)
    print("LAYER 3 SEMANTIC VERSE MATCHER — TEST")
    print("=" * 60)
    print(f"Threshold     : {args.threshold}")
    print(f"Segment size  : {args.segment_words} words")
    print(f"Overlap       : {args.overlap_words} words")
    print(f"Videos        : {len(GROUND_TRUTH)}")

    matcher = Layer3SemanticMatcher(
        threshold=args.threshold,
        cache_path="layer3_index.json"
    )
    matcher.build_index(force_rebuild=args.rebuild)

    # ── Run evaluation ──────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("RUNNING EVALUATION")
    print(f"{'='*60}")

    all_results = []
    total_tp = total_fp = total_fn = 0

    for filename, expected_verses in GROUND_TRUTH.items():
        transcript_path = os.path.join(args.transcripts, filename)

        if not os.path.exists(transcript_path):
            print(f"\n  [SKIP] File not found: {transcript_path}")
            continue

        result = evaluate_video(
            transcript_path=transcript_path,
            expected_verses=expected_verses,
            matcher=matcher,
            words_per_segment=args.segment_words,
            debug=args.debug
        )
        all_results.append(result)
        total_tp += result["tp"]
        total_fp += result["fp"]
        total_fn += result["fn"]

    # ── Aggregate results ───────────────────────────────────────────────
    precision, recall, f1 = compute_prf1(total_tp, total_fp, total_fn)

    print(f"\n{'='*60}")
    print("AGGREGATE RESULTS (Layer 3 — English corpus)")
    print(f"{'='*60}")
    print(f"Videos evaluated  : {len(all_results)}")
    print(f"Total Expected     : {total_tp + total_fn}")
    print(f"Total Detected     : {total_tp + total_fp}")
    print(f"True Positives     : {total_tp}")
    print(f"False Positives    : {total_fp}")
    print(f"False Negatives    : {total_fn}")
    print()
    print(f"Precision : {precision}")
    print(f"Recall    : {recall}")
    print(f"F1        : {f1}")
    print(f"{'='*60}")

    # ── Paper table row ─────────────────────────────────────────────────
    print(f"\nPAPER TABLE ROW:")
    print(f"| Layer 3 | English (paraphrase) | {precision} | {recall} | {f1} |")

    # ── Save results ────────────────────────────────────────────────────
    output = {
        "config": {
            "threshold": args.threshold,
            "segment_words": args.segment_words,
            "overlap_words": args.overlap_words,
        },
        "summary": {
            "videos": len(all_results),
            "total_tp": total_tp,
            "total_fp": total_fp,
            "total_fn": total_fn,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        },
        "per_video": all_results
    }
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to: {args.output}")

    # ── Threshold tuning hint ───────────────────────────────────────────
    if f1 < 0.7:
        print(f"\nHINT: F1={f1} is below 0.7. Try tuning:")
        print(f"  Lower threshold (currently {args.threshold}):")
        print(f"    python test_layer3.py --threshold 0.35")
        print(f"  Larger segments (currently {args.segment_words} words):")
        print(f"    python test_layer3.py --segment-words 400")
        print(f"  Debug mode to see what's happening:")
        print(f"    python test_layer3.py --debug")


if __name__ == "__main__":
    main()
