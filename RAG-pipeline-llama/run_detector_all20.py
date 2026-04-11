"""
run_detector_all20.py
─────────────────────
Runs the 3-layer śloka detector on all 20 Kannada transcripts.

Filename mapping (matches your actual files in data/transcripts/):
  Video 1  → gita-1st video.txt      ← pilot set (manual annotations)
  Video 2  → gita-2nd video.txt
  Video 3  → gita-3rd video.txt
  Video 4  → gita-4th video.txt
  Video 5  → gita-5th video.txt
  Video 6  → gita-6th video.txt
  Video 7  → kannada_07.txt          ← new set (light annotations)
  ...
  Video 20 → kannada_20.txt

Usage (just run from inside RAG-pipeline-llama/):
  python run_detector_all20.py

Custom paths:
  python run_detector_all20.py --transcripts data/transcripts --annotations data/annotations.csv
"""

import os
import sys
import json
import argparse
import csv
from pathlib import Path

# ── filename map — matches your actual files ───────────────────────────────────
FILENAMES = {
    1:  "gita-1st video.txt",
    2:  "gita-2nd video.txt",
    3:  "gita-3rd video.txt",
    4:  "gita-4th video.txt",
    5:  "gita-5th video.txt",
    6:  "gita-6th video.txt",
    7:  "kannada_07.txt",
    8:  "kannada_08.txt",
    9:  "kannada_09.txt",
    10: "kannada_10.txt",
    11: "kannada_11.txt",
    12: "kannada_12.txt",
    13: "kannada_13.txt",
    14: "kannada_14.txt",
    15: "kannada_15.txt",
    16: "kannada_16.txt",
    17: "kannada_17.txt",
    18: "kannada_18.txt",
    19: "kannada_19.txt",
    20: "kannada_20.txt",
}

# ── video_file column values in annotations.csv → numeric video id ────────────
VIDEO_FILE_TO_ID = {
    "gita-1st_video":  "1",
    "gita-2nd_video":  "2",
    "gita-3rd_video":  "3",
    "gita-4th_video":  "4",
    "gita-5th_video":  "5",
    "gita-6th_video":  "6",
    "kannada_07":      "7",
    "kannada_08":      "8",
    "kannada_09":      "9",
    "kannada_10":      "10",
    "kannada_11":      "11",
    "kannada_12":      "12",
    "kannada_13":      "13",
    "kannada_14":      "14",
    "kannada_15":      "15",
    "kannada_16":      "16",
    "kannada_17":      "17",
    "kannada_18":      "18",
    "kannada_19":      "19",
    "kannada_20":      "20",
}

# ── try to import your existing detector ──────────────────────────────────────
try:
    from shloka_detector import ShlokaDetector
    DETECTOR_AVAILABLE = True
except ImportError:
    DETECTOR_AVAILABLE = False
    print("[WARN] shloka_detector.py not found — using stub (returns no verses).")
    print("       Make sure shloka_detector.py is in the same folder.\n")


class _StubDetector:
    """Placeholder so the script runs even without shloka_detector.py."""
    def detect(self, text: str):
        return []


# ─────────────────────────────────────────────────────────────────────────────
# ANNOTATIONS LOADER  (fixed: reads video_file column, skips "-" verse refs)
# ─────────────────────────────────────────────────────────────────────────────
def load_annotations(annotations_path: str) -> dict:
    """
    Reads annotations.csv → { "1": ["BG 2.1", "BG 2.2"], "2": [...], ... }

    Your CSV layout:
      video_file, verse_ref, speaker, sanskrit_shloka, ...
      verse_ref = "-" means intro/closing row — skipped
    """
    annotations = {str(i): [] for i in range(1, 21)}

    if not os.path.exists(annotations_path):
        print(f"[WARN] Annotations file not found: {annotations_path}")
        print("       Evaluating with empty ground truth (all FP, zero TP).")
        return annotations

    with open(annotations_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # support both column names just in case
            video_file = row.get("video_file", row.get("video_id", "")).strip()
            if not video_file:
                continue

            # map "gita-1st_video" → "1" etc.
            vid_id = VIDEO_FILE_TO_ID.get(video_file)
            if vid_id is None:
                continue  # unrecognised video, skip silently

            verse = row.get("verse_ref", "").strip()
            # skip blank or placeholder rows
            if not verse or verse == "-":
                continue

            # avoid duplicates (same verse can appear in multiple rows)
            if verse not in annotations[vid_id]:
                annotations[vid_id].append(verse)

    # show what was loaded so you can verify
    print("[INFO] Ground-truth annotations loaded:")
    for vid_id in sorted(annotations.keys(), key=int):
        verses = annotations[vid_id]
        if verses:
            print(f"  Video {int(vid_id):>2}: {verses}")
    print()

    return annotations


# ─────────────────────────────────────────────────────────────────────────────
# METRICS
# ─────────────────────────────────────────────────────────────────────────────
def compute_metrics(expected: list, detected: list) -> dict:
    expected_set = set(v.strip() for v in expected)
    detected_set = set(v.strip() for v in detected)

    tp = len(expected_set & detected_set)
    fp = len(detected_set - expected_set)
    fn = len(expected_set - detected_set)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) > 0 else 0.0)

    return {
        "tp": tp, "fp": fp, "fn": fn,
        "precision": round(precision, 4),
        "recall":    round(recall,    4),
        "f1":        round(f1,        4),
    }


def aggregate_metrics(video_results: list) -> dict:
    if not video_results:
        return {"total_tp": 0, "total_fp": 0, "total_fn": 0,
                "micro_precision": 0.0, "micro_recall": 0.0, "micro_f1": 0.0,
                "macro_precision": 0.0, "macro_recall": 0.0, "macro_f1": 0.0}

    total_tp = sum(r["tp"] for r in video_results)
    total_fp = sum(r["fp"] for r in video_results)
    total_fn = sum(r["fn"] for r in video_results)

    micro_p = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    micro_r = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    micro_f = (2 * micro_p * micro_r / (micro_p + micro_r)
               if (micro_p + micro_r) > 0 else 0.0)

    macro_p = sum(r["precision"] for r in video_results) / len(video_results)
    macro_r = sum(r["recall"]    for r in video_results) / len(video_results)
    macro_f = sum(r["f1"]        for r in video_results) / len(video_results)

    return {
        "total_tp": total_tp, "total_fp": total_fp, "total_fn": total_fn,
        "micro_precision": round(micro_p, 4),
        "micro_recall":    round(micro_r, 4),
        "micro_f1":        round(micro_f, 4),
        "macro_precision": round(macro_p, 4),
        "macro_recall":    round(macro_r, 4),
        "macro_f1":        round(macro_f, 4),
    }


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Run sloka detector on all 20 Kannada transcripts and report P/R/F1"
    )
    parser.add_argument(
        "--transcripts", default="data/transcripts",
        help="Folder with transcript .txt files  (default: data/transcripts)"
    )
    parser.add_argument(
        "--annotations", default="data/annotations.csv",
        help="Ground-truth CSV  (default: data/annotations.csv)"
    )
    parser.add_argument(
        "--output", default="results/detector_kannada_full.json",
        help="Output JSON path  (default: results/detector_kannada_full.json)"
    )
    parser.add_argument(
        "--pilot-count", type=int, default=6,
        help="How many videos are the pilot set  (default: 6)"
    )
    args = parser.parse_args()

    transcript_dir  = Path(args.transcripts)
    annotations_csv = args.annotations
    output_path     = Path(args.output)
    pilot_count     = args.pilot_count

    if not transcript_dir.exists():
        sys.exit(f"[ERROR] Transcript directory not found: {transcript_dir}\n"
                 f"        Pass the correct path with --transcripts <path>")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # ── initialise detector ───────────────────────────────────────────────────
    if DETECTOR_AVAILABLE:
        detector = ShlokaDetector()
        print("[INFO] ShlokaDetector loaded OK")
    else:
        detector = _StubDetector()

    # ── load annotations ──────────────────────────────────────────────────────
    annotations = load_annotations(annotations_csv)

    # ── run detector on all 20 videos ─────────────────────────────────────────
    all_results = []

    print(f"{'#':<5} {'File':<26} {'Exp':>4} {'Det':>4} "
          f"{'TP':>4} {'FP':>4} {'FN':>4} "
          f"{'P':>7} {'R':>7} {'F1':>7}  Set")
    print("─" * 85)

    for vid_num in range(1, 21):
        filename = FILENAMES[vid_num]
        txt_file = transcript_dir / filename
        vid_set  = "pilot" if vid_num <= pilot_count else "new"

        if not txt_file.exists():
            print(f"{vid_num:<5} {filename:<26} [FILE NOT FOUND — skipped]")
            continue

        transcript_text = txt_file.read_text(encoding="utf-8")

        # run detector
        raw_detections = detector.detect(transcript_text)

        # normalise: handles dict output {"verse_ref": "BG 2.X"} or plain strings
        if raw_detections and isinstance(raw_detections[0], dict):
            detected_verses = [
                d.get("verse_ref", d.get("verse", d.get("id", "")))
                for d in raw_detections
            ]
        else:
            detected_verses = [str(d) for d in raw_detections]

        expected_verses = annotations.get(str(vid_num), [])
        metrics = compute_metrics(expected_verses, detected_verses)

        result = {
            "video_id":        vid_num,
            "filename":        filename,
            "set":             vid_set,
            "expected_verses": expected_verses,
            "detected_verses": detected_verses,
            **metrics,
        }
        all_results.append(result)

        print(f"{vid_num:<5} {filename:<26} {len(expected_verses):>4} {len(detected_verses):>4} "
              f"{metrics['tp']:>4} {metrics['fp']:>4} {metrics['fn']:>4} "
              f"{metrics['precision']:>7.4f} {metrics['recall']:>7.4f} {metrics['f1']:>7.4f}"
              f"  {vid_set}")

    # ── aggregate ─────────────────────────────────────────────────────────────
    pilot_results = [r for r in all_results if r["set"] == "pilot"]
    new_results   = [r for r in all_results if r["set"] == "new"]

    pilot_agg = aggregate_metrics(pilot_results)
    new_agg   = aggregate_metrics(new_results)
    full_agg  = aggregate_metrics(all_results)

    print("\n" + "=" * 85)
    print(f"{'AGGREGATE SUMMARY':^85}")
    print("=" * 85)
    print(f"{'Corpus':<22} {'Videos':>7} {'TP':>6} {'FP':>6} {'FN':>6} "
          f"{'Precision':>10} {'Recall':>8} {'F1':>8}")
    print("-" * 85)

    def _row(label, n, agg):
        print(f"{label:<22} {n:>7} {agg['total_tp']:>6} {agg['total_fp']:>6} {agg['total_fn']:>6} "
              f"{agg['micro_precision']:>10.4f} {agg['micro_recall']:>8.4f} {agg['micro_f1']:>8.4f}")

    _row("Kannada pilot",  len(pilot_results), pilot_agg)
    _row("Kannada new",    len(new_results),   new_agg)
    _row("Kannada full",   len(all_results),   full_agg)
    print("=" * 85)

    # ── paper table ───────────────────────────────────────────────────────────
    print("\n── Paper Table (copy-paste ready) ──────────────────────────────────────────")
    print(f"  Kannada pilot ({len(pilot_results):>2} videos) | "
          f"P={pilot_agg['micro_precision']:.4f}  "
          f"R={pilot_agg['micro_recall']:.4f}  "
          f"F1={pilot_agg['micro_f1']:.4f}")
    print(f"  Kannada new   ({len(new_results):>2} videos) | "
          f"P={new_agg['micro_precision']:.4f}  "
          f"R={new_agg['micro_recall']:.4f}  "
          f"F1={new_agg['micro_f1']:.4f}")
    print(f"  Kannada full  ({len(all_results):>2} videos) | "
          f"P={full_agg['micro_precision']:.4f}  "
          f"R={full_agg['micro_recall']:.4f}  "
          f"F1={full_agg['micro_f1']:.4f}")

    # ── save JSON ─────────────────────────────────────────────────────────────
    output = {
        "config": {
            "transcript_dir":  str(transcript_dir),
            "annotations_csv": annotations_csv,
            "pilot_count":     pilot_count,
        },
        "per_video": all_results,
        "aggregate": {
            "pilot": {**pilot_agg, "n_videos": len(pilot_results)},
            "new":   {**new_agg,   "n_videos": len(new_results)},
            "full":  {**full_agg,  "n_videos": len(all_results)},
        },
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\n[INFO] Full results saved → {output_path}")


if __name__ == "__main__":
    main()