"""
Śloka Detector Evaluation
Run detector on all 6 Kannada transcripts and compare against manual annotations.
Outputs per-video and aggregate P/R/F1.
"""
import csv
import json
import os
import sys
from shloka_detector import ShlokaDetector, evaluate_detector

TRANSCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
ANNOTATIONS_FILE = os.path.join(TRANSCRIPTS_DIR, "data", "annotations.csv")

VIDEO_FILES = [
    "data/transcripts/gita-1st video.txt",
    "data/transcripts/gita-2nd video.txt",
    "data/transcripts/gita-3rd video.txt",
    "data/transcripts/gita-4th video.txt",
    "data/transcripts/gita-5th video.txt",
    "data/transcripts/gita-6th video.txt",
]

def load_ground_truth(annotations_file):
    """Parse annotations.csv → {video_file: [verse_refs]} for shloka_explanation rows."""
    ground_truth = {}
    with open(annotations_file, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            vid = row["video_file"].strip()
            ref = row["verse_ref"].strip()
            section = row["section_type"].strip().lower()
            if section == "shloka_explanation" and ref and ref != "-":
                ground_truth.setdefault(vid, [])
                if ref not in ground_truth[vid]:
                    ground_truth[vid].append(ref)
    return ground_truth

def main():
    detector = ShlokaDetector()
    ground_truth = load_ground_truth(ANNOTATIONS_FILE)

    all_results = []
    aggregate_tp = 0
    aggregate_fp = 0
    aggregate_fn = 0

    print("=" * 65)
    print("ŚLOKA DETECTOR EVALUATION — 6 Kannada Transcripts")
    print("=" * 65)

    for fname in VIDEO_FILES:
        fpath = os.path.join(TRANSCRIPTS_DIR, fname)
        if not os.path.exists(fpath):
            print(f"\n[SKIP] {fname} — file not found")
            continue

        with open(fpath, encoding="utf-8") as f:
            text = f.read()

        # Video key in annotations: "gita-1st_video" format
        # fname may be "data/transcripts/gita-1st video.txt" — normalize it
        basename = os.path.basename(fname)           # "gita-1st video.txt"
        vid_key = basename.replace(".txt", "").replace(" ", "_")  # "gita-1st_video"
        expected = ground_truth.get(vid_key, [])

        eval_result = evaluate_detector(detector, text, expected)

        tp = len(eval_result["true_positives"])
        fp = len(eval_result["false_positives"])
        fn = len(eval_result["false_negatives"])
        aggregate_tp += tp
        aggregate_fp += fp
        aggregate_fn += fn

        print(f"\n{'─'*65}")
        print(f"Video : {fname}")
        print(f"Expected verses : {sorted(expected)}")
        print(f"Detected verses : {sorted(d['ref'] for d in eval_result['detections'])}")
        print(f"True Positives  : {sorted(eval_result['true_positives'])}")
        print(f"False Positives : {sorted(eval_result['false_positives'])}")
        print(f"False Negatives : {sorted(eval_result['false_negatives'])}")
        print(f"Precision={eval_result['precision']:.4f}  Recall={eval_result['recall']:.4f}  F1={eval_result['f1']:.4f}")

        all_results.append({
            "video": fname,
            "expected": sorted(expected),
            "detected": sorted(d['ref'] for d in eval_result['detections']),
            **eval_result
        })

    # Aggregate metrics
    agg_precision = aggregate_tp / (aggregate_tp + aggregate_fp) if (aggregate_tp + aggregate_fp) > 0 else 0
    agg_recall    = aggregate_tp / (aggregate_tp + aggregate_fn) if (aggregate_tp + aggregate_fn) > 0 else 0
    agg_f1        = 2 * agg_precision * agg_recall / (agg_precision + agg_recall) if (agg_precision + agg_recall) > 0 else 0

    print(f"\n{'='*65}")
    print("AGGREGATE RESULTS (across all 6 videos)")
    print(f"{'='*65}")
    print(f"Total Expected  : {aggregate_tp + aggregate_fn}")
    print(f"Total Detected  : {aggregate_tp + aggregate_fp}")
    print(f"True Positives  : {aggregate_tp}")
    print(f"False Positives : {aggregate_fp}")
    print(f"False Negatives : {aggregate_fn}")
    print(f"\nPrecision : {agg_precision:.4f}")
    print(f"Recall    : {agg_recall:.4f}")
    print(f"F1        : {agg_f1:.4f}")
    print(f"{'='*65}")

    # Save results
    output = {
        "per_video": all_results,
        "aggregate": {
            "precision": round(agg_precision, 4),
            "recall": round(agg_recall, 4),
            "f1": round(agg_f1, 4),
            "true_positives": aggregate_tp,
            "false_positives": aggregate_fp,
            "false_negatives": aggregate_fn,
        }
    }
    out_path = os.path.join(TRANSCRIPTS_DIR, "results", "detector_eval.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"\nResults saved to: {out_path}")

if __name__ == "__main__":
    main()
