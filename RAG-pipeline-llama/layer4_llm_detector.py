#!/usr/bin/env python3
"""
Layer 4: LLM-based Verse Detector
==================================
Sends transcript segments to an LLM and asks it to identify which
Bhagavad Gita Chapter 2 verse is being discussed.

This is the KEY EXPERIMENT for the paper ? comparing:
  - Layer 1 (string matching, F1=0.92 on Kannada)
  - Layer 3 (embedding similarity, F1=0.48 on English)
  - Layer 4 (LLM identification, F1=?? on English)  <-- THIS FILE

Usage:
  # With OpenAI (recommended for you):
  export OPENAI_API_KEY="sk-..."
  python layer4_llm_detector.py --provider openai --model gpt-4o-mini

  # With Ollama (for teammates):
  python layer4_llm_detector.py --provider ollama --model llama3.1:8b

  # Run on a single video to test:
  python layer4_llm_detector.py --provider openai --model gpt-4o-mini --video video_02

  # Adjust segment size (default 200 words):
  python layer4_llm_detector.py --provider openai --model gpt-4o-mini --segment-words 300
"""

import json
import re
import os
import sys
import time
import argparse
from pathlib import Path
from typing import List, Dict, Optional, Tuple

from dotenv import load_dotenv

load_dotenv()


# ??? Ground Truth (from advaita_ground_truth.py) ???????????????????????

GROUND_TRUTH = {
    "video_01": list(range(1, 11)),    # BG 2.1-2.10  (Intro episode)
    "video_02": [11, 12],              # BG 2.11-2.12
    "video_03": [13, 14, 15],          # BG 2.13-2.15
    "video_04": [16],                  # BG 2.16
    "video_05": [17, 18],              # BG 2.17-2.18
    "video_06": [19],                  # BG 2.19
    "video_07": [20, 21, 22],          # BG 2.20-2.22
    "video_08": [22, 23, 24, 25],      # BG 2.22-2.25
    "video_09": [26, 27],              # BG 2.26-2.27
    "video_10": [28, 29],              # BG 2.28-2.29
}

# ??? LLM Clients ???????????????????????????????????????????????????????

def call_openai(prompt: str, model: str = "gpt-4o-mini") -> str:
    """Call OpenAI API."""
    from openai import OpenAI

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set — check the repo-root .env file")
    client = OpenAI(api_key=api_key)
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=50,
    )
    return resp.choices[0].message.content.strip()


def call_ollama(prompt: str, model: str = "llama3.1:8b") -> str:
    """Call Ollama local API."""
    import requests
    resp = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.0, "num_predict": 50},
        },
        timeout=120,
    )
    resp.raise_for_status()
    return resp.json()["response"].strip()


def call_lm(prompt: str, provider: str, model: str) -> str:
    """Route to the right LLM backend."""
    if provider == "openai":
        return call_openai(prompt, model)
    elif provider == "ollama":
        return call_ollama(prompt, model)
    else:
        raise ValueError(f"Unknown provider: {provider}")


# ??? Transcript Handling ???????????????????????????????????????????????

def load_transcript(path: Path) -> str:
    """Load transcript text, handling both key formats."""
    with open(path) as f:
        data = json.load(f)
    # Handle the key inconsistency across transcripts
    text = data.get("text", data.get("full_transcript", ""))
    if not text and "segments" in data:
        # Fall back to concatenating segments
        text = " ".join(
            seg.get("text", "") for seg in data["segments"]
        )
    return text.strip()


def segment_text(text: str, segment_words: int = 200) -> List[Dict]:
    """Split transcript into overlapping segments of ~N words."""
    words = text.split()
    segments = []
    step = segment_words  # non-overlapping for now; can add overlap later
    for i in range(0, len(words), step):
        chunk_words = words[i:i + segment_words]
        segments.append({
            "index": len(segments),
            "start_word": i,
            "end_word": min(i + segment_words, len(words)),
            "text": " ".join(chunk_words),
        })
    return segments


# ??? Verse Detection Prompt ???????????????????????????????????????????

DETECTION_PROMPT = """You are an expert on the Bhagavad Gita Chapter 2. 
Read the following passage from a lecture/discourse and identify which specific verse(s) from Bhagavad Gita Chapter 2 (verses 2.1 through 2.72) the speaker is explaining or discussing.

Rules:
- Reply with ONLY the verse number(s), e.g. "2.17" or "2.22, 2.23"
- If the passage discusses a general theme without focusing on a specific verse, reply "none"
- If you are unsure, reply "none"
- Only identify verses from Chapter 2 (2.1 to 2.72)
- Do NOT explain your reasoning, just output the verse number(s)

Passage:
\"\"\"
{segment_text}
\"\"\"

Verse(s):"""


def detect_verse_llm(segment_text: str, provider: str, model: str) -> List[int]:
    """Ask the LLM which verse a segment discusses. Returns list of verse numbers."""
    prompt = DETECTION_PROMPT.format(segment_text=segment_text)
    try:
        response = call_lm(prompt, provider, model)
    except Exception as e:
        print(f"  [LLM error: {e}]")
        return []

    return parse_verse_response(response)


def parse_verse_response(response: str) -> List[int]:
    """Parse LLM response like '2.17' or '2.22, 2.23' or 'none' into verse numbers."""
    response = response.lower().strip()
    if "none" in response or not response:
        return []

    verses = []
    # Match patterns like 2.17, 2.22, BG 2.17, verse 17, etc.
    patterns = [
        r'2\.(\d+)',           # "2.17"
        r'verse\s*(\d+)',      # "verse 17"
        r'(?:bg|gita)\s*2?\.?(\d+)',  # "BG 17" or "Gita 2.17"
    ]
    for pat in patterns:
        for m in re.finditer(pat, response):
            v = int(m.group(1))
            if 1 <= v <= 72:
                verses.append(v)

    # If no pattern matched but there's a bare number
    if not verses:
        for m in re.finditer(r'\b(\d+)\b', response):
            v = int(m.group(1))
            if 1 <= v <= 72:
                verses.append(v)

    return sorted(set(verses))


# ??? Evaluation ????????????????????????????????????????????????????????

def evaluate_video(
    video_id: str,
    transcript_path: Path,
    provider: str,
    model: str,
    segment_words: int = 200,
    verbose: bool = True,
) -> Dict:
    """Run verse detection on a single video and compare to ground truth."""

    gt_verses = set(GROUND_TRUTH.get(video_id, []))
    if not gt_verses:
        print(f"  WARNING: No ground truth for {video_id}, skipping.")
        return {}

    text = load_transcript(transcript_path)
    segments = segment_text(text, segment_words)

    if verbose:
        print(f"\n{'='*60}")
        print(f"  {video_id} | GT verses: {sorted(gt_verses)} | {len(segments)} segments")
        print(f"{'='*60}")

    # Collect all predicted verses across segments
    all_predictions = []
    segment_results = []

    for seg in segments:
        predicted = detect_verse_llm(seg["text"], provider, model)
        all_predictions.extend(predicted)
        segment_results.append({
            "segment_index": seg["index"],
            "predicted_verses": predicted,
            "text_preview": seg["text"][:100] + "...",
        })
        if verbose and predicted:
            print(f"  seg {seg['index']:3d} ? {predicted}")

        # Small delay to avoid rate limits (OpenAI)
        if provider == "openai":
            time.sleep(0.1)

    # Aggregate: which verses were detected at least once?
    predicted_set = set(all_predictions)

    # Also track vote counts (how many segments mentioned each verse)
    from collections import Counter
    vote_counts = Counter(all_predictions)

    # Compute P/R/F1
    tp = len(predicted_set & gt_verses)
    fp = len(predicted_set - gt_verses)
    fn = len(gt_verses - predicted_set)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    result = {
        "video_id": video_id,
        "ground_truth": sorted(gt_verses),
        "predicted": sorted(predicted_set),
        "vote_counts": dict(vote_counts),
        "tp": tp, "fp": fp, "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "num_segments": len(segments),
        "segment_results": segment_results,
    }

    if verbose:
        print(f"\n  Predicted: {sorted(predicted_set)}")
        print(f"  GT:        {sorted(gt_verses)}")
        print(f"  TP={tp} FP={fp} FN={fn}")
        print(f"  P={precision:.3f} R={recall:.3f} F1={f1:.3f}")
        if vote_counts:
            print(f"  Votes: {dict(vote_counts.most_common())}")

    return result


def run_full_evaluation(
    transcripts_dir: Path,
    provider: str,
    model: str,
    segment_words: int = 200,
    video_filter: Optional[str] = None,
    verbose: bool = True,
) -> Dict:
    """Run on all videos and compute aggregate metrics."""

    results = []
    total_tp, total_fp, total_fn = 0, 0, 0

    for video_id in sorted(GROUND_TRUTH.keys()):
        if video_filter and video_id != video_filter:
            continue

        # Try both naming conventions
        candidates = [
            transcripts_dir / f"{video_id}.json",
            transcripts_dir / f"english/{video_id}.json",
            transcripts_dir / f"advaita/{video_id}.json",
        ]
        transcript_path = None
        for c in candidates:
            if c.exists():
                transcript_path = c
                break

        if not transcript_path:
            print(f"  Transcript not found for {video_id}, tried: {[str(c) for c in candidates]}")
            continue

        r = evaluate_video(video_id, transcript_path, provider, model, segment_words, verbose)
        if r:
            results.append(r)
            total_tp += r["tp"]
            total_fp += r["fp"]
            total_fn += r["fn"]

    # Macro-average
    if results:
        macro_p = sum(r["precision"] for r in results) / len(results)
        macro_r = sum(r["recall"] for r in results) / len(results)
        macro_f1 = sum(r["f1"] for r in results) / len(results)
    else:
        macro_p = macro_r = macro_f1 = 0.0

    # Micro-average
    micro_p = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    micro_r = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    micro_f1 = 2 * micro_p * micro_r / (micro_p + micro_r) if (micro_p + micro_r) > 0 else 0.0

    summary = {
        "provider": provider,
        "model": model,
        "segment_words": segment_words,
        "num_videos": len(results),
        "micro": {"precision": micro_p, "recall": micro_r, "f1": micro_f1,
                  "tp": total_tp, "fp": total_fp, "fn": total_fn},
        "macro": {"precision": macro_p, "recall": macro_r, "f1": macro_f1},
        "per_video": results,
    }

    print(f"\n{'='*60}")
    print(f"  AGGREGATE RESULTS ({provider}/{model}, {segment_words}w segments)")
    print(f"{'='*60}")
    print(f"  Videos evaluated: {len(results)}")
    print(f"  Micro ? P={micro_p:.3f} R={micro_r:.3f} F1={micro_f1:.3f} (TP={total_tp} FP={total_fp} FN={total_fn})")
    print(f"  Macro ? P={macro_p:.3f} R={macro_r:.3f} F1={macro_f1:.3f}")

    return summary


# ??? Vote-filtered evaluation ?????????????????????????????????????????

def run_with_vote_filter(
    transcripts_dir: Path,
    provider: str,
    model: str,
    segment_words: int = 200,
    min_votes: int = 2,
    verbose: bool = True,
) -> Dict:
    """Same as full eval but only count a verse as 'detected' if it
    appears in >= min_votes segments. Reduces false positives from
    one-off LLM hallucinations."""

    from collections import Counter

    results = []
    total_tp, total_fp, total_fn = 0, 0, 0

    for video_id in sorted(GROUND_TRUTH.keys()):
        gt_verses = set(GROUND_TRUTH[video_id])

        candidates = [
            transcripts_dir / f"{video_id}.json",
            transcripts_dir / f"english/{video_id}.json",
            transcripts_dir / f"advaita/{video_id}.json",
        ]
        transcript_path = None
        for c in candidates:
            if c.exists():
                transcript_path = c
                break
        if not transcript_path:
            continue

        text = load_transcript(transcript_path)
        segments = segment_text(text, segment_words)

        all_preds = []
        for seg in segments:
            predicted = detect_verse_llm(seg["text"], provider, model)
            all_preds.extend(predicted)
            if provider == "openai":
                time.sleep(0.1)

        # Vote filter
        counts = Counter(all_preds)
        predicted_set = {v for v, c in counts.items() if c >= min_votes}

        tp = len(predicted_set & gt_verses)
        fp = len(predicted_set - gt_verses)
        fn = len(gt_verses - predicted_set)

        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0

        results.append({"video_id": video_id, "p": p, "r": r, "f1": f1,
                        "tp": tp, "fp": fp, "fn": fn, "votes": dict(counts)})
        total_tp += tp
        total_fp += fp
        total_fn += fn

        if verbose:
            print(f"  {video_id}: GT={sorted(gt_verses)} Pred={sorted(predicted_set)} "
                  f"P={p:.3f} R={r:.3f} F1={f1:.3f}")

    micro_p = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    micro_r = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    micro_f1 = 2 * micro_p * micro_r / (micro_p + micro_r) if (micro_p + micro_r) > 0 else 0.0

    print(f"\n  VOTE-FILTERED (min_votes={min_votes})")
    print(f"  Micro ? P={micro_p:.3f} R={micro_r:.3f} F1={micro_f1:.3f}")

    return {"min_votes": min_votes, "micro_p": micro_p, "micro_r": micro_r,
            "micro_f1": micro_f1, "per_video": results}


# ??? Main ??????????????????????????????????????????????????????????????

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Layer 4: LLM-based verse detector")
    parser.add_argument("--provider", choices=["openai", "ollama"], default="openai",
                        help="LLM provider (default: openai)")
    parser.add_argument("--model", default=None,
                        help="Model name (default: gpt-4o-mini for openai, llama3.1:8b for ollama)")
    parser.add_argument("--transcripts-dir", type=Path, default=Path("data/transcripts"),
                        help="Path to transcripts directory")
    parser.add_argument("--segment-words", type=int, default=200,
                        help="Segment size in words (default: 200)")
    parser.add_argument("--video", type=str, default=None,
                        help="Run on single video (e.g. video_02)")
    parser.add_argument("--min-votes", type=int, default=0,
                        help="Min segment votes to count a verse as detected (0=no filter)")
    parser.add_argument("--output", type=Path, default=None,
                        help="Save results JSON to this path")
    parser.add_argument("--quiet", action="store_true", help="Less output")

    args = parser.parse_args()

    # Default model per provider
    if args.model is None:
        args.model = "gpt-4o-mini" if args.provider == "openai" else "llama3.1:8b"

    print(f"Layer 4 LLM Verse Detector")
    print(f"Provider: {args.provider} | Model: {args.model}")
    print(f"Transcripts: {args.transcripts_dir}")
    print(f"Segment size: {args.segment_words} words")

    if args.min_votes >= 2:
        summary = run_with_vote_filter(
            args.transcripts_dir, args.provider, args.model,
            args.segment_words, args.min_votes, verbose=not args.quiet,
        )
    else:
        summary = run_full_evaluation(
            args.transcripts_dir, args.provider, args.model,
            args.segment_words, args.video, verbose=not args.quiet,
        )

    # Save results
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\nResults saved to {args.output}")
    elif not args.video:
        # Auto-save
        out_path = Path(f"results/layer4_{args.provider}_{args.segment_words}w.json")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\nResults auto-saved to {out_path}")
