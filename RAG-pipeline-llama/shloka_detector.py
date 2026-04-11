"""
Śloka Detection Engine — "Shazam for Ślokas"
==============================================
Given raw Kannada ASR transcript text, this module:
  1. Scans for Sanskrit verse fragments using character n-gram fingerprinting
  2. Fuzzy-matches detected fragments against the canonical verse database
  3. Returns verse boundaries with confidence scores
  4. Identifies structural cues (padavibhāga markers, closing phrases)

This is the core novel contribution of the project.

How it works:
  - Sanskrit in Kannada script has distinctive character patterns:
    heavy conjuncts (ಕ್ಷ, ಜ್ಞ), visarga (ಃ), anusvara (ಂ), halanta (್)
  - We compute a "Sanskrit density score" for sliding windows of text
  - High-density windows are candidate verse fragments
  - Candidates are fuzzy-matched against the verse database
  - Structural cues (ಪದವಿಭಾಗ, ಶ್ಲೋಕ, ಕೃಷ್ಣಾರ್ಪಣ) help confirm boundaries

Usage:
    from shloka_detector import ShlokaDetector
    detector = ShlokaDetector()
    results = detector.detect(transcript_text)
    for r in results:
        print(f"{r['verse_ref']} at position {r['start']}–{r['end']} (confidence: {r['confidence']:.2f})")
"""
import re
from collections import Counter
from rapidfuzz import fuzz, process
from verse_database import VERSES_BG_CH2, FIRST_WORD_INDEX


class ShlokaDetector:
    """Detects and identifies Sanskrit śloka fragments in Kannada ASR text."""

    # --- Structural cue patterns ---
    SHLOKA_MARKERS = [
        r"ಶ್ಲೋಕ",           # "śloka"
        r"ಪದ\s*ವಿಭಾಗ",      # "padavibhāga" (word-by-word split)
        r"ಪದವಿಭಾಗ",
        r"ಮತ್ತೊಮ್ಮೆ\s*ಶ್ಲೋಕ",  # "once more, the śloka"
        r"ಶ್ಲೋಕ\s*ಹೀಗಿದೆ",    # "the śloka is as follows"
        r"ಅಧ್ಯಾಯ",           # "chapter"
    ]

    CLOSING_MARKERS = [
        r"ಕೃಷ್ಣಾರ್ಪಣ",       # "offered to Krishna"
        r"ಶ್ರೀ\s*ಕೃಷ್ಣಾರ್ಪಣ",
        r"ಹರಿಃ\s*ಓಂ",        # "Hari Om"
        r"ಗುರುಭ್ಯೋ\s*ನಮಃ",   # "salutations to the guru"
    ]

    INTRO_MARKERS = [
        r"ಶ್ರೀ\s*ಗುರುಭ್ಯೋ\s*ನಮಃ",
        r"ಹಿಂದಿನ\s*ಭಾಗ",      # "previous part"
        r"ಕಳೆದ\s*ಸಲ",         # "last time"
    ]

    # Sanskrit-indicative Kannada characters (higher density = likely Sanskrit)
    SANSKRIT_HEAVY_CHARS = set("ಃಂ್ಷಕ್ಷಜ್ಞಶ್ರೀ")
    CONJUNCT_PATTERN = re.compile(r"[ಕ-ಹ]್[ಕ-ಹ]")  # halanta + consonant = conjunct
    VISARGA_ANUSVARA = re.compile(r"[ಃಂ]")

    def __init__(self, verses=None, window_size=80, stride=20,
                 sanskrit_threshold=0.12, match_threshold=55):
        """
        Args:
            verses: list of verse dicts from verse_database.py
            window_size: character window for scanning (default 80 chars ≈ half a śloka)
            stride: step size for sliding window
            sanskrit_threshold: min Sanskrit density to flag a window (0-1)
            match_threshold: min fuzzy match score (0-100) to identify a verse
        """
        self.verses = verses or VERSES_BG_CH2
        self.window_size = window_size
        self.stride = stride
        self.sanskrit_threshold = sanskrit_threshold
        self.match_threshold = match_threshold

        # Pre-build search corpus: all canonical verse texts
        self.verse_texts = {v["ref"]: v["kannada"] for v in self.verses}
        self.verse_refs = list(self.verse_texts.keys())
        self.verse_corpus = list(self.verse_texts.values())

    def sanskrit_density(self, text):
        """
        Compute a score (0-1) indicating how likely this text is Sanskrit
        written in Kannada script vs native Kannada.

        Signals:
        - High conjunct density (consonant clusters like ್ + consonant)
        - Visarga (ಃ) and anusvara (ಂ) frequency
        - Halanta (್) frequency
        - Absence of common Kannada function words
        """
        if len(text) < 10:
            return 0.0

        chars = len(text)

        # Count Sanskrit-indicative features
        conjuncts = len(self.CONJUNCT_PATTERN.findall(text))
        visarga_anusvara = len(self.VISARGA_ANUSVARA.findall(text))
        halanta_count = text.count("್")

        # Common Kannada function words (their presence = more likely Kannada)
        kannada_words = ["ಅಂದ್ರೆ", "ಅಂತ", "ಇದು", "ಅದು", "ಏನು", "ಹೇಗೆ",
                         "ಮಾಡಿ", "ಇದ್ದ", "ಆಗಿ", "ನಮ್ಮ", "ನಿಮ್ಮ", "ಅವರ",
                         "ಹೇಳ", "ನೋಡ", "ಬಂದ", "ಹೋಗ", "ಕೊಡ", "ಬೇಕು",
                         "ಇಲ್ಲ", "ಅಲ್ಲ", "ಹೌದು", "ಯಾಕೆ", "ಅಲ್ಲಿ", "ಇಲ್ಲಿ"]
        kn_word_count = sum(1 for w in kannada_words if w in text)

        # Sanskrit density formula
        sanskrit_score = (
            (conjuncts * 3 + visarga_anusvara * 2 + halanta_count * 1.5) / chars
        )

        # Penalize for Kannada function words
        kannada_penalty = min(kn_word_count * 0.03, 0.15)

        return max(0.0, min(1.0, sanskrit_score - kannada_penalty))

    def find_sanskrit_windows(self, text):
        """
        Scan text with sliding window, return regions with high Sanskrit density.
        Returns list of (start_idx, end_idx, density_score).
        """
        windows = []
        text_len = len(text)

        for i in range(0, text_len - self.window_size, self.stride):
            window = text[i:i + self.window_size]
            density = self.sanskrit_density(window)

            if density >= self.sanskrit_threshold:
                windows.append((i, i + self.window_size, density))

        # Merge overlapping windows
        if not windows:
            return []

        merged = [windows[0]]
        for start, end, density in windows[1:]:
            prev_start, prev_end, prev_density = merged[-1]
            if start <= prev_end + self.stride:
                # Merge: extend and take max density
                merged[-1] = (prev_start, max(prev_end, end), max(prev_density, density))
            else:
                merged.append((start, end, density))

        return merged

    def fuzzy_match_verse(self, candidate_text):
        """
        Match a candidate Sanskrit fragment against the verse database.
        Returns (verse_ref, score, matched_verse_text) or None.
        """
        # Quick check: does candidate start with any known first word?
        candidate_words = candidate_text.strip().split()[:3]
        for word in candidate_words:
            if word in FIRST_WORD_INDEX:
                ref = FIRST_WORD_INDEX[word]
                score = fuzz.partial_ratio(candidate_text, self.verse_texts[ref])
                if score >= self.match_threshold:
                    return ref, score, self.verse_texts[ref]

        # Full fuzzy search against all verses
        result = process.extractOne(
            candidate_text,
            self.verse_corpus,
            scorer=fuzz.partial_ratio,
            score_cutoff=self.match_threshold
        )

        if result:
            matched_text, score, idx = result
            ref = self.verse_refs[idx]
            return ref, score, matched_text

        return None

    def find_structural_cues(self, text):
        """Find structural markers (padavibhāga, closing phrases, etc.)."""
        cues = []

        for pattern in self.SHLOKA_MARKERS:
            for m in re.finditer(pattern, text):
                cues.append({
                    "type": "shloka_marker",
                    "text": m.group(),
                    "position": m.start(),
                    "end": m.end()
                })

        for pattern in self.CLOSING_MARKERS:
            for m in re.finditer(pattern, text):
                cues.append({
                    "type": "closing_marker",
                    "text": m.group(),
                    "position": m.start(),
                    "end": m.end()
                })

        for pattern in self.INTRO_MARKERS:
            for m in re.finditer(pattern, text):
                cues.append({
                    "type": "intro_marker",
                    "text": m.group(),
                    "position": m.start(),
                    "end": m.end()
                })

        return sorted(cues, key=lambda x: x["position"])

    def scan_first_words(self, text):
        """
        Pass 2: Scan text for known first words of any verse.
        This catches verses that the density-based scanner misses.
        Returns list of (position, verse_ref, match_score).
        """
        hits = []
        words = text.split()

        for i, word in enumerate(words):
            if word in FIRST_WORD_INDEX:
                ref = FIRST_WORD_INDEX[word]
                # Get position in original text
                pos = text.find(word)
                # Extract ~120 chars starting from this word for fuzzy matching
                candidate = text[pos:pos + 150]
                canonical = self.verse_texts.get(ref, "")
                score = fuzz.partial_ratio(candidate, canonical)

                if score >= self.match_threshold:
                    hits.append((pos, ref, score, candidate))

        return hits

    def detect(self, text):
        """
        Main detection pipeline (two-pass).
        Pass 1: Sanskrit density windows + fuzzy match
        Pass 2: First-word scanning for verses missed by Pass 1
        Returns list of detected ślokas with positions and confidence.
        """
        results = []

        # --- Pass 1: Sanskrit density windows ---
        sanskrit_windows = self.find_sanskrit_windows(text)

        for start, end, density in sanskrit_windows:
            expanded_start = max(0, start - 20)
            expanded_end = min(len(text), end + 40)
            candidate = text[expanded_start:expanded_end]

            # Try matching overlapping sub-windows within large merged windows
            if (end - start) > 200:
                # Large window — scan sub-segments
                for sub_start in range(start, end - 60, 40):
                    sub_end = min(sub_start + 150, len(text))
                    sub_candidate = text[sub_start:sub_end]
                    match = self.fuzzy_match_verse(sub_candidate)
                    if match:
                        ref, score, matched_text = match
                        if not any(r["verse_ref"] == ref for r in results):
                            confidence = min(1.0, (score / 100) * 0.7 + density * 0.3)
                            results.append({
                                "verse_ref": ref,
                                "start": sub_start,
                                "end": sub_end,
                                "matched_text": sub_candidate.strip(),
                                "canonical_text": matched_text,
                                "match_score": score,
                                "sanskrit_density": round(density, 3),
                                "confidence": round(confidence, 3),
                                "detection_method": "density_subscan",
                            })
            else:
                match = self.fuzzy_match_verse(candidate)
                if match:
                    ref, score, matched_text = match
                    if not any(r["verse_ref"] == ref for r in results):
                        confidence = min(1.0, (score / 100) * 0.7 + density * 0.3)
                        results.append({
                            "verse_ref": ref,
                            "start": expanded_start,
                            "end": expanded_end,
                            "matched_text": candidate.strip(),
                            "canonical_text": matched_text,
                            "match_score": score,
                            "sanskrit_density": round(density, 3),
                            "confidence": round(confidence, 3),
                            "detection_method": "density_match",
                        })

        # --- Pass 2: First-word scanning ---
        first_word_hits = self.scan_first_words(text)
        for pos, ref, score, candidate in first_word_hits:
            if not any(r["verse_ref"] == ref for r in results):
                density = self.sanskrit_density(candidate[:80])
                confidence = min(1.0, (score / 100) * 0.6 + 0.2)
                results.append({
                    "verse_ref": ref,
                    "start": pos,
                    "end": min(pos + 150, len(text)),
                    "matched_text": candidate.strip(),
                    "canonical_text": self.verse_texts.get(ref, ""),
                    "match_score": score,
                    "sanskrit_density": round(density, 3),
                    "confidence": round(confidence, 3),
                    "detection_method": "first_word_scan",
                })

        # Step 3: Find structural cues
        cues = self.find_structural_cues(text)

        # Step 4: Enrich results with nearby structural cues
        for result in results:
            nearby_cues = [
                c for c in cues
                if abs(c["position"] - result["start"]) < 500
            ]
            result["structural_cues"] = [c["type"] for c in nearby_cues]

        # Step 5: Detect section boundaries
        result_with_sections = self._assign_sections(text, results, cues)

        return sorted(result_with_sections, key=lambda x: x["start"])

    def _assign_sections(self, text, verse_results, cues):
        """Assign section types based on position relative to cues."""
        text_len = len(text)

        for result in verse_results:
            pos = result["start"]

            # Check if near intro markers
            intro_cues = [c for c in cues if c["type"] == "intro_marker" and c["position"] < pos]
            closing_cues = [c for c in cues if c["type"] == "closing_marker" and c["position"] > pos]

            if intro_cues and (pos - intro_cues[-1]["position"]) < 1000:
                result["section_context"] = "near_introduction"
            elif closing_cues and (closing_cues[0]["position"] - pos) < 500:
                result["section_context"] = "near_closing"
            else:
                result["section_context"] = "explanation_body"

        return verse_results

    def detect_and_chunk(self, text, video_file="unknown"):
        """
        Full pipeline: detect ślokas and create chunks automatically.
        Returns chunks similar to 01_chunk.py output but WITHOUT manual annotation.
        """
        detections = self.detect(text)
        cues = self.find_structural_cues(text)
        chunks = []
        chunk_id = 0

        # Sort detections by position
        detections = sorted(detections, key=lambda d: d["start"])

        # Create intro chunk (text before first detection)
        if detections:
            first_start = detections[0]["start"]
            if first_start > 100:
                chunk_id += 1
                chunks.append({
                    "chunk_id": f"auto_{chunk_id:03d}",
                    "video_file": video_file,
                    "verse_ref": None,
                    "section_type": "introduction",
                    "text": text[:first_start].strip(),
                    "detection_method": "structural",
                    "confidence": 0.8,
                })

        # Create verse chunks
        for i, det in enumerate(detections):
            chunk_id += 1

            # Determine chunk boundaries:
            # start = where this verse was detected
            # end = where next verse starts, or next structural cue, or +2000 chars
            chunk_start = det["start"]
            if i + 1 < len(detections):
                chunk_end = detections[i + 1]["start"]
            else:
                # Find closing marker after this verse
                closing = [c for c in cues if c["type"] == "closing_marker"
                          and c["position"] > det["end"]]
                if closing:
                    chunk_end = closing[0]["position"]
                else:
                    chunk_end = min(det["start"] + 3000, len(text))

            chunk_text = text[chunk_start:chunk_end].strip()

            chunks.append({
                "chunk_id": f"auto_{chunk_id:03d}",
                "video_file": video_file,
                "verse_ref": det["verse_ref"],
                "section_type": "shloka_explanation",
                "text": chunk_text,
                "sanskrit_detected": det["matched_text"],
                "canonical_verse": det["canonical_text"],
                "match_score": det["match_score"],
                "confidence": det["confidence"],
                "detection_method": "automatic",
                "start_pos": chunk_start,
                "end_pos": chunk_end,
            })

        # Create closing chunk if closing markers exist after last verse
        if detections:
            last_end = detections[-1]["end"]
            closing_cues = [c for c in cues if c["type"] == "closing_marker"
                          and c["position"] > last_end]
            if closing_cues:
                chunk_id += 1
                close_start = closing_cues[0]["position"]
                chunks.append({
                    "chunk_id": f"auto_{chunk_id:03d}",
                    "video_file": video_file,
                    "verse_ref": None,
                    "section_type": "closing",
                    "text": text[close_start:].strip(),
                    "detection_method": "structural",
                    "confidence": 0.9,
                })

        return chunks


def evaluate_detector(detector, transcript_text, ground_truth_verses):
    """
    Evaluate detection accuracy against manual annotations.
    Returns precision, recall, F1, and per-verse results.
    """
    detections = detector.detect(transcript_text)
    detected_refs = set(d["verse_ref"] for d in detections)
    expected_refs = set(ground_truth_verses)

    true_positives = detected_refs & expected_refs
    false_positives = detected_refs - expected_refs
    false_negatives = expected_refs - detected_refs

    precision = len(true_positives) / len(detected_refs) if detected_refs else 0
    recall = len(true_positives) / len(expected_refs) if expected_refs else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "true_positives": sorted(true_positives),
        "false_positives": sorted(false_positives),
        "false_negatives": sorted(false_negatives),
        "num_detected": len(detected_refs),
        "num_expected": len(expected_refs),
        "detections": [
            {"ref": d["verse_ref"], "confidence": d["confidence"], "score": d["match_score"]}
            for d in detections
        ]
    }


# --- CLI for testing ---
if __name__ == "__main__":
    import sys
    import json

    detector = ShlokaDetector()

    if len(sys.argv) > 1:
        # Test on a transcript file
        filepath = sys.argv[1]
        with open(filepath, "r", encoding="utf-8") as f:
            text = f.read()

        print(f"Scanning: {filepath} ({len(text)} chars)")
        print("=" * 60)

        results = detector.detect(text)
        print(f"\nDetected {len(results)} śloka(s):\n")
        for r in results:
            print(f"  {r['verse_ref']:10s} | confidence={r['confidence']:.2f} | "
                  f"score={r['match_score']:3d} | density={r['sanskrit_density']:.3f}")
            print(f"  {'':10s} | pos={r['start']}–{r['end']}")
            print(f"  {'':10s} | cues: {r.get('structural_cues', [])}")
            print()

        # Auto-chunk
        chunks = detector.detect_and_chunk(text, video_file=filepath)
        print(f"\nAuto-generated {len(chunks)} chunks:")
        for c in chunks:
            ref = c.get("verse_ref") or c["section_type"]
            conf = c.get("confidence", "N/A")
            print(f"  {c['chunk_id']}: {ref} (conf={conf}) [{len(c['text'])} chars]")

    else:
        # Quick test with embedded sample
        sample = """ಶ್ರೀ ಗುರುಭ್ಯೋ ನಮಃ ಹರಿಃ ಓಂ ಭಗವದ್ಗೀತೆಯ ಎರಡನೆಯ ಅಧ್ಯಾಯದ ಈ ಶ್ಲೋಕದಲ್ಲಿ ಶ್ರೀಕೃಷ್ಣ ಹೇಳ್ತಿದ್ದಾನೆ ನತ್ವೇವಾಹಂ ಜಾತುನಾಸಂ ನತ್ವಂ ನೇಮೇ ಜನಾಧಿಪಾಹ ನಚೈವ ನ ಭವಿಷ್ಯಾಮಃ ಸರ್ವೇ ವಯಮತಃ ಪರಂ ಈ ಶ್ಲೋಕದಲ್ಲಿ ಶ್ರೀಕೃಷ್ಣ ಆತ್ಮ ನಿತ್ಯ ಅಂತ ಹೇಳ್ತಿದ್ದಾನೆ ಶರೀರಕ್ಕೆ ನಾಶ ಆತ್ಮನಿಗೆ ಎಂದು ನಾಶ ಇಲ್ಲ ಶ್ರೀ ಕೃಷ್ಣಾರ್ಪಣಮಸ್ತು"""

        print("Quick test with embedded sample:")
        print("=" * 60)

        results = detector.detect(sample)
        for r in results:
            print(f"  Detected: {r['verse_ref']} (confidence={r['confidence']:.2f})")

        cues = detector.find_structural_cues(sample)
        print(f"\n  Structural cues: {[c['type'] for c in cues]}")

        chunks = detector.detect_and_chunk(sample, "test_video")
        print(f"\n  Auto-chunks: {len(chunks)}")
        for c in chunks:
            ref = c.get("verse_ref") or c["section_type"]
            print(f"    {c['chunk_id']}: {ref}")
