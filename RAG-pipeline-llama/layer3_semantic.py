"""
layer3_semantic.py
==================
Layer 3 of the 3-layer śloka detection system.

PURPOSE:
  Detects which Bhagavad Gita verse a transcript segment is about,
  even when the speaker never recites Sanskrit — only paraphrases in English.

HOW IT WORKS:
  1. At startup: embed all verse translations using BGE-M3 → build index
  2. At runtime: embed a transcript segment → find closest verse by cosine similarity
  3. If similarity >= threshold → return verse_ref + confidence
  4. If below threshold → return None (no confident match)

USED FOR:
  - English Advaita transcripts (Vedanta Society of NY)
  - Any discourse where speaker explains without reciting Sanskrit

USAGE:
  from layer3_semantic import Layer3SemanticMatcher

  matcher = Layer3SemanticMatcher()
  matcher.build_index()  # do this once at startup

  verse, score = matcher.detect("the soul is never born and never dies")
  # Returns: ("BG 2.20", 0.72)
"""

import json
import os
import sys
import requests
import numpy as np
from typing import Optional

# Add parent dir to path so config is accessible
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    import config
    OLLAMA_BASE_URL = config.OLLAMA_BASE_URL
    EMBED_MODEL = config.EMBED_MODEL  # should be "bge-m3"
except Exception:
    # Fallback defaults if config not available
    OLLAMA_BASE_URL = "http://localhost:11434"
    EMBED_MODEL = "bge-m3"

from verse_translations import VERSE_TRANSLATIONS


# ─────────────────────────────────────────────
# EMBEDDING HELPER
# ─────────────────────────────────────────────

def embed_text(text: str) -> Optional[np.ndarray]:
    """
    Embed a single text string using BGE-M3 via Ollama.
    Returns numpy array of shape (1024,) or None on failure.
    """
    try:
        resp = requests.post(
            f"{OLLAMA_BASE_URL}/api/embeddings",
            json={"model": EMBED_MODEL, "prompt": text},
            timeout=60
        )
        resp.raise_for_status()
        vec = resp.json().get("embedding", [])
        if not vec:
            return None
        return np.array(vec, dtype=np.float32)
    except Exception as e:
        print(f"[Layer3] Embedding error: {e}")
        return None


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two vectors."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


# ─────────────────────────────────────────────
# LAYER 3 MATCHER
# ─────────────────────────────────────────────

class Layer3SemanticMatcher:
    """
    Semantic verse matcher using pre-computed translation embeddings.
    
    Workflow:
      1. build_index() — called once at startup, embeds all translations
      2. detect(segment) — called per transcript segment
    """

    def __init__(self, threshold: float = 0.45, cache_path: str = "layer3_index.json"):
        """
        Args:
            threshold: minimum cosine similarity to accept a match (0.0-1.0)
                       0.45 is a reasonable starting point — tune after testing
            cache_path: where to save/load the pre-computed index
                        so you don't re-embed every run
        """
        self.threshold = threshold
        self.cache_path = cache_path
        # index: { "BG 2.11": [[...embedding...], [...embedding...]], ... }
        self.index: dict[str, list[np.ndarray]] = {}
        self.is_built = False

    def build_index(self, force_rebuild: bool = False) -> None:
        """
        Pre-compute BGE-M3 embeddings for all verse translations.
        Saves to cache so subsequent runs are instant.
        
        Args:
            force_rebuild: if True, ignore cache and re-embed everything
        """
        # Try loading from cache first
        if not force_rebuild and os.path.exists(self.cache_path):
            print(f"[Layer3] Loading index from cache: {self.cache_path}")
            self._load_cache()
            return

        print(f"[Layer3] Building verse embedding index...")
        print(f"         Model: {EMBED_MODEL}")
        print(f"         Verses: {len(VERSE_TRANSLATIONS)}")
        print(f"         This takes ~5-10 minutes on first run, cached after.\n")

        total_translations = sum(len(v) for v in VERSE_TRANSLATIONS.values())
        done = 0

        for verse_ref, translations in VERSE_TRANSLATIONS.items():
            embeddings = []
            for text in translations:
                vec = embed_text(text)
                if vec is not None:
                    embeddings.append(vec)
                done += 1
                print(f"  [{done}/{total_translations}] Embedded: {verse_ref}", end="\r")

            if embeddings:
                self.index[verse_ref] = embeddings
            else:
                print(f"\n  [WARN] No embeddings for {verse_ref} — skipping")

        print(f"\n[Layer3] Index built: {len(self.index)} verses")
        self._save_cache()
        self.is_built = True

    def detect(self, segment: str) -> tuple[Optional[str], float]:
        """
        Detect which verse a transcript segment is about.
        
        Args:
            segment: a chunk of transcript text (~200-400 words)
        
        Returns:
            (verse_ref, confidence) if match found, e.g. ("BG 2.13", 0.67)
            (None, best_score)      if no match above threshold
        """
        if not self.is_built and not self.index:
            raise RuntimeError("Index not built. Call build_index() first.")

        # Embed the transcript segment
        segment_vec = embed_text(segment)
        if segment_vec is None:
            return None, 0.0

        best_verse = None
        best_score = 0.0
        scores = {}

        for verse_ref, verse_embeddings in self.index.items():
            # Compare segment against ALL translations for this verse
            # Take the MAX score — if any translation matches, the verse matches
            verse_scores = [
                cosine_similarity(segment_vec, ve)
                for ve in verse_embeddings
            ]
            verse_score = max(verse_scores)
            scores[verse_ref] = verse_score

            if verse_score > best_score:
                best_score = verse_score
                best_verse = verse_ref

        # Only return if above threshold
        if best_score >= self.threshold:
            return best_verse, round(best_score, 4)
        else:
            return None, round(best_score, 4)

    def detect_top_k(self, segment: str, k: int = 3) -> list[tuple[str, float]]:
        """
        Return top-k verse matches with scores.
        Useful for debugging and analysis.
        
        Returns list of (verse_ref, score) sorted by score descending.
        """
        if not self.is_built and not self.index:
            raise RuntimeError("Index not built. Call build_index() first.")

        segment_vec = embed_text(segment)
        if segment_vec is None:
            return []

        scores = {}
        for verse_ref, verse_embeddings in self.index.items():
            verse_scores = [cosine_similarity(segment_vec, ve) for ve in verse_embeddings]
            scores[verse_ref] = max(verse_scores)

        sorted_verses = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [(v, round(s, 4)) for v, s in sorted_verses[:k]]

    # ── Cache management ────────────────────────────────────────────────

    def _save_cache(self) -> None:
        """Save embeddings to JSON cache file."""
        serializable = {
            verse_ref: [vec.tolist() for vec in vecs]
            for verse_ref, vecs in self.index.items()
        }
        with open(self.cache_path, "w") as f:
            json.dump(serializable, f)
        print(f"[Layer3] Index cached to: {self.cache_path}")

    def _load_cache(self) -> None:
        """Load embeddings from JSON cache file."""
        with open(self.cache_path, "r") as f:
            raw = json.load(f)
        self.index = {
            verse_ref: [np.array(vec, dtype=np.float32) for vec in vecs]
            for verse_ref, vecs in raw.items()
        }
        self.is_built = True
        print(f"[Layer3] Loaded {len(self.index)} verses from cache.")
