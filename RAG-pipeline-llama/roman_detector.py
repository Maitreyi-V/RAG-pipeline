"""
Romanised Sanskrit verse detector — the English counterpart to shloka_detector.

The Kannada transcripts render recited Sanskrit in Kannada script, so Layer 1
matches against verse_database's `kannada` field. The English (Sarvam)
transcripts render the same Sanskrit in Roman letters -- "Tam tatha kripaya
vishtam" -- which shares no characters with the Kannada forms. Matching English
units against the Kannada database therefore scores 0.000 for a trivial reason.

This module derives a Roman form of every verse from its Devanagari (never
hand-entered, so it cannot drift from the canonical text) and fuzzy-matches
against it, using the loose conventions ASR actually produces: r-vocalic -> "ri",
retroflex/palatal sibilants -> "sh", long vowels collapsed, diacritics stripped.
"""
import re
import unicodedata

from rapidfuzz import fuzz
from indic_transliteration.sanscript import transliterate, DEVANAGARI, IAST

from verse_database import VERSES_BG_CH2

# IAST -> the loose ASCII that ASR systems actually emit
_SUBS = {"ṛ": "ri", "ṝ": "ri", "ḷ": "li", "ṣ": "sh", "ś": "sh",
         "ñ": "n", "ṅ": "n", "ṇ": "n", "ṭ": "t", "ḍ": "d",
         "ṁ": "m", "ṃ": "m", "ḥ": "h",
         "ā": "a", "ī": "i", "ū": "u", "ē": "e", "ō": "o"}


def romanize(text: str) -> str:
    """Devanagari -> loose ASCII roman, matching ASR spelling conventions."""
    s = transliterate(text, DEVANAGARI, IAST).lower()
    for k, v in _SUBS.items():
        s = s.replace(k, v)
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    return re.sub(r"[^a-z ]+", " ", s)


def normalize_asr(text: str) -> str:
    """Same normalisation applied to transcript text so both sides align."""
    s = unicodedata.normalize("NFKD", text.lower())
    s = "".join(c for c in s if not unicodedata.combining(c))
    return re.sub(r"[^a-z ]+", " ", s)


class RomanShlokaDetector:
    def __init__(self, match_threshold: int = 65):
        self.match_threshold = match_threshold
        self.verses = {v["ref"]: re.sub(r"\s+", " ", romanize(v["devanagari"])).strip()
                       for v in VERSES_BG_CH2}

    def detect(self, text: str):
        """-> [{'verse_ref': 'BG 2.1', 'score': 72.7}, ...] above threshold."""
        hay = re.sub(r"\s+", " ", normalize_asr(text)).strip()
        if not hay:
            return []
        out = []
        for ref, verse in self.verses.items():
            score = fuzz.partial_ratio(verse, hay)
            if score >= self.match_threshold:
                out.append({"verse_ref": ref, "score": round(score, 1)})
        out.sort(key=lambda d: -d["score"])
        return out
