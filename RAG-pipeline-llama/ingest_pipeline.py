"""
ingest_pipeline.py
==================
Full ingestion pipeline for new Bhagavad Gita lecture videos.

Pipeline:
  1. Download audio from YouTube (yt-dlp) OR accept an uploaded audio file
  2. Transcribe with faster-whisper (timestamped segments)
  3. Detect verse boundaries (tradition-aware)
  4. Build śloka-aware chunks with start/end timestamps
  5. Embed and upsert into ChromaDB

Public API:
  ingest_from_url(youtube_url, tradition, collection, progress_callback)
  ingest_from_file(audio_path, tradition, collection, video_label, youtube_url, progress_callback)
"""

import os
import re
import sys
import hashlib

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config


# ─────────────────────────────────────────────
# YOUTUBE DOWNLOAD
# ─────────────────────────────────────────────

def download_youtube_audio(url: str) -> tuple:
    """
    Download best-quality audio from a YouTube URL.
    Returns: (audio_path, video_id, title)
    Caches to AUDIO_CACHE_DIR — re-downloads are skipped.
    """
    import yt_dlp

    os.makedirs(config.AUDIO_CACHE_DIR, exist_ok=True)

    with yt_dlp.YoutubeDL({"quiet": True}) as ydl:
        info = ydl.extract_info(url, download=False)
        video_id = info.get("id") or hashlib.md5(url.encode()).hexdigest()[:12]
        title = info.get("title", "unknown")

    # Check cache first
    for ext in ("m4a", "webm", "mp3", "opus", "ogg"):
        cached = os.path.join(config.AUDIO_CACHE_DIR, f"{video_id}.{ext}")
        if os.path.exists(cached):
            return cached, video_id, title

    ydl_opts = {
        "format": "bestaudio[ext=m4a]/bestaudio/best",
        "outtmpl": os.path.join(config.AUDIO_CACHE_DIR, f"{video_id}.%(ext)s"),
        "quiet": True,
        "no_warnings": True,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])

    for ext in ("m4a", "webm", "mp3", "opus", "ogg"):
        candidate = os.path.join(config.AUDIO_CACHE_DIR, f"{video_id}.{ext}")
        if os.path.exists(candidate):
            return candidate, video_id, title

    raise FileNotFoundError(f"yt-dlp finished but no audio file found for video_id={video_id}")


# ─────────────────────────────────────────────
# TRANSCRIPTION
# ─────────────────────────────────────────────

def transcribe(audio_path: str, language: str = None) -> dict:
    """
    Transcribe an audio file with faster-whisper.
    Returns: {"text": str, "segments": [{"start", "end", "text"}, ...]}
    """
    from faster_whisper import WhisperModel

    model = WhisperModel(config.WHISPER_MODEL, device="cpu", compute_type="int8")
    kwargs = {"beam_size": 5, "vad_filter": True}
    if language:
        kwargs["language"] = language

    raw_segments, _info = model.transcribe(audio_path, **kwargs)

    segments = []
    parts = []
    for seg in raw_segments:
        segments.append({"start": seg.start, "end": seg.end, "text": seg.text})
        parts.append(seg.text)

    return {"text": " ".join(parts), "segments": segments}


# ─────────────────────────────────────────────
# POSITION ↔ TIMESTAMP MAPPING
# ─────────────────────────────────────────────

def _build_offset_map(segments: list) -> list:
    """
    Returns list of (char_start, char_end, time_start, time_end) per segment,
    mirroring how segments are concatenated with a single space separator.
    """
    offset_map = []
    pos = 0
    for seg in segments:
        text = seg["text"]
        start, end = pos, pos + len(text)
        offset_map.append((start, end, seg["start"], seg["end"]))
        pos = end + 1  # space separator
    return offset_map


def _pos_to_time(char_pos: int, offset_map: list) -> float:
    """Return the timestamp (seconds) for a character position in the full transcript."""
    for seg_start, seg_end, t_start, t_end in offset_map:
        if seg_start <= char_pos < seg_end:
            ratio = (char_pos - seg_start) / max(1, seg_end - seg_start)
            return t_start + ratio * (t_end - t_start)
    return offset_map[-1][3] if offset_map else 0.0


# ─────────────────────────────────────────────
# VERSE MENTION DETECTION
# ─────────────────────────────────────────────

def _english_verse_mentions(text: str, chapter: int = 2) -> list:
    """Detect verse mentions in English (Advaita) transcripts."""
    valid = set(range(1, 73))
    text_lower = text.lower()
    mentions = []

    num_patterns = [
        (r'\b(?:verse|shloka|sloka|stanza)\s*(?:number\s*)?(\d{1,2})\b', 1),
        (r'\b(\d{1,2})(?:st|nd|rd|th)\s+(?:verse|shloka|sloka|stanza)\b', 1),
        (r'\bchapter\s*2\s*(?:,?\s*)?(?:verse|shloka)?\s*(\d{1,2})\b', 1),
    ]
    for pat, grp in num_patterns:
        for m in re.finditer(pat, text_lower):
            num = int(m.group(grp))
            if num in valid:
                mentions.append({
                    "verse_ref": f"BG {chapter}.{num}",
                    "position": m.start(),
                    "matched_text": m.group(),
                })

    ordinals = {
        'eleventh': 11, 'twelfth': 12, 'thirteenth': 13, 'fourteenth': 14,
        'fifteenth': 15, 'sixteenth': 16, 'seventeenth': 17, 'eighteenth': 18,
        'nineteenth': 19, 'twentieth': 20, 'twenty-first': 21, 'twenty-second': 22,
        'twenty-third': 23, 'twenty-fourth': 24, 'twenty-fifth': 25,
        'twenty-sixth': 26, 'twenty-seventh': 27, 'twenty-eighth': 28,
        'twenty-ninth': 29, 'thirtieth': 30,
    }
    for word, num in ordinals.items():
        pat = rf'\b{re.escape(word)}\s+(?:verse|shloka|sloka)\b'
        for m in re.finditer(pat, text_lower):
            mentions.append({
                "verse_ref": f"BG {chapter}.{num}",
                "position": m.start(),
                "matched_text": m.group(),
            })

    seen = set()
    unique = []
    for m in sorted(mentions, key=lambda x: x["position"]):
        if not any(abs(m["position"] - p) < 20 for p in seen):
            unique.append(m)
            seen.add(m["position"])
    return unique


def _kannada_verse_mentions(text: str) -> list:
    """Detect śloka boundaries in Kannada (Dvaita) transcripts using ShlokaDetector."""
    from shloka_detector import ShlokaDetector
    detector = ShlokaDetector()
    return [
        {
            "verse_ref": d["verse_ref"],
            "position": d["start"],
            "matched_text": d.get("matched_text", ""),
        }
        for d in detector.detect(text)
    ]


# ─────────────────────────────────────────────
# ŚLOKA-AWARE CHUNKING WITH TIMESTAMPS
# ─────────────────────────────────────────────

def _chunk_by_mentions(
    full_text: str,
    mentions: list,
    offset_map: list,
    video_id: str,
    youtube_url: str,
    tradition: str,
    min_words: int = 80,
    max_words: int = 800,
    fallback_words: int = 400,
) -> list:
    """
    Split full_text into verse-aware chunks using detected mention positions.
    Each chunk carries start_time_sec / end_time_sec derived from offset_map.
    """
    text_len = len(full_text)
    chunks = []

    def _make(text_slice, start_pos, end_pos, verse_ref, section_type):
        start_sec = _pos_to_time(start_pos, offset_map)
        end_sec = _pos_to_time(min(end_pos, text_len - 1), offset_map)
        idx = len(chunks)
        return {
            "chunk_id": f"{video_id}_{idx:03d}",
            "video_id": video_id,
            "youtube_url": youtube_url,
            "tradition": tradition,
            "verse_ref": verse_ref or "",
            "section_type": section_type,
            "text": text_slice.strip(),
            "start_time_sec": round(start_sec, 1),
            "end_time_sec": round(end_sec, 1),
            "speaker": "Auto-transcribed",
            "video_file": video_id,
        }

    if not mentions:
        # No verse boundaries found — fixed-size fallback
        words = full_text.split()
        pos = 0
        for i in range(0, len(words), fallback_words):
            window_words = words[i:i + fallback_words]
            window_text = " ".join(window_words)
            chunks.append(_make(window_text, pos, pos + len(window_text), None, "fallback_chunk"))
            pos += len(window_text) + 1
        return chunks

    boundaries = sorted(mentions, key=lambda m: m["position"])

    # Intro chunk (before first mention)
    first_pos = boundaries[0]["position"]
    if first_pos > 200:
        intro = full_text[:first_pos]
        if len(intro.split()) >= min_words:
            chunks.append(_make(intro, 0, first_pos, None, "introduction"))

    # Verse chunks
    for i, mention in enumerate(boundaries):
        start = mention["position"]
        end = boundaries[i + 1]["position"] if i + 1 < len(boundaries) else text_len
        segment = full_text[start:end]
        word_count = len(segment.split())

        if word_count < min_words:
            # Too short — absorb into next chunk by skipping; it'll be included in i+1's range
            continue

        if word_count > max_words:
            # Over-long — split with 50-word overlap
            words = segment.split()
            stride = max_words - 50
            sub_pos = start
            for j in range(0, len(words), stride):
                sub_words = words[j:j + max_words]
                sub_text = " ".join(sub_words)
                chunks.append(_make(sub_text, sub_pos, sub_pos + len(sub_text),
                                    mention["verse_ref"], "shloka_explanation"))
                sub_pos += len(sub_text)
        else:
            chunks.append(_make(segment, start, end,
                                mention["verse_ref"], "shloka_explanation"))

    return chunks


# ─────────────────────────────────────────────
# EMBED + INDEX
# ─────────────────────────────────────────────

def _embed(text: str) -> list:
    import requests
    resp = requests.post(
        f"{config.OLLAMA_BASE_URL}/api/embeddings",
        json={"model": config.EMBED_MODEL, "prompt": text},
        timeout=60,
    )
    resp.raise_for_status()
    return resp.json()["embedding"]


def index_chunks(chunks: list, collection) -> int:
    """Embed and upsert chunks into ChromaDB. Returns the count successfully indexed."""
    indexed = 0
    for chunk in chunks:
        if not chunk["text"].strip():
            continue
        try:
            embedding = _embed(chunk["text"])
            meta = {k: chunk[k] for k in (
                "tradition", "verse_ref", "section_type", "video_id",
                "youtube_url", "start_time_sec", "end_time_sec",
                "speaker", "video_file",
            )}
            collection.upsert(
                ids=[chunk["chunk_id"]],
                embeddings=[embedding],
                documents=[chunk["text"]],
                metadatas=[meta],
            )
            indexed += 1
        except Exception as e:
            print(f"  Warning: could not index chunk {chunk['chunk_id']}: {e}")
    return indexed


# ─────────────────────────────────────────────
# PUBLIC API
# ─────────────────────────────────────────────

def _run_pipeline(
    audio_path: str,
    tradition: str,
    collection,
    video_id: str,
    title: str,
    youtube_url: str,
    progress_callback,
) -> dict:
    """Shared core: transcribe → detect → chunk → index."""
    def _cb(step, detail=""):
        if progress_callback:
            progress_callback(step, detail)

    _cb("transcribe", f"Transcribing with Whisper ({config.WHISPER_MODEL})…")
    lang = config.WHISPER_LANGUAGE_MAP.get(tradition)
    transcript = transcribe(audio_path, language=lang)
    n_segs = len(transcript["segments"])
    duration_sec = transcript["segments"][-1]["end"] if transcript["segments"] else 0
    _cb("transcribe", f"Done — {n_segs} segments, {duration_sec / 60:.1f} min")

    _cb("detect", "Detecting verse boundaries…")
    offset_map = _build_offset_map(transcript["segments"])
    if tradition == "Advaita":
        mentions = _english_verse_mentions(transcript["text"])
    else:
        mentions = _kannada_verse_mentions(transcript["text"])
    _cb("detect", f"Found {len(mentions)} verse mention(s)")

    _cb("chunk", "Building śloka-aware chunks…")
    chunks = _chunk_by_mentions(
        transcript["text"], mentions, offset_map,
        video_id=video_id, youtube_url=youtube_url, tradition=tradition,
    )
    _cb("chunk", f"Created {len(chunks)} chunk(s)")

    _cb("index", f"Indexing {len(chunks)} chunks into ChromaDB…")
    indexed = index_chunks(chunks, collection)
    _cb("index", f"Indexed {indexed} chunks ✓")

    return {
        "video_id": video_id,
        "title": title,
        "num_chunks": indexed,
        "duration_sec": duration_sec,
    }


def ingest_from_url(
    youtube_url: str,
    tradition: str,
    collection,
    progress_callback=None,
) -> dict:
    """
    Download a YouTube lecture, transcribe it, detect verses, and index into ChromaDB.

    Args:
        youtube_url:       YouTube watch URL
        tradition:         "Advaita" or "Dvaita"
        collection:        ChromaDB collection object
        progress_callback: optional callable(step: str, detail: str)

    Returns: {"video_id", "title", "num_chunks", "duration_sec"}
    """
    def _cb(step, detail=""):
        if progress_callback:
            progress_callback(step, detail)

    _cb("download", "Downloading audio from YouTube…")
    audio_path, video_id, title = download_youtube_audio(youtube_url)
    _cb("download", f"Downloaded: {title}")

    return _run_pipeline(audio_path, tradition, collection,
                         video_id, title, youtube_url, progress_callback)


def ingest_from_file(
    audio_path: str,
    tradition: str,
    collection,
    video_label: str = None,
    youtube_url: str = "",
    progress_callback=None,
) -> dict:
    """
    Transcribe a local audio file, detect verses, and index into ChromaDB.

    Args:
        audio_path:        path to audio file (mp3 / wav / m4a / etc.)
        tradition:         "Advaita" or "Dvaita"
        collection:        ChromaDB collection object
        video_label:       display name / identifier (defaults to filename stem)
        youtube_url:       optional originating YouTube URL (for deep-link timestamps)
        progress_callback: optional callable(step: str, detail: str)

    Returns: {"video_id", "title", "num_chunks", "duration_sec"}
    """
    stem = os.path.splitext(os.path.basename(audio_path))[0]
    video_id = re.sub(r"[^a-zA-Z0-9_-]", "_", video_label or stem)
    title = video_label or stem

    return _run_pipeline(audio_path, tradition, collection,
                         video_id, title, youtube_url, progress_callback)
