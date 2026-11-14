#!/usr/bin/env python3
"""
transcribe_new.py — Transcribe Chinmaya (English) videos using Sarvam Batch API.
Saves .txt and .json in data/transcripts_new/. Resumes automatically if interrupted.

Safeguards against cut-off transcripts:
  * --no-playlist so yt-dlp downloads ONLY the single video (URLs carry &list=...)
  * long audio is split into <=40 min chunks, all sent in ONE job, then stitched in order
  * long poll timeout (2 hours) instead of 30 min
  * ALL output json files are read (not just the first) and sorted by chunk number
  * raw Sarvam outputs are kept in data/transcripts_new/raw/<name>/ (never deleted)
  * sanity check on words-per-minute; a suspiciously short transcript is NOT saved
    as "done", so the next run retries it
"""
import os, sys, json, subprocess
from pathlib import Path

SARVAM_API_KEY = os.environ.get("SARVAM_API_KEY")
if not SARVAM_API_KEY:
    print('ERROR: Run: export SARVAM_API_KEY="sk_..."')
    sys.exit(1)

try:
    from sarvamai import SarvamAI
except ImportError:
    print("Run: pip install sarvamai --break-system-packages")
    sys.exit(1)

client = SarvamAI(api_subscription_key=SARVAM_API_KEY)

OUT_DIR = Path("data/transcripts_new")
OUT_DIR.mkdir(parents=True, exist_ok=True)
RAW_DIR = OUT_DIR / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

AUDIO_DIR = Path("data/raw_audio")
AUDIO_DIR.mkdir(parents=True, exist_ok=True)

CHUNK_SECONDS = 40 * 60      # split anything longer than this
POLL_TIMEOUT = 7200          # 2 hours
MIN_WORDS_PER_MIN = 40       # below this, transcript is assumed truncated

VIDEOS = {
    "en_chinmaya_day06": ("https://www.youtube.com/watch?v=omDmaalmgwA", "en-IN"),
    "en_chinmaya_day07": ("https://www.youtube.com/watch?v=EcNP1Od3N8M", "en-IN"),
    "en_chinmaya_day08": ("https://www.youtube.com/watch?v=TkKpn2O-s80", "en-IN"),
    "en_chinmaya_day09": ("https://www.youtube.com/watch?v=OrvzNCFLIM4", "en-IN"),
}

# ---------- previously done (kept for reference) ----------
# "kn_brahmanyachar_day01": ("https://youtu.be/SwBF8M5tB2k", "kn-IN"),
# "kn_brahmanyachar_day02": ("https://youtu.be/XYkANTXEys4", "kn-IN"),
# "kn_brahmanyachar_day03": ("https://youtu.be/ErAVN60kR9E", "kn-IN"),
# "kn_brahmanyachar_day04": ("https://youtu.be/w8pNS7QiXOg", "kn-IN"),
# "kn_brahmanyachar_day05": ("https://youtu.be/wQYaNItpvNo", "kn-IN"),
# "kn_brahmanyachar_day06": ("https://youtu.be/tPIy6zOH350", "kn-IN"),
# "kn_brahmanyachar_day07": ("https://youtu.be/sMXw-JHv_CY", "kn-IN"),
# "kn_brahmanyachar_day08": ("https://youtu.be/Tdv9thMsnpU", "kn-IN"),
# "en_chinmaya_day03":      ("https://youtu.be/tFNZLFpjDs4", "en-IN"),
# "en_chinmaya_day04":      ("https://youtu.be/v2VaTR4D2FI", "en-IN"),
# "kn_brahmanyachar_day02b":("https://youtu.be/gtF6OYxs3ps", "kn-IN"),
# "en_iskcon":              ("https://youtu.be/UoZEVYw459M", "en-IN"),


def get_duration(path):
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
        capture_output=True, text=True, check=True)
    return float(out.stdout.strip())


def download_audio(name, url):
    out_path = AUDIO_DIR / f"{name}.mp3"
    if out_path.exists() and out_path.stat().st_size > 0:
        print(f"  [skip download] {name}.mp3 already exists")
        return out_path
    print(f"  Downloading {url} ...")
    subprocess.run([
        "yt-dlp", "--no-playlist", "-x", "--audio-format", "mp3",
        "-o", str(AUDIO_DIR / f"{name}.%(ext)s"), url
    ], check=True)
    if not out_path.exists():
        raise Exception(f"Download finished but {out_path} not found")
    return out_path


def split_audio(name, audio_path):
    """Return list of audio chunk paths (just [audio_path] if short enough)."""
    duration = get_duration(audio_path)
    print(f"  Duration: {duration/60:.1f} min")
    if duration <= CHUNK_SECONDS + 60:
        return [audio_path], duration

    chunk_dir = AUDIO_DIR / f"{name}_chunks"
    chunk_dir.mkdir(exist_ok=True)
    existing = sorted(chunk_dir.glob(f"{name}_part*.mp3"))
    if not existing:
        print(f"  Splitting into {CHUNK_SECONDS//60}-min chunks ...")
        subprocess.run([
            "ffmpeg", "-y", "-i", str(audio_path),
            "-f", "segment", "-segment_time", str(CHUNK_SECONDS),
            "-c", "copy", "-reset_timestamps", "1",
            str(chunk_dir / f"{name}_part%03d.mp3")
        ], check=True, capture_output=True)
        existing = sorted(chunk_dir.glob(f"{name}_part*.mp3"))
    print(f"  {len(existing)} chunks")
    return existing, duration


def extract_text(raw):
    if isinstance(raw, list):
        return " ".join(extract_text(c) for c in raw).strip()
    if isinstance(raw, dict):
        if "transcript" in raw and isinstance(raw["transcript"], str):
            return raw["transcript"].strip()
        # fallback: diarized entries, if present
        entries = (raw.get("diarized_transcript") or {}).get("entries")
        if entries:
            return " ".join(e.get("transcript", "") for e in entries).strip()
        return ""
    return str(raw).strip()


def transcribe_batch(name, audio_path, lang):
    txt_path = OUT_DIR / f"{name}.txt"
    json_path = OUT_DIR / f"{name}.json"
    if txt_path.exists() and json_path.exists():
        print(f"  [skip transcribe] {name} already done")
        return

    chunks, duration = split_audio(name, audio_path)

    raw_dir = RAW_DIR / name
    raw_dir.mkdir(parents=True, exist_ok=True)

    # Reuse raw outputs from an earlier run if present
    out_files = sorted(raw_dir.glob("*.json"))
    if len(out_files) < len(chunks):
        print(f"  Submitting batch job for {name} ({lang}), {len(chunks)} file(s) ...")
        job = client.speech_to_text_job.create_job(
            model="saaras:v3",
            mode="transcribe",
            language_code=lang,
        )
        job.upload_files(file_paths=[str(c) for c in chunks])
        job.start()
        print("  Waiting for job to complete ...")
        job.wait_until_complete(poll_interval=10, timeout=POLL_TIMEOUT)
        job.download_outputs(output_dir=str(raw_dir))
        out_files = sorted(raw_dir.glob("*.json"))

    if not out_files:
        raise Exception(f"No output json found in {raw_dir}")
    if len(out_files) < len(chunks):
        raise Exception(f"Expected {len(chunks)} outputs, got {len(out_files)} — not saving")

    # sorted() on zero-padded names (…_part000, …_part001) keeps chunks in order
    parts = []
    for f in out_files:
        t = extract_text(json.loads(f.read_text(encoding="utf-8")))
        print(f"    {f.name}: {len(t.split())} words")
        parts.append(t)
    transcript = " ".join(p for p in parts if p)

    # Sanity check: refuse to mark as done if it looks truncated
    words = len(transcript.split())
    wpm = words / (duration / 60)
    print(f"  Total: {words} words over {duration/60:.1f} min ({wpm:.0f} wpm)")
    if wpm < MIN_WORDS_PER_MIN:
        raise Exception(f"Transcript looks cut off ({wpm:.0f} wpm) — not saved, rerun to retry")

    txt_path.write_text(transcript, encoding="utf-8")
    json_path.write_text(
        json.dumps({"name": name, "lang": lang, "duration_min": round(duration / 60, 1),
                    "word_count": words, "transcript": transcript},
                   ensure_ascii=False, indent=2),
        encoding="utf-8"
    )
    print(f"  Saved {txt_path} and {json_path}")


for name, (url, lang) in VIDEOS.items():
    print(f"\n=== {name} ===")
    try:
        audio = download_audio(name, url)
        transcribe_batch(name, audio, lang)
    except Exception as e:
        print(f"  ERROR: {e} — skipping, will resume next run")

print("\nAll done!")