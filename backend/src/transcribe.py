from __future__ import annotations
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from pathlib import Path
from typing import Dict, Any
from tqdm import tqdm

from backend.src.config import get_settings
from backend.src.utils import safe_stem, write_json

def _transcribe_local(mp3_path: Path, model_size: str = "small") -> Dict[str, Any]:

    from faster_whisper import WhisperModel

    model = WhisperModel(model_size, device="cpu", compute_type="int8")
    segments, info = model.transcribe(
        str(mp3_path),
        language=None,
        vad_filter=True,
    )

    seg_list = []
    for seg in segments:
        seg_list.append({
            "start": float(seg.start),
            "end": float(seg.end),
            "text": (seg.text or "").strip(),
        })

    return {
        "source_audio": mp3_path.name,
        "language": getattr(info, "language", None),
        "duration": getattr(info, "duration", None),
        "segments": seg_list,
        "provider": "local_faster_whisper",
    }

def _transcribe_openai(mp3_path: Path) -> Dict[str, Any]:
    s = get_settings()
    if not s.openai_api_key:
        raise RuntimeError("OPENAI_API_KEY is missing in .env")

    from openai import OpenAI
    client = OpenAI(api_key=s.openai_api_key)

    # NOTE: OpenAI timestamp/segment formats can vary by API/version.
    # We try verbose_json; if segments aren't present, we still store text.
    with mp3_path.open("rb") as f:
        resp = client.audio.transcriptions.create(
          model=s.openai_asr_model,
          file=f,
          response_format="verbose_json",
          language="kn",  # FORCE Kannada
         prompt="Transcribe this audio in Kannada only."
)


    data = resp.model_dump() if hasattr(resp, "model_dump") else dict(resp)

    segments = data.get("segments") or []
    seg_list = []
    for seg in segments:
        seg_list.append({
            "start": float(seg.get("start", 0.0)),
            "end": float(seg.get("end", 0.0)),
            "text": (seg.get("text") or "").strip(),
        })

    return {
        "source_audio": mp3_path.name,
        "language": data.get("language"),
        "duration": data.get("duration"),
        "text": data.get("text"),
        "segments": seg_list,
        "provider": "openai",
    }

def transcribe_all(model_size: str = "small") -> Dict[str, Any]:
    s = get_settings()
    audio_files = sorted(s.audio_dir.glob("*.mp3"))
    if not audio_files:
        return {"ok": False, "message": f"No .mp3 files found in {s.audio_dir}. Run process_videos.py first."}

    done = 0
    for mp3 in tqdm(audio_files, desc="Transcribing"):
        out = s.transcripts_dir / (safe_stem(mp3.stem) + ".json")
        if out.exists():
            continue

        if s.transcribe_provider == "openai":
            payload = _transcribe_openai(mp3)
        else:
            payload = _transcribe_local(mp3, model_size=model_size)

        write_json(out, payload)
        done += 1

    return {"ok": True, "num_audio": len(audio_files), "new_transcripts": done, "transcripts_dir": str(s.transcripts_dir)}

if __name__ == "__main__":
    print(transcribe_all())
