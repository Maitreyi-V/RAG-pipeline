from __future__ import annotations
import subprocess
from pathlib import Path
from typing import Dict, Any
from tqdm import tqdm

from backend.src.config import get_settings
from backend.src.utils import safe_stem

def mp4_to_mp3(mp4_path: Path, mp3_path: Path) -> None:
    mp3_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-y",
        "-i", str(mp4_path),
        "-vn",
        "-acodec", "libmp3lame",
        "-ar", "44100",
        "-ac", "2",
        "-b:a", "128k",
        str(mp3_path),
    ]
    subprocess.run(cmd, check=True, capture_output=True)

def process_videos() -> Dict[str, Any]:
    s = get_settings()
    mp4_files = sorted(s.videos_dir.glob("*.mp4"))
    if not mp4_files:
        return {"ok": False, "message": f"No .mp4 files found in {s.videos_dir}"}

    converted = 0
    for mp4 in tqdm(mp4_files, desc="MP4 -> MP3"):
        out_name = safe_stem(mp4.stem) + ".mp3"
        mp3 = s.audio_dir / out_name
        if mp3.exists():
            continue
        mp4_to_mp3(mp4, mp3)
        converted += 1

    return {
        "ok": True,
        "num_mp4": len(mp4_files),
        "converted_now": converted,
        "audio_dir": str(s.audio_dir),
    }

if __name__ == "__main__":
    print(process_videos())
