from __future__ import annotations
from pathlib import Path
from typing import Dict, Any, List
from tqdm import tqdm

from backend.src.config import get_settings
from backend.src.utils import read_json, write_jsonl, safe_stem

TARGET_CHUNK_SECONDS = 90.0   # tune
MIN_CHUNK_SECONDS = 30.0

def make_chunks_for_transcript(transcript_path: Path) -> List[Dict[str, Any]]:
    data = read_json(transcript_path)
    segments = data.get("segments", [])
    video_id = transcript_path.stem

    chunks: List[Dict[str, Any]] = []
    buf_text = []
    buf_start = None
    buf_end = None

    def flush():
        nonlocal buf_text, buf_start, buf_end
        if not buf_text or buf_start is None or buf_end is None:
            return
        duration = buf_end - buf_start
        if duration < MIN_CHUNK_SECONDS and chunks:
            # merge tiny tail into previous chunk
            chunks[-1]["text"] += " " + " ".join(buf_text).strip()
            chunks[-1]["end"] = buf_end
        else:
            chunks.append({
                "id": f"{video_id}_{len(chunks):05d}",
                "source_type": "video",
                "video_id": video_id,
                "start": float(buf_start),
                "end": float(buf_end),
                "text": " ".join(buf_text).strip(),
            })
        buf_text = []
        buf_start = None
        buf_end = None

    for seg in segments:
        txt = (seg.get("text") or "").strip()
        if not txt:
            continue
        st = float(seg.get("start", 0.0))
        en = float(seg.get("end", st))

        if buf_start is None:
            buf_start = st
            buf_end = en
            buf_text = [txt]
            continue

        # if adding this segment exceeds target duration -> flush
        if (en - buf_start) >= TARGET_CHUNK_SECONDS:
            flush()
            buf_start = st
            buf_end = en
            buf_text = [txt]
        else:
            buf_end = en
            buf_text.append(txt)

    flush()
    return chunks

def make_all_chunks() -> Dict[str, Any]:
    s = get_settings()
    transcripts = sorted(s.transcripts_dir.glob("*.json"))
    if not transcripts:
        return {"ok": False, "message": f"No transcripts found in {s.transcripts_dir}. Run transcribe.py first."}

    new_files = 0
    total_chunks = 0
    for tpath in tqdm(transcripts, desc="Chunking"):
        out = s.chunks_dir / (safe_stem(tpath.stem) + ".jsonl")
        if out.exists():
            continue
        chunks = make_chunks_for_transcript(tpath)
        write_jsonl(out, chunks)
        new_files += 1
        total_chunks += len(chunks)

    return {"ok": True, "num_transcripts": len(transcripts), "new_chunk_files": new_files, "total_new_chunks": total_chunks}

if __name__ == "__main__":
    print(make_all_chunks())
