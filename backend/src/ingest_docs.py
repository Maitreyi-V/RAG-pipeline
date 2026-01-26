from __future__ import annotations
from pathlib import Path
from typing import Dict, Any, List
import re
import uuid

from backend.src.config import get_settings
from backend.src.utils import write_jsonl

def _clean_text(t: str) -> str:
    t = t.replace("\r\n", "\n").replace("\r", "\n")
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t.strip()

def _chunk_by_chars(text: str, chunk_size: int = 1200, overlap: int = 200) -> List[str]:
    # simple and robust chunking for Kannada text
    chunks = []
    i = 0
    n = len(text)
    while i < n:
        j = min(n, i + chunk_size)
        chunk = text[i:j].strip()
        if chunk:
            chunks.append(chunk)
        i = max(i + chunk_size - overlap, i + 1)
    return chunks

def ingest_txt_docs() -> Dict[str, Any]:
    s = get_settings()
    docs_dir = s.data_dir / "docs"
    docs_dir.mkdir(parents=True, exist_ok=True)

    txt_files = sorted(docs_dir.glob("*.txt"))
    if not txt_files:
        return {"ok": False, "message": f"No .txt files in {docs_dir}. Put your Kannada txt there."}

    out_path = s.chunks_dir / "docs_chunks.jsonl"
    rows: List[Dict[str, Any]] = []

    for fp in txt_files:
        raw = fp.read_text(encoding="utf-8", errors="ignore")
        text = _clean_text(raw)
        parts = _chunk_by_chars(text)

        for idx, chunk in enumerate(parts):
            rows.append({
                "id": f"doc::{fp.stem}::{idx}::{uuid.uuid4().hex[:8]}",
                "source_type": "doc",
                "doc_name": fp.name,
                "start": 0.0,
                "end": 0.0,
                "lang": "kn",
                "text": chunk,
            })

    write_jsonl(out_path, rows)
    return {"ok": True, "num_docs": len(txt_files), "num_chunks": len(rows), "chunks_file": str(out_path)}

if __name__ == "__main__":
    print(ingest_txt_docs())
