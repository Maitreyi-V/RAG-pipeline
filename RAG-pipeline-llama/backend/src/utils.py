from __future__ import annotations
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List

def safe_stem(name: str) -> str:
    # filesystem-safe id
    name = re.sub(r"[^\w\-\.]+", "_", name, flags=re.UNICODE).strip("_")
    return name[:180] if len(name) > 180 else name

def ms_to_mmss(ms: float) -> str:
    # ms can be float seconds too; we treat input as seconds
    sec = int(round(ms))
    m = sec // 60
    s = sec % 60
    return f"{m:02d}:{s:02d}"

def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")

def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))

def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows

def detect_language(text: str) -> str:
    """
    Very simple script-based detection:
    - Kannada: U+0C80–U+0CFF
    - Devanagari (Hindi/Sanskrit): U+0900–U+097F
    else English
    """
    for ch in text:
        o = ord(ch)
        if 0x0C80 <= o <= 0x0CFF:
            return "kn"
        if 0x0900 <= o <= 0x097F:
            return "hi"
    return "en"

def lang_label(code: str) -> str:
    return {"kn": "Kannada", "hi": "Hindi", "en": "English"}.get(code, "English")
