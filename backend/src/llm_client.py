from __future__ import annotations
from typing import Dict, Any, List
import requests

from backend.src.config import get_settings
from backend.src.utils import detect_language, lang_label

def _sec_to_mmss(sec: float) -> str:
    sec = max(0.0, float(sec))
    m = int(sec // 60)
    s = int(sec % 60)
    return f"{m:02d}:{s:02d}"

def _build_context(hits):
    lines = []
    for h in hits:
        meta = h.get("meta", {}) or {}
        source_type = meta.get("source_type", "doc")

        text = h.get("text", "")

        # ✅ VIDEO → include timestamps
        if source_type == "video":
            vid = meta.get("video_id", "video")
            st = _sec_to_mmss(meta.get("start", 0.0))
            en = _sec_to_mmss(meta.get("end", 0.0))
            lines.append(f"[{vid} @ {st}-{en}] {text}")

        # ✅ DOC / BOOK → NO timestamps
        else:
            doc = meta.get("doc_name", "document")
            lines.append(f"[{doc}] {text}")

    return "\n\n".join(lines)

def _openai_chat(system: str, user: str) -> str:
    s = get_settings()
    if not s.openai_api_key:
        raise RuntimeError("OPENAI_API_KEY is missing in .env")

    from openai import OpenAI
    client = OpenAI(api_key=s.openai_api_key)

    resp = client.chat.completions.create(
        model=s.openai_chat_model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=0.2,
    )
    return resp.choices[0].message.content or ""

def _lmstudio_chat(system: str, user: str) -> str:
    s = get_settings()
    url = s.lmstudio_base_url.rstrip("/") + "/chat/completions"
    payload = {
        "model": s.lmstudio_model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "temperature": 0.2,
    }
    resp = requests.post(url, json=payload, timeout=120)
    resp.raise_for_status()
    data = resp.json()
    return data["choices"][0]["message"]["content"]

def _ollama_chat(system: str, user: str) -> str:
    import requests

    payload = {
        "model": "llama3",
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "stream": False
    }

    r = requests.post(
        "http://localhost:11434/api/chat",
        json=payload,
        timeout=300,
    )
    r.raise_for_status()

    return r.json()["message"]["content"]

def generate_answer(question: str, hits: List[Dict[str, Any]]) -> Dict[str, Any]:
    s = get_settings()
    qlang = detect_language(question)
    target_lang = lang_label(qlang)

    context = _build_context(hits)

    system = (
        "You are a helpful teaching assistant for Bhagavad Gita / philosophy lectures. "
        "Use ONLY the provided context to answer. "
        "If the answer is not in the context, say clearly: 'Not found in the uploaded content.' "
        f"Answer in {target_lang}. "
        "Always include citations by referencing the brackets like [video_id @ mm:ss-mm:ss] from the context."
    )

    user = (
        f"Question: {question}\n\n"
        f"Context:\n{context}\n\n"
        "Answer:"
    )

    try:
        if s.llm_provider == "openai":
            text = _openai_chat(system, user)
        elif s.llm_provider == "lmstudio":
            text = _lmstudio_chat(system, user)
        elif s.llm_provider == "ollama":
            text = _ollama_chat(system, user)
        else:
            return {"ok": False, "message": "LLM_PROVIDER must be 'openai', 'lmstudio', or 'ollama'."}

        return {"ok": True, "answer": text, "lang": qlang}
    except Exception as e:
        return {"ok": False, "message": str(e)}
