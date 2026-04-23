"""
backend/src/llm_client.py

KEY FIXES:
- _ollama_chat now passes num_predict=1500 so answers are not truncated at default ~128 tokens
- System prompt instructs LLM to write detailed, abstractive answers (not one-liners)
- _lmstudio_chat and _openai_chat also updated for consistency
"""
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

        if source_type == "video":
            vid = meta.get("video_id", "video")
            st = _sec_to_mmss(meta.get("start", 0.0))
            en = _sec_to_mmss(meta.get("end", 0.0))
            lines.append(f"[{vid} @ {st}-{en}]\n{text}")
        else:
            doc = meta.get("doc_name", "document")
            lines.append(f"[{doc}]\n{text}")

    return "\n\n".join(lines)


# ---------------------------------------------------------------------------
# SYSTEM PROMPT
# ---------------------------------------------------------------------------
# FIX: The old system prompt was too terse — it only said "use ONLY the context"
# with no instruction to be detailed. This caused the LLM to produce 1–2 sentence
# answers. The new prompt explicitly demands a thorough, abstractive response.
_SYSTEM_PROMPT_TEMPLATE = """\
You are a knowledgeable teaching assistant for Bhagavad Gita and philosophy lectures.

Your task is to give DETAILED, THOROUGH, SYNTHESIZED answers — not one-liners.
This is called ABSTRACTIVE answering: you read the retrieved context and explain
the ideas IN YOUR OWN WORDS, like a teacher explaining to a student.

STRICT RULES:
1. Use ONLY information present in the provided context. Do not add external knowledge.
2. If the answer is truly not in the context, say:
   "This specific detail is not covered in the uploaded content, but based on what is
   available: ..." and share what IS there.
3. Answer in {target_lang}.
4. Always include citations in the format [video_id @ mm:ss-mm:ss] or [document_name]
   exactly as they appear in the context headers.
5. Write AT LEAST 4–6 sentences. Cover: what the source says, why it matters,
   and the deeper meaning or implication.
6. Do NOT simply copy-paste text from the context. Synthesise and explain.

Structure (use markdown):
**Answer:** (2–4 sentences explaining the core idea in your own words)
**From the Discourse:** (2–3 sentences citing specific points from the retrieved segments,
   with timestamps/source in brackets)
**Deeper Meaning:** (1–2 sentences on the philosophical significance)
"""


def _get_system_prompt(target_lang: str) -> str:
    return _SYSTEM_PROMPT_TEMPLATE.format(target_lang=target_lang)


# ---------------------------------------------------------------------------
# LLM BACKENDS
# ---------------------------------------------------------------------------

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
        temperature=0.3,
        max_tokens=1500,        # FIX: was not set → defaulted to ~256
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
        "temperature": 0.3,
        "max_tokens": 1500,     # FIX: was not set → defaulted to ~256
    }
    resp = requests.post(url, json=payload, timeout=180)
    resp.raise_for_status()
    data = resp.json()
    return data["choices"][0]["message"]["content"]


def _ollama_chat(system: str, user: str) -> str:
    """
    FIX: Added num_predict=1500 to Ollama options.
    Without this, Ollama uses its default (often 128–256 tokens) which causes
    answers to be cut off after 1–2 sentences, making the system look extractive.
    """
    payload = {
        "model": "llama3",
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "stream": False,
        "options": {
            "temperature": 0.3,
            "num_predict": 1500,    # FIX: KEY CHANGE — was missing entirely
            "top_p": 0.9,
            "repeat_penalty": 1.1,
        }
    }

    r = requests.post(
        "http://localhost:11434/api/chat",
        json=payload,
        timeout=300,    # longer timeout for detailed answers
    )
    r.raise_for_status()
    return r.json()["message"]["content"]


# ---------------------------------------------------------------------------
# PUBLIC ENTRY POINT
# ---------------------------------------------------------------------------

def generate_answer(question: str, hits: List[Dict[str, Any]]) -> Dict[str, Any]:
    s = get_settings()
    qlang = detect_language(question)
    target_lang = lang_label(qlang)

    context = _build_context(hits)

    system = _get_system_prompt(target_lang)

    # The user message explicitly requests a detailed answer — this counters
    # the LLM's default tendency to give brief fill-in-the-blank completions.
    user = (
        f"Question: {question}\n\n"
        f"Context from the uploaded discourse:\n{context}\n\n"
        "Please write a DETAILED, SYNTHESIZED answer (minimum 4–6 sentences). "
        "Explain the teaching thoroughly in your own words and cite the sources."
    )

    try:
        if s.llm_provider == "openai":
            text = _openai_chat(system, user)
        elif s.llm_provider == "lmstudio":
            text = _lmstudio_chat(system, user)
        elif s.llm_provider == "ollama":
            text = _ollama_chat(system, user)
        else:
            return {
                "ok": False,
                "message": "LLM_PROVIDER must be 'openai', 'lmstudio', or 'ollama'."
            }

        return {"ok": True, "answer": text, "lang": qlang}

    except Exception as e:
        return {"ok": False, "message": str(e)}
