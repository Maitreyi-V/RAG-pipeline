"""
judge_gpt.py
============
A trustworthy faithfulness / groundedness judge for the Gita RAG eval.

WHY THIS EXISTS
---------------
The old judge in 05_evaluate.py asked llama3 to output a single 0-3 digit with
num_predict=10. A small local model on a coarse scale hedges to "2" almost every
time, so the result was a flat wall of 2s (groundedness: {"2": 72}) — i.e. NO
signal at all. This module replaces that with:

  1. A GPT-4o-mini judge (different model family from the llama3 GENERATOR, so the
     model is not grading its own homework).
  2. CLAIM-LEVEL grounding: we decompose each answer into atomic factual claims,
     then check each claim against the retrieved context (supported / unsupported
     / partially-supported). Groundedness = fraction of claims supported. This is
     a real continuous score AND it lists the actual hallucinated claims.
  3. A 0-5 faithfulness score (answer vs. expected answer) with a one-line reason,
     which forces the judge to discriminate instead of hedging.

It reuses the same OPENAI_API_KEY pattern already used in layer4_llm_detector.py.

USAGE (imported by 05_evaluate.py; can also be run as a smoke test):
    from judge_gpt import judge_groundedness_claims, judge_faithfulness_gpt
"""
from __future__ import annotations
import json
import os
import time
from typing import List, Dict, Any, Optional

from dotenv import load_dotenv

load_dotenv()

JUDGE_MODEL = os.environ.get("JUDGE_MODEL", "gpt-4o-mini")


def _client():
    """Lazy OpenAI client — same pattern as layer4_llm_detector.call_openai."""
    from openai import OpenAI

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set — check the repo-root .env file")
    return OpenAI(api_key=api_key)


def _chat_json(prompt: str, model: str = None, max_retries: int = 3) -> Optional[dict]:
    """Call the judge and parse a JSON object from the response. Returns None on failure."""
    model = model or JUDGE_MODEL
    client = _client()
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a careful, literal evaluation judge. "
                                                  "You always respond with a single valid JSON object and nothing else."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.0,
                response_format={"type": "json_object"},
            )
            text = resp.choices[0].message.content.strip()
            return json.loads(text)
        except Exception as e:
            print(f"    [judge retry {attempt+1}/{max_retries}] {e}")
            time.sleep(1.5 * (attempt + 1))
    return None


# ─────────────────────────────────────────────────────────────────────────────
# 1. CLAIM-LEVEL GROUNDEDNESS  (the core hallucination test)
# ─────────────────────────────────────────────────────────────────────────────

_CLAIM_PROMPT = """You are auditing whether an AI answer is GROUNDED in the retrieved context.

Do this in two stages and return JSON.

STAGE 1 — Decompose the ANSWER into atomic factual claims. A claim is a single,
checkable statement (e.g. "Arjuna's eyes were filled with tears"). Ignore filler,
hedging, and verse-reference labels. Aim for 2-8 claims.

STAGE 2 — For EACH claim, decide its support based ONLY on the RETRIEVED CONTEXT
below (not your own knowledge of the Gita):
  "supported"  = the context clearly states or directly entails the claim
  "partial"    = the context hints at it but does not fully establish it
  "unsupported"= the claim is NOT in the context (this is a hallucination, even if
                 it happens to be true in general)

Return EXACTLY this JSON shape:
{{
  "claims": [
    {{"claim": "<text>", "verdict": "supported|partial|unsupported", "evidence": "<short quote or 'none'>"}}
  ]
}}

QUESTION:
{question}

RETRIEVED CONTEXT:
{context}

AI ANSWER TO AUDIT:
{answer}
"""


def judge_groundedness_claims(answer: str, retrieved_chunks: List[Dict[str, Any]],
                              question: str) -> Optional[Dict[str, Any]]:
    """
    Returns:
      {
        "groundedness": float in [0,1],          # fraction supported (partial = 0.5)
        "n_claims": int,
        "n_supported": int, "n_partial": int, "n_unsupported": int,
        "unsupported_claims": [list of hallucinated claim strings],
        "claims": [full per-claim breakdown]
      }
    or None if the judge call failed or there was no context.
    """
    context_parts = []
    for i, c in enumerate(retrieved_chunks):
        meta = c.get("metadata", {}) or {}
        verse = c.get("verse_ref") or meta.get("verse_ref", "?")
        text = c.get("document", c.get("text", "")).strip()
        if text:
            context_parts.append(f"[Chunk {i+1} — {verse}]\n{text}")
    if not context_parts:
        return None
    context = "\n\n".join(context_parts)

    out = _chat_json(_CLAIM_PROMPT.format(question=question, context=context, answer=answer))
    if not out or "claims" not in out:
        return None

    claims = out["claims"]
    if not claims:
        return None

    weight = {"supported": 1.0, "partial": 0.5, "unsupported": 0.0}
    n_sup = sum(1 for c in claims if c.get("verdict") == "supported")
    n_par = sum(1 for c in claims if c.get("verdict") == "partial")
    n_uns = sum(1 for c in claims if c.get("verdict") == "unsupported")
    score = sum(weight.get(c.get("verdict"), 0.0) for c in claims) / len(claims)

    return {
        "groundedness": round(score, 4),
        "n_claims": len(claims),
        "n_supported": n_sup,
        "n_partial": n_par,
        "n_unsupported": n_uns,
        "unsupported_claims": [c.get("claim", "") for c in claims if c.get("verdict") == "unsupported"],
        "claims": claims,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 2. FAITHFULNESS  (answer vs. expected answer — did we get the right info?)
# ─────────────────────────────────────────────────────────────────────────────

_FAITH_PROMPT = """You are grading a Q&A system for the Bhagavad Gita, Chapter 2.
Compare the GENERATED answer against the EXPECTED (gold) answer.

Score 0-5 on whether the generated answer conveys the same key information as the gold:
  5 = all key points correct, no errors
  4 = all key points present, trivial wording differences
  3 = most key points present, one minor omission or inaccuracy
  2 = partially correct, a key point missing or wrong
  1 = mostly wrong, only a fragment correct
  0 = wrong, irrelevant, or empty

Return EXACTLY this JSON:
{{"score": <int 0-5>, "reason": "<one short sentence>"}}

QUESTION:
{question}

EXPECTED (GOLD) ANSWER:
{expected}

GENERATED ANSWER:
{answer}
"""


def judge_faithfulness_gpt(answer: str, expected_answer: str,
                           question: str) -> Optional[Dict[str, Any]]:
    """Returns {"faithfulness": int 0-5, "reason": str} or None."""
    if not expected_answer or expected_answer.strip().lower() in ("", "nan"):
        return None
    out = _chat_json(_FAITH_PROMPT.format(
        question=question, expected=expected_answer, answer=answer))
    if not out or "score" not in out:
        return None
    try:
        score = int(out["score"])
    except (TypeError, ValueError):
        return None
    return {"faithfulness": max(0, min(5, score)), "reason": out.get("reason", "")}


# ─────────────────────────────────────────────────────────────────────────────
# Smoke test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    fake_chunks = [{
        "verse_ref": "BG 2.1",
        "document": "Sanjaya said: seeing Arjuna overwhelmed with compassion, his eyes "
                    "full of tears and despondent, Madhusudana spoke these words.",
        "metadata": {"verse_ref": "BG 2.1"},
    }]
    ans = ("Sanjaya describes Arjuna's eyes as filled with tears and overwhelmed by "
           "compassion. Arjuna later becomes a great archer who wins the war.")
    print("Groundedness test (2nd claim should be unsupported):")
    print(json.dumps(judge_groundedness_claims(ans, fake_chunks,
          "How does Sanjaya describe Arjuna's eyes?"), indent=2, ensure_ascii=False))
    print("\nFaithfulness test:")
    print(json.dumps(judge_faithfulness_gpt(ans,
          "Sanjaya describes Arjuna's eyes as tear-filled and full of compassion.",
          "How does Sanjaya describe Arjuna's eyes?"), indent=2, ensure_ascii=False))
