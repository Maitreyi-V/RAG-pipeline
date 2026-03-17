"""
Step 2: Translation Pipeline
=============================
Translates Kannada text in each chunk to English using either:
  - Google Translate (fast, free via deep-translator)
  - Ollama/Llama3 (slower, better for domain-specific terms)

The English translation is stored alongside the original Kannada text.
The English summary from annotations is used as the primary embedding text,
with translated Kannada text as supplementary context.

Usage:
    python 02_translate.py
    python 02_translate.py --method google
    python 02_translate.py --method ollama
"""
import json
import os
import sys
import time
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config


def translate_google(text, src="kn", dest="en"):
    """Translate using Google Translate via deep-translator."""
    from deep_translator import GoogleTranslator
    try:
        # deep-translator has a 5000 char limit, chunk if needed
        if len(text) > 4500:
            parts = [text[i:i+4500] for i in range(0, len(text), 4500)]
            translated_parts = []
            for part in parts:
                result = GoogleTranslator(source=src, target=dest).translate(part)
                translated_parts.append(result)
                time.sleep(0.5)  # rate limit
            return " ".join(translated_parts)
        return GoogleTranslator(source=src, target=dest).translate(text)
    except Exception as e:
        print(f"    Google translation error: {e}")
        return None


def translate_ollama(text, src_lang="Kannada", dest_lang="English"):
    """Translate using Llama3 via Ollama (better for philosophical terms)."""
    import requests
    prompt = f"""Translate the following {src_lang} text about the Bhagavad Gita into {dest_lang}. 
Preserve Sanskrit terms (like dharma, atma, shloka) as-is. 
Keep the translation faithful to the philosophical meaning in the Dvaita tradition.

Text to translate:
{text}

English translation:"""
    
    try:
        resp = requests.post(
            f"{config.OLLAMA_BASE_URL}/api/generate",
            json={"model": config.LLM_MODEL, "prompt": prompt, "stream": False},
            timeout=120
        )
        resp.raise_for_status()
        return resp.json().get("response", "").strip()
    except Exception as e:
        print(f"    Ollama translation error: {e}")
        return None


def build_embedding_text(chunk):
    """
    Build the English text that will be used for embedding.
    Priority: annotation English summary > translated Kannada > short summary
    Concatenates multiple sources for richer semantic representation.
    """
    parts = []

    # Verse reference as prefix
    if chunk.get("verse_ref"):
        parts.append(f"Bhagavad Gita {chunk['verse_ref']}")

    # Speaker context
    if chunk.get("speaker") and chunk["speaker"] not in ["-", ""]:
        parts.append(f"Speaker: {chunk['speaker']}")

    # Primary: English explanation summary from annotation
    if chunk.get("explanation_summary_en"):
        parts.append(chunk["explanation_summary_en"])

    # Secondary: Translated Kannada text (adds detail)
    if chunk.get("kannada_translated_en"):
        # Only add if substantially different from summary
        parts.append(f"Detailed discourse: {chunk['kannada_translated_en']}")

    # Tertiary: Short summary
    if chunk.get("short_summary_en") and not chunk.get("explanation_summary_en"):
        parts.append(chunk["short_summary_en"])

    # Sanskrit shloka (helps with exact-verse queries)
    if chunk.get("sanskrit_shloka"):
        parts.append(f"Sanskrit: {chunk['sanskrit_shloka']}")

    return "\n".join(parts)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", default=config.TRANSLATION_METHOD,
                       choices=["google", "ollama"])
    args = parser.parse_args()

    if not os.path.exists(config.CHUNKS_JSON):
        print("ERROR: chunks.json not found. Run 01_chunk.py first.")
        sys.exit(1)

    with open(config.CHUNKS_JSON, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    print(f"Loaded {len(chunks)} chunks")
    print(f"Translation method: {args.method}")

    translate_fn = translate_google if args.method == "google" else translate_ollama
    translated_count = 0
    skipped_count = 0

    for i, chunk in enumerate(chunks):
        label = chunk.get("verse_ref") or chunk.get("section_type", "unknown")
        print(f"\n[{i+1}/{len(chunks)}] {chunk['chunk_id']}: {label}")

        # Translate Kannada text if present
        if chunk.get("kannada_text"):
            print(f"  Translating Kannada text ({len(chunk['kannada_text'])} chars)...")
            translation = translate_fn(chunk["kannada_text"])
            if translation:
                chunk["kannada_translated_en"] = translation
                translated_count += 1
                print(f"  ✓ Translated ({len(translation)} chars)")
            else:
                chunk["kannada_translated_en"] = None
                print(f"  ✗ Translation failed")
        else:
            chunk["kannada_translated_en"] = None
            skipped_count += 1
            print(f"  ○ No Kannada text to translate")

        # Build the combined English embedding text
        chunk["embedding_text"] = build_embedding_text(chunk)
        print(f"  Embedding text: {len(chunk['embedding_text'])} chars")

        # Small delay for API rate limiting
        if args.method == "google":
            time.sleep(0.3)

    # Save updated chunks
    with open(config.CHUNKS_JSON, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)

    print(f"\n{'='*60}")
    print(f"Translation complete:")
    print(f"  - Translated: {translated_count}")
    print(f"  - Skipped (no Kannada): {skipped_count}")
    print(f"  - Failed: {len(chunks) - translated_count - skipped_count}")
    print(f"Saved to: {config.CHUNKS_JSON}")


if __name__ == "__main__":
    main()
