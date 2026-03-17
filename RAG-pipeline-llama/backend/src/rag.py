from __future__ import annotations
from typing import Any, Dict, List
from pathlib import Path

import chromadb
from chromadb.config import Settings as ChromaSettings

from backend.src.config import get_settings
from backend.src.utils import read_jsonl

def _get_client():
    s = get_settings()
    return chromadb.PersistentClient(
        path=str(s.chroma_dir),
        settings=ChromaSettings(anonymized_telemetry=False),
    )

def _embed_texts_openai(texts: List[str]) -> List[List[float]]:
    s = get_settings()
    if not s.openai_api_key:
        raise RuntimeError("OPENAI_API_KEY is missing in .env")

    from openai import OpenAI
    client = OpenAI(api_key=s.openai_api_key)

    # Batch embeddings (keep batches modest)
    out: List[List[float]] = []
    batch_size = 96
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        resp = client.embeddings.create(
            model=s.openai_embed_model,
            input=batch,
        )
        # response order matches input order
        out.extend([d.embedding for d in resp.data])
    return out

def _embed_texts_sentence(texts: List[str]) -> List[List[float]]:
    s = get_settings()
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(s.embed_model)
    emb = model.encode(texts, batch_size=32, normalize_embeddings=True, show_progress_bar=True)
    return emb.tolist()

def _embed_texts_ollama(texts: List[str]) -> List[List[float]]:
    import requests

    embeddings = []
    for t in texts:
        r = requests.post(
            "http://localhost:11434/api/embeddings",
            json={
                "model": "bge-m3",
                "prompt": t
            },
            timeout=60,
        )
        r.raise_for_status()
        embeddings.append(r.json()["embedding"])
    return embeddings

def _embed_texts(texts: List[str]) -> List[List[float]]:
    s = get_settings()
    if s.embed_provider == "ollama":
        return _embed_texts_ollama(texts)
    if s.embed_provider == "sentence":
        return _embed_texts_sentence(texts)
    if s.embed_provider == "openai":
        return _embed_texts_openai(texts)
    raise ValueError("EMBED_PROVIDER must be 'ollama', 'sentence', or 'openai'")

def build_index(reset: bool | None = None) -> Dict[str, Any]:
    s = get_settings()
    if reset is None:
        reset = s.reset_index

    client = _get_client()

    if reset:
        try:
            client.delete_collection(s.collection)
        except Exception:
            pass

    col = client.get_or_create_collection(
        name=s.collection,
        metadata={"hnsw:space": "cosine"},
    )

    chunk_files = sorted(s.chunks_dir.glob("*.jsonl"))
    if not chunk_files:
        return {"ok": False, "message": f"No chunk files found in {s.chunks_dir}. Run make_chunks.py first."}

    ids: List[str] = []
    docs: List[str] = []
    metas: List[Dict[str, Any]] = []

    for cf in chunk_files:
        rows = read_jsonl(cf)
        for r in rows:
            text = (r.get("text") or "").strip()
            if not text:
                continue
            ids.append(r["id"])
            docs.append(text)
            meta = {
                "source_type": r.get("source_type", "video"),
                "video_id": r.get("video_id"),
                "doc_name": r.get("doc_name"),
                "start": float(r.get("start", 0.0) or 0.0),
                "end": float(r.get("end", 0.0) or 0.0),
                "lang": r.get("lang"),
            }

            # ✅ Chroma does NOT allow None in metadata
            meta = {k: v for k, v in meta.items() if v is not None}

            # optional: ensure video_id is always a string if present
            if "video_id" in meta:
                meta["video_id"] = str(meta["video_id"])

            metas.append(meta)


    embeddings = _embed_texts(docs)

    col.add(
        ids=ids,
        documents=docs,
        metadatas=metas,
        embeddings=embeddings,
    )

    return {"ok": True, "num_files": len(chunk_files), "num_chunks": len(ids), "collection": s.collection}

def query_index(query: str, top_k: int | None = None) -> Dict[str, Any]:
    s = get_settings()
    if top_k is None:
        top_k = s.top_k

    client = _get_client()
    col = client.get_or_create_collection(name=s.collection, metadata={"hnsw:space": "cosine"})

    q_emb = _embed_texts([query])[0]

    res = col.query(
        query_embeddings=[q_emb],
        n_results=int(top_k),
        include=["documents", "metadatas", "distances"],
    )

    hits = []
    for i in range(len(res["ids"][0])):
        hits.append({
            "id": res["ids"][0][i],
            "text": res["documents"][0][i],
            "meta": res["metadatas"][0][i],
            "distance": float(res["distances"][0][i]),  # lower is better
        })

    return {"ok": True, "query": query, "hits": hits}
