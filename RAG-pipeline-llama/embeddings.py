"""
Shared local embedding backend.

Runs BAAI/bge-m3 through sentence-transformers without Ollama.
All indexing and retrieval code must use this module.
"""

from functools import lru_cache
from typing import Sequence

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

import config


def _select_device() -> str:
    requested = config.EMBED_DEVICE.strip().lower()

    if requested != "auto":
        return requested

    if torch.cuda.is_available():
        return "cuda"

    if torch.backends.mps.is_available():
        return "mps"

    return "cpu"


@lru_cache(maxsize=1)
def get_model() -> SentenceTransformer:
    device = _select_device()
    print(f"[Embeddings] Loading {config.EMBED_MODEL} on {device}")
    return SentenceTransformer(config.EMBED_MODEL, device=device)


def embed_texts(
    texts: Sequence[str],
    batch_size: int | None = None,
) -> list[list[float]]:
    cleaned = [str(text).strip() for text in texts]

    if not cleaned:
        return []

    if any(not text for text in cleaned):
        raise ValueError("Cannot embed empty text")

    vectors = get_model().encode(
        cleaned,
        batch_size=batch_size or config.EMBED_BATCH_SIZE,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )

    return np.asarray(vectors, dtype=np.float32).tolist()


def embed_text(text: str) -> list[float]:
    return embed_texts([text])[0]