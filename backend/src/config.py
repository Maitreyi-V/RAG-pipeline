from __future__ import annotations
import os
from dataclasses import dataclass
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

@dataclass(frozen=True)
class Settings:
    data_dir: Path
    videos_dir: Path
    audio_dir: Path
    transcripts_dir: Path
    chunks_dir: Path
    chroma_dir: Path

    embed_model: str
    collection: str

    llm_provider: str
    lmstudio_base_url: str
    lmstudio_model: str

    top_k: int
    reset_index: bool

def get_settings() -> Settings:
    data_dir = Path(os.getenv("DATA_DIR", "backend/data"))
    videos_dir = data_dir / "videos"
    audio_dir = data_dir / "audio"
    transcripts_dir = data_dir / "transcripts"
    chunks_dir = data_dir / "chunks"
    chroma_dir = Path(os.getenv("CHROMA_DIR", str(data_dir / "chroma_db")))

    for d in [videos_dir, audio_dir, transcripts_dir, chunks_dir, chroma_dir]:
        d.mkdir(parents=True, exist_ok=True)

    return Settings(
        data_dir=data_dir,
        videos_dir=videos_dir,
        audio_dir=audio_dir,
        transcripts_dir=transcripts_dir,
        chunks_dir=chunks_dir,
        chroma_dir=chroma_dir,
        embed_model=os.getenv("EMBED_MODEL", "BAAI/bge-m3"),
        collection=os.getenv("COLLECTION", "capstone_rag"),
        llm_provider=os.getenv("LLM_PROVIDER", "lmstudio"),
        lmstudio_base_url=os.getenv("LM_STUDIO_BASE_URL", "http://localhost:1234/v1"),
        lmstudio_model=os.getenv("LM_STUDIO_MODEL", "local-model"),
        top_k=int(os.getenv("TOP_K", "3")),
        reset_index=os.getenv("RESET_INDEX", "true").lower() == "true",
    )
