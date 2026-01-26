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

    collection: str
    top_k: int
    reset_index: bool

    # Providers
    llm_provider: str           # lmstudio | openai
    embed_provider: str         # sentence | openai
    transcribe_provider: str    # local | openai

    # Local embedding model (SentenceTransformers)
    embed_model: str

    # LM Studio
    lmstudio_base_url: str
    lmstudio_model: str

    # OpenAI
    openai_api_key: str
    openai_chat_model: str
    openai_embed_model: str
    openai_asr_model: str


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

        collection=os.getenv("COLLECTION", "capstone_rag"),
        top_k=int(os.getenv("TOP_K", "3")),
        reset_index=os.getenv("RESET_INDEX", "true").lower() == "true",

        llm_provider=os.getenv("LLM_PROVIDER", "lmstudio").lower(),
        embed_provider=os.getenv("EMBED_PROVIDER", "sentence").lower(),
        transcribe_provider=os.getenv("TRANSCRIBE_PROVIDER", "local").lower(),

        embed_model=os.getenv("EMBED_MODEL", "BAAI/bge-m3"),

        lmstudio_base_url=os.getenv("LM_STUDIO_BASE_URL", "http://localhost:1234/v1"),
        lmstudio_model=os.getenv("LM_STUDIO_MODEL", "local-model"),

        openai_api_key=os.getenv("OPENAI_API_KEY", ""),
        openai_chat_model=os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini"),
        openai_embed_model=os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small"),
        openai_asr_model=os.getenv("OPENAI_ASR_MODEL", "whisper-1"),
    )
