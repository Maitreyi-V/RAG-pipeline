# RAG Pipeline – Kannada Gita Capstone (MVP)

Local pipeline:
MP4 videos -> MP3 audio -> faster-whisper transcription (timestamps) -> chunking -> BGE-M3 embeddings -> ChromaDB -> Streamlit UI + LM Studio answer.

## What goes in GitHub?
Code, requirements, scripts, README  
Do NOT commit videos/audio/transcripts/chroma_db (see .gitignore)

---

## 1) Setup (Mac)

### Install ffmpeg
```bash
brew install ffmpeg
