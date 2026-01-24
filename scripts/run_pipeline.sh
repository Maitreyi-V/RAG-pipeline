#!/usr/bin/env bash
set -e

source .venv/bin/activate

echo "1) MP4 -> MP3"
python -m backend.src.process_videos

echo "2) Transcribe MP3 -> transcripts"
python -m backend.src.transcribe

echo "3) Transcript -> chunks"
python -m backend.src.make_chunks

echo "4) Build Chroma index"
python -c "from backend.src.rag import build_index; print(build_index(reset=True))"

echo "5) Run Streamlit"
streamlit run app.py
