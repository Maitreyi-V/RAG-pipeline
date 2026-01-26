.\.venv\Scripts\activate

Write-Host "1) MP4 -> MP3"
python -m backend.src.process_videos

Write-Host "2) Transcribe MP3 -> transcripts"
python -m backend.src.transcribe

Write-Host "3) Transcript -> chunks"
python -m backend.src.make_chunks

Write-Host "4) Build Chroma index"
python -c "from backend.src.rag import build_index; print(build_index(reset=True))"

Write-Host "5) Run Streamlit"
streamlit run app.py
