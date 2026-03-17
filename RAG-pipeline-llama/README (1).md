# Gita RAG Pipeline
## Śloka-Aware, Multilingual Q&A for Bhagavad Gita Discourses

A retrieval-augmented generation (RAG) pipeline for answering questions about Bhagavad Gita Chapter 2 (verses 2.1–2.14) based on Kannada discourses from the Dvaita (Madhva) Vedānta tradition.

### Core Novelty
- **Śloka-aware chunking** — each chunk maps to exactly one Sanskrit verse + its Kannada explanation
- **English as semantic bridge** — embeddings are on English translations for cross-lingual retrieval
- **Tradition-tagged answers** — responses explicitly cite the Dvaita interpretation
- **Bilingual Q&A** — accepts questions in English or Kannada

---

## Architecture

```
Kannada Audio Discourses (6 videos)
        ↓ (Sarvam AI - already done)
Plain Text Transcripts (.txt)
        ↓
┌─────────────────────────────────────────┐
│  01_chunk.py    - Śloka-aware chunking  │  ← uses annotations.csv
│  02_translate.py - Kannada → English    │  ← Google Translate / Ollama
│  03_index.py    - BGE-M3 → ChromaDB    │  ← embed English text
│  04_retrieve.py - Vector search + LLM  │  ← Llama3 generates answers
│  05_evaluate.py - Precision@k + Faith  │  ← 74 gold test questions
│  06_app.py      - Streamlit UI         │
└─────────────────────────────────────────┘
```

---

## Quick Start

### Prerequisites
```bash
# Python packages
pip install chromadb deep-translator openpyxl pandas streamlit requests

# Ollama (local LLM server)
curl -fsSL https://ollama.com/install.sh | sh
ollama pull llama3
ollama pull bge-m3
```

### File Setup
Place these files in `data/`:
```
data/
├── annotations.csv          # ← provided (śloka annotations from Google Sheet)
├── transcripts/
│   ├── gita-1st_video.txt   # ← your 6 transcript files
│   ├── gita-2nd_video.txt
│   ├── gita-3rd_video.txt
│   ├── gita-4th_video.txt
│   ├── gita-5th_video.txt
│   └── gita-6th_video.txt
└── test_questions.xlsx       # ← your 74 gold test questions
```

### Run the Pipeline
```bash
# Step 1: Create śloka-aware chunks from annotations + transcripts
python 01_chunk.py

# Step 2: Translate Kannada text to English
python 02_translate.py                  # uses Google Translate (fast)
python 02_translate.py --method ollama  # uses Llama3 (better for domain terms)

# Step 3: Embed and index into ChromaDB
python 03_index.py          # first time
python 03_index.py --reset  # rebuild from scratch

# Step 4: Ask questions
python 04_retrieve.py "What does Krishna say about the soul?"
python 04_retrieve.py "ಅರ್ಜುನನ ದುಃಖಕ್ಕೆ ಕಾರಣ ಏನು?"
python 04_retrieve.py --interactive

# Step 5: Run evaluation on 74 test questions
python 05_evaluate.py
python 05_evaluate.py --questions-only  # skip LLM faithfulness scoring

# Step 6: Launch Streamlit UI
streamlit run 06_app.py
```

---

## Pipeline Details

### 01_chunk.py — Śloka-Aware Chunking
- Reads `annotations.csv` (your manual śloka annotations)
- For each annotation row, tries to extract the matching Kannada segment from the transcript
- Creates one chunk per: introduction, śloka explanation, or closing
- Output: `data/chunks.json` (~26 chunks for 6 videos)

### 02_translate.py — Translation Pipeline
- Translates Kannada discourse text → English for each chunk
- Builds a combined `embedding_text` field per chunk:
  - Verse reference + speaker + English summary + translated Kannada + Sanskrit śloka
- This combined text is what gets embedded (not raw Kannada)

### 03_index.py — Embedding & Indexing
- Uses BGE-M3 (via Ollama) to generate embeddings from English text
- Stores vectors + full metadata in ChromaDB with cosine similarity
- Metadata includes: verse_ref, speaker, section_type, tradition, video_file

### 04_retrieve.py — Retrieval + Answer Generation
- Detects query language (English/Kannada)
- If Kannada: translates to English for semantic search
- Retrieves top-k chunks by cosine similarity
- Feeds retrieved context to Llama3 with a Dvaita-aware system prompt
- Answer includes verse citations and tradition-specific commentary

### 05_evaluate.py — Evaluation
- Loads 74 gold test questions from Excel
- Measures:
  - **Retrieval Precision@k** (k=1,3,5): Is the correct verse in top-k?
  - **Śloka Identification Accuracy**: Is the correct verse retrieved at all?
  - **Answer Faithfulness** (LLM-judged): Does the answer match expected content?
- Breaks down results by question type (Factual, Interpretive, Cross-verse, etc.)

### 06_app.py — Streamlit UI
- Interactive web interface for asking questions
- Shows answer + cited verses + source details
- Displays Sanskrit śloka, padavibhāga, and original Kannada text

---

## Configuration

Edit `config.py` to customize:
- `LLM_MODEL` — which Ollama model for generation (default: llama3)
- `EMBED_MODEL` — which model for embeddings (default: bge-m3)
- `TOP_K` — how many chunks to retrieve (default: 5)
- `TRANSLATION_METHOD` — "google" or "ollama"
- `SIMILARITY_THRESHOLD` — minimum similarity to include (default: 0.3)

---

## Ablation Experiments (for the paper)

The pipeline is designed to support these ablation studies:

1. **Chunking comparison**: Modify `01_chunk.py` to produce fixed-time or fixed-char chunks instead of śloka-aware
2. **Translation impact**: In `03_index.py`, embed raw Kannada vs. English translations vs. concatenated
3. **Retrieval method**: Add BM25 (e.g., via `rank_bm25`) alongside vector search
4. **LLM comparison**: Change `config.LLM_MODEL` to test different models (llama3, gemma, etc.)

---

## Project Info
- **Course**: UE23CS320A – Capstone Project Phase 1 & 2
- **University**: PES University, Bengaluru
- **Guide**: Prof Badri Prasad VR
- **Team**: Maitreyi V, Meghana Gajendran, Meghana S, Prerana M.P
