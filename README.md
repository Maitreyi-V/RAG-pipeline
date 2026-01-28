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
```

## 🛠️ Prerequisites

1️⃣ Install Python (3.10+ recommended)
```
python --version
```

2️⃣ Install Ollama
```
https://ollama.com/download
```
After install, verify:
```
ollama --version
```

### Pull Required Models (MANDATORY)
🔹 Embedding Model
```
ollama pull bge-m3
```
🔹 LLM Model
```
ollama pull llama3
```

Verify:
```
ollama list
```

You should see:
```
bge-m3
llama3
```

📦 Project Setup
3️⃣ Clone Repository
```
git clone <-repo-url->
cd RAG-pipeline
```

Switch to llama branch:
```
git checkout llama
```

4️⃣ Create Virtual Environment
```
python -m venv venv
```

Activate:

Windows
```
venv\Scripts\activate
```

Mac/Linux
```
source venv/bin/activate
```

5️⃣ Install Dependencies
```
pip install -r requirements.txt
```

⚙️ Environment Configuration

Create a .env file in project root:
```
# ---- Core paths ----
DATA_DIR=backend/data
CHROMA_DIR=backend/data/chroma_db
COLLECTION=capstone_rag
TOP_K=1
RESET_INDEX=true

# ---- Providers ----
LLM_PROVIDER=ollama
EMBED_PROVIDER=ollama
TRANSCRIBE_PROVIDER=local

# ---- Ollama models ----
OLLAMA_LLM_MODEL=llama3
OLLAMA_EMBED_MODEL=bge-m3
```

##  Resetting / Deleting Existing ChromaDB (IMPORTANT)

ChromaDB stores vector embeddings.  
If **any of the following change**, you **MUST delete and rebuild** the database:

- Embedding model (OpenAI → BGE-M3 / Ollama)
- Chunking logic
- Source documents
- Language of documents
- Corrupted or mismatched vectors

---

###  When is deletion REQUIRED?

| Situation | Required |
|--------|---------|
| Changed embedding model | ✅ YES |
| Added / removed documents | ✅ YES |
| Switching OpenAI ↔ Ollama | ✅ YES |
| Getting dimension mismatch errors | ✅ YES |
| Just asking new questions | ❌ NO |

---

### 🗑️ Delete Existing ChromaDB

ChromaDB is stored here:
backend/data/chroma_db   


### **Windows (PowerShell)**
```powershell
Remove-Item -Recurse -Force backend\data\chroma_db
```

### **Mac / Linux**
```
rm -rf backend/data/chroma_db
```

🏗️ Build the RAG Index (IMPORTANT)

Run these in order:
```
python -m backend.src.ingest_docs
```
```
python -c "from backend.src.rag import build_index; print(build_index(reset=True))"
```
✔️ This creates embeddings and stores them in ChromaDB.

🚀 Run the Application
```
streamlit run app.py
```
Open browser:
```
http://localhost:8501
```
