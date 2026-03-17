"""
Configuration for Gita RAG Pipeline
"""
import os

# --- Paths ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
TRANSCRIPTS_DIR = os.path.join(DATA_DIR, "transcripts")
ANNOTATIONS_CSV = os.path.join(DATA_DIR, "annotations.csv")
TEST_QUESTIONS_XLSX = os.path.join(DATA_DIR, "test_questions.xlsx")
CHUNKS_JSON = os.path.join(DATA_DIR, "chunks.json")
CHROMA_DIR = os.path.join(DATA_DIR, "chroma_db")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

# --- Ollama ---
OLLAMA_BASE_URL = "http://localhost:11434"
LLM_MODEL = "llama3"            # for answer generation
EMBED_MODEL = "bge-m3"          # for embeddings

# --- ChromaDB ---
COLLECTION_NAME = "gita_shloka_chunks"

# --- Translation ---
TRANSLATION_METHOD = "google"   # "google" | "ollama"
# google uses deep-translator (free, fast)
# ollama uses LLM_MODEL for translation (slower, more accurate for domain terms)

# --- Retrieval ---
TOP_K = 5                       # number of chunks to retrieve
SIMILARITY_THRESHOLD = 0.3      # minimum similarity score (lower = more permissive)

# --- LLM Prompt ---
SYSTEM_PROMPT = """You are a scholarly assistant specializing in the Bhagavad Gita, 
specifically Chapter 2 as explained in the Dvaita (Madhva) Vedānta tradition through 
Kannada discourses.

When answering questions:
1. Always cite the specific verse reference (e.g., BG 2.11) that your answer is based on.
2. Ground your answer in the retrieved discourse content — do not add external knowledge.
3. Mention the Dvaita tradition perspective when relevant.
4. If the user asks in Kannada, respond in Kannada. If in English, respond in English.
5. If the retrieved context does not contain enough information to answer, say so honestly.

Format your answer as:
**Verse Reference:** [BG X.Y]
**Answer:** [Your grounded answer]
**Dvaita Perspective:** [Any tradition-specific insight from the discourse]
"""

# --- Evaluation ---
EVAL_K_VALUES = [1, 3, 5]       # for precision@k measurement
