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
AUDIO_CACHE_DIR = os.path.join(DATA_DIR, "audio_cache")

# --- Ollama ---
OLLAMA_BASE_URL = "http://localhost:11434"
LLM_MODEL = "llama3"            # for answer generation
EMBED_MODEL = "bge-m3"          # for embeddings

# --- Whisper ---
WHISPER_MODEL = "small"         # base | small | medium | large-v3
WHISPER_LANGUAGE_MAP = {        # tradition → expected audio language for Whisper
    "Advaita": "en",
    "Dvaita": "kn",
}

# --- ChromaDB ---
COLLECTION_NAME = "gita_shloka_chunks"

# --- Translation ---
TRANSLATION_METHOD = "google"   # "google" | "ollama"

# --- Retrieval ---
TOP_K = 5                         # number of chunks to retrieve (was 1 in .env — too low!)
SIMILARITY_THRESHOLD = 0.40       # slightly more permissive to get richer context

# --- LLM Generation ---
# Increase max tokens so the LLM is not cut off mid-answer.
# llama3 context window is large; 1500 tokens gives a thorough paragraph-level response.
LLM_MAX_TOKENS = 600    # safe for CPU — gives 4-6 sentences without timing out
LLM_TEMPERATURE = 0.3

# --- System Prompt ---
# FIX: The old prompt forced a rigid 3-line template which made answers look extractive.
# The new prompt instructs the LLM to produce a FULL, DETAILED, SYNTHESIZED (abstractive)
# answer — explaining in its own words, using all retrieved chunks, like a scholar would.
SUMMARY_PROMPT = """You are a scholarly teaching assistant for the Bhagavad Gita, Chapter 2.

Given the retrieved discourse segments from BOTH Dvaita and Advaita traditions,
write a clear, comprehensive SUMMARY that answers the user's question.

RULES:
1. Synthesize information from ALL provided segments.
2. Write 4-6 sentences minimum. Be thorough and educational.
3. Cite verse references (e.g., BG 2.13) when relevant.
4. Do NOT separate by tradition here — just give the best overall answer.
5. If the question is in Kannada, respond in Kannada.
6. IMPORTANT: If the question is NOT about the Bhagavad Gita, Vedanta, or Indian philosophy,
   say clearly: "This question is outside the scope of the Bhagavad Gita discourses in our corpus."
   Do NOT try to connect unrelated topics to the Gita. Stay honest and grounded.
"""

# --- Evaluation ---
EVAL_K_VALUES = [1, 3, 5]       # for precision@k measurement
