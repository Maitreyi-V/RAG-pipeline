#!/usr/bin/env bash
# =============================================================================
# Chunking ablation: śloka-aware (verse-boundary) vs fixed-size (300-word)
# -----------------------------------------------------------------------------
# Reproduces the pending ablation in ONE command. For each chunking strategy it
#   (1) clears + re-indexes the Advaita chunks into ChromaDB,
#   (2) runs the 48-question Advaita eval (retrieval-only P@k, no API key),
#   (3) captures stdout to eval_<strategy>.txt and structured metrics to JSON,
# then prints a side-by-side P@1/P@3/P@5 comparison.
#
# WHY retrieval-only (--questions-only): the ablation isolates whether verse-
# boundary chunks improve *retrieval*. That is exactly P@k. It needs Ollama
# (bge-m3 embeddings) but NOT OpenAI. To also compare answer quality, re-run
# with:  JUDGE=gpt ./run_chunking_ablation.sh   (needs OPENAI_API_KEY).
#
# Usage:
#   ./run_chunking_ablation.sh              # retrieval-only (default)
#   JUDGE=gpt ./run_chunking_ablation.sh    # also score answer quality
#
# Prereqs (step zero — this repo is not runnable until these exist):
#   - Ollama running with bge-m3 pulled:   ollama serve & ; ollama pull bge-m3
#   - Python deps installed:               pip install -r requirements.txt
#   - Run from the RAG-pipeline-llama/ directory.
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")"

JUDGE="${JUDGE:-none}"                       # none = retrieval-only ; gpt = + answer quality
QUESTIONS="advaita_test_questions.json"
TRADITION="Advaita"
PY="${PYTHON:-python3}"

if [[ "$JUDGE" == "none" ]]; then EVAL_JUDGE_FLAG="--questions-only"; else EVAL_JUDGE_FLAG="--judge $JUDGE"; fi

echo "==================================================================="
echo " CHUNKING ABLATION   (judge mode: $JUDGE)"
echo "==================================================================="

# ---- Step zero: fail fast with a clear message if the env isn't ready -------
if ! curl -s http://localhost:11434/api/tags >/dev/null 2>&1; then
  echo "ERROR: Ollama is not reachable at http://localhost:11434"
  echo "       Start it:  ollama serve   (and:  ollama pull bge-m3)"
  exit 1
fi
if [[ "$JUDGE" == "gpt" && -z "${OPENAI_API_KEY:-}" ]]; then
  echo "ERROR: JUDGE=gpt needs OPENAI_API_KEY exported."
  exit 1
fi

run_variant () {
  local name="$1"; local index_cmd="$2"
  echo ""
  echo "-------------------------------------------------------------------"
  echo " VARIANT: $name"
  echo "-------------------------------------------------------------------"
  echo ">> (re)indexing:  $index_cmd"
  # Index step: clears Advaita chunks then re-indexes with this strategy.
  eval "$index_cmd" | tee "index_${name}.log"

  echo ">> evaluating $QUESTIONS  (tradition=$TRADITION, no-ref)"
  $PY 05_evaluate.py \
      $EVAL_JUDGE_FLAG \
      --questions-file "$QUESTIONS" \
      --tradition "$TRADITION" \
      --no-ref \
      --output "results/eval_${name}.json" \
      | tee "eval_${name}.txt"
}

run_variant "fixed_size"   "$PY index_advaita.py --clear-advaita --words-per-chunk 300 --overlap-words 75"
run_variant "shloka_aware" "$PY index_advaita_shloka_aware.py --clear-advaita --min-chunk-words 100 --max-chunk-words 800"

# ---- Side-by-side comparison ------------------------------------------------
echo ""
echo "==================================================================="
echo " ABLATION SUMMARY"
echo "==================================================================="
$PY - <<'PYEOF'
import json, os
def load(p):
    if not os.path.exists(p): return None
    d = json.load(open(p)); return d.get("summary", d)
fx = load("results/eval_fixed_size.json")
sa = load("results/eval_shloka_aware.json")
def g(d,k): return "n/a" if d is None or d.get(k) is None else f"{d[k]:.4f}"
def n_chunks(log):
    # best-effort parse of chunk count from the index log
    try:
        for line in open(log):
            for tok in ("chunks indexed","chunks created","total chunks","Indexed"):
                if tok.lower() in line.lower(): return line.strip()
    except FileNotFoundError: pass
    return "(see index log)"
rows = [
    ("metric", "fixed_size", "shloka_aware"),
    ("P@1", g(fx,"precision_at_1"), g(sa,"precision_at_1")),
    ("P@3", g(fx,"precision_at_3"), g(sa,"precision_at_3")),
    ("P@5", g(fx,"precision_at_5"), g(sa,"precision_at_5")),
    ("shloka_acc", g(fx,"shloka_identification_accuracy"), g(sa,"shloka_identification_accuracy")),
    ("avg_similarity", g(fx,"avg_retrieval_similarity"), g(sa,"avg_retrieval_similarity")),
]
w = [max(len(str(r[i])) for r in rows) for i in range(3)]
for r in rows:
    print("  " + " | ".join(str(r[i]).ljust(w[i]) for i in range(3)))
print()
print("  fixed_size   chunks: " + n_chunks("index_fixed_size.log"))
print("  shloka_aware chunks: " + n_chunks("index_shloka_aware.log"))
print()
print("  Full stdout: eval_fixed_size.txt / eval_shloka_aware.txt")
print("  Structured : results/eval_fixed_size.json / results/eval_shloka_aware.json")
PYEOF
echo "Done."
