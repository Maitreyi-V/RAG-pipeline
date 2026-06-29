"""
Step 5: Evaluation
==================
Runs the gold test questions through the pipeline and measures:
  1. Retrieval Precision@k (k=1,3,5) — does the right verse come back?
  2. Śloka Identification Accuracy — is the correct verse reference cited?
  3. Answer Faithfulness — does the answer match the expected answer? (LLM-judged)
  4. Answer Groundedness — does the answer stay within the retrieved context? (LLM-judged)
  5. Retrieval Similarity — average cosine similarity of retrieved chunks

JUDGES
------
  --judge gpt   (DEFAULT) : GPT-4o-mini claim-level judge (judge_gpt.py).
                            faithfulness 0-5, groundedness 0-1 (fraction of claims supported).
                            Requires OPENAI_API_KEY in the environment.
  --judge llama           : the old llama3 0-3 judge (kept only to reproduce old numbers).
  --judge none            : skip answer-quality judging (retrieval metrics only).

ABLATION
--------
  --no-context : CLOSED-BOOK generation. The LLM answers from its own memory with
                 NO retrieved context. Compare faithfulness/groundedness against a
                 normal run: if they barely drop, the system is leaning on the
                 model's parametric knowledge rather than your transcripts.

Usage:
    python 05_evaluate.py                                   # gpt judge, with context
    python 05_evaluate.py --judge none                      # retrieval only (fast)
    python 05_evaluate.py --no-context                      # closed-book ablation
    python 05_evaluate.py --no-ref                          # honest (verse refs stripped)
    python 05_evaluate.py --questions-file data/advaita_questions.json
    python 05_evaluate.py --tradition Dvaita                # scope retrieval to one tradition
    python 05_evaluate.py --judge llama --output results/old_llama_eval.json
"""

import json
import os
import sys
import argparse
import time
from collections import defaultdict

import requests

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config

from retrieve_lib import retrieve_for_eval, retrieve_chunks, build_context, \
    generate_answer, generate_answer_no_context, detect_language

# GPT judge (Phase 1 fix). Imported lazily-safe: if openai isn't installed and the
# user picks --judge llama or none, we never call it.
try:
    from judge_gpt import judge_groundedness_claims, judge_faithfulness_gpt
    _HAS_GPT_JUDGE = True
except Exception as _e:
    _HAS_GPT_JUDGE = False
    _GPT_IMPORT_ERROR = _e


# ─────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────

def load_test_questions(xlsx_path):
    """Load test questions from Excel file."""
    import pandas as pd
    df = pd.read_excel(xlsx_path)
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    questions = []
    for _, row in df.iterrows():
        q = {
            "question_id":       str(row.get("question_id", "")),
            "question_en":       str(row.get("question_en", "")),
            "question_en_no_ref":str(row.get("question_en_no_ref", "")),
            "question_kn":       str(row.get("question_kn", "")),
            "expected_verse":    str(row.get("expected_verse", "")),
            "expected_answer_en":str(row.get("expected_answer_en", "")),
            "question_type":     str(row.get("question_type", "")),
        }
        if q["question_en"] and q["question_en"] != "nan":
            questions.append(q)
    return questions


def load_test_questions_json(json_path):
    """Load test questions from JSON file (e.g., Advaita questions)."""
    with open(json_path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    questions = []
    for r in raw:
        expected = r.get("expected_verses", [])
        if isinstance(expected, list):
            expected = ", ".join(expected)
        q = {
            "question_id":        r.get("id", ""),
            "question_en":        r.get("question_en_no_ref", ""),
            "question_en_no_ref": r.get("question_en_no_ref", ""),
            "question_kn":        "",
            "expected_verse":     expected,
            "expected_answer_en": "",   # Advaita JSON has no gold answer → faithfulness skipped
            "question_type":      r.get("type", ""),
        }
        if q["question_en"]:
            questions.append(q)
    return questions


# ─────────────────────────────────────────────
# RETRIEVAL METRICS
# ─────────────────────────────────────────────

def _flatten_verses(verse_list):
    """Expand comma-separated verse labels ('BG 2.13, BG 2.14') into individual refs."""
    result = set()
    for entry in verse_list:
        for v in entry.split(","):
            v = v.strip()
            if v:
                result.add(v)
    return result


def compute_precision_at_k(retrieved_verses, expected_verse, k):
    expected_set = set(v.strip() for v in expected_verse.split(",") if v.strip())
    if not expected_set:
        return None
    top_k_flat = _flatten_verses(retrieved_verses[:k])
    return 1.0 if expected_set & top_k_flat else 0.0


def compute_shloka_accuracy(retrieved_verses, expected_verse):
    expected_set = set(v.strip() for v in expected_verse.split(",") if v.strip())
    return 1.0 if expected_set & _flatten_verses(retrieved_verses) else 0.0


# ─────────────────────────────────────────────
# OLD LLAMA JUDGES (kept to reproduce old numbers — known to be unreliable)
# ─────────────────────────────────────────────

def _call_ollama_digit(prompt):
    try:
        resp = requests.post(
            f"{config.OLLAMA_BASE_URL}/api/generate",
            json={"model": config.LLM_MODEL, "prompt": prompt, "stream": False,
                  "options": {"temperature": 0.0, "num_predict": 10}},
            timeout=300)
        resp.raise_for_status()
        text = resp.json().get("response", "").strip()
        for char in text:
            if char.isdigit() and int(char) <= 3:
                return int(char)
        return None
    except Exception as e:
        print(f"    [llama judge error] {e}")
        return None


def judge_faithfulness_llm(answer, expected_answer, question):
    prompt = f"""You are evaluating a Q&A system about the Bhagavad Gita.
Question: {question}
Expected Answer: {expected_answer}
Generated Answer: {answer}
Rate faithfulness 0-3 (0=wrong, 3=fully faithful). Respond with ONLY the number."""
    return _call_ollama_digit(prompt)


def judge_groundedness_llm(answer, retrieved_chunks, question):
    context_parts = []
    for i, c in enumerate(retrieved_chunks):
        verse = c.get("verse_ref", "?")
        text = c.get("document", c.get("text", "")).strip()
        if text:
            context_parts.append(f"[Chunk {i+1} — {verse}]:\n{text}")
    if not context_parts:
        return None
    context = "\n\n".join(context_parts)
    prompt = f"""You are evaluating whether an AI answer is grounded in the provided context.
Question: {question}
Retrieved Context:
{context}
Generated Answer: {answer}
Rate groundedness 0-3 (0=hallucination, 3=fully grounded). Respond with ONLY the number."""
    return _call_ollama_digit(prompt)


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--judge", default="gpt", choices=["gpt", "llama", "none"],
                        help="Answer-quality judge backend (default: gpt = GPT-4o-mini claim judge)")
    parser.add_argument("--no-context", action="store_true",
                        help="CLOSED-BOOK ablation: generate answers with NO retrieved context")
    parser.add_argument("--questions-only", action="store_true",
                        help="(alias for --judge none) retrieval metrics only")
    parser.add_argument("--no-ref", action="store_true",
                        help="Use question_en_no_ref (verse refs stripped) for honest eval")
    parser.add_argument("--lang", default="en", choices=["en", "kn"],
                        help="Which question language to use (ignored if --no-ref)")
    parser.add_argument("--tradition", default=None, choices=["Dvaita", "Advaita"],
                        help="Scope retrieval to one tradition (default: no filter)")
    parser.add_argument("--questions-file", default=None,
                        help="Load questions from JSON instead of Excel")
    parser.add_argument("--include-xlsx", action="store_true",
                        help="Also include Excel questions when using --questions-file")
    parser.add_argument("--output", default=None,
                        help="Output path (default: auto-named so frozen results aren't overwritten)")
    args = parser.parse_args()

    if args.questions_only:
        args.judge = "none"

    # Resolve judge backend
    judge_backend = args.judge
    if judge_backend == "gpt":
        if not _HAS_GPT_JUDGE:
            print(f"ERROR: --judge gpt needs judge_gpt.py + the openai package.")
            print(f"       import failed with: {_GPT_IMPORT_ERROR}")
            print(f"       Fix: pip install openai   and   export OPENAI_API_KEY=sk-...")
            sys.exit(1)
        if not os.environ.get("OPENAI_API_KEY"):
            print("ERROR: --judge gpt needs OPENAI_API_KEY in the environment.")
            print("       export OPENAI_API_KEY=sk-...   (or use --judge none / --judge llama)")
            sys.exit(1)

    import chromadb

    # ── Load questions (FIXED: no more duplicate overwrite) ────────────
    if args.questions_file:
        questions = load_test_questions_json(args.questions_file)
        print(f"Loaded {len(questions)} questions from {args.questions_file}")
        if args.include_xlsx and os.path.exists(config.TEST_QUESTIONS_XLSX):
            xlsx_q = load_test_questions(config.TEST_QUESTIONS_XLSX)
            questions.extend(xlsx_q)
            print(f"  + {len(xlsx_q)} from Excel = {len(questions)} total")
    else:
        if not os.path.exists(config.TEST_QUESTIONS_XLSX):
            print(f"ERROR: Test questions not found: {config.TEST_QUESTIONS_XLSX}")
            sys.exit(1)
        questions = load_test_questions(config.TEST_QUESTIONS_XLSX)
        print(f"Loaded {len(questions)} test questions from Excel")

    gen_mode = "no_context" if args.no_context else "with_context"
    print(f"Judge backend   : {judge_backend}")
    print(f"Generation mode : {gen_mode}")
    if args.tradition:
        print(f"Tradition filter: {args.tradition}")
    if args.no_ref:
        print("Question mode   : HONEST (verse refs stripped)")
    else:
        print(f"Question mode   : STANDARD ({args.lang})")

    # Auto-name output so frozen results are never clobbered
    if args.output is None:
        tag = f"{judge_backend}_{gen_mode}"
        if args.no_ref:
            tag += "_noref"
        if args.tradition:
            tag += f"_{args.tradition.lower()}"
        args.output = os.path.join(config.RESULTS_DIR, f"eval_{tag}.json")

    # ── Connect to ChromaDB ─────────────────────────────────────────────
    client = chromadb.PersistentClient(path=config.CHROMA_DIR)
    try:
        collection = client.get_collection(config.COLLECTION_NAME)
    except Exception:
        print("ERROR: ChromaDB collection not found. Run 03_index.py first.")
        sys.exit(1)
    print(f"ChromaDB collection: {collection.count()} documents\n")

    # ── Metrics containers ──────────────────────────────────────────────
    results = []
    metrics = {
        "precision_at_k":   {k: [] for k in config.EVAL_K_VALUES},
        "shloka_accuracy":  [],
        "faithfulness":     [],
        "groundedness":     [],
        "n_unsupported":    [],
        "similarity":       [],
        "by_type": defaultdict(lambda: {
            "precision_1": [], "precision_3": [], "precision_5": [],
            "shloka_acc": [], "faithfulness": [], "groundedness": [], "similarity": []
        })
    }

    # ── Evaluation loop ─────────────────────────────────────────────────
    for i, q in enumerate(questions):
        qid            = q["question_id"]
        expected_verse = q["expected_verse"]
        qtype          = q["question_type"]

        if args.no_ref:
            query = q.get("question_en_no_ref", "") or q["question_en"]
        elif args.lang == "kn":
            query = q.get("question_kn", q["question_en"])
        else:
            query = q["question_en"]

        print(f"[{i+1}/{len(questions)}] Q{qid} ({qtype}): {query[:80]}...")

        # ── Retrieve (always — we need chunks for groundedness + metrics) ──
        try:
            retrieved_chunks = retrieve_chunks(query, collection,
                                               top_k=config.TOP_K, tradition=args.tradition)
        except Exception as e:
            print(f"  ERROR during retrieval: {e}")
            results.append({"question_id": qid, "error": str(e)})
            continue

        # ── Generate answer (normal vs closed-book ablation) ──────────────
        lang = detect_language(query)
        if args.no_context:
            answer = generate_answer_no_context(query, lang)
        else:
            context = build_context(retrieved_chunks)
            answer = generate_answer(query, context, lang)

        retrieved_verses = [c["verse_ref"] for c in retrieved_chunks if c.get("verse_ref")]
        similarities = [c.get("similarity") for c in retrieved_chunks if c.get("similarity") is not None]
        avg_sim = round(sum(similarities) / len(similarities), 4) if similarities else None

        result = {
            "question_id":      qid,
            "question":         query,
            "question_type":    qtype,
            "expected_verse":   expected_verse,
            "retrieved_verses": retrieved_verses,
            "answer":           answer,
            "generation_mode":  gen_mode,
            "retrieved_chunks_text": [c.get("document", c.get("text", "")) for c in retrieved_chunks],
            "retrieval_similarities": similarities,
            "avg_similarity":   avg_sim,
        }

        if avg_sim is not None:
            metrics["similarity"].append(avg_sim)
            metrics["by_type"][qtype]["similarity"].append(avg_sim)

        for k in config.EVAL_K_VALUES:
            pk = compute_precision_at_k(retrieved_verses, expected_verse, k)
            result[f"precision_at_{k}"] = pk
            if pk is not None:
                metrics["precision_at_k"][k].append(pk)
                metrics["by_type"][qtype][f"precision_{k}"].append(pk)

        sa = compute_shloka_accuracy(retrieved_verses, expected_verse)
        result["shloka_accuracy"] = sa
        metrics["shloka_accuracy"].append(sa)
        metrics["by_type"][qtype]["shloka_acc"].append(sa)

        # ── Answer-quality judging ────────────────────────────────────────
        if judge_backend == "gpt":
            # groundedness: claim-level (always, if we have chunks)
            g = judge_groundedness_claims(answer, retrieved_chunks, query)
            if g is not None:
                result["groundedness"] = g["groundedness"]
                result["groundedness_detail"] = {
                    "n_claims": g["n_claims"], "n_supported": g["n_supported"],
                    "n_partial": g["n_partial"], "n_unsupported": g["n_unsupported"],
                    "unsupported_claims": g["unsupported_claims"],
                }
                metrics["groundedness"].append(g["groundedness"])
                metrics["n_unsupported"].append(g["n_unsupported"])
                metrics["by_type"][qtype]["groundedness"].append(g["groundedness"])

            # faithfulness: only if a gold answer exists
            if q.get("expected_answer_en") and q["expected_answer_en"].strip().lower() not in ("", "nan"):
                f = judge_faithfulness_gpt(answer, q["expected_answer_en"], query)
                if f is not None:
                    result["faithfulness"] = f["faithfulness"]
                    result["faithfulness_reason"] = f["reason"]
                    metrics["faithfulness"].append(f["faithfulness"])
                    metrics["by_type"][qtype]["faithfulness"].append(f["faithfulness"])

        elif judge_backend == "llama":
            if q.get("expected_answer_en") and q["expected_answer_en"].strip().lower() not in ("", "nan"):
                fs = judge_faithfulness_llm(answer, q["expected_answer_en"], query)
                result["faithfulness"] = fs
                if fs is not None:
                    metrics["faithfulness"].append(fs)
                    metrics["by_type"][qtype]["faithfulness"].append(fs)
            gs = judge_groundedness_llm(answer, retrieved_chunks, query)
            result["groundedness"] = gs
            if gs is not None:
                metrics["groundedness"].append(gs)
                metrics["by_type"][qtype]["groundedness"].append(gs)
            time.sleep(0.3)

        results.append(result)

        # ── Per-question print ────────────────────────────────────────────
        sim_str = f"{avg_sim:.3f}" if avg_sim is not None else "N/A"
        print(f"  Expected: {expected_verse} | Retrieved: {retrieved_verses[:3]}")
        print(f"  P@1={result.get('precision_at_1')} | Śloka={sa} | AvgSim={sim_str}", end="")
        if judge_backend != "none":
            print(f" | Faith={result.get('faithfulness','—')} | Ground={result.get('groundedness','—')}")
        else:
            print()

    # ── Aggregate summary ────────────────────────────────────────────────
    if judge_backend == "gpt":
        faith_scale, ground_scale = "0-5", "0-1"
    elif judge_backend == "llama":
        faith_scale, ground_scale = "0-3", "0-3"
    else:
        faith_scale, ground_scale = None, None

    summary = {
        "total_questions": len(questions),
        "evaluated":       len(results),
        "judge_backend":   judge_backend,
        "generation_mode": gen_mode,
        "tradition_filter": args.tradition,
        "eval_mode":       "no_ref" if args.no_ref else "standard",
        "faithfulness_scale": faith_scale,
        "groundedness_scale": ground_scale,
    }

    for k in config.EVAL_K_VALUES:
        vals = metrics["precision_at_k"][k]
        summary[f"precision_at_{k}"] = round(sum(vals) / len(vals), 4) if vals else 0

    vals = metrics["shloka_accuracy"]
    summary["shloka_identification_accuracy"] = round(sum(vals) / len(vals), 4) if vals else 0

    if metrics["similarity"]:
        summary["avg_retrieval_similarity"] = round(sum(metrics["similarity"]) / len(metrics["similarity"]), 4)

    if metrics["faithfulness"]:
        fv = metrics["faithfulness"]
        summary["avg_faithfulness"] = round(sum(fv) / len(fv), 4)
        summary["n_faithfulness_judged"] = len(fv)
        top = 5 if judge_backend == "gpt" else 3
        summary["faithfulness_distribution"] = {str(s): fv.count(s) for s in range(top + 1)}

    if metrics["groundedness"]:
        gv = metrics["groundedness"]
        summary["avg_groundedness"] = round(sum(gv) / len(gv), 4)
        summary["n_groundedness_judged"] = len(gv)
        if judge_backend == "gpt" and metrics["n_unsupported"]:
            nu = metrics["n_unsupported"]
            summary["avg_unsupported_claims_per_answer"] = round(sum(nu) / len(nu), 3)
            summary["answers_fully_grounded"] = sum(1 for x in gv if x >= 0.999)
            summary["answers_with_hallucination"] = sum(1 for x in nu if x > 0)

    summary["by_question_type"] = {}
    for qtype, tm in metrics["by_type"].items():
        ts = {}
        for metric_name, vals in tm.items():
            if vals:
                ts[metric_name] = round(sum(vals) / len(vals), 4)
        ts["count"] = len(tm.get("shloka_acc", []))
        summary["by_question_type"][qtype] = ts

    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "detailed_results": results}, f,
                  ensure_ascii=False, indent=2)

    # ── Print summary ─────────────────────────────────────────────────────
    print(f"\n{'='*60}\nEVALUATION SUMMARY\n{'='*60}")
    print(f"Questions       : {summary['total_questions']}")
    print(f"Judge / Gen     : {judge_backend} / {gen_mode}")
    print(f"Question mode   : {summary['eval_mode']}\n")
    for k in config.EVAL_K_VALUES:
        print(f"Precision@{k}  : {summary.get(f'precision_at_{k}')}")
    print(f"Śloka Accuracy  : {summary.get('shloka_identification_accuracy')}")
    if "avg_retrieval_similarity" in summary:
        print(f"Avg Similarity  : {summary['avg_retrieval_similarity']}")
    if "avg_faithfulness" in summary:
        print(f"Avg Faithfulness: {summary['avg_faithfulness']} ({faith_scale})  n={summary['n_faithfulness_judged']}")
        print(f"  Distribution  : {summary['faithfulness_distribution']}")
    if "avg_groundedness" in summary:
        print(f"Avg Groundedness: {summary['avg_groundedness']} ({ground_scale})  n={summary['n_groundedness_judged']}")
        if "avg_unsupported_claims_per_answer" in summary:
            print(f"  Unsupported claims/answer : {summary['avg_unsupported_claims_per_answer']}")
            print(f"  Answers fully grounded    : {summary['answers_fully_grounded']}")
            print(f"  Answers w/ hallucination  : {summary['answers_with_hallucination']}")

    print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()