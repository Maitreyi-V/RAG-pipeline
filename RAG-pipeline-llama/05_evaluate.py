"""
Step 5: Evaluation
==================
Runs the 74 gold test questions through the pipeline and measures:
  1. Retrieval Precision@k (k=1,3,5) — does the right verse come back?
  2. Śloka Identification Accuracy — is the correct verse reference cited?
  3. Answer Faithfulness — does the answer match the expected answer? (LLM-judged)
  4. Answer Groundedness — does the answer stay within the retrieved context? (LLM-judged)
  5. Retrieval Similarity — average cosine similarity of retrieved chunks

Input: test_questions.xlsx with columns:
  question_id, question_en, question_en_no_ref, question_kn,
  expected_verse, expected_answer_en, question_type

Usage:
    python 05_evaluate.py
    python 05_evaluate.py --questions-only       # Skip LLM scoring (fast)
    python 05_evaluate.py --no-ref               # Use verse-ref-stripped questions (honest eval)
    python 05_evaluate.py --output results/eval_results.json
    python 05_evaluate.py --no-ref --questions-only   # Fastest honest retrieval eval
"""

from html import parser
import json
import os
import sys
import argparse
import time
import requests
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config

from retrieve_lib import retrieve_for_eval


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
            "question_en_no_ref":str(row.get("question_en_no_ref", "")),  # verse-ref stripped
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
    import json as json_mod
    with open(json_path, "r", encoding="utf-8") as f:
        raw = json_mod.load(f)
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
            "expected_answer_en": "",
            "question_type":      r.get("type", ""),
        }
        if q["question_en"]:
            questions.append(q)
    return questions


# ─────────────────────────────────────────────
# RETRIEVAL METRICS
# ─────────────────────────────────────────────

def _flatten_verses(verse_list):
    """Expand comma-separated verse labels (e.g. 'BG 2.13, BG 2.14') into individual refs."""
    result = set()
    for entry in verse_list:
        for v in entry.split(","):
            v = v.strip()
            if v:
                result.add(v)
    return result


def compute_precision_at_k(retrieved_verses, expected_verse, k):
    """
    Precision@k: is the expected verse in the top-k retrieved verses?
    For cross-verse questions (multiple expected verses), any match counts.
    Returns None if no expected verse is set (e.g. General questions).
    Each element of retrieved_verses may itself be a comma-separated string
    of verse refs (multi-verse chunk labels), so we flatten before comparing.
    """
    expected_set = set(v.strip() for v in expected_verse.split(",") if v.strip())
    if not expected_set:
        return None

    top_k_flat = _flatten_verses(retrieved_verses[:k])
    return 1.0 if expected_set & top_k_flat else 0.0


def compute_shloka_accuracy(retrieved_verses, expected_verse):
    """Is the expected verse anywhere in the full retrieved set?
    Handles multi-verse chunk labels (comma-separated) by flattening first.
    """
    expected_set = set(v.strip() for v in expected_verse.split(",") if v.strip())
    return 1.0 if expected_set & _flatten_verses(retrieved_verses) else 0.0


# ─────────────────────────────────────────────
# LLM JUDGES
# ─────────────────────────────────────────────

def _call_ollama(prompt):
    """Shared helper for LLM judge calls."""
    try:
        resp = requests.post(
            f"{config.OLLAMA_BASE_URL}/api/generate",
            json={
                "model": config.LLM_MODEL,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0.0, "num_predict": 10}
            },
            timeout=300
        )
        resp.raise_for_status()
        text = resp.json().get("response", "").strip()
        for char in text:
            if char.isdigit() and int(char) <= 3:
                return int(char)
        return None
    except Exception as e:
        print(f"    [LLM judge error] {e}")
        return None


def judge_faithfulness_llm(answer, expected_answer, question):
    """
    Faithfulness: does the generated answer agree with the expected answer?
    Measures answer quality — did we get the right information?

    Scale 0-3:
      0 = Completely wrong or irrelevant
      1 = Partially correct, missing key points
      2 = Mostly correct, minor inaccuracies
      3 = Fully faithful, covers all key points
    """
    prompt = f"""You are evaluating a Q&A system about the Bhagavad Gita.
Given the question, expected answer, and generated answer, rate faithfulness.

Question: {question}
Expected Answer: {expected_answer}
Generated Answer: {answer}

Rate faithfulness 0-3:
0 = Completely wrong or irrelevant
1 = Partially correct but missing key points
2 = Mostly correct with minor inaccuracies
3 = Fully faithful and covers the key points

Respond with ONLY the number (0, 1, 2, or 3)."""

    return _call_ollama(prompt)


def judge_groundedness_llm(answer, retrieved_chunks, question):
    """
    Groundedness: does the answer stay within what the retrieved chunks say?
    Measures hallucination — did the LLM add claims not in the context?

    Scale 0-3:
      0 = Major claims NOT in context (clear hallucination)
      1 = Mostly from context but has unsupported claims
      2 = Mostly grounded, only minor additions
      3 = Fully grounded — every claim traceable to context
    """
    # Build readable context from chunks
    context_parts = []
    for i, c in enumerate(retrieved_chunks):
        verse = c.get("verse_ref", "?")
        text  = c.get("text", c.get("document", "")).strip()
        if text:
            context_parts.append(f"[Chunk {i+1} — {verse}]:\n{text}")

    if not context_parts:
        return None  # nothing to judge against

    context = "\n\n".join(context_parts)

    prompt = f"""You are evaluating whether an AI answer is grounded in the provided context.
The answer should ONLY use information present in the retrieved context below.

Question: {question}

Retrieved Context:
{context}

Generated Answer: {answer}

Rate groundedness 0-3:
0 = Answer contains major claims NOT in context (hallucination)
1 = Mostly from context but has some unsupported claims
2 = Mostly grounded, only minor additions beyond context
3 = Fully grounded — every claim is traceable to the context above

Respond with ONLY the number (0, 1, 2, or 3)."""

    return _call_ollama(prompt)


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--questions-only", action="store_true",
                        help="Only evaluate retrieval metrics, skip LLM scoring")
    parser.add_argument("--no-ref", action="store_true",
                        help="Use question_en_no_ref (verse refs stripped) for honest eval")
    parser.add_argument("--output", default=os.path.join(config.RESULTS_DIR, "eval_results.json"))
    parser.add_argument("--lang", default="en", choices=["en", "kn"],
                        help="Which question language to use (ignored if --no-ref)")
    parser.add_argument("--questions-file", default=None,
                    help="Load questions from JSON instead of Excel")
    parser.add_argument("--include-xlsx", action="store_true",
                    help="Also include Excel questions when using --questions-file")
    args = parser.parse_args()

    import chromadb

    # ── Load questions ──────────────────────────────────────────────────
    questions = []
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
    print(f"Loaded {len(questions)} test questions")

    questions = load_test_questions(config.TEST_QUESTIONS_XLSX)
    print(f"Loaded {len(questions)} test questions")

    if args.no_ref:
        print("Mode: HONEST EVAL — using verse-ref-stripped questions (question_en_no_ref)")
    else:
        print(f"Mode: STANDARD EVAL — using question_en ({args.lang})")

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
        "total": len(questions),
        "precision_at_k":    {k: [] for k in config.EVAL_K_VALUES},
        "shloka_accuracy":   [],
        "faithfulness_scores":  [],
        "groundedness_scores":  [],
        "similarity_scores":    [],
        "by_type": defaultdict(lambda: {
            "precision_1": [], "precision_3": [], "precision_5": [],
            "shloka_acc": [], "faithfulness": [], "groundedness": [],
            "similarity": []
        })
    }

    # ── Evaluation loop ─────────────────────────────────────────────────
    for i, q in enumerate(questions):
        qid            = q["question_id"]
        expected_verse = q["expected_verse"]
        qtype          = q["question_type"]

        # Choose which question text to use
        if args.no_ref:
            no_ref_q = q.get("question_en_no_ref", "")
            if not no_ref_q or no_ref_q == "nan":
                # Fall back to standard if stripped version doesn't exist
                query = q["question_en"]
                print(f"  [WARN] No no-ref version for Q{qid}, using standard")
            else:
                query = no_ref_q
        elif args.lang == "kn":
            query = q.get("question_kn", q["question_en"])
        else:
            query = q["question_en"]

        print(f"[{i+1}/{len(questions)}] Q{qid} ({qtype}): {query[:80]}...")

        # ── Retrieve ──────────────────────────────────────────────────
        try:
            retrieved_chunks, answer = retrieve_for_eval(query, collection)
        except Exception as e:
            print(f"  ERROR during retrieval: {e}")
            results.append({"question_id": qid, "error": str(e)})
            continue

        retrieved_verses = [
            c["verse_ref"] for c in retrieved_chunks if c.get("verse_ref")
        ]

        # Similarity scores (if retrieve_lib returns them)
        similarities = [
            c.get("similarity") for c in retrieved_chunks
            if c.get("similarity") is not None
        ]
        avg_sim = round(sum(similarities) / len(similarities), 4) if similarities else None

        # ── Build result record ───────────────────────────────────────
        result = {
            "question_id":           qid,
            "question":              query,
            "question_type":         qtype,
            "expected_verse":        expected_verse,
            "retrieved_verses":      retrieved_verses,
            "answer":                answer,
            "retrieved_chunks_text": [
                c.get("text", c.get("document", "")) for c in retrieved_chunks
            ],
            "retrieval_similarities": similarities,
            "avg_similarity":         avg_sim,
        }

        if avg_sim is not None:
            metrics["similarity_scores"].append(avg_sim)
            metrics["by_type"][qtype]["similarity"].append(avg_sim)

        # ── Precision@k ───────────────────────────────────────────────
        for k in config.EVAL_K_VALUES:
            pk = compute_precision_at_k(retrieved_verses, expected_verse, k)
            result[f"precision_at_{k}"] = pk
            if pk is not None:
                metrics["precision_at_k"][k].append(pk)
                metrics["by_type"][qtype][f"precision_{k}"].append(pk)

        # ── Śloka accuracy ────────────────────────────────────────────
        sa = compute_shloka_accuracy(retrieved_verses, expected_verse)
        result["shloka_accuracy"] = sa
        metrics["shloka_accuracy"].append(sa)
        metrics["by_type"][qtype]["shloka_acc"].append(sa)

        # ── LLM judges (faithfulness + groundedness) ──────────────────
        if not args.questions_only and q.get("expected_answer_en"):

            # Faithfulness: answer vs expected_answer
            f_score = judge_faithfulness_llm(answer, q["expected_answer_en"], query)
            result["faithfulness"] = f_score
            if f_score is not None:
                metrics["faithfulness_scores"].append(f_score)
                metrics["by_type"][qtype]["faithfulness"].append(f_score)

            # Groundedness: answer vs retrieved_chunks
            g_score = judge_groundedness_llm(answer, retrieved_chunks, query)
            result["groundedness"] = g_score
            if g_score is not None:
                metrics["groundedness_scores"].append(g_score)
                metrics["by_type"][qtype]["groundedness"].append(g_score)

            time.sleep(0.5)  # avoid hammering Ollama

        results.append(result)

        # ── Per-question print ────────────────────────────────────────
        p1 = result.get("precision_at_1", "N/A")
        p3 = result.get("precision_at_3", "N/A")
        sim_str = f"{avg_sim:.3f}" if avg_sim is not None else "N/A"
        print(f"  Expected: {expected_verse} | Retrieved: {retrieved_verses[:3]}")
        print(f"  P@1={p1} | P@3={p3} | Śloka={sa} | AvgSim={sim_str}")
        if not args.questions_only:
            print(f"  Faithfulness={result.get('faithfulness','N/A')} | "
                  f"Groundedness={result.get('groundedness','N/A')}")

    # ── Aggregate summary ────────────────────────────────────────────────
    summary = {
        "total_questions": len(questions),
        "evaluated":       len(results),
        "eval_mode":       "no_ref" if args.no_ref else "standard",
    }

    for k in config.EVAL_K_VALUES:
        vals = metrics["precision_at_k"][k]
        summary[f"precision_at_{k}"] = round(sum(vals) / len(vals), 4) if vals else 0

    vals = metrics["shloka_accuracy"]
    summary["shloka_identification_accuracy"] = round(sum(vals) / len(vals), 4) if vals else 0

    vals = metrics["similarity_scores"]
    if vals:
        summary["avg_retrieval_similarity"] = round(sum(vals) / len(vals), 4)

    vals = metrics["faithfulness_scores"]
    if vals:
        summary["avg_faithfulness"] = round(sum(vals) / len(vals), 4)
        summary["faithfulness_distribution"] = {
            str(s): vals.count(s) for s in range(4)
        }

    vals = metrics["groundedness_scores"]
    if vals:
        summary["avg_groundedness"] = round(sum(vals) / len(vals), 4)
        summary["groundedness_distribution"] = {
            str(s): vals.count(s) for s in range(4)
        }

    # By question type
    summary["by_question_type"] = {}
    for qtype, tm in metrics["by_type"].items():
        ts = {}
        for metric_name, vals in tm.items():
            if vals:
                ts[metric_name] = round(sum(vals) / len(vals), 4)
        ts["count"] = len(tm.get("shloka_acc", []))
        summary["by_question_type"][qtype] = ts

    # ── Save ─────────────────────────────────────────────────────────────
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    output_data = {"summary": summary, "detailed_results": results}
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    # ── Print summary ─────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("EVALUATION SUMMARY")
    print(f"{'='*60}")
    print(f"Total questions : {summary['total_questions']}")
    print(f"Eval mode       : {summary['eval_mode']}")
    print()

    for k in config.EVAL_K_VALUES:
        print(f"Precision@{k}  : {summary.get(f'precision_at_{k}', 'N/A')}")
    print(f"Śloka Accuracy  : {summary.get('shloka_identification_accuracy', 'N/A')}")

    if "avg_retrieval_similarity" in summary:
        print(f"Avg Similarity  : {summary['avg_retrieval_similarity']}")
    if "avg_faithfulness" in summary:
        print(f"Avg Faithfulness: {summary['avg_faithfulness']} / 3.0")
        print(f"  Distribution  : {summary['faithfulness_distribution']}")
    if "avg_groundedness" in summary:
        print(f"Avg Groundedness: {summary['avg_groundedness']} / 3.0")
        print(f"  Distribution  : {summary['groundedness_distribution']}")

    print(f"\nBy Question Type:")
    for qtype, ts in summary.get("by_question_type", {}).items():
        print(f"  {qtype} (n={ts.get('count','?')}): "
              f"P@1={ts.get('precision_1','N/A')} | "
              f"Śloka={ts.get('shloka_acc','N/A')} | "
              f"Faith={ts.get('faithfulness','N/A')} | "
              f"Ground={ts.get('groundedness','N/A')}")

    print(f"\nResults saved to: {args.output}")

    # ── Paper table hint ──────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("PAPER TABLE (copy this):")
    print(f"{'='*60}")
    mode = "No-Ref" if args.no_ref else "Standard"
    print(f"| Metric              | {mode} |")
    print(f"|---------------------|---------|")
    for k in config.EVAL_K_VALUES:
        print(f"| Precision@{k}        | {summary.get(f'precision_at_{k}','—'):>7} |")
    print(f"| Śloka Accuracy      | {summary.get('shloka_identification_accuracy','—'):>7} |")
    if "avg_faithfulness" in summary:
        print(f"| Faithfulness (/3)   | {summary['avg_faithfulness']:>7} |")
    if "avg_groundedness" in summary:
        print(f"| Groundedness (/3)   | {summary['avg_groundedness']:>7} |")


if __name__ == "__main__":
    main()