"""
Step 5: Evaluation
==================
Runs the 74 gold test questions through the pipeline and measures:
  1. Retrieval Precision@k (k=1,3,5) — does the right verse come back?
  2. Śloka Identification Accuracy — is the correct verse reference cited?
  3. Answer Faithfulness — does the answer match the expected answer? (LLM-judged)

Input: test_questions.xlsx with columns:
  question_id, question_en, question_kn, expected_verse, expected_answer_en, question_type

Usage:
    python 05_evaluate.py
    python 05_evaluate.py --questions-only    # Just evaluate retrieval, skip LLM scoring
    python 05_evaluate.py --output results/eval_results.json
"""
import json
import os
import sys
import argparse
import time
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config

# Import retrieve function from step 4
from retrieve_lib import retrieve_for_eval


def load_test_questions(xlsx_path):
    """Load test questions from Excel file."""
    import pandas as pd
    df = pd.read_excel(xlsx_path)

    # Normalize column names
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    questions = []
    for _, row in df.iterrows():
        q = {
            "question_id": str(row.get("question_id", "")),
            "question_en": str(row.get("question_en", "")),
            "question_kn": str(row.get("question_kn", "")),
            "expected_verse": str(row.get("expected_verse", "")),
            "expected_answer_en": str(row.get("expected_answer_en", "")),
            "question_type": str(row.get("question_type", "")),
        }
        if q["question_en"] and q["question_en"] != "nan":
            questions.append(q)

    return questions


def compute_precision_at_k(retrieved_verses, expected_verse, k):
    """
    Precision@k: is the expected verse in the top-k retrieved verses?
    For cross-verse questions (multiple expected verses), check any match.
    """
    expected_set = set(v.strip() for v in expected_verse.split(",") if v.strip())
    top_k_verses = set(retrieved_verses[:k])

    if not expected_set:
        return None  # can't evaluate without expected verse

    hits = expected_set & top_k_verses
    return 1.0 if hits else 0.0


def compute_shloka_accuracy(retrieved_verses, expected_verse):
    """Is the expected verse in the retrieved set at all?"""
    expected_set = set(v.strip() for v in expected_verse.split(",") if v.strip())
    retrieved_set = set(retrieved_verses)
    return 1.0 if expected_set & retrieved_set else 0.0


def judge_faithfulness_llm(answer, expected_answer, question):
    """Use LLM to judge if the generated answer is faithful to the expected answer."""
    import requests

    prompt = f"""You are evaluating a Q&A system about the Bhagavad Gita.
Given the question, expected answer, and generated answer, rate the faithfulness.

Question: {question}
Expected Answer: {expected_answer}
Generated Answer: {answer}

Rate the faithfulness on a scale of 0-3:
0 = Completely wrong or irrelevant
1 = Partially correct but missing key points
2 = Mostly correct with minor inaccuracies
3 = Fully faithful and covers the key points

Respond with ONLY the number (0, 1, 2, or 3)."""

    try:
        resp = requests.post(
            f"{config.OLLAMA_BASE_URL}/api/generate",
            json={"model": config.LLM_MODEL, "prompt": prompt, "stream": False,
                  "options": {"temperature": 0.0, "num_predict": 10}},
            timeout=60
        )
        resp.raise_for_status()
        text = resp.json().get("response", "").strip()
        # Extract first digit
        for char in text:
            if char.isdigit() and int(char) <= 3:
                return int(char)
        return None
    except:
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--questions-only", action="store_true",
                       help="Only evaluate retrieval, skip LLM faithfulness scoring")
    parser.add_argument("--output", default=os.path.join(config.RESULTS_DIR, "eval_results.json"))
    parser.add_argument("--lang", default="en", choices=["en", "kn"],
                       help="Which question language to use")
    args = parser.parse_args()

    import chromadb

    # Load test questions
    if not os.path.exists(config.TEST_QUESTIONS_XLSX):
        print(f"ERROR: Test questions not found: {config.TEST_QUESTIONS_XLSX}")
        sys.exit(1)

    questions = load_test_questions(config.TEST_QUESTIONS_XLSX)
    print(f"Loaded {len(questions)} test questions")

    # Connect to ChromaDB
    client = chromadb.PersistentClient(path=config.CHROMA_DIR)
    try:
        collection = client.get_collection(config.COLLECTION_NAME)
    except:
        print("ERROR: ChromaDB collection not found. Run 03_index.py first.")
        sys.exit(1)

    print(f"ChromaDB collection: {collection.count()} documents")

    # Run evaluation
    results = []
    metrics = {
        "total": len(questions),
        "precision_at_k": {k: [] for k in config.EVAL_K_VALUES},
        "shloka_accuracy": [],
        "faithfulness_scores": [],
        "by_type": defaultdict(lambda: {"precision_1": [], "precision_3": [], "precision_5": [],
                                        "shloka_acc": [], "faithfulness": []})
    }

    for i, q in enumerate(questions):
        qid = q["question_id"]
        query = q["question_en"] if args.lang == "en" else q.get("question_kn", q["question_en"])
        expected_verse = q["expected_verse"]
        qtype = q["question_type"]

        print(f"\n[{i+1}/{len(questions)}] Q{qid} ({qtype}): {query[:80]}...")

        # Retrieve
        try:
            retrieved_chunks, answer = retrieve_for_eval(query, collection)
        except Exception as e:
            print(f"  ERROR: {e}")
            results.append({"question_id": qid, "error": str(e)})
            continue

        retrieved_verses = [c["verse_ref"] for c in retrieved_chunks if c.get("verse_ref")]

        # Compute metrics
        result = {
            "question_id": qid,
            "question": query,
            "question_type": qtype,
            "expected_verse": expected_verse,
            "retrieved_verses": retrieved_verses,
            "answer": answer,
        }

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

        # Faithfulness (LLM-judged)
        if not args.questions_only and q.get("expected_answer_en"):
            score = judge_faithfulness_llm(answer, q["expected_answer_en"], query)
            result["faithfulness"] = score
            if score is not None:
                metrics["faithfulness_scores"].append(score)
                metrics["by_type"][qtype]["faithfulness"].append(score)
            time.sleep(0.5)

        results.append(result)

        # Print summary for this question
        p1 = result.get("precision_at_1", "N/A")
        p3 = result.get("precision_at_3", "N/A")
        print(f"  Expected: {expected_verse} | Retrieved: {retrieved_verses[:3]}")
        print(f"  P@1={p1} | P@3={p3} | Shloka_Acc={sa}")

    # Compute aggregate metrics
    summary = {
        "total_questions": len(questions),
        "evaluated": len(results),
    }

    for k in config.EVAL_K_VALUES:
        vals = metrics["precision_at_k"][k]
        summary[f"precision_at_{k}"] = round(sum(vals) / len(vals), 4) if vals else 0

    vals = metrics["shloka_accuracy"]
    summary["shloka_identification_accuracy"] = round(sum(vals) / len(vals), 4) if vals else 0

    vals = metrics["faithfulness_scores"]
    if vals:
        summary["avg_faithfulness"] = round(sum(vals) / len(vals), 4)
        summary["faithfulness_distribution"] = {
            str(s): vals.count(s) for s in range(4)
        }

    # By question type
    summary["by_question_type"] = {}
    for qtype, type_metrics in metrics["by_type"].items():
        type_summary = {}
        for metric_name, vals in type_metrics.items():
            if vals:
                type_summary[metric_name] = round(sum(vals) / len(vals), 4)
        type_summary["count"] = len(type_metrics.get("shloka_acc", []))
        summary["by_question_type"][qtype] = type_summary

    # Save results
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    output = {"summary": summary, "detailed_results": results}
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    # Print summary
    print(f"\n{'='*60}")
    print("EVALUATION SUMMARY")
    print(f"{'='*60}")
    print(f"Total questions: {summary['total_questions']}")
    for k in config.EVAL_K_VALUES:
        print(f"Precision@{k}: {summary.get(f'precision_at_{k}', 'N/A')}")
    print(f"Śloka ID Accuracy: {summary.get('shloka_identification_accuracy', 'N/A')}")
    if "avg_faithfulness" in summary:
        print(f"Avg Faithfulness: {summary['avg_faithfulness']} / 3.0")

    print(f"\nBy Question Type:")
    for qtype, ts in summary.get("by_question_type", {}).items():
        print(f"  {qtype} (n={ts.get('count', '?')}): P@1={ts.get('precision_1', 'N/A')}, "
              f"Shloka={ts.get('shloka_acc', 'N/A')}")

    print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()
