# scripts/run_eval.py

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Dict, Any

import requests  # make sure 'requests' is installed in your venv

from ai_tutor.config import Config
from ai_tutor.data_utils import load_eval_dataset, QAExample


def simple_score(gold: str, pred: str) -> float:
    """
    Lexical overlap scoring focused on content words.

    - 1.0 if >= 40% of content words in the gold answer appear in the prediction.
    - 0.5 if >= 20% overlap.
    - 0.0 otherwise.
    """

    if not pred:
        return 0.0

    import re

    def tokenize(text: str) -> list[str]:
        text = text.lower()
        # Replace basic punctuation with spaces
        text = re.sub(r"[.,;:!?()\[\]\"']", " ", text)
        return [w for w in text.split() if w]

    # Very small stopword list is enough for this use
    STOPWORDS = {
        "a", "an", "the", "is", "are", "was", "were",
        "it", "that", "this", "of", "to", "in", "for",
        "and", "or", "as", "on", "at", "by", "from",
        "with", "you", "your", "their", "its", "be",
        "can", "will", "when", "while", "if", "then",
    }

    gold_tokens = [w for w in tokenize(gold) if w not in STOPWORDS]
    pred_tokens = tokenize(pred)

    if not gold_tokens or not pred_tokens:
        return 0.0

    gold_set = set(gold_tokens)
    pred_set = set(pred_tokens)

    overlap = gold_set & pred_set
    overlap_ratio = len(overlap) / len(gold_set)

    if overlap_ratio >= 0.4:
        return 1.0
    if overlap_ratio >= 0.2:
        return 0.5
    return 0.0


def score_with_tutor_style(gold: str, pred: str) -> float:
    """
    Wraps simple_score() and adds small bonuses for the tutor signature:
    - +0.25 if the answer includes an example
    - +0.25 if the answer mentions a common mistake / warning
    """
    base = simple_score(gold, pred)
    ans = (pred or "").lower()

    example_bonus = 0.0
    mistake_bonus = 0.0

    # Heuristics for examples
    if (
        "for example" in ans
        or "for instance" in ans
        or "e.g." in ans
        or "example:" in ans
    ):
        example_bonus = 0.25

    # Heuristics for common mistakes / warnings
    if (
        "common mistake" in ans
        or "often confused" in ans
        or "be careful" in ans
        or "frequent bug" in ans
        or "a common bug" in ans
    ):
        mistake_bonus = 0.25

    return min(1.0, base + example_bonus + mistake_bonus)


def call_chat_api(question: str, use_finetuned: bool) -> str:
    """
    Call the FastAPI /chat endpoint and return the 'answer' string.
    We always disable RAG for eval so we are just measuring the generator.
    """
    url = f"http://{Config.api_host}:{Config.api_port}/chat"
    payload = {
        "question": question,
        "use_finetuned": use_finetuned,
        "use_rag": False,
    }

    try:
        resp = requests.post(url, json=payload, timeout=300)
    except Exception as e:
        raise RuntimeError(f"Error calling /chat API: {e}") from e

    if resp.status_code != 200:
        raise RuntimeError(f"/chat returned status {resp.status_code}: {resp.text}")

    data = resp.json()
    # Expected shape: {"question": "...", "answer": "...", "model_type": "...", "used_rag": false, ...}
    return data.get("answer", "")


def run_eval(max_samples: int | None = None) -> Dict[str, Any]:
    print("=== Evaluation Script ===")
    eval_path: Path = Config.eval_results_path
    print(f"Eval results path: {eval_path}")
    print(f"Max samples:       {max_samples if max_samples is not None else 'ALL'}\n")

    # Load eval examples from data/val/val.jsonl
    examples: List[QAExample] = load_eval_dataset(max_samples=max_samples)
    if not examples:
        print("No evaluation examples found. Exiting.")
        return {}

    rows: List[Dict[str, Any]] = []
    base_total = 0.0
    ft_total = 0.0

    for idx, ex in enumerate(examples, start=1):
        print(f"[{idx}/{len(examples)}] Q: {ex.question}")

        # Base model (no finetune, no RAG)
        base_answer = call_chat_api(ex.question, use_finetuned=False)

        # Finetuned model (no RAG)
        ft_answer = call_chat_api(ex.question, use_finetuned=True)

        # Use tutor-style scoring (simple_score + bonuses)
        base_score = score_with_tutor_style(ex.answer, base_answer)
        ft_score = score_with_tutor_style(ex.answer, ft_answer)

        base_total += base_score
        ft_total += ft_score

        rows.append(
            {
                "question": ex.question,
                "gold_answer": ex.answer,
                "base_answer": base_answer,
                "finetuned_answer": ft_answer,
                "base_score": float(base_score),
                "finetuned_score": float(ft_score),
            }
        )

    num_samples = len(examples)
    base_avg = base_total / num_samples
    ft_avg = ft_total / num_samples

    results: Dict[str, Any] = {
        "num_samples": num_samples,
        "base_score": base_avg,
        "finetuned_score": ft_avg,
        "results": rows,
    }

    # Ensure directory exists
    eval_path.parent.mkdir(parents=True, exist_ok=True)
    eval_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    print("\nEvaluation complete.")
    print(f"Base model score:       {base_avg:.4f}")
    print(f"Fine-tuned model score: {ft_avg:.4f}")
    print(f"Num samples:            {num_samples}")

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Run evaluation for AI Tutor models via the /chat API.")
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum number of eval samples to use (default: all)",
    )
    args = parser.parse_args()

    run_eval(max_samples=args.max_samples)


if __name__ == "__main__":
    main()
