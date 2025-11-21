
from __future__ import annotations

import argparse
from pathlib import Path

from ai_tutor.config import Config
from ai_tutor.eval.evaluator import EvaluationResult, run_evaluation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run evaluation for base vs fine-tuned models.")
    parser.add_argument(
        "--output-path",
        type=str,
        default=str(Config.eval_results_path),
        help="Path to write eval_results.json.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=20,
        help="Maximum number of evaluation samples.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print("=== Evaluation Script ===")
    print(f"Eval results path: {output_path}")
    print(f"Max samples:       {args.max_samples}")
    print()

    result: EvaluationResult = run_evaluation(
        max_samples=args.max_samples,
        output_path=output_path,
    )

    print("Evaluation complete.")
    print(f"Base model score:       {result.base_score:.4f}")
    print(f"Fine-tuned model score: {result.finetuned_score:.4f}")
    print(f"Num samples:            {result.num_samples}")


if __name__ == "__main__":
    main()
