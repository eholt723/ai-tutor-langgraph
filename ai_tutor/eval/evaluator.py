# ai_tutor/eval/evaluator.py

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List

from ai_tutor.data_utils import QAExample, load_eval_dataset
from ai_tutor.models.base_loader import load_base_model
from ai_tutor.models.lora_loader import load_finetuned_model
from ai_tutor.models.inference import generate_answer


@dataclass
class EvaluationResult:
    base_score: float
    finetuned_score: float
    num_samples: int


def simple_scoring(reference: str, prediction: str) -> float:

    reference = reference.lower()
    prediction = prediction.lower()

    return 1.0 if any(word in prediction for word in reference.split()[:3]) else 0.0


def run_evaluation(max_samples: int, output_path: Path) -> EvaluationResult:

    eval_data: List[QAExample] = load_eval_dataset(max_samples=max_samples)

    base_model, base_tokenizer = load_base_model()
    ft_model, ft_tokenizer = load_finetuned_model()

    base_scores = []
    ft_scores = []

    for example in eval_data:
        base_pred = generate_answer(base_model, base_tokenizer, example.question, example.context)
        ft_pred = generate_answer(ft_model, ft_tokenizer, example.question, example.context)

        base_scores.append(simple_scoring(example.answer, base_pred))
        ft_scores.append(simple_scoring(example.answer, ft_pred))

    result = EvaluationResult(
        base_score=sum(base_scores) / len(base_scores),
        finetuned_score=sum(ft_scores) / len(ft_scores),
        num_samples=len(eval_data),
    )

    output = {
        "base_score": result.base_score,
        "finetuned_score": result.finetuned_score,
        "num_samples": result.num_samples,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    return result
