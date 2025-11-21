# ai_tutor/data_utils.py

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any

from ai_tutor.config import Config


@dataclass
class QAExample:
    """Simple representation of a question-answer example for the tutor."""

    question: str
    context: str | None
    answer: str


def load_training_dataset() -> List[QAExample]:

    examples = [
        QAExample(
            question="What is a variable in programming?",
            context=None,
            answer="A variable is a named storage location for data that can change during program execution.",
        ),
        QAExample(
            question="What does a 'for loop' do?",
            context=None,
            answer="A for loop repeats a block of code for each item in a sequence or for a specific number of iterations.",
        ),
    ]
    return examples


def load_eval_dataset(max_samples: int | None = None) -> List[QAExample]:

    examples = load_training_dataset()
    if max_samples is not None:
        examples = examples[:max_samples]
    return examples


def prepare_all_splits() -> None:

    base = Config.data_dir
    for sub in ["raw", "processed", "train", "val", "test", "live_eval"]:
        path = base / sub
        path.mkdir(parents=True, exist_ok=True)

    print(f"Prepared data folder structure under: {Config.data_dir}")


def as_dict(example: QAExample) -> Dict[str, Any]:
    """Utility to convert QAExample to a plain dict."""
    return {
        "question": example.question,
        "context": example.context,
        "answer": example.answer,
    }
