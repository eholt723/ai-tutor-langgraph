# ai_tutor/rag/ingest.py

from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass
class ReferenceDoc:
    """Simple reference document used for RAG."""
    id: str
    title: str
    content: str


def ingest_reference_corpus() -> List[ReferenceDoc]:

    docs = [
        ReferenceDoc(
            id="prog_basics",
            title="Programming Basics",
            content=(
                "A variable is a named storage location for data. "
                "Control structures like if statements and loops control the flow of a program."
            ),
        ),
        ReferenceDoc(
            id="loops_intro",
            title="Loops in Programming",
            content=(
                "For loops iterate a fixed number of times or over items in a collection. "
                "While loops repeat as long as a condition is true."
            ),
        ),
        ReferenceDoc(
            id="functions_intro",
            title="Functions",
            content=(
                "Functions are reusable blocks of code that perform a specific task. "
                "They can take input parameters and return a value."
            ),
        ),
    ]
    return docs
