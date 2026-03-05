# ai_tutor/graph/state.py

from __future__ import annotations

from typing import Optional, TypedDict


class GraphState(TypedDict, total=False):

    # Config info / metadata
    config_summary: str

    # Data preview
    data_preview: str

    # Evaluation
    eval_summary: str

    # RAG
    rag_status: str

    # Chat
    last_question: Optional[str]
    last_answer_base: Optional[str]
    last_answer_finetuned: Optional[str]
    last_answer_with_rag: Optional[str]
