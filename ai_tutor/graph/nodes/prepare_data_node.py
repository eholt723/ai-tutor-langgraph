# ai_tutor/graph/nodes/prepare_data_node.py

from __future__ import annotations

from ai_tutor.data_utils import load_training_dataset
from ai_tutor.graph.workflow import GraphState


def prepare_data_node(state: GraphState) -> GraphState:
    examples = load_training_dataset()
    preview_count = min(3, len(examples))
    preview_lines = []

    for i, ex in enumerate(examples[:preview_count], start=1):
        preview_lines.append(f"{i}. Q: {ex.question}")
        preview_lines.append(f"   A: {ex.answer}")

    state["data_preview"] = "\n".join(preview_lines) if preview_lines else "No examples loaded."
    return state
