# ai_tutor/graph/nodes/evaluate_node.py

from __future__ import annotations

from ai_tutor.graph.state import GraphState


def evaluate_node(state: GraphState) -> GraphState:
    # Full evaluation requires the FastAPI server to be running.
    # Run: .venv/bin/python scripts/run_eval.py
    state["eval_summary"] = (
        "Evaluation skipped in pipeline. "
        "Run scripts/run_eval.py against a live server for full eval results."
    )
    return state
