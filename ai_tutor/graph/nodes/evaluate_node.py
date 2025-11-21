# ai_tutor/graph/nodes/evaluate_node.py

from __future__ import annotations

from pathlib import Path

from ai_tutor.config import Config
from ai_tutor.eval.evaluator import run_evaluation
from ai_tutor.graph.workflow import GraphState


def evaluate_node(state: GraphState) -> GraphState:

    output_path: Path = Config.eval_results_path
    result = run_evaluation(max_samples=10, output_path=output_path)

    summary = (
        f"Evaluation complete on {result.num_samples} samples.\n"
        f"Base score:       {result.base_score:.3f}\n"
        f"Fine-tuned score: {result.finetuned_score:.3f}\n"
        f"Results saved to: {output_path}"
    )

    state["eval_summary"] = summary
    return state
