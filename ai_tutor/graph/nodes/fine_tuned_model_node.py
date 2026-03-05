# ai_tutor/graph/nodes/fine_tuned_model_node.py

from __future__ import annotations

from ai_tutor.graph.state import GraphState


def load_models_node(state: GraphState) -> GraphState:
    # llama_backend loads models lazily via @lru_cache on first generate call.
    # No explicit loading step needed here.
    return state
