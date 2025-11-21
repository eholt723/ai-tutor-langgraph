# ai_tutor/graph/nodes/fine_tuned_model_node.py

from __future__ import annotations

from ai_tutor.models.base_loader import load_base_model
from ai_tutor.models.lora_loader import load_finetuned_model
from ai_tutor.graph.workflow import GraphState


def load_models_node(state: GraphState) -> GraphState:

    base_model, base_tokenizer = load_base_model()
    ft_model, ft_tokenizer = load_finetuned_model()

    state["base_model"] = base_model
    state["base_tokenizer"] = base_tokenizer
    state["ft_model"] = ft_model
    state["ft_tokenizer"] = ft_tokenizer

    return state
