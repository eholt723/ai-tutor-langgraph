# ai_tutor/graph/nodes/load_config_node.py

from __future__ import annotations

from ai_tutor.config import Config
from ai_tutor.graph.workflow import GraphState


def load_config_node(state: GraphState) -> GraphState:
    summary_lines = [
        f"Project root: {Config.project_root}",
        f"Base model ID: {Config.base_model_id}",
        f"Base model path: {Config.base_model_path}",
        f"LoRA adapter path: {Config.lora_adapter_path}",
        f"RAG index path: {Config.rag_index_path}",
    ]
    state["config_summary"] = "\n".join(summary_lines)
    return state
