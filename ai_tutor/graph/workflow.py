# ai_tutor/graph/workflow.py

from __future__ import annotations

from langgraph.graph import StateGraph, END

from ai_tutor.graph.state import GraphState
from ai_tutor.graph.nodes import (
    load_config_node,
    prepare_data_node,
    load_models_node,
    evaluate_node,
    build_rag_index_node,
    chat_node,
)


def build_workflow_app():
    graph = StateGraph(GraphState)

    graph.add_node("load_config", load_config_node)
    graph.add_node("prepare_data", prepare_data_node)
    graph.add_node("load_models", load_models_node)
    graph.add_node("evaluate", evaluate_node)
    graph.add_node("build_rag_index", build_rag_index_node)
    graph.add_node("chat", chat_node)

    # Simple linear pipeline
    graph.set_entry_point("load_config")
    graph.add_edge("load_config", "prepare_data")
    graph.add_edge("prepare_data", "load_models")
    graph.add_edge("load_models", "evaluate")
    graph.add_edge("evaluate", "build_rag_index")
    graph.add_edge("build_rag_index", "chat")
    graph.add_edge("chat", END)

    return graph.compile()
