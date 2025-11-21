# ai_tutor/graph/workflow.py

from __future__ import annotations

from typing import TypedDict, Optional, Any

from langgraph.graph import StateGraph, END

from ai_tutor.graph.nodes import (
    load_config_node,
    prepare_data_node,
    load_models_node,
    evaluate_node,
    build_rag_index_node,
    chat_node,
)


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

    # Internal caches / handles
    base_model: Any
    base_tokenizer: Any
    ft_model: Any
    ft_tokenizer: Any


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
