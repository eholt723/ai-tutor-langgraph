# ai_tutor/graph/nodes/build_rag_index_node.py

from __future__ import annotations

from ai_tutor.rag.ingest import ingest_reference_corpus
from ai_tutor.rag.store import save_vector_store
from ai_tutor.graph.workflow import GraphState


def build_rag_index_node(state: GraphState) -> GraphState:

    docs = ingest_reference_corpus()
    vs = save_vector_store(docs, rebuild=False)

    state["rag_status"] = (
        f"RAG index ready with {len(docs)} docs "
        f"(model={vs.model_name})."
    )
    return state
