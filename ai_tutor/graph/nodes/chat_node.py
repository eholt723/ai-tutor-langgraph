# ai_tutor/graph/nodes/chat_node.py

from __future__ import annotations

from ai_tutor.graph.state import GraphState
from ai_tutor.llama_backend import generate_answer
from ai_tutor.rag.retriever import retrieve_context


def chat_node(state: GraphState) -> GraphState:

    question = "What is a variable in programming?"

    base_answer, _ = generate_answer(question, use_finetuned=False)
    ft_answer, _ = generate_answer(question, use_finetuned=True)

    contexts = retrieve_context(question, top_k=2)
    combined_context = (
        "\n\n".join([f"{title}: {text}" for title, text in contexts])
        if contexts
        else None
    )
    rag_answer, _ = generate_answer(question, use_finetuned=True, context=combined_context)

    state["last_question"] = question
    state["last_answer_base"] = base_answer
    state["last_answer_finetuned"] = ft_answer
    state["last_answer_with_rag"] = rag_answer

    return state
