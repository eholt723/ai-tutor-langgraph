# ai_tutor/graph/nodes/chat_node.py

from __future__ import annotations

from ai_tutor.graph.workflow import GraphState
from ai_tutor.models.inference import generate_answer
from ai_tutor.rag.retriever import retrieve_context


def chat_node(state: GraphState) -> GraphState:

    question = "What is a variable in programming?"

    # Base answer
    base_model = state.get("base_model")
    base_tokenizer = state.get("base_tokenizer")
    ft_model = state.get("ft_model")
    ft_tokenizer = state.get("ft_tokenizer")

    base_answer = generate_answer(base_model, base_tokenizer, question)

    # Fine-tuned answer (may be same initially)
    ft_answer = generate_answer(ft_model, ft_tokenizer, question)

    # RAG context
    contexts = retrieve_context(question, top_k=2)
    combined_context = "\n\n".join([f"{title}: {text}" for title, text in contexts]) if contexts else None

    rag_answer = generate_answer(ft_model, ft_tokenizer, question, combined_context)

    state["last_question"] = question
    state["last_answer_base"] = base_answer
    state["last_answer_finetuned"] = ft_answer
    state["last_answer_with_rag"] = rag_answer

    return state
