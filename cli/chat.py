# cli/chat.py

from __future__ import annotations

from ai_tutor.models.base_loader import load_base_model
from ai_tutor.models.lora_loader import load_finetuned_model
from ai_tutor.models.inference import generate_answer
from ai_tutor.rag.retriever import retrieve_context


def main() -> None:
    print("=== AI Tutor Chat ===")
    print("Type 'exit' to quit.\n")

    base_model, base_tokenizer = load_base_model()
    ft_model, ft_tokenizer = load_finetuned_model()

    while True:
        question = input("You: ").strip()
        if not question:
            continue
        if question.lower() in {"exit", "quit"}:
            break

        # Base answer
        base_ans = generate_answer(base_model, base_tokenizer, question)

        # Fine-tuned answer
        ft_ans = generate_answer(ft_model, ft_tokenizer, question)

        # RAG answer
        contexts = retrieve_context(question, top_k=3)
        context_text = "\n\n".join([text for _, text in contexts]) if contexts else None
        rag_ans = generate_answer(ft_model, ft_tokenizer, question, context_text)

        print("\n[Base model]")
        print(base_ans)
        print("\n[Fine-tuned model]")
        print(ft_ans)
        print("\n[Fine-tuned + RAG]")
        print(rag_ans)
        print("\n" + "-" * 40 + "\n")


if __name__ == "__main__":
    main()
