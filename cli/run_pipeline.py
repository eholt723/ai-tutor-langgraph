# cli/run_pipeline.py

from __future__ import annotations

from ai_tutor.graph.workflow import build_workflow_app, GraphState


def main() -> None:
    print("=== AI Tutor Pipeline (LangGraph) ===")

    app = build_workflow_app()
    initial_state: GraphState = {}

    final_state = app.invoke(initial_state)

    print("\n[Config]")
    print(final_state.get("config_summary", "(no config summary)"))

    print("\n[Data Preview]")
    print(final_state.get("data_preview", "(no data preview)"))

    print("\n[Evaluation]")
    print(final_state.get("eval_summary", "(no eval summary)"))

    print("\n[RAG Status]")
    print(final_state.get("rag_status", "(no RAG status)"))

    print("\n[Chat Demo]")
    print(f"Question: {final_state.get('last_question', '(none)')}\n")
    print("Base model answer:")
    print(final_state.get("last_answer_base", "(no answer)"))
    print("\nFine-tuned model answer:")
    print(final_state.get("last_answer_finetuned", "(no answer)"))
    print("\nFine-tuned + RAG answer:")
    print(final_state.get("last_answer_with_rag", "(no answer)"))


if __name__ == "__main__":
    main()
