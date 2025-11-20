# AI Tutor LangGraph Pipeline

This project implements an end-to-end AI Tutor workflow featuring fine-tuning, evaluation, RAG, and a LangGraph-based orchestration layer. The system is designed to demonstrate a complete ML pipeline during a live demo while keeping all heavy computations offline and pre-processed.

The workflow includes:
- Preparing a small public educational dataset.
- Fine-tuning a small model using QLoRA (performed offline once).
- Loading base and fine-tuned weights for comparison.
- Running evaluation on a tiny test set for live demonstration.
- Building and loading a RAG index from reference materials.
- A LangGraph workflow that stitches together all pipeline steps.
- Optional lightweight web UI and a terminal-based demo.

The project is designed to run locally on modest hardware and remain front-end light enough to work on older machines.

## Goals

1. Demonstrate an applied ML workflow: fine-tuning, evaluation, and retrieval.
2. Provide a reproducible local inference and RAG pipeline.
3. Use LangGraph to make the workflow modular and observable.
4. Offer both a terminal demo and an optional web UI for interviews.
5. Keep the frontend minimal and mobile-friendly.

## Repository Structure (High Level)

# ai-tutor-langgraph
