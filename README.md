---
title: AI Tutor
emoji: 🎓
colorFrom: indigo
colorTo: blue
sdk: docker
pinned: false
---

# AI Tutor — TinyLlama + LoRA + LangGraph + RAG

A lightweight AI tutor backend built as a full end-to-end ML engineering project covering fine-tuning, quantized inference, retrieval-augmented generation, LangGraph orchestration, and evaluation. Runs on modest hardware, deploys to the cloud.

---

## Stack

| Layer | Technology |
|-------|-----------|
| Model | TinyLlama-1.1B-Chat (GGUF, quantized) |
| Fine-tuning | QLoRA (offline, `scripts/fine_tune_qlora.py`) |
| Inference | llama.cpp via `llama-cpp-python` |
| RAG | sentence-transformers + cosine similarity (custom FAISS-free store) |
| Orchestration | LangGraph |
| API | FastAPI |
| Frontend | GitHub Pages (`docs/index.html`) |
| Deployment | Hugging Face Spaces (Docker) |
| Model hosting | Hugging Face Hub (`eholt723/ai-tutor-models`) |

---

## Phase 1 — FastAPI + llama.cpp Backend (Complete)

- TinyLlama-1.1B-Chat converted to GGUF and quantized
- Custom LoRA adapter trained offline with QLoRA and applied at runtime via llama.cpp
- Two inference modes side-by-side: base model vs fine-tuned tutor mode
- Structured 3-part tutor answers: Core Idea / Step-by-Step Example / Common Mistake
- FastAPI backend with `/health` and `/chat` endpoints
- CORS configured for GitHub Pages and HF Space frontend origins
- Dockerized and deployed to Hugging Face Spaces

### API

| Endpoint | Description |
|----------|-------------|
| `GET /health` | Health check |
| `POST /chat` | Tutoring endpoint — base or fine-tuned mode, optional prompt debug |

---

## Phase 2 — LangGraph Orchestration + RAG (In Progress)

### LangGraph Pipeline (`cli/run_pipeline.py`)

Linear workflow demonstrating multi-step orchestration:

```
load_config → prepare_data → load_models → evaluate → build_rag_index → chat
```

Each step is an isolated LangGraph node passing typed state.

### RAG

- Reference corpus ingested via `ai_tutor/rag/ingest.py`
- Embedded with `sentence-transformers/all-MiniLM-L6-v2`
- Stored as a cosine-similarity vector store (no FAISS dependency)
- Index built with: `scripts/build_rag_index.py [--rebuild]`
- Live `/chat` RAG injection is in progress

### Evaluation

API-based evaluation via `scripts/run_eval.py`:

- Loads examples from `data/val/val.jsonl`
- Calls the live `/chat` endpoint for base and fine-tuned answers
- Lexical overlap scoring + tutor-style bonuses (example present, mistake mentioned)
- Results saved to `artifacts/eval/eval_results.json`

```bash
# Start server first, then:
.venv/bin/python scripts/run_eval.py --max-samples 20
```

---

## Tests

128 tests across 9 files — run with:

```bash
.venv/bin/python -m pytest tests/ -v
```

Coverage includes:

| Area | What's tested |
|------|---------------|
| API endpoints | Happy path, input validation, CORS allow/deny |
| llama backend | Normalization, contraction expansion (parametrized), restructuring, meta stripping |
| RAG | Cosine similarity math, retriever integration (fixture-based, no model download) |
| LangGraph nodes | Config, data, model loader, evaluate nodes |
| Prompts | Base and fine-tuned prompt builders |
| Scoring | Lexical overlap scorer, tutor-style bonus logic |
| Config | Env-driven config loading |

---

## Project Structure

```
ai_tutor/
  config.py              # Central config (env-driven)
  llama_backend.py       # llama.cpp inference + prompt cleanup
  prompts.py             # Prompt templates (base and finetuned modes)
  data_utils.py          # QAExample dataclass, dataset loaders
  web/api.py             # FastAPI app
  rag/
    ingest.py            # Reference corpus loader
    store.py             # Vector store build/save/load
    retriever.py         # Cosine similarity retrieval
  graph/
    workflow.py          # LangGraph StateGraph definition
    nodes/               # One file per pipeline node

scripts/
  fine_tune_qlora.py     # Offline QLoRA fine-tuning
  build_rag_index.py     # Build/rebuild the RAG vector index
  run_eval.py            # Evaluation against live API

cli/
  run_pipeline.py        # Run the full LangGraph pipeline end-to-end

tests/
  conftest.py            # Shared fixtures (api_client, vector store, mocks)
  test_api.py
  test_llama_backend.py
  test_rag_retriever.py
  ...

data/
  train/train.jsonl      # Training data
  val/val.jsonl          # Evaluation data

docs/
  index.html             # GitHub Pages frontend
```

---

## Local Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Models are not included in the repo. For local inference, download them from
[eholt723/ai-tutor-models](https://huggingface.co/eholt723/ai-tutor-models) and place at:

```
models/gguf/tinyllama-q4_0.gguf
models/lora_gguf/tinyllama-tutor-lora-q8_0.gguf
```

## Run Locally

```bash
# Start API server
.venv/bin/python -m uvicorn ai_tutor.web.api:app --host 0.0.0.0 --port 8000

# Build RAG index
.venv/bin/python scripts/build_rag_index.py

# Run full LangGraph pipeline
.venv/bin/python cli/run_pipeline.py

# Run evaluation (server must be running)
.venv/bin/python scripts/run_eval.py
```

## Deploy (Hugging Face Spaces)

Models are hosted on HF Hub and downloaded automatically at container startup.

```bash
# Push to HF Space
git push hf main
```

The Space builds from the `Dockerfile` and runs `startup.sh`, which pulls both
GGUF files from `eholt723/ai-tutor-models` before starting uvicorn on port 7860.
