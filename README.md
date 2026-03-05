# AI Tutor — TinyLlama + LoRA + LangGraph + RAG

A lightweight AI tutor backend capable of running on modest hardware and deploying to the cloud. Built as a full end-to-end ML engineering project covering fine-tuning, quantized inference, retrieval-augmented generation, LangGraph orchestration, and evaluation.

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
| Deployment | Azure Container Apps (Docker) |

---

## Phase 1 — FastAPI + llama.cpp Backend (Complete)

- TinyLlama-1.1B-Chat converted to GGUF and quantized
- Custom LoRA adapter trained offline with QLoRA and applied at runtime via llama.cpp
- Two inference modes: base model and fine-tuned tutor mode
- Structured 3-part tutor answers: Core Idea / Step-by-Step Example / Common Mistake
- FastAPI backend with `/health` and `/chat` endpoints
- CORS configured for GitHub Pages frontend
- Dockerized and deployable to Azure Container Apps

### API

| Endpoint | Description |
|----------|-------------|
| `GET /health` | Health check |
| `POST /chat` | Tutoring endpoint — base or fine-tuned mode, optional RAG, optional prompt debug |

---

## Phase 2 — LangGraph Orchestration + RAG (Implemented)

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
- Retrieval integrated into the `/chat` endpoint via `use_rag` flag
- Index built with: `scripts/build_rag_index.py [--rebuild]`

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
  eval/                  # (eval handled via scripts/run_eval.py)

scripts/
  fine_tune_qlora.py     # Offline QLoRA fine-tuning
  build_rag_index.py     # Build/rebuild the RAG vector index
  run_eval.py            # Evaluation against live API

cli/
  run_pipeline.py        # Run the full LangGraph pipeline end-to-end

data/
  train/train.jsonl      # Training data
  val/val.jsonl          # Evaluation data

docs/
  index.html             # GitHub Pages frontend
```

---

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env   # set BASE_MODEL_PATH, LORA_ADAPTER_PATH, etc.
```

## Run

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
