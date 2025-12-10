# AI Tutor Backend — TinyLlama + LoRA (Phase 1)  
### LangGraph-Powered Tutor Pipeline (Phase 2)

This project implements a lightweight AI Tutor backend capable of running efficiently on modest hardware and scaling seamlessly in the cloud.  
It features a custom fine-tuned **LoRA adapter**, a quantized **TinyLlama GGUF** model running through **llama.cpp**, and a clean **FastAPI** interface deployed on **Azure Container Apps**.

The system demonstrates ML engineering skills across fine-tuning, model conversion, CPU-optimized inference, Dockerization, and managed cloud deployment.

---

# Phase 1 — Current System (Fully Implemented)

Phase 1 focuses on delivering a **complete, deployable AI tutor backend** using a small but capable model.

## Core Features Implemented

### **Model**
- **TinyLlama-1.1B Chat** converted to **GGUF**
- **Custom LoRA fine-tuning** trained offline
- LoRA adapter converted to **GGUF** and applied dynamically in llama.cpp
- Supports:
  - **Base model mode**
  - **Fine-tuned tutor mode**

### **Inference**
- CPU-optimized inference using **llama.cpp**
- Fast startup (small model, quantized)
- Tutor-style answer generation in finetuned mode

### **Backend API**
Powered by **FastAPI**, exposing:

| Endpoint | Description |
|---------|-------------|
| `GET /health` | Health check |
| `POST /chat` | Main tutoring endpoint (base vs finetuned) |

# Phase 2 — LangGraph Workflow + RAG Pipeline (Coming Soon)

Phase 2 will expand the system from a simple model-inference backend into a full tutoring pipeline, demonstrating orchestration, retrieval, evaluation, and multi-step reasoning using **LangGraph**.

This phase will show how educational AI assistants can move beyond single-shot responses and adopt structured reasoning loops—improving clarity, grounding, and helpfulness.

---

## Planned Additions

### 1. LangGraph Tutor Workflow
A graph-based orchestration system that allows the model to execute multi-step reasoning:

- Intent classification  
- Decomposition of complex questions  
- Lookup of relevant examples or definitions  
- Draft-and-revise answer loops  
- Error-checking and correction passes  

The LangGraph workflow will wrap the model and RAG components into a predictable, testable pipeline.

---

### 2. Retrieval-Augmented Generation (RAG)
Add a lightweight retrieval layer to give the tutor access to curated reference material.

Planned features:

- Preprocessing a small set of educational texts (Python basics, CS fundamentals)  
- Embedding with `sentence-transformers`  
- Vector indexing using FAISS  
- Runtime retrieval inside the LangGraph workflow  
- Transparent inclusion of retrieved passages in the answer chain  

RAG will remain modular so the demo can easily compare:

- Model-only  
- Model + RAG  
- Model + RAG + LangGraph reasoning  

---

### 3. Expanded Evaluation Framework
Phase 2 adds:

- 20–50 curated evaluation questions  
- Automatic scoring for clarity and correctness  
- Side-by-side comparison of:
  - Base model  
  - Fine-tuned LoRA model  
  - LangGraph-enhanced model  
  - RAG-assisted responses  

---

### 4. Frontend Enhancements

- Toggle between Base / Fine-Tuned / LangGraph modes  
- Display of retrieved context when RAG is enabled  
- Model comparison interface for demo purposes  

---

### 5. Optional Extensions

- Voice input/output  
- Chat history persistence  
- Live telemetry (token usage, latency, retrieval hits)

---

## Phase 2 Goals

- A complete ML engineering workflow: fine-tuning → RAG → orchestration → evaluation  
- Real tutor-like behaviors (planning, checking its own work, improving answers)  
- A clean, cloud-deployable architecture  
- A compelling technical story for academic, professional, or portfolio presentation
