# AI Tutor – TinyLlama + LoRA Backend (with optional LangGraph Pipeline)

This project implements a lightweight AI Tutor system capable of running entirely on modest hardware while also supporting cloud deployment.  
The system includes fine-tuning (LoRA), evaluation utilities, and a FastAPI backend powered by **TinyLlama running through llama.cpp (GGUF)**.

A quantized TinyLlama model and a matching LoRA adapter power the *tutor-style* responses.  
The backend is deployed using **Azure Container Apps**, which provides auto-scale-to-zero for near-free hosting on a student subscription.

---

## 🔍 Current Phase (Live Demo Architecture)

The deployed live system uses:

- **TinyLlama-1.1B Chat (GGUF format)**
- **Custom fine-tuned LoRA adapter** (converted to GGUF)
- **llama.cpp inference backend** for fast CPU-only generation
- **FastAPI** service providing:
  - `/chat`
  - `/health`
- Deployment on **Azure Container Apps**
- Comparison mode: **base vs fine-tuned tutor**

No LangGraph workflow is active in Phase 1.  
RAG is reserved for Phase 2.

**Phase 1 inference flow:**

