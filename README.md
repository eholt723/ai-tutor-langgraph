# AI Tutor Backend — TinyLlama + LoRA (Phase 1)  
### with Roadmap for LangGraph-Powered Tutor Pipeline (Phase 2)

This project implements a lightweight AI Tutor backend capable of running efficiently on modest hardware and scaling seamlessly in the cloud.  
It features a custom fine-tuned **LoRA adapter**, a quantized **TinyLlama GGUF** model running through **llama.cpp**, and a clean **FastAPI** interface deployed on **Azure Container Apps**.

The system is designed to demonstrate real ML engineering skills across fine-tuning, model conversion, CPU-optimized inference, Dockerization, and managed cloud deployment.

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

**Request fields:**
```json
{
  "question": "What is a variable?",
  "use_finetuned": true
}
