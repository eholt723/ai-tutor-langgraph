# ai_tutor/web/api.py

from __future__ import annotations

from typing import Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from ai_tutor.llama_backend import generate_llama_answer

app = FastAPI(title="AI Tutor TinyLlama Backend")


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    question: str
    use_finetuned: bool = True
    use_rag: bool = False  # kept for compatibility with the frontend


class ChatResponse(BaseModel):
    question: str
    answer: str
    model_type: str
    used_rag: bool
    context_preview: Optional[str] = None


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    # All logic is in llama_backend; this just routes the request.
    answer = generate_llama_answer(
        question=req.question,
        use_finetuned=req.use_finetuned,
    )

    model_type = "finetuned-llama-lora" if req.use_finetuned else "base-llama"

    return ChatResponse(
        question=req.question,
        answer=answer,
        model_type=model_type,
        used_rag=False,          # RAG is Phase 2
        context_preview=None,    # kept for schema compatibility
    )
