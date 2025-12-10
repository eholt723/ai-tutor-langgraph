from typing import Optional

from fastapi import FastAPI
from pydantic import BaseModel

from ai_tutor.llama_backend import generate_answer
from ai_tutor.prompts import build_prompt  # for prompt_debug


app = FastAPI()


class ChatRequest(BaseModel):
    question: str
    use_finetuned: bool = False
    use_rag: bool = False  # ignored for now
    debug_prompt: bool = False  # NEW: ask API to return the full prompt


class ChatResponse(BaseModel):
    question: str
    answer: str
    model_type: str
    used_rag: bool
    context_preview: Optional[str] = None
    prompt_debug: Optional[str] = None  # NEW: echoes the prompt when requested


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    # Phase 1: RAG is off, but the flag is kept for later
    context: Optional[str] = None

    # Build the prompt explicitly so we can optionally return it
    mode = "finetuned" if req.use_finetuned else "base"
    prompt = build_prompt(
        question=req.question,
        mode=mode,
        context=context,
    )

    # Core generation path
    answer, model_type = generate_answer(
        question=req.question,
        use_finetuned=req.use_finetuned,
        context=context,
    )

    return ChatResponse(
        question=req.question,
        answer=answer,
        model_type=model_type,
        used_rag=False,
        context_preview=context,
        prompt_debug=prompt if req.debug_prompt else None,
    )
