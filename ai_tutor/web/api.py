# ai_tutor/web/api.py

from __future__ import annotations

from typing import Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from ai_tutor.config import Config
from ai_tutor.eval.evaluator import run_evaluation
from ai_tutor.models.base_loader import load_base_model
from ai_tutor.models.lora_loader import load_finetuned_model
from ai_tutor.models.inference import generate_answer
from ai_tutor.rag.retriever import retrieve_context

app = FastAPI(title="AI Tutor LangGraph API")


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
    use_rag: bool = False


class ChatResponse(BaseModel):
    question: str
    answer: str
    model_type: str
    used_rag: bool
    context_preview: Optional[str] = None


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/run-eval")
def run_eval(max_samples: int = 10):
    result = run_evaluation(max_samples=max_samples, output_path=Config.eval_results_path)
    return {
        "base_score": result.base_score,
        "finetuned_score": result.finetuned_score,
        "num_samples": result.num_samples,
        "results_path": str(Config.eval_results_path),
    }


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    if req.use_finetuned:
        model, tokenizer = load_finetuned_model()
        model_type = "finetuned"
    else:
        model, tokenizer = load_base_model()
        model_type = "base"

    context_preview = None
    context = None
    if req.use_rag:
        contexts = retrieve_context(req.question, top_k=3)
        context_preview = "\n\n".join(
            [f"{title}: {text[:200]}..." for title, text in contexts]
        )
        context = "\n\n".join([text for _, text in contexts])

    # Finetuned -> tutor style prompt, base -> neutral prompt
    answer = generate_answer(
        model,
        tokenizer,
        req.question,
        context,
        tutor_style=req.use_finetuned,
    )

    return ChatResponse(
        question=req.question,
        answer=answer,
        model_type=model_type,
        used_rag=req.use_rag,
        context_preview=context_preview,
    )
