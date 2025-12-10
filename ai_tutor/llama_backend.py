# ai_tutor/llama_backend.py

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Optional, Tuple

from llama_cpp import Llama

# These paths are relative to the project root where you run uvicorn
BASE_GGUF = Path("models/gguf/tinyllama-q4_0.gguf")
LORA_GGUF = Path("models/lora_gguf/tinyllama-tutor-lora-q8_0.gguf")


@lru_cache(maxsize=1)
def get_base_model() -> Llama:
    """
    Base TinyLlama (no LoRA) – used when use_finetuned=False.
    """
    if not BASE_GGUF.exists():
        raise RuntimeError(f"Base GGUF model not found at {BASE_GGUF}")

    llm = Llama(
        model_path=str(BASE_GGUF),
        n_ctx=1024,
        n_threads=2,  # adjust for your CPU / ACA cores
        logits_all=False,
        use_mmap=True,
        use_mlock=False,
        verbose=False,
    )
    return llm


@lru_cache(maxsize=1)
def get_finetuned_model() -> Llama:
    """
    TinyLlama + your tutor LoRA – used when use_finetuned=True.
    """
    if not BASE_GGUF.exists():
        raise RuntimeError(f"Base GGUF model not found at {BASE_GGUF}")
    if not LORA_GGUF.exists():
        raise RuntimeError(f"LoRA GGUF adapter not found at {LORA_GGUF}")

    llm = Llama(
        model_path=str(BASE_GGUF),
        lora_path=str(LORA_GGUF),
        n_ctx=1024,
        n_threads=2,
        logits_all=False,
        use_mmap=True,
        use_mlock=False,
        verbose=False,
    )
    return llm


def generate_llama_answer(
    question: str,
    context: Optional[str],
    use_finetuned: bool,
) -> Tuple[str, str]:
    """
    Unified entrypoint for the API.

    Returns:
        answer_text, model_type_string
    """

    if use_finetuned:
        llm = get_finetuned_model()
        model_type = "finetuned-llama-lora"

        # Enforce the 3-step tutor structure
        system = (
            "You are a patient programming tutor for beginners.\n"
            "For every answer, ALWAYS use this exact 3-part structure:\n"
            "1. Core idea: Explain the concept in 1–3 short sentences.\n"
            "2. Example: Give a short, clear Python code example.\n"
            "3. Common mistake: Describe one typical beginner mistake and how to avoid it.\n"
            "Do not add any other sections, introductions, or conclusions. "
            "Just the three numbered items in order."
        )
        temperature = 0.5
    else:
        llm = get_base_model()
        model_type = "base-llama"

        system = (
            "You are a helpful programming assistant. "
            "Answer clearly and concisely using simple language."
        )
        temperature = 0.3

    if context:
        prompt = (
            f"{system}\n\n"
            f"Relevant context:\n{context}\n\n"
            f"Student: {question}\n"
            "Tutor:"
        )
    else:
        prompt = (
            f"{system}\n\n"
            f"Student: {question}\n"
            "Tutor:"
        )

    result = llm(
        prompt,
        max_tokens=256,
        temperature=temperature,
        top_p=0.9,
        stop=["Student:", "Tutor:"],
        echo=False,
    )

    answer = result["choices"][0]["text"].strip()
    return answer, model_type
