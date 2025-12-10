# ai_tutor/llama_backend.py

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Optional

from llama_cpp import Llama

from .prompts import build_prompt


BASE_GGUF = Path("models/gguf/tinyllama-q4_0.gguf")
LORA_GGUF = Path("models/lora_gguf/tinyllama-tutor-lora-q8_0.gguf")


@lru_cache(maxsize=1)
def get_base_model() -> Llama:
    if not BASE_GGUF.exists():
        raise RuntimeError(f"Base GGUF model not found at {BASE_GGUF}")

    return Llama(
        model_path=str(BASE_GGUF),
        n_ctx=1024,
        n_threads=2,
        logits_all=False,
        use_mmap=True,
        use_mlock=False,
        verbose=False,
    )


@lru_cache(maxsize=1)
def get_finetuned_model() -> Llama:
    if not BASE_GGUF.exists():
        raise RuntimeError(f"Base GGUF model not found at {BASE_GGUF}")
    if not LORA_GGUF.exists():
        raise RuntimeError(f"LoRA GGUF adapter not found at {LORA_GGUF}")

    return Llama(
        model_path=str(BASE_GGUF),
        lora_path=str(LORA_GGUF),
        n_ctx=1024,
        n_threads=2,
        logits_all=False,
        use_mmap=True,
        use_mlock=False,
        verbose=False,
    )


def generate_answer(
    question: str,
    use_finetuned: bool = False,
    context: Optional[str] = None,
    max_tokens: int = 384,
) -> tuple[str, str]:
    mode = "finetuned" if use_finetuned else "base"
    model = get_finetuned_model() if use_finetuned else get_base_model()

    prompt = build_prompt(question=question, mode=mode, context=context)

    temperature = 0.5 if use_finetuned else 0.7

    output = model(
        prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=0.9,
        repeat_penalty=1.1,
        stop=["</s>", "[/INST]"],
    )

    text = output["choices"][0]["text"].strip()

    # >>> THIS IS THE DEBUG WIRING <<<
    if use_finetuned:
        text = "DEBUG-FINETUNED-ACTIVE\n\n" + text

    model_type = "finetuned-llama-lora" if use_finetuned else "base-llama"
    return text, model_type
