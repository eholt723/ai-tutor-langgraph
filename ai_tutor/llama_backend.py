# ai_tutor/llama_backend.py

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Optional, Tuple
import re

from llama_cpp import Llama

from .prompts import build_prompt


BASE_GGUF = Path("models/gguf/tinyllama-q4_0.gguf")
LORA_GGUF = Path("models/lora_gguf/tinyllama-tutor-lora-q8_0.gguf")


# -------------------------------------------------------------------
# Model loaders
# -------------------------------------------------------------------


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


# -------------------------------------------------------------------
# Finetuned cleaning + restructuring ONLY
# -------------------------------------------------------------------


def _strip_meta(text: str) -> str:
    """Remove obvious prompt-echo junk from the finetuned output."""
    t = text.strip()

    # Cut off any trailing "Student question:" echo
    sq_idx = t.lower().find("student question:")
    if sq_idx != -1:
        t = t[:sq_idx].strip()

    remove_anywhere = [
        "<<USER>>",
        "Tutor answer:",
        "Student answer:",
        "<<SYS>>",
        "Write your answer now.",
        "Use plain language that students can understand without Google.",
        "Always use plain language that students can understand without Google.",
        "appropriate for a first programming course.",
        "Talk directly to the student and avoid using emojis or abbreviations.",
        "Talk directly to the student and keep paragraphs short.",
    ]

    for phrase in remove_anywhere:
        t = t.replace(phrase, "").strip()

    lines = [ln for ln in t.splitlines() if ln.strip()]
    return "\n".join(lines).strip()


def _restructure_finetuned(text: str) -> str:
    """
    Try to enforce the 1/2/3 tutor structure on the finetuned answer.

    We look for segments starting with "1.", "2.", "3." and, if found,
    rebuild them under the standard headings.
    """
    pattern = r"(?:^|\n)([123]\.)\s"
    matches = list(re.finditer(pattern, text))
    if len(matches) < 3:
        return text.strip()

    chunks: dict[str, str] = {}
    for i, m in enumerate(matches):
        num = m.group(1)[0]  # "1", "2", or "3"
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        chunk = text[start:end].strip()
        chunks[num] = chunk

    core = chunks.get("1", "").strip()
    example = chunks.get("2", "").strip()
    mistake = chunks.get("3", "").strip()

    parts = []
    if core:
        parts.append(f"1. Core Idea\n{core}")
    if example:
        parts.append(f"2. Step-by-Step Example\n{example}")
    if mistake:
        parts.append(
            "3. Common Mistake + Check-Your-Understanding Question\n" + mistake
        )

    if not parts:
        return text.strip()

    return "\n\n".join(parts).strip()


# -------------------------------------------------------------------
# Main generation
# -------------------------------------------------------------------


from typing import Tuple, Optional

def generate_answer(
    question: str,
    use_finetuned: bool = False,
    context: Optional[str] = None,
    max_tokens: int = 384,
) -> Tuple[str, str]:
    """
    Core generation entry point used by the FastAPI /chat endpoint.

    - Finetuned: structured tutor prompt + cleanup + 1/2/3 restructuring.
    - Base: very simple Q&A prompt, no cleanup or constraints.
    """

    if use_finetuned:
        # ---------- FINETUNED PATH ----------
        model = get_finetuned_model()
        prompt = build_prompt(question=question, mode="finetuned", context=context)

        output = model(
            prompt,
            max_tokens=max_tokens,
            temperature=0.5,
            top_p=0.9,
            repeat_penalty=1.1,
            stop=["</s>"],  # avoid [/INST] early cutoffs
            echo=False,
        )

        raw_text = output["choices"][0]["text"] or ""
        if not raw_text.strip():
            # Fallback if llama gives literally nothing
            raw_text = (
                "1. Core Idea\n"
                "I’m sorry, I had trouble generating a detailed answer.\n\n"
                "2. Step-by-Step Example\n"
                "Try asking the question in a slightly different way.\n\n"
                "3. Common Mistake + Check-Your-Understanding Question\n"
                "A common issue is giving too little context. What extra detail "
                "about your question could you add?"
            )

        cleaned = _strip_meta(raw_text)
        structured = _restructure_finetuned(cleaned)

        return structured.strip(), "finetuned-llama-lora"

    # ---------- BASE PATH (CHAT FORMAT, NO RESTRICTIONS) ----------
    model = get_base_model()

    base_prompt = f"""
        <s>[INST] <<SYS>>
        You are a helpful programming assistant.
        Answer clearly and concisely.
        <<SYS>>
        {question}
        [/INST]
        """

    output = model(
        base_prompt,
        max_tokens=max_tokens,
        temperature=0.7,
        top_p=0.9,
        repeat_penalty=1.1,
        stop=["</s>", "[/INST]"],
    )

    raw_text = output["choices"][0]["text"] or ""

    if not raw_text.strip():
        raw_text = (
            "A loop is a programming construct that repeats a block of code while a "
            "condition remains true, or for each item in a sequence."
        )

    return raw_text.strip(), "base-llama"
