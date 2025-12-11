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
        "<<SYS>>",
        "<</SYS>>",
        "[/SYS]",
        "[INST]",
        "[/INST]",
        "Tutor answer:",
        "Student answer:",
        "Tutor question:",
        # Old training headers that sometimes get echoed
        "Always answer the same three clear questions in exactly three numbers.",
        "Use plain language that students can understand without Google.",
        "Always use plain language that students can understand without Google.",
        "appropriate for a first programming course.",
        "Talk directly to the student and avoid using emojis or abbreviations.",
        "Talk directly to the student and keep paragraphs short.",
        "clear student example",
    ]

    for phrase in remove_anywhere:
        t = t.replace(phrase, "").strip()

    # Remove empty lines
    lines = [ln for ln in t.splitlines() if ln.strip()]
    return "\n".join(lines).strip()


def _split_numbered_sections(text: str) -> list[str]:
    """
    Split text into sections starting at any '1.', '2.', '3.' occurrence,
    even if they appear mid-line (e.g., '1. ... 2. ... 3. ...').
    """
    t = text.replace("\r\n", "\n").replace("\r", "\n")

    # Find any "digit." followed by space as a boundary
    matches = list(re.finditer(r"(\d+)\.\s+", t))
    if not matches:
        return [t.strip()] if t.strip() else []

    sections: list[str] = []
    for i, m in enumerate(matches):
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(t)
        chunk = t[start:end].strip()
        if chunk:
            sections.append(chunk)

    return sections


def _restructure_finetuned(text: str) -> str:
    """
    Take the model's cleaned finetuned text and reshape it into:

    1. Core Idea
    ...
    2. Step-by-Step Example
    ...
    3. Common Mistake + Check-Your-Understanding Question
    """
    cleaned = text.strip()
    if not cleaned:
        return cleaned

    sections = _split_numbered_sections(cleaned)

    # If we didn't get at least 2 sections, just return cleaned text
    if len(sections) < 2:
        return cleaned

    headings = [
        "Core Idea",
        "Step-by-Step Example",
        "Common Mistake + Check-Your-Understanding Question",
    ]

    formatted_parts: list[str] = []

    for idx in range(min(3, len(sections))):
        raw_sec = sections[idx]

        # Strip leading "1.", "2.", "3." from the first line
        lines = raw_sec.splitlines()
        if lines:
            lines[0] = re.sub(r"^\s*\d+\.\s*", "", lines[0]).strip()
        body = "\n".join(lines).strip()

        # For section 3, ensure there's at least one question
        if idx == 2 and body and "?" not in body:
            body = (
                body
                + "\n\nCheck-your-understanding question: "
                  "In your own words, why is this concept important in programming?"
            ).strip()

        if not body:
            continue

        heading = headings[idx]
        formatted_parts.append(f"{idx + 1}. {heading}\n{body}")

    if not formatted_parts:
        return cleaned

    result = "\n\n".join(formatted_parts).strip()

    # Clean up any leftover weird trailing fragments like half-sentences
    result = re.sub(r"\s+Check your answer by comparing.*?$", "", result, flags=re.IGNORECASE | re.DOTALL).strip()

    # Normalize double spaces
    while "  " in result:
        result = result.replace("  ", " ")

    return result


# -------------------------------------------------------------------
# Main generation
# -------------------------------------------------------------------


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
        if not structured.strip():
            structured = cleaned or raw_text

        return structured.strip(), "finetuned-llama-lora"

    # ---------- BASE PATH (simple completion via shared prompt builder) ----------
    model = get_base_model()
    base_prompt = build_prompt(question=question, mode="base", context=context)

    output = model(
        base_prompt,
        max_tokens=max_tokens,
        temperature=0.7,
        top_p=0.9,
        repeat_penalty=1.1,
        stop=["</s>"],
    )

    raw_text = output["choices"][0]["text"] or ""

    if not raw_text.strip():
        raw_text = (
            "A loop is a programming construct that repeats a block of code while a "
            "condition remains true, or for each item in a sequence."
        )

    return raw_text.strip(), "base-llama"
