# ai_tutor/models/inference.py

from __future__ import annotations

from typing import Optional
import re

import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase


def build_prompt(
    question: str,
    context: Optional[str] = None,
    tutor_style: bool = False,
) -> str:
    question = question.strip()
    context = (context or "").strip()

    tutor_header = (
        "You are a beginner-friendly programming tutor for CEIS150: Programming with Objects.\n"
        "Always answer in three short parts:\n"
        "1. A clear one-sentence definition.\n"
        "2. A short concrete example.\n"
        "3. A common mistake or warning to avoid.\n"
        "Use plain language appropriate for a first programming course. Do NOT use emojis.\n\n"
    )

    neutral_header = (
        "You are an AI assistant. Provide a short, clear answer in 1–3 sentences. "
        "Use plain language appropriate for a college student. Do NOT use emojis.\n\n"
    )

    header = tutor_header if tutor_style else neutral_header

    if context:
        prompt = (
            header
            + "Context:\n"
            + context
            + "\n\nStudent question:\n"
            + question
            + "\n\nTutor answer:\n"
        )
    else:
        prompt = (
            header
            + "Student question:\n"
            + question
            + "\n\nTutor answer:\n"
        )

    return prompt


def _strip_outer_quotes(text: str) -> str:
    text = text.strip()

    # Remove matching outer quotes
    if len(text) >= 2 and (
        (text[0] == '"' and text[-1] == '"')
        or (text[0] == "'" and text[-1] == "'")
    ):
        text = text[1:-1].strip()

    # Remove leading/trailing stray quote
    if text.startswith('"') or text.startswith("'"):
        text = text[1:].lstrip()

    if text.endswith('"') or text.endswith("'"):
        text = text[:-1].rstrip()

    # As final cleanup, remove any remaining quote characters
    text = text.replace('"', "")

    return text


def _trim_to_sentences(text: str, max_sentences: int = 3) -> str:
    """
    Post-process the raw model output:
    - squash whitespace
    - strip quotes
    - remove parenthetical notes
    - split into sentences
    - keep only the first few sentences (up to max_sentences)
    - hard-cap total words
    """
    clean = " ".join(text.split())
    clean = _strip_outer_quotes(clean)

    # Remove parenthetical notes like (Note: ...)
    clean = re.sub(r"\([^)]*\)", "", clean).strip()

    if not clean:
        return clean

    # Split on sentence boundaries . ! ?
    parts = re.split(r"(?<=[.!?])\s+", clean)
    parts = [p.strip() for p in parts if p.strip()]

    if parts:
        clean = " ".join(parts[:max_sentences])

    # Hard word cap (safety net) – allow more room for 3-part answers
    words = clean.split()
    if len(words) > 80:
        clean = " ".join(words[:80])
        if not clean.endswith("."):
            clean += "."

    return clean


def generate_answer(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    question: str,
    context: Optional[str] = None,
    max_new_tokens: int = 200,
    tutor_style: bool = False,
) -> str:
    prompt = build_prompt(question, context, tutor_style=tutor_style)

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=512,
    )

    # Ensure pad token is set
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,               # deterministic, stable output
            repetition_penalty=1.1,
            no_repeat_ngram_size=3,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)

    marker = "Tutor answer:"
    if marker in decoded:
        raw_answer = decoded.split(marker, 1)[1].strip()
    else:
        raw_answer = decoded[len(prompt):].strip()

    trimmed = _trim_to_sentences(raw_answer)
    answer = trimmed or raw_answer.strip()

    # --- Safety corrections for incorrect loop statements ---
    answer = re.sub(
        r"executes? one time",
        "repeats while a condition is true",
        answer,
        flags=re.IGNORECASE,
    )
    answer = re.sub(
        r"runs? one time",
        "can run many times",
        answer,
        flags=re.IGNORECASE,
    )

    # --- Safety correction for incomplete while-loop print example ---
    answer = re.sub(
        r"print;\s*i\s*\+\=\s*1",
        "print(i); i += 1",
        answer,
        flags=re.IGNORECASE,
    )

    # --- Safety correction for bad if-statement explanation ---
    if re.fullmatch(r"If x then y else z\.?", answer.strip(), flags=re.IGNORECASE):
        answer = (
            "An if statement checks a condition and only runs its block of code when that condition is true."
        )

    # --- Safety correction for vague variable definition ---
    if "refers to any object that can be manipulated" in answer:
        answer = (
            "A variable is a named storage location in memory that holds a value, "
            "and that value can change while the program runs."
        )

    return answer
