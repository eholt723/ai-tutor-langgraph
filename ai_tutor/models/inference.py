# ai_tutor/models/inference.py

from __future__ import annotations

from typing import Optional
import re

import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase


def build_prompt(question: str, context: Optional[str] = None) -> str:
    question = question.strip()
    context = (context or "").strip()

    header = (
        "You are an AI software tutor for CEIS150: Programming with Objects. "
        "Answer the student’s question directly in 2–4 short sentences. "
        "Use plain language appropriate for a first programming course. "
        "Do NOT repeat sentences. Do NOT restate the question. "
        "Do NOT use emojis.\n\n"
    )

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


def _trim_to_sentences(text: str, max_sentences: int = 1) -> str:
    """
    Post-process the raw model output:
    - squash whitespace
    - strip quotes
    - remove parenthetical notes
    - split into sentences
    - keep only the first
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
        clean = parts[0]  # only the first sentence

    # Hard word cap (safety net)
    words = clean.split()
    if len(words) > 40:
        clean = " ".join(words[:40])
        if not clean.endswith("."):
            clean += "."

    return clean


def generate_answer(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    question: str,
    context: Optional[str] = None,
    max_new_tokens: int = 80,
) -> str:
    prompt = build_prompt(question, context)

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

