# ai_tutor/models/inference.py

from __future__ import annotations

from typing import Optional
import re

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def _build_tutor_prompt(question: str, context: Optional[str] = None) -> str:
    """
    Build the *same* tutor prompt we used during training for the finetuned model.
    """
    header = (
        "You are a beginner-friendly programming tutor for CEIS150: Programming with Objects.\n"
        "Always answer in three short parts:\n"
        "1. A clear one-sentence definition.\n"
        "2. A short concrete example.\n"
        "3. A common mistake or warning to avoid.\n"
        "Use plain language appropriate for a first programming course. Do NOT use emojis.\n\n"
    )

    parts = [header]
    if context:
        parts.append("Here is some helpful reference material:\n")
        parts.append(context.strip())
        parts.append("\n\n")

    parts.append("Student question:\n")
    parts.append(question.strip())
    parts.append("\n\nTutor answer:\n")

    return "".join(parts)


def _build_neutral_prompt(question: str, context: Optional[str] = None) -> str:
    """
    Simpler neutral prompt for the base model.
    """
    parts = []
    if context:
        parts.append("Context:\n")
        parts.append(context.strip())
        parts.append("\n\n")

    parts.append("Question:\n")
    parts.append(question.strip())
    parts.append("\n\nAnswer:\n")
    return "".join(parts)


def _postprocess_tutor_answer(raw: str) -> str:
    """
    Post-process the tutor-style answer:

    1. Chop off any hallucinated extra prompts like "student question:" or "tutor answer:".
    2. Keep only the first '1.', '2.', '3.' segments and drop the rest.
    """

    # Collapse whitespace first
    text = " ".join(raw.split())

    # 1) Hard cut at any prompt-like markers the model may have hallucinated.
    lower = text.lower()
    cut_markers = ["student question:", "tutor answer:"]
    cut_at = None
    for marker in cut_markers:
        idx = lower.find(marker)
        if idx != -1:
            if cut_at is None or idx < cut_at:
                cut_at = idx

    if cut_at is not None:
        text = text[:cut_at].strip()
        lower = text.lower()

    # 2) Now split on numbered segments and keep at most 1., 2., 3.
    parts = re.split(r"(?=\b[123]\.\s)", text)

    numbered = []
    seen = set()

    for part in parts:
        p = part.strip()
        if len(p) >= 2 and p[0] in "123" and p[1] == ".":
            n = p[0]
            if n not in seen:
                seen.add(n)
                numbered.append(p)
        if len(seen) == 3:
            break

    if numbered:
        return " ".join(numbered)

    # Fallback: if we didn't detect numbered items, return the cleaned text
    return text


def generate_answer(
    model,
    tokenizer,
    question: str,
    context: Optional[str] = None,
    tutor_style: bool = True,
    max_new_tokens: int = 120,  # slightly shorter to reduce rambling
) -> str:
    """
    Generate an answer from the given model/tokenizer.

    - If tutor_style is True (finetuned model), we use the CEIS150 tutor prompt.
    - If tutor_style is False (base model), we use a simpler neutral prompt.
    """

    if tutor_style:
        prompt = _build_tutor_prompt(question, context)
    else:
        prompt = _build_neutral_prompt(question, context)

    # Encode the full prompt
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        add_special_tokens=True,
    )

    # Move to the same device as the model
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,          # deterministic for now
            no_repeat_ngram_size=4,
            repetition_penalty=1.2,
            pad_token_id=tokenizer.eos_token_id,
        )

    full_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    # Strip the prompt part, keep only the model's continuation
    if tutor_style:
        split_marker = "Tutor answer:\n"
    else:
        split_marker = "Answer:\n"

    if split_marker in full_text:
        answer = full_text.split(split_marker, 1)[1]
    else:
        # Fallback in case something odd happens
        answer = full_text[len(prompt) :]

    answer = answer.strip()

    if tutor_style:
        answer = _postprocess_tutor_answer(answer)

    return answer
