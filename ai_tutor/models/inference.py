# ai_tutor/models/inference.py

from __future__ import annotations

from typing import Optional, Any

import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase


def build_prompt(question: str, context: Optional[str] = None) -> str:
    """
    Build the same style of prompt we used during fine-tuning.
    This is critical so the LoRA adapter 'recognizes' the pattern.
    """
    question = question.strip()
    context = (context or "").strip()

    if context:
        prompt = (
            "You are an AI programming tutor.\n\n"
            f"Context:\n{context}\n\n"
            f"Student question:\n{question}\n\n"
            "Tutor answer:\n"
        )
    else:
        prompt = (
            "You are an AI programming tutor.\n\n"
            f"Student question:\n{question}\n\n"
            "Tutor answer:\n"
        )

    return prompt


def generate_answer(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    question: str,
    context: Optional[str] = None,
    max_new_tokens: int = 128,
    temperature: float = 0.7,
    top_p: float = 0.9,
) -> str:
    """
    Generate an answer using the given model and tokenizer.
    """
    prompt = build_prompt(question, context)

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=512,
    )

    # Move to model device
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            pad_token_id=tokenizer.eos_token_id,
        )

    full_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Return only the part after "Tutor answer:"
    marker = "Tutor answer:"
    if marker in full_text:
        return full_text.split(marker, 1)[1].strip()
    else:
        # Fallback if formatting changes
        return full_text[len(prompt) :].strip()
