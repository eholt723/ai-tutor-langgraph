# ai_tutor/models/inference.py

from __future__ import annotations

from typing import Any


def generate_answer(model: Any, tokenizer: Any, question: str, context: str | None = None) -> str:

    if context:
        prompt = f"Context: {context}\n\nQuestion: {question}\nAnswer:"
    else:
        prompt = f"Question: {question}\nAnswer:"

    inputs = tokenizer(prompt, return_tensors="pt")

    output = model.generate(
        **inputs,
        max_new_tokens=128,
        do_sample=False,
        temperature=0.0,
    )

    text = tokenizer.decode(output[0], skip_special_tokens=True)
    return text[len(prompt):].strip()
