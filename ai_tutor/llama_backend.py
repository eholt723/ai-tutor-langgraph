# ai_tutor/llama_backend.py

from __future__ import annotations

from pathlib import Path
from typing import Optional

from llama_cpp import Llama

# Paths to your GGUF files (adjust if needed)
BASE_MODEL_PATH = Path("models/gguf/tinyllama-q4_0.gguf")
LORA_ADAPTER_PATH = Path("models/lora_gguf/tinyllama-tutor-lora-q8_0.gguf")

_llm: Optional[Llama] = None
_lora_attempted: bool = False  # so we don't keep retrying if attaching fails


def _ensure_loaded(use_finetuned: bool) -> Llama:
    """
    Load the base TinyLlama GGUF once.
    If finetuned mode is requested, try to attach the LoRA adapter.
    If LoRA attaching fails (no support in this build), continue with base weights.
    """
    global _llm, _lora_attempted

    if _llm is None:
        if not BASE_MODEL_PATH.is_file():
            raise RuntimeError(f"Base GGUF not found at {BASE_MODEL_PATH}")
        _llm = Llama(
            model_path=str(BASE_MODEL_PATH),
            n_ctx=2048,
            n_threads=0,  # auto
        )

    # Try to attach LoRA the first time finetuned is used
    if use_finetuned and not _lora_attempted:
        _lora_attempted = True  # avoid spamming attempts every call

        if not LORA_ADAPTER_PATH.is_file():
            print(
                f"[llama_backend] WARNING: LoRA GGUF not found at {LORA_ADAPTER_PATH}, "
                "running finetuned mode without adapter."
            )
            return _llm

        attached = False

        # Newer llama-cpp-python exposes lora_adapter
        try:
            if hasattr(_llm, "lora_adapter"):
                _llm.lora_adapter(str(LORA_ADAPTER_PATH), scale=1.0)
                attached = True
                print("[llama_backend] LoRA adapter attached via lora_adapter().")
        except Exception as e:
            print(f"[llama_backend] WARNING: lora_adapter() call failed: {e!r}")

        # Some builds only expose _lora_adapter
        if not attached:
            try:
                if hasattr(_llm, "_lora_adapter"):
                    _llm._lora_adapter(str(LORA_ADAPTER_PATH), scale=1.0)  # type: ignore[attr-defined]
                    attached = True
                    print("[llama_backend] LoRA adapter attached via _lora_adapter().")
            except Exception as e:
                print(f"[llama_backend] WARNING: _lora_adapter() call failed: {e!r}")

        if not attached:
            print(
                "[llama_backend] WARNING: Could not attach LoRA adapter. "
                "Finetuned mode will use tutor-style prompting only."
            )

    return _llm


def _build_messages(question: str, tutor_style: bool):
    """
    Build chat messages so that finetuned mode ALWAYS uses
    the 3-part tutor signature.
    """
    if tutor_style:
        system = (
            "You are a friendly programming tutor for beginners.\n"
            "Your ENTIRE answer must be EXACTLY three numbered lines, nothing else:\n"
            "1. Core idea: one short sentence explaining the concept.\n"
            "2. Example (Python): one short sentence plus a tiny inline Python example in backticks.\n"
            "3. Common mistake: one short sentence about a typical beginner mistake and how to avoid it.\n"
            "Rules:\n"
            "- Do NOT use code blocks or ``` fences.\n"
            "- Do NOT add extra paragraphs, introductions, or summaries.\n"
            "- Each line must start with '1.', '2.', and '3.' exactly.\n"
            "- Keep each line to one or two sentences maximum."
        )
    else:
        system = (
            "You are a concise programming assistant. "
            "Answer the question clearly in a few sentences without extra formatting."
        )

    return [
        {"role": "system", "content": system},
        {"role": "user", "content": question},
    ]


def generate_llama_answer(question: str, use_finetuned: bool) -> str:
    """
    Main entry point used by the API.
    Base vs finetuned is controlled by `use_finetuned`.
    """
    llm = _ensure_loaded(use_finetuned=use_finetuned)
    messages = _build_messages(question, tutor_style=use_finetuned)

    result = llm.create_chat_completion(
        messages=messages,
        max_tokens=160,
        temperature=0.4 if use_finetuned else 0.5,
        top_p=0.9,
    )

    return result["choices"][0]["message"]["content"].strip()
