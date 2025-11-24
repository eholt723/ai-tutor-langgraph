# ai_tutor/models/lora_loader.py

from __future__ import annotations

from pathlib import Path
from typing import Any, Tuple

import torch
from peft import PeftModel, PeftConfig
from ai_tutor.config import Config
from ai_tutor.models.base_loader import load_base_model


def load_finetuned_model() -> Tuple[Any, Any]:
    """
    Load a base model + LoRA adapter (if present).

    If no adapter exists, fallback to the base model so that
    the rest of the pipeline continues to work.
    """

    adapter_path = Path(Config.lora_adapter_path)

    if not adapter_path.exists() or not any(adapter_path.iterdir()):
        print("[LoRA] Adapter path not found or empty:")
        print(f"       {adapter_path}")
        print("[LoRA] Falling back to base model.\n")

        # Use your normal base loader
        model, tokenizer = load_base_model()
        return model, tokenizer

    print(f"[LoRA] Loading LoRA adapter from: {adapter_path}")

    # If adapter exists, load the correct base model
    peft_config = PeftConfig.from_pretrained(adapter_path)

    # Use your standard base loader for consistency
    base_model, tokenizer = load_base_model()

    # Device selection
    device_arg = "auto" if torch.cuda.is_available() else "cpu"

    # Attach LoRA weights
    model = PeftModel.from_pretrained(
        base_model,
        adapter_path,
        device_map=device_arg,
    )

    return model, tokenizer
