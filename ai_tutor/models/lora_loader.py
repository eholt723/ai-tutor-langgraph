# ai_tutor/models/lora_loader.py

from __future__ import annotations

from pathlib import Path
from typing import Any, Tuple

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel, PeftConfig

from ai_tutor.config import Config
from ai_tutor.models.base_loader import load_base_model


def load_finetuned_model() -> Tuple[Any, Any]:
    """
    Load the base model and attach a LoRA adapter if present.

    - If the adapter directory is missing or empty, fall back to the base model.
    - If the adapter exists, we always use Config.base_model_id as the base
      model name (e.g. 'distilgpt2') to match the LoRA weights trained in Colab.
    """

    adapter_path = Path(Config.lora_adapter_path)

    # 1. Fallback: no adapter -> just use the base model
    if not adapter_path.exists() or not any(adapter_path.iterdir()):
        print("[LoRA] Adapter path not found or empty:")
        print(f"       {adapter_path}")
        print("[LoRA] Falling back to base model.\n")

        model, tokenizer = load_base_model()
        return model, tokenizer

    # 2. Adapter exists: load matching base model and attach LoRA
    print(f"[LoRA] Loading LoRA adapter from: {adapter_path}")

    # Read adapter config (for logging/debugging only)
    peft_config = PeftConfig.from_pretrained(adapter_path)
    print(f"[LoRA] Adapter was trained with base: {peft_config.base_model_name_or_path}")

    # IMPORTANT: always use Config.base_model_id so it matches your Colab training
    base_model_id = Config.base_model_id
    print(f"[LoRA] Using base model id from Config: {base_model_id}")

    # Load tokenizer and base model
    tokenizer = AutoTokenizer.from_pretrained(base_model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    device_arg = "auto" if torch.cuda.is_available() else "cpu"

    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        device_map=device_arg,
        low_cpu_mem_usage=True,
    )

    # Attach LoRA weights on top of the base model
    model = PeftModel.from_pretrained(
        base_model,
        adapter_path,
        device_map=device_arg,
    )

    model.eval()
    return model, tokenizer
