# ai_tutor/models/lora_loader.py

from __future__ import annotations

from pathlib import Path
from typing import Any, Tuple

from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel, PeftConfig

from ai_tutor.config import Config


def load_finetuned_model() -> Tuple[Any, Any]:
    """
    Try to load a LoRA-adapted model.

    If no adapter is found at Config.lora_adapter_path, we gracefully fall
    back to loading the base model instead, so the rest of the pipeline
    still works for demos.
    """
    adapter_path = Path(Config.lora_adapter_path)

    if not adapter_path.exists() or not any(adapter_path.iterdir()):
        # Fallback: just use the base model as a stand-in for "fine-tuned"
        print("[LoRA] Adapter path not found or empty:")
        print(f"       {adapter_path}")
        print("[LoRA] Falling back to base model as the 'fine-tuned' model.\n")

        model_id = Config.base_model_id
        print(f"[LoRA] Loading base model as fallback: {model_id}")
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="cpu",
            low_cpu_mem_usage=True,
        )
        return model, tokenizer

    print(f"[LoRA] Loading LoRA adapter from: {adapter_path}")

    peft_config = PeftConfig.from_pretrained(adapter_path)
    tokenizer = AutoTokenizer.from_pretrained(peft_config.base_model_name_or_path)

    base_model = AutoModelForCausalLM.from_pretrained(
        peft_config.base_model_name_or_path,
        device_map="cpu",
        low_cpu_mem_usage=True,
    )

    model = PeftModel.from_pretrained(base_model, adapter_path)

    return model, tokenizer
