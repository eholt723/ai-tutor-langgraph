# ai_tutor/models/lora_loader.py

from __future__ import annotations

from pathlib import Path
from typing import Any, Tuple

from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel, PeftConfig

from ai_tutor.config import Config


def load_finetuned_model() -> Tuple[Any, Any]:


    adapter_path = Path(Config.lora_adapter_path)

    if not adapter_path.exists():
        print("WARNING: LoRA adapter not found. Falling back to base model.")
        from .base_loader import load_base_model

        return load_base_model()

    print(f"Loading LoRA adapter from: {adapter_path}")

    peft_config = PeftConfig.from_pretrained(adapter_path)
    tokenizer = AutoTokenizer.from_pretrained(peft_config.base_model_name_or_path)

    base_model = AutoModelForCausalLM.from_pretrained(
        peft_config.base_model_name_or_path,
        device_map="cpu",
        low_cpu_mem_usage=True,
    )

    model = PeftModel.from_pretrained(base_model, adapter_path)

    return model, tokenizer
