
from __future__ import annotations

from pathlib import Path
from typing import Any, Tuple

from transformers import AutoTokenizer, AutoModelForCausalLM

from ai_tutor.config import Config


def load_base_model() -> Tuple[Any, Any]:

    model_id = Config.base_model_id
    model_path = Path(Config.base_model_path)

    print(f"Loading base model: {model_id}")
    print(f"Model path: {model_path}")

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="cpu",
        low_cpu_mem_usage=True,
    )

    return model, tokenizer
