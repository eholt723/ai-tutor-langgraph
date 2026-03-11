#!/bin/bash
set -e

python - <<'EOF'
import os
from huggingface_hub import hf_hub_download

REPO_ID = os.getenv("HF_MODEL_REPO", "eholt723/ai-tutor-models")

os.makedirs("models/gguf", exist_ok=True)
os.makedirs("models/lora_gguf", exist_ok=True)

base_path = "models/gguf/tinyllama-q4_0.gguf"
lora_path = "models/lora_gguf/tinyllama-tutor-lora-q8_0.gguf"

if not os.path.exists(base_path):
    print(f"Downloading base model from {REPO_ID}...", flush=True)
    hf_hub_download(
        repo_id=REPO_ID,
        filename="tinyllama-q4_0.gguf",
        local_dir="models/gguf",
        local_dir_use_symlinks=False,
    )
    print("Base model ready.", flush=True)
else:
    print("Base model already present, skipping download.", flush=True)

if not os.path.exists(lora_path):
    print(f"Downloading LoRA adapter from {REPO_ID}...", flush=True)
    hf_hub_download(
        repo_id=REPO_ID,
        filename="tinyllama-tutor-lora-q8_0.gguf",
        local_dir="models/lora_gguf",
        local_dir_use_symlinks=False,
    )
    print("LoRA adapter ready.", flush=True)
else:
    print("LoRA adapter already present, skipping download.", flush=True)
EOF

exec uvicorn ai_tutor.web.api:app --host 0.0.0.0 --port 7860
