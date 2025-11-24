
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

# Load environment variables from .env if present
load_dotenv()


@dataclass(frozen=True)
class ConfigClass:

    # Base project paths
    project_root: Path = Path(__file__).resolve().parents[1]
    data_dir: Path = project_root / "data"
    live_eval_dir: Path = data_dir / "live_eval"
    models_dir: Path = project_root / "models"
    artifacts_dir: Path = project_root / "artifacts"
    rag_index_dir: Path = project_root / "rag_index"

    # Model identifiers / paths
    base_model_id: str = os.getenv("BASE_MODEL_ID", "gpt2-medium")
    base_model_path: Path = Path(os.getenv("BASE_MODEL_PATH", str(models_dir / "base_model")))
    lora_adapter_path: Path = Path(os.getenv("LORA_ADAPTER_PATH", str(models_dir / "lora_adapter")))

    #Embedding model for RAG
    embedding_model_id: str = os.getenv("EMBEDDING_MODEL_ID", "sentence-transformers/all-MiniLM-L6-v2")

    # RAG index path
    rag_index_path: Path = Path(os.getenv("RAG_INDEX_PATH", str(rag_index_dir)))

    # Eval results
    eval_results_path: Path = artifacts_dir / "eval" / "eval_results.json"

    # API configuration
    api_host: str = os.getenv("API_HOST", "127.0.0.1")
    api_port: int = int(os.getenv("API_PORT", "8000"))

    # Dataset configuration
    dataset_name: str = os.getenv("DATASET_NAME", "ai_tutor_demo_dataset")
    # You can later set this to a real HF dataset ID.


Config = ConfigClass()

