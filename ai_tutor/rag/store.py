# ai_tutor/rag/store.py

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np
from sentence_transformers import SentenceTransformer

from ai_tutor.config import Config
from ai_tutor.rag.ingest import ReferenceDoc


@dataclass
class VectorStore:
    model_name: str
    embeddings: np.ndarray  # shape: (num_docs, dim)
    texts: List[str]
    ids: List[str]
    titles: List[str]


def _get_embedder(model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> SentenceTransformer:
    return SentenceTransformer(model_name)


def build_vector_store(docs: List[ReferenceDoc]) -> VectorStore:
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    embedder = _get_embedder(model_name)

    texts = [doc.content for doc in docs]
    ids = [doc.id for doc in docs]
    titles = [doc.title for doc in docs]

    embeddings = embedder.encode(texts, convert_to_numpy=True)

    return VectorStore(
        model_name=model_name,
        embeddings=embeddings,
        texts=texts,
        ids=ids,
        titles=titles,
    )


def save_vector_store(docs: List[ReferenceDoc], rebuild: bool = False) -> VectorStore:

    index_dir: Path = Config.rag_index_path
    index_dir.mkdir(parents=True, exist_ok=True)
    index_file = index_dir / "vector_store.pkl"

    if index_file.exists() and not rebuild:
        return load_vector_store()

    vs = build_vector_store(docs)

    with open(index_file, "wb") as f:
        pickle.dump(vs, f)

    return vs


def load_vector_store() -> VectorStore:
    index_dir: Path = Config.rag_index_path
    index_file = index_dir / "vector_store.pkl"

    if not index_file.exists():
        raise FileNotFoundError(f"Vector store not found at {index_file}. Run build_rag_index.py first.")

    with open(index_file, "rb") as f:
        vs: VectorStore = pickle.load(f)

    return vs
