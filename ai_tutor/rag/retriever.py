# ai_tutor/rag/retriever.py

from __future__ import annotations

from typing import List, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer

from ai_tutor.rag.store import VectorStore, load_vector_store


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a_norm = a / (np.linalg.norm(a, axis=-1, keepdims=True) + 1e-8)
    b_norm = b / (np.linalg.norm(b, axis=-1, keepdims=True) + 1e-8)
    return np.dot(a_norm, b_norm.T)


def retrieve_context(question: str, top_k: int = 3) -> List[Tuple[str, str]]:

    vs: VectorStore = load_vector_store()
    embedder = SentenceTransformer(vs.model_name)

    query_emb = embedder.encode([question], convert_to_numpy=True)
    sims = _cosine_similarity(query_emb, vs.embeddings)[0]

    top_indices = np.argsort(-sims)[:top_k]

    results: List[Tuple[str, str]] = []
    for idx in top_indices:
        title = vs.titles[idx]
        text = vs.texts[idx]
        results.append((title, text))

    return results
