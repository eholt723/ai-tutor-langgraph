"""
Shared fixtures for the AI Tutor test suite.
"""
from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest
from fastapi.testclient import TestClient

from ai_tutor.web.api import app
from ai_tutor.rag.store import VectorStore


@pytest.fixture
def api_client() -> TestClient:
    """FastAPI test client — reused across API tests."""
    return TestClient(app)


@pytest.fixture
def mock_generate():
    """
    Callable that returns a canned (answer, model_type) tuple.
    Use as a drop-in for ai_tutor.web.api.generate_answer.
    """
    return MagicMock(
        return_value=(
            "A variable is a named storage location for data.",
            "finetuned-llama-lora",
        )
    )


@pytest.fixture
def tiny_vector_store() -> VectorStore:
    """
    Minimal in-memory VectorStore for RAG tests.
    Uses pre-computed unit vectors so no model download is needed.
    """
    texts = [
        "A variable stores a value and has a name.",
        "A loop repeats code while a condition is true.",
        "A function is a reusable block of code.",
    ]
    titles = ["Variables", "Loops", "Functions"]
    ids = ["doc-0", "doc-1", "doc-2"]
    # Simple 3-d unit vectors — cosine similarity is deterministic
    embeddings = np.eye(3, dtype=np.float32)

    return VectorStore(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        embeddings=embeddings,
        texts=texts,
        ids=ids,
        titles=titles,
    )
