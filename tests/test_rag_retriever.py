"""
Tests for the RAG cosine-similarity retriever.

All tests use the tiny_vector_store fixture from conftest.py —
no sentence-transformer model is downloaded, no disk I/O required.
"""
from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest

from ai_tutor.rag.retriever import _cosine_similarity, retrieve_context


# ---------------------------------------------------------------------------
# Unit: cosine similarity math
# ---------------------------------------------------------------------------


class TestCosineSimilarity:
    def test_identical_vectors_score_one(self):
        v = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
        score = _cosine_similarity(v, v)[0][0]
        assert abs(score - 1.0) < 1e-5

    def test_orthogonal_vectors_score_zero(self):
        a = np.array([[1.0, 0.0]], dtype=np.float32)
        b = np.array([[0.0, 1.0]], dtype=np.float32)
        score = _cosine_similarity(a, b)[0][0]
        assert abs(score) < 1e-5

    def test_opposite_vectors_score_negative_one(self):
        a = np.array([[1.0, 0.0]], dtype=np.float32)
        b = np.array([[-1.0, 0.0]], dtype=np.float32)
        score = _cosine_similarity(a, b)[0][0]
        assert abs(score + 1.0) < 1e-5

    def test_zero_vector_does_not_raise(self):
        """Zero vector is guarded by the +1e-8 epsilon — should not divide by zero."""
        z = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)
        v = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
        score = _cosine_similarity(z, v)[0][0]
        assert np.isfinite(score)

    @pytest.mark.parametrize("top_k", [1, 2, 3])
    def test_top_k_returns_correct_count(self, top_k):
        a = np.ones((1, 4), dtype=np.float32)
        b = np.eye(4, dtype=np.float32)
        sims = _cosine_similarity(a, b)[0]
        indices = np.argsort(-sims)[:top_k]
        assert len(indices) == top_k


# ---------------------------------------------------------------------------
# Integration: retrieve_context with mocked store + embedder
# ---------------------------------------------------------------------------


class TestRetrieveContext:
    def test_returns_correct_number_of_results(self, tiny_vector_store):
        # Query embedding matches the first basis vector → doc-0 ranks first
        query_emb = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)

        with patch("ai_tutor.rag.retriever.load_vector_store", return_value=tiny_vector_store), \
             patch("ai_tutor.rag.retriever.SentenceTransformer") as mock_st:
            mock_st.return_value.encode.return_value = query_emb
            results = retrieve_context("What is a variable?", top_k=2)

        assert len(results) == 2

    def test_returns_tuples_of_title_and_text(self, tiny_vector_store):
        query_emb = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)

        with patch("ai_tutor.rag.retriever.load_vector_store", return_value=tiny_vector_store), \
             patch("ai_tutor.rag.retriever.SentenceTransformer") as mock_st:
            mock_st.return_value.encode.return_value = query_emb
            results = retrieve_context("What is a variable?", top_k=1)

        title, text = results[0]
        assert isinstance(title, str) and title
        assert isinstance(text, str) and text

    def test_top_result_matches_query_intent(self, tiny_vector_store):
        """Query vector aligned with doc-0 (Variables) should rank it first."""
        query_emb = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)

        with patch("ai_tutor.rag.retriever.load_vector_store", return_value=tiny_vector_store), \
             patch("ai_tutor.rag.retriever.SentenceTransformer") as mock_st:
            mock_st.return_value.encode.return_value = query_emb
            results = retrieve_context("What is a variable?", top_k=1)

        assert results[0][0] == "Variables"

    def test_top_k_one_returns_single_result(self, tiny_vector_store):
        query_emb = np.array([[0.0, 1.0, 0.0]], dtype=np.float32)

        with patch("ai_tutor.rag.retriever.load_vector_store", return_value=tiny_vector_store), \
             patch("ai_tutor.rag.retriever.SentenceTransformer") as mock_st:
            mock_st.return_value.encode.return_value = query_emb
            results = retrieve_context("How does a loop work?", top_k=1)

        assert len(results) == 1

    def test_missing_store_raises_file_not_found(self):
        with patch(
            "ai_tutor.rag.retriever.load_vector_store",
            side_effect=FileNotFoundError("Vector store not found"),
        ):
            with pytest.raises(FileNotFoundError):
                retrieve_context("What is a variable?")
