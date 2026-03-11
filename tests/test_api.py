from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from ai_tutor.web.api import app

client = TestClient(app)

MOCK_ANSWER = ("A variable is a named storage location for data.", "finetuned-llama-lora")


# ---------------------------------------------------------------------------
# Health endpoint
# ---------------------------------------------------------------------------


class TestHealthEndpoint:
    def test_returns_200(self):
        response = client.get("/health")
        assert response.status_code == 200

    def test_returns_ok_status(self):
        response = client.get("/health")
        assert response.json() == {"status": "ok"}


# ---------------------------------------------------------------------------
# Chat endpoint — happy path
# ---------------------------------------------------------------------------


class TestChatEndpoint:
    def test_base_mode_returns_200(self):
        with patch("ai_tutor.web.api.generate_answer", return_value=MOCK_ANSWER):
            response = client.post("/chat", json={"question": "What is a variable?"})
        assert response.status_code == 200

    def test_response_contains_answer(self):
        with patch("ai_tutor.web.api.generate_answer", return_value=MOCK_ANSWER):
            response = client.post("/chat", json={"question": "What is a loop?"})
        data = response.json()
        assert "answer" in data
        assert data["answer"] == MOCK_ANSWER[0]

    def test_response_contains_model_type(self):
        with patch("ai_tutor.web.api.generate_answer", return_value=MOCK_ANSWER):
            response = client.post("/chat", json={"question": "What is a function?"})
        data = response.json()
        assert "model_type" in data

    def test_question_echoed_in_response(self):
        question = "What is a class?"
        with patch("ai_tutor.web.api.generate_answer", return_value=MOCK_ANSWER):
            response = client.post("/chat", json={"question": question})
        assert response.json()["question"] == question

    def test_used_rag_false_by_default(self):
        with patch("ai_tutor.web.api.generate_answer", return_value=MOCK_ANSWER):
            response = client.post("/chat", json={"question": "What is inheritance?"})
        assert response.json()["used_rag"] is False

    def test_prompt_debug_returned_when_requested(self):
        with patch("ai_tutor.web.api.generate_answer", return_value=MOCK_ANSWER):
            response = client.post(
                "/chat",
                json={"question": "What is a loop?", "debug_prompt": True},
            )
        data = response.json()
        assert data["prompt_debug"] is not None
        assert "What is a loop?" in data["prompt_debug"]

    def test_prompt_debug_null_when_not_requested(self):
        with patch("ai_tutor.web.api.generate_answer", return_value=MOCK_ANSWER):
            response = client.post("/chat", json={"question": "What is a loop?"})
        assert response.json()["prompt_debug"] is None

    def test_fixture_client_works(self, api_client):
        """Verify the shared api_client fixture produces the same results."""
        with patch("ai_tutor.web.api.generate_answer", return_value=MOCK_ANSWER):
            response = api_client.post("/chat", json={"question": "What is a loop?"})
        assert response.status_code == 200


# ---------------------------------------------------------------------------
# Chat endpoint — input validation edge cases
# ---------------------------------------------------------------------------


class TestChatInputValidation:
    def test_missing_question_field_rejected(self):
        response = client.post("/chat", json={})
        assert response.status_code == 422

    def test_non_string_question_rejected(self):
        response = client.post("/chat", json={"question": 42})
        assert response.status_code == 422

    def test_null_question_rejected(self):
        response = client.post("/chat", json={"question": None})
        assert response.status_code == 422

    @pytest.mark.parametrize("question", [
        "What is a variable?",
        "  whitespace padded  ",
        "What is a loop?" * 20,        # long input
        "What's a héritàge class?",    # unicode
        "print('hello'); import os",   # code-like input
    ])
    def test_various_valid_questions_return_200(self, question):
        with patch("ai_tutor.web.api.generate_answer", return_value=MOCK_ANSWER):
            response = client.post("/chat", json={"question": question})
        assert response.status_code == 200

    def test_finetuned_flag_accepted(self):
        with patch("ai_tutor.web.api.generate_answer", return_value=MOCK_ANSWER):
            response = client.post(
                "/chat",
                json={"question": "What is a class?", "use_finetuned": True},
            )
        assert response.status_code == 200

    def test_generate_answer_receives_correct_finetuned_flag(self):
        with patch("ai_tutor.web.api.generate_answer", return_value=MOCK_ANSWER) as mock_gen:
            client.post("/chat", json={"question": "What is a loop?", "use_finetuned": True})
        _, kwargs = mock_gen.call_args
        assert kwargs.get("use_finetuned") is True


# ---------------------------------------------------------------------------
# CORS configuration
# ---------------------------------------------------------------------------


class TestCORSHeaders:
    @pytest.mark.parametrize("origin", [
        "https://eholt723.github.io",
        "https://eholt723-ai-tutor.hf.space",
    ])
    def test_allowed_origins_receive_cors_header(self, origin):
        with patch("ai_tutor.web.api.generate_answer", return_value=MOCK_ANSWER):
            response = client.post(
                "/chat",
                json={"question": "What is a variable?"},
                headers={"Origin": origin},
            )
        assert response.headers.get("access-control-allow-origin") == origin

    def test_disallowed_origin_does_not_receive_cors_header(self):
        with patch("ai_tutor.web.api.generate_answer", return_value=MOCK_ANSWER):
            response = client.post(
                "/chat",
                json={"question": "What is a variable?"},
                headers={"Origin": "https://evil.example.com"},
            )
        assert response.headers.get("access-control-allow-origin") != "https://evil.example.com"
