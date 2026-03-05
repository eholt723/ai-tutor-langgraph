from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from ai_tutor.web.api import app

client = TestClient(app)

MOCK_ANSWER = ("A variable is a named storage location for data.", "finetuned-llama-lora")


class TestHealthEndpoint:
    def test_returns_200(self):
        response = client.get("/health")
        assert response.status_code == 200

    def test_returns_ok_status(self):
        response = client.get("/health")
        assert response.json() == {"status": "ok"}


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

    def test_empty_question_rejected(self):
        response = client.post("/chat", json={})
        assert response.status_code == 422
