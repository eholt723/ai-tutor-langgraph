import pytest

from ai_tutor.prompts import build_prompt


QUESTION = "What is a loop?"
CONTEXT = "Loops repeat code blocks."


class TestBaseModePrompt:
    def test_contains_question(self):
        prompt = build_prompt(QUESTION, mode="base")
        assert QUESTION in prompt

    def test_ends_with_answer_marker(self):
        prompt = build_prompt(QUESTION, mode="base")
        assert prompt.strip().endswith("A:")

    def test_no_inst_tags(self):
        prompt = build_prompt(QUESTION, mode="base")
        assert "[INST]" not in prompt

    def test_context_included_when_provided(self):
        prompt = build_prompt(QUESTION, mode="base", context=CONTEXT)
        assert CONTEXT in prompt

    def test_no_context_when_none(self):
        prompt = build_prompt(QUESTION, mode="base", context=None)
        assert "reference notes" not in prompt.lower() or CONTEXT not in prompt


class TestFinetunedModePrompt:
    def test_contains_question(self):
        prompt = build_prompt(QUESTION, mode="finetuned")
        assert QUESTION in prompt

    def test_has_inst_tags(self):
        prompt = build_prompt(QUESTION, mode="finetuned")
        assert "[INST]" in prompt
        assert "[/INST]" in prompt

    def test_has_sys_tags(self):
        prompt = build_prompt(QUESTION, mode="finetuned")
        assert "<<SYS>>" in prompt
        assert "<</SYS>>" in prompt

    def test_mentions_three_sections(self):
        prompt = build_prompt(QUESTION, mode="finetuned")
        assert "Core Idea" in prompt
        assert "Step-by-Step Example" in prompt
        assert "Common Mistake" in prompt

    def test_context_included_when_provided(self):
        prompt = build_prompt(QUESTION, mode="finetuned", context=CONTEXT)
        assert CONTEXT in prompt

    def test_no_context_block_when_none(self):
        prompt = build_prompt(QUESTION, mode="finetuned", context=None)
        assert CONTEXT not in prompt
