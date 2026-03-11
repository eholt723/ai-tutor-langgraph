import pytest

from ai_tutor.llama_backend import _normalize_question, _restructure_finetuned, _split_numbered_sections, _strip_meta


# ---------------------------------------------------------------------------
# Parametrized contraction expansion — one table drives all cases
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("raw,expected_fragment", [
    ("what's a loop?",          "What is a loop?"),
    ("how's a function defined?", "How does a function defined?"),
    ("why's this important?",   "Why is this important?"),
    ("where's the error?",      "Where is the error?"),
    ("who's responsible?",      "Who is responsible?"),
    ("it's a variable",         "It is a variable?"),
    ("don't use globals",       "Do not use globals?"),
    ("doesn't work",            "Does not work?"),
    ("can't compile",           "Cannot compile?"),
    ("isn't valid syntax",      "Is not valid syntax?"),
])
def test_normalize_expands_contraction(raw, expected_fragment):
    assert _normalize_question(raw) == expected_fragment


@pytest.mark.parametrize("raw", [
    "what is a variable",
    "explain loops",
    "define a function",
])
def test_normalize_always_ends_with_question_mark(raw):
    assert _normalize_question(raw).endswith("?")


@pytest.mark.parametrize("raw", [
    "what is a variable?",
    "How does a loop work?",
    "Why is indentation required?",
])
def test_normalize_no_double_question_mark(raw):
    assert _normalize_question(raw).count("?") == 1


class TestNormalizeQuestion:
    def test_expands_whats_contraction(self):
        assert _normalize_question("what's a loop?") == "What is a loop?"

    def test_capitalizes_first_letter(self):
        assert _normalize_question("what is a variable?")[0].isupper()

    def test_adds_question_mark(self):
        result = _normalize_question("what is a function")
        assert result.endswith("?")

    def test_does_not_double_question_mark(self):
        result = _normalize_question("What is a class?")
        assert result.count("?") == 1

    def test_already_formal_unchanged(self):
        q = "What is a variable in Python?"
        assert _normalize_question(q) == q

    def test_expands_dont(self):
        result = _normalize_question("why don't we use global variables?")
        assert "do not" in result

    def test_strips_whitespace(self):
        result = _normalize_question("  what's a loop?  ")
        assert result == "What is a loop?"


class TestStripMeta:
    def test_removes_student_question_echo(self):
        text = "A variable stores data. Student question: What is a variable?"
        result = _strip_meta(text)
        assert "Student question:" not in result
        assert "A variable stores data." in result

    def test_removes_inst_tags(self):
        text = "[INST] some junk [/INST] Real answer here."
        result = _strip_meta(text)
        assert "[INST]" not in result
        assert "[/INST]" not in result

    def test_removes_tutor_answer_label(self):
        result = _strip_meta("Tutor answer: Here is the explanation.")
        assert "Tutor answer:" not in result

    def test_strips_empty_lines(self):
        text = "Line one.\n\n\nLine two."
        result = _strip_meta(text)
        lines = result.splitlines()
        assert all(line.strip() for line in lines)

    def test_plain_text_unchanged(self):
        text = "A loop repeats code while a condition is true."
        result = _strip_meta(text)
        assert result == text


class TestSplitNumberedSections:
    def test_splits_three_sections(self):
        text = "1. First section. 2. Second section. 3. Third section."
        sections = _split_numbered_sections(text)
        assert len(sections) == 3

    def test_no_numbers_returns_whole_text(self):
        text = "No numbered sections here at all."
        sections = _split_numbered_sections(text)
        assert len(sections) == 1
        assert sections[0] == text.strip()

    def test_empty_string_returns_empty_list(self):
        assert _split_numbered_sections("") == []

    def test_sections_start_with_number(self):
        text = "1. Core idea. 2. Example. 3. Mistake."
        sections = _split_numbered_sections(text)
        for section in sections:
            assert section[0].isdigit()


class TestRestructureFinetuned:
    def test_three_sections_formatted(self):
        text = "1. Core idea here. 2. Example code here. 3. Common mistake here."
        result = _restructure_finetuned(text)
        assert "Core Idea" in result
        assert "Step-by-Step Example" in result
        assert "Common Mistake" in result

    def test_empty_input_returns_empty(self):
        assert _restructure_finetuned("") == ""

    def test_adds_check_question_when_missing(self):
        text = "1. Core idea. 2. Example here. 3. Watch out for this mistake."
        result = _restructure_finetuned(text)
        assert "?" in result

    def test_no_double_spaces(self):
        text = "1. Core idea text.  2. Example text.  3. Mistake text."
        result = _restructure_finetuned(text)
        assert "  " not in result

    def test_single_section_falls_back_to_cleaned(self):
        text = "Just a single paragraph with no numbered sections at all."
        result = _restructure_finetuned(text)
        assert result  # non-empty, graceful fallback
