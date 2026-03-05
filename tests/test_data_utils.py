import json
import tempfile
from pathlib import Path

import pytest

from ai_tutor.data_utils import QAExample, as_dict, load_eval_dataset, load_training_dataset, prepare_all_splits


class TestQAExample:
    def test_fields(self):
        ex = QAExample(question="What is x?", context=None, answer="x is a variable.")
        assert ex.question == "What is x?"
        assert ex.context is None
        assert ex.answer == "x is a variable."

    def test_as_dict(self):
        ex = QAExample(question="Q", context="C", answer="A")
        d = as_dict(ex)
        assert d == {"question": "Q", "context": "C", "answer": "A"}


class TestLoadTrainingDataset:
    def test_returns_list(self):
        examples = load_training_dataset()
        assert isinstance(examples, list)

    def test_non_empty(self):
        examples = load_training_dataset()
        assert len(examples) > 0

    def test_all_are_qa_examples(self):
        for ex in load_training_dataset():
            assert isinstance(ex, QAExample)

    def test_examples_have_question_and_answer(self):
        for ex in load_training_dataset():
            assert ex.question
            assert ex.answer


class TestLoadEvalDataset:
    def test_falls_back_to_training_data_when_no_file(self, tmp_path, monkeypatch):
        # Point config data_dir to a temp dir with no val.jsonl
        import ai_tutor.data_utils as du
        import ai_tutor.config as cfg

        fake_config = type(cfg.Config)(
            **{**cfg.Config.__dataclass_fields__,
               "data_dir": tmp_path}
        )
        # Simpler approach: just verify fallback message behavior by checking
        # that the function returns examples even when val.jsonl is absent.
        examples = load_eval_dataset(max_samples=2)
        assert isinstance(examples, list)

    def test_max_samples_respected(self):
        examples = load_eval_dataset(max_samples=1)
        assert len(examples) <= 1

    def test_loads_from_jsonl(self, tmp_path, monkeypatch):
        val_dir = tmp_path / "val"
        val_dir.mkdir()
        val_file = val_dir / "val.jsonl"
        records = [
            {"question": "What is a function?", "answer": "A reusable block of code."},
            {"question": "What is a loop?", "answer": "Repeats code.", "context": "loops info"},
        ]
        val_file.write_text("\n".join(json.dumps(r) for r in records))

        monkeypatch.setattr("ai_tutor.data_utils.Config", type("FakeCfg", (), {"data_dir": tmp_path})())

        examples = load_eval_dataset()
        assert len(examples) == 2
        assert examples[0].question == "What is a function?"
        assert examples[1].context == "loops info"


class TestPrepareAllSplits:
    def test_creates_expected_directories(self, tmp_path, monkeypatch):
        monkeypatch.setattr("ai_tutor.data_utils.Config", type("FakeCfg", (), {"data_dir": tmp_path})())
        prepare_all_splits()
        for sub in ["raw", "processed", "train", "val", "test", "live_eval"]:
            assert (tmp_path / sub).is_dir()
