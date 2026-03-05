from ai_tutor.graph.nodes.evaluate_node import evaluate_node
from ai_tutor.graph.nodes.fine_tuned_model_node import load_models_node
from ai_tutor.graph.nodes.load_config_node import load_config_node
from ai_tutor.graph.nodes.prepare_data_node import prepare_data_node


class TestLoadConfigNode:
    def test_sets_config_summary(self):
        state = load_config_node({})
        assert "config_summary" in state

    def test_summary_contains_project_root(self):
        state = load_config_node({})
        assert "Project root:" in state["config_summary"]

    def test_summary_contains_model_id(self):
        state = load_config_node({})
        assert "Base model ID:" in state["config_summary"]

    def test_returns_dict(self):
        assert isinstance(load_config_node({}), dict)


class TestPrepareDataNode:
    def test_sets_data_preview(self):
        state = prepare_data_node({})
        assert "data_preview" in state

    def test_preview_is_non_empty_string(self):
        state = prepare_data_node({})
        assert isinstance(state["data_preview"], str)
        assert state["data_preview"].strip()

    def test_preview_contains_question_label(self):
        state = prepare_data_node({})
        assert "Q:" in state["data_preview"]


class TestLoadModelsNode:
    def test_is_noop_returns_state(self):
        initial = {"config_summary": "test"}
        result = load_models_node(initial)
        assert result == initial

    def test_does_not_add_unexpected_keys(self):
        result = load_models_node({})
        assert result == {}


class TestEvaluateNode:
    def test_sets_eval_summary(self):
        state = evaluate_node({})
        assert "eval_summary" in state

    def test_summary_mentions_run_eval(self):
        state = evaluate_node({})
        assert "run_eval.py" in state["eval_summary"]

    def test_returns_dict(self):
        assert isinstance(evaluate_node({}), dict)
