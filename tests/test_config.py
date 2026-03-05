from pathlib import Path

from ai_tutor.config import Config


def test_project_root_is_path():
    assert isinstance(Config.project_root, Path)


def test_project_root_exists():
    assert Config.project_root.exists()


def test_derived_paths_are_under_project_root():
    assert Config.data_dir == Config.project_root / "data"
    assert Config.models_dir == Config.project_root / "models"
    assert Config.rag_index_dir == Config.project_root / "rag_index"


def test_base_model_id_is_string():
    assert isinstance(Config.base_model_id, str)
    assert len(Config.base_model_id) > 0


def test_embedding_model_id_is_string():
    assert isinstance(Config.embedding_model_id, str)


def test_api_port_is_int():
    assert isinstance(Config.api_port, int)


def test_eval_results_path_has_json_extension():
    assert Config.eval_results_path.suffix == ".json"
