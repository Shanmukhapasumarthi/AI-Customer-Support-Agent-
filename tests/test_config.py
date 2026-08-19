from __future__ import annotations
import pytest
from pydantic import ValidationError
from app.config import PROJECT_ROOT, Settings, get_settings


def test_project_root_is_correct() -> None:
    """PROJECT_ROOT should point at the folder containing requirements.txt."""
    assert (PROJECT_ROOT / "requirements.txt").exists()
    assert (PROJECT_ROOT / "app").is_dir()


def test_settings_load_from_env_file() -> None:
    """The real .env should load and produce sane values."""
    settings = get_settings()
    assert settings.groq_api_key, "GROQ_API_KEY is empty -- did you create .env?"
    assert settings.groq_model
    assert 0.0 <= settings.llm_temperature <= 2.0
    assert settings.retriever_top_k > 0


def test_get_settings_is_cached() -> None:
    """get_settings() must return the SAME object, not a fresh parse."""
    assert get_settings() is get_settings()


def test_paths_are_absolute() -> None:
    """Path properties must be absolute so cwd never matters."""
    settings = get_settings()
    assert settings.knowledge_base_path.is_absolute()
    assert settings.vector_store_path.is_absolute()
    assert settings.sqlite_path.is_absolute()
    assert settings.sqlite_path.name.endswith(".db")


def test_missing_api_key_is_rejected() -> None:
    """A required field with no value must raise, not silently default."""
    with pytest.raises(ValidationError):
        # _env_file=None disables .env loading so the field is truly absent.
        Settings(_env_file=None, groq_api_key=None)  # type: ignore[arg-type]


def test_overlap_larger_than_chunk_is_rejected() -> None:
    """Our custom validator must catch this classic RAG misconfiguration."""
    with pytest.raises(ValidationError, match="must be smaller than"):
        Settings(
            _env_file=None,
            groq_api_key="test-key",
            chunk_size=100,
            chunk_overlap=200,
        )


def test_invalid_log_level_is_rejected() -> None:
    with pytest.raises(ValidationError, match="LOG_LEVEL must be one of"):
        Settings(_env_file=None, groq_api_key="test-key", log_level="LOUD")


def test_log_level_is_uppercased() -> None:
    """Lowercase in .env should still work."""
    s = Settings(_env_file=None, groq_api_key="test-key", log_level="debug")
    assert s.log_level == "DEBUG"
