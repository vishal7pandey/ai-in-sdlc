"""Unit tests for configuration loading."""

from src.config import settings


def test_config_loads_defaults():
    assert settings.APP_NAME == "req-eng-platform"
    assert settings.OPENAI_MODEL
    assert settings.DATABASE_URI.startswith("postgresql")
