"""Application configuration powered by Pydantic settings."""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Centralized environment configuration."""

    # OpenAI
    OPENAI_API_KEY: str = "sk-placeholder"
    OPENAI_MODEL: str = "gpt-4-turbo-preview"
    OPENAI_EMBEDDING_MODEL: str = "text-embedding-3-small"

    # Database & cache
    DATABASE_URI: str = "postgresql+asyncpg://reqeng:password@localhost:5432/reqengdb"
    REDIS_URL: str = "redis://localhost:6379/0"
    CHROMA_HOST: str = "localhost"
    CHROMA_PORT: int = 8001

    # App metadata
    APP_ENV: str = "development"
    APP_NAME: str = "req-eng-platform"
    APP_VERSION: str = "1.0.0"
    LOG_LEVEL: str = "DEBUG"

    # Security
    SECRET_KEY: str = "development-secret-key"
    JWT_SECRET: str = "development-jwt-secret"
    JWT_ALGORITHM: str = "HS256"
    JWT_EXPIRY_HOURS: int = 24

    # CORS
    CORS_ORIGINS: str = "http://localhost:3000,http://localhost:8000"

    # Agent tuning
    MAX_ITERATIONS: int = 10
    CONFIDENCE_THRESHOLD: float = 0.60
    VALIDATION_AMBIGUITY_THRESHOLD: float = 0.75
    SEMANTIC_SIMILARITY_THRESHOLD: float = 0.85

    # LangSmith optional
    LANGSMITH_API_KEY: str | None = None
    LANGSMITH_PROJECT: str = "req-eng-local"
    LANGSMITH_TRACING: bool = False
    LANGSMITH_ENDPOINT: str = "https://api.smith.langchain.com"

    model_config = SettingsConfigDict(env_file=".env", case_sensitive=True)


@lru_cache
def get_settings() -> Settings:
    return Settings()  # pragma: no cover - simple factory


settings = get_settings()
