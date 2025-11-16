"""Unit tests for health endpoint."""

import pytest

from src.api.routes import health


@pytest.mark.asyncio
async def test_health_endpoint(monkeypatch):
    async def _ok():
        return True

    for name in ("check_postgres_health", "check_redis_health", "check_chromadb_health"):
        monkeypatch.setattr(health, name, _ok)

    response = await health.health_check()

    assert response.status == "healthy"
    assert response.services == {
        "postgres": "up",
        "redis": "up",
        "chromadb": "up",
    }
