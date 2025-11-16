"""Health check endpoint."""

from __future__ import annotations

import logging
from typing import Literal

from fastapi import APIRouter
from pydantic import BaseModel

from src.config import settings
from src.storage.postgres import check_postgres_health
from src.storage.redis_cache import check_redis_health
from src.storage.vectorstore import check_chromadb_health

logger = logging.getLogger("reqeng.health")

router = APIRouter()


class HealthResponse(BaseModel):
    status: Literal["healthy", "degraded", "unhealthy"]
    services: dict[str, Literal["up", "down"]]
    version: str


@router.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Report health for dependent services."""
    services = {
        "postgres": "up" if await check_postgres_health() else "down",
        "redis": "up" if await check_redis_health() else "down",
        "chromadb": "up" if await check_chromadb_health() else "down",
    }

    if all(status == "up" for status in services.values()):
        summary = "healthy"
    elif any(status == "up" for status in services.values()):
        summary = "degraded"
    else:
        summary = "unhealthy"

    logger.info("Health status: %s | %s", summary, services)

    return HealthResponse(status=summary, services=services, version=settings.APP_VERSION)
