"""Redis cache helpers."""

from __future__ import annotations

from redis.asyncio import Redis

from src.config import settings

_redis_client: Redis | None = None


def get_redis() -> Redis:
    global _redis_client
    if _redis_client is None:
        _redis_client = Redis.from_url(settings.REDIS_URL, encoding="utf-8", decode_responses=True)
    return _redis_client


async def init_redis() -> None:
    await get_redis().ping()


async def check_redis_health() -> bool:
    try:
        await get_redis().ping()
        return True
    except Exception:
        return False
