"""HTTP logging middleware for FastAPI."""

import logging
import time
from collections.abc import Callable

from fastapi import Request, Response

logger = logging.getLogger("reqeng.http")


async def logging_middleware(
    request: Request, call_next: Callable[[Request], Response]
) -> Response:
    """Log incoming HTTP requests and their response metadata."""
    start = time.perf_counter()
    logger.info("➡️ %s %s", request.method, request.url.path)

    response = await call_next(request)

    duration_ms = (time.perf_counter() - start) * 1000
    logger.info(
        "⬅️ %s %s -> %s (%.2f ms)",
        request.method,
        request.url.path,
        response.status_code,
        duration_ms,
    )

    response.headers["X-Process-Time-ms"] = f"{duration_ms:.2f}"
    return response
