"""Authentication-related dependencies for FastAPI routes.

This is intentionally lightweight for STORY-004: it provides a minimal
"current user" abstraction that can later be replaced with real auth.
"""

from __future__ import annotations

from fastapi import Header


async def get_current_user(x_user_id: str | None = Header(default=None)) -> str:
    """Return the current user's identifier.

    For now we accept an optional ``X-User-Id`` header and fall back to a
    generic ``"anonymous"`` user when not provided. This keeps the
    orchestrator and sessions API wired for per-user context without
    introducing full authentication yet.
    """

    return x_user_id or "anonymous"
