"""Session management endpoints (stubbed for STORY-001)."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, status

router = APIRouter()


async def _not_implemented() -> None:
    raise HTTPException(
        status_code=status.HTTP_501_NOT_IMPLEMENTED,
        detail={
            "error": "not_implemented",
            "message": "This endpoint will be implemented in subsequent stories",
        },
    )


@router.post("/")
async def create_session():
    """Stub create session endpoint."""
    await _not_implemented()


@router.get("/")
async def list_sessions():
    """Stub list sessions endpoint."""
    await _not_implemented()


@router.get("/{session_id}")
async def get_session(session_id: str):
    """Stub get session endpoint."""
    await _not_implemented()


@router.post("/{session_id}/messages")
async def post_session_message(session_id: str):
    """Stub send message endpoint."""
    await _not_implemented()
