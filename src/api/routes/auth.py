"""Authentication routes (stubbed)."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, status

router = APIRouter()


@router.post("/login")
async def login() -> None:
    """Placeholder login endpoint for future stories."""
    raise HTTPException(
        status_code=status.HTTP_501_NOT_IMPLEMENTED,
        detail={
            "error": "not_implemented",
            "message": "This endpoint will be implemented in subsequent stories",
        },
    )
