"""Chat-related schemas and utilities."""

from __future__ import annotations

from datetime import datetime  # noqa: TCH003
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

ChatRole = Literal["user", "assistant", "system"]


class Message(BaseModel):
    """Immutable chat message exchanged between user and agents."""

    id: str
    role: ChatRole
    content: str = Field(..., min_length=1)
    timestamp: datetime
    metadata: dict[str, str] | None = None

    model_config = ConfigDict(frozen=True)


__all__ = ["ChatRole", "Message"]
