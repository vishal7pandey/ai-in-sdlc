"""Chat-related schemas and utilities."""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, ConfigDict, Field

if TYPE_CHECKING:  # pragma: no cover
    from datetime import datetime
else:  # pragma: no cover - runtime alias for Pydantic
    datetime = import_module("datetime").datetime

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
