"""API-facing request/response schemas for the orchestrator endpoints."""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING

from pydantic import BaseModel

if TYPE_CHECKING:  # pragma: no cover
    from . import GraphState, Message
else:  # pragma: no cover - runtime aliasing for Pydantic
    GraphState = import_module("src.schemas").GraphState
    Message = import_module("src.schemas").Message


class SendMessageRequest(BaseModel):
    """Request body for sending a chat message to a session.

    For the MVP we accept a plain text message and an optional project name
    (used only when creating a brand new session state).
    """

    message: str
    project_name: str | None = None


class ChatMessageResponse(BaseModel):
    """Response payload after invoking the orchestrator graph for a turn."""

    session_id: str
    project_name: str
    assistant_message: Message
    state: GraphState
