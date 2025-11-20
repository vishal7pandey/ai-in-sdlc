"""API-facing request/response schemas for the orchestrator endpoints."""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel

if TYPE_CHECKING:  # pragma: no cover
    from datetime import datetime

    from . import GraphState, Message
else:  # pragma: no cover - runtime aliasing for Pydantic
    datetime = import_module("datetime").datetime
    GraphState = import_module("src.schemas").GraphState
    Message = import_module("src.schemas").Message


class CreateSessionRequest(BaseModel):
    """Request body for creating a new session."""

    project_name: str
    user_id: str | None = None
    metadata: dict[str, Any] | None = None


class SessionResponse(BaseModel):
    """Basic session representation returned from session endpoints."""

    id: str
    project_name: str
    user_id: str
    status: str
    created_at: datetime
    updated_at: datetime | None = None


class SessionDetailResponse(SessionResponse):
    """Session detail including latest orchestrator state when available."""

    state: GraphState | None = None


class HumanReviewDecision(BaseModel):
    """Payload capturing a human decision at the review interrupt point."""

    approval_status: Literal["pending", "approved", "revision_requested"]
    review_feedback: str | None = None


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


class OrchestratorTurnResponse(BaseModel):
    """Unified response for orchestrator turns, including interrupts."""

    status: Literal["ok", "interrupt"]
    interrupt_type: str | None = None
    state: GraphState


class RDResponse(BaseModel):
    """Response model for RD generation and retrieval endpoints."""

    session_id: str
    version: int
    content: str
    format: Literal["markdown", "json", "pdf"]
    status: Literal["draft", "under_review", "approved"]
