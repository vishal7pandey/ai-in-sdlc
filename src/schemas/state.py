"""LangGraph state schemas shared across agents."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, ConfigDict, Field

if TYPE_CHECKING:
    from .chat import Message
    from .requirement import Requirement

ApprovalStatus = Literal["pending", "approved", "revision_requested"]


class GraphState(BaseModel):
    """Immutable representation of the orchestrator state passed between agents."""

    session_id: str
    project_name: str
    user_id: str

    chat_history: list[Message] = Field(default_factory=list)
    current_turn: int = 0

    requirements: list[Requirement] = Field(default_factory=list)
    inferred_requirements: list[Requirement] = Field(default_factory=list)
    extracted_topics: list[str] = Field(default_factory=list)
    pending_clarifications: list[str] = Field(default_factory=list)
    ambiguous_items: list[str] = Field(default_factory=list)

    validation_issues: list[dict[str, Any]] = Field(default_factory=list)
    confidence: float = Field(1.0, ge=0.0, le=1.0)
    last_next_action: str = "continue_eliciting"
    last_sentiment: str = "neutral"

    rd_draft: str | None = None
    rd_version: int = 0

    approval_status: ApprovalStatus = "pending"
    review_feedback: str | None = None
    extraction_metadata: dict[str, Any] | None = None

    last_agent: str = "system"
    iterations: int = 0
    error_count: int = 0

    checkpoint_id: str | None = None
    parent_checkpoint_id: str | None = None
    correlation_id: str | None = None
    conversation_context: dict[str, Any] | None = None
    ambiguity_assessment: dict[str, Any] | None = None

    model_config = ConfigDict(frozen=True)

    def with_updates(self, **updates: Any) -> GraphState:
        """Return a copy of the state with provided fields updated."""

        return self.model_copy(update=updates)

    @classmethod
    def field_names(cls) -> list[str]:
        """Return a list of valid GraphState field names."""

        return list(cls.model_fields.keys())


__all__ = ["ApprovalStatus", "GraphState"]
