"""Schemas used by the conversational agent."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, model_validator

NextAction = Literal["continue_eliciting", "extract_requirements", "clarify", "wait_for_input"]
Sentiment = Literal["positive", "neutral", "negative"]


class ConversationalResponse(BaseModel):
    message: str = Field(..., min_length=1)
    next_action: NextAction
    clarifying_questions: list[str] | None = None
    confidence: float = Field(..., ge=0.0, le=1.0)
    extracted_topics: list[str] = Field(default_factory=list)
    sentiment: Sentiment = "neutral"

    @model_validator(mode="after")
    def validate_clarifications(
        self,
    ) -> ConversationalResponse:  # pragma: no cover - simple validation
        if self.next_action == "clarify" and not self.clarifying_questions:
            raise ValueError("clarifying_questions required when next_action is 'clarify'")
        if self.next_action != "clarify" and self.clarifying_questions:
            raise ValueError("clarifying_questions must be empty unless next_action is 'clarify'")
        return self


__all__ = ["ConversationalResponse", "NextAction", "Sentiment"]
