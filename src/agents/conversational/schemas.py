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
        if self.next_action == "clarify":
            # If we're in a clarification step but the model omitted questions,
            # treat that as an empty list rather than an error.
            if self.clarifying_questions is None:
                self.clarifying_questions = []
        else:
            # For non-clarification actions, ignore any stray clarifying
            # questions instead of failing validation.
            if self.clarifying_questions:
                self.clarifying_questions = None
        return self


__all__ = ["ConversationalResponse", "NextAction", "Sentiment"]
