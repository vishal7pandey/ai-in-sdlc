"""Requirement-related schemas."""

from __future__ import annotations

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, ConfigDict, Field, field_validator


class RequirementType(str, Enum):
    """Enumerated requirement classifications."""

    FUNCTIONAL = "functional"
    NON_FUNCTIONAL = "non_functional"
    BUSINESS = "business"
    SECURITY = "security"
    DATA = "data"
    INTERFACE = "interface"
    CONSTRAINT = "constraint"


class Priority(str, Enum):
    """Priority scale for captured requirements."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    MUST = "must"


class RequirementItem(BaseModel):
    """Immutable representation of a discovered requirement with validation."""

    id: str = Field(..., pattern=r"^REQ(?:-INF)?-\d{3}$")
    title: str = Field(..., min_length=10, max_length=500)
    type: RequirementType = RequirementType.FUNCTIONAL
    actor: str = Field(..., min_length=1, max_length=200)
    action: str = Field(..., min_length=5)
    condition: str | None = None
    acceptance_criteria: list[str] = Field(default_factory=list, min_length=1)
    priority: Priority = Priority.MEDIUM
    confidence: float = Field(0.0, ge=0.0, le=1.0)
    inferred: bool = False
    rationale: str = Field(..., min_length=20)
    source_refs: list[str] = Field(default_factory=list, min_length=1)
    created_at: datetime = Field(default_factory=datetime.utcnow)

    model_config = ConfigDict(frozen=True)

    @field_validator("acceptance_criteria")
    @classmethod
    def _validate_criteria(cls, value: list[str]) -> list[str]:
        for criterion in value:
            if len(criterion.strip()) < 10:
                raise ValueError(f"Acceptance criterion too short: {criterion!r}")
        return value

    @field_validator("source_refs")
    @classmethod
    def _validate_source_refs(cls, value: list[str]) -> list[str]:
        normalized: list[str] = []
        for ref in value:
            # Accept the canonical "chat:turn:N" format as-is.
            if ref.startswith("chat:turn:"):
                normalized.append(ref)
                continue

            # Also accept shorthand like "chat:N" and normalize it to
            # "chat:turn:N" so minor prompt drift doesn't break parsing.
            if ref.startswith("chat:"):
                suffix = ref.split(":", 1)[1]
                if suffix.isdigit():
                    normalized.append(f"chat:turn:{suffix}")
                    continue

            raise ValueError(f"Invalid source reference: {ref}")

        return normalized


# Backwards compatibility alias used throughout the codebase
Requirement = RequirementItem


__all__ = [
    "Priority",
    "Requirement",
    "RequirementItem",
    "RequirementType",
]
