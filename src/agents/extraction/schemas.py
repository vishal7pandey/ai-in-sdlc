"""Schemas for extraction agent outputs."""

from __future__ import annotations

from dataclasses import dataclass, field
from importlib import import_module
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

if TYPE_CHECKING:  # pragma: no cover - type checking only
    from src.schemas.requirement import RequirementItem
else:  # pragma: no cover - runtime alias to avoid circular imports
    RequirementItem = import_module("src.schemas.requirement").RequirementItem


class ExtractionMetadata(BaseModel):
    """Operational metadata returned by the extraction agent."""

    tokens_used: int = 0
    duration_ms: float = 0.0
    model: str = ""


class ExtractionOutput(BaseModel):
    """Validated structure parsed from LLM extraction output."""

    requirements: list[RequirementItem] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0, default=0.5)
    ambiguous_items: list[str] = Field(default_factory=list)
    metadata: ExtractionMetadata = Field(default_factory=ExtractionMetadata)


@dataclass(slots=True)
class ExtractionPromptContext:
    """Helper dataclass for prompt building."""

    recent_messages: list[str] = field(default_factory=list)
    requirements_so_far: int = 0
    conversation_summary: str = ""
    pending_clarifications: list[str] = field(default_factory=list)
    ambiguous_terms: list[str] = field(default_factory=list)
    project_name: str = ""
    sentiment: str = "neutral"


__all__ = [
    "ExtractionMetadata",
    "ExtractionOutput",
    "ExtractionPromptContext",
]
