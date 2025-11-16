"""Prompt builder for the Extraction Agent."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from .schemas import ExtractionPromptContext

if TYPE_CHECKING:
    from src.schemas import GraphState

TEMPLATE_PATH = Path("src/templates/prompts/extraction.txt")


def _all_messages(state: GraphState) -> list[str]:
    return [f"{msg.role.upper()}: {msg.content}" for msg in state.chat_history]


class ExtractionPromptBuilder:
    """Builds deterministic prompts for converting chat into requirements."""

    def __init__(self, template_path: Path | None = None) -> None:
        self.template_path = template_path or TEMPLATE_PATH
        self.template = self.template_path.read_text(encoding="utf-8")

    def build(self, state: GraphState, *, format_instructions: str = "") -> str:
        context = self._derive_context(state)
        pending = "\n".join(f"- {item}" for item in context.pending_clarifications) or "None"
        ambiguous = "\n".join(f"- {term}" for term in context.ambiguous_terms) or "None"
        recent_messages = "\n".join(context.recent_messages) or "(no recent messages)"

        instructions = format_instructions or "Provide JSON output"
        return self.template.format(
            project_name=context.project_name or state.project_name,
            requirements_so_far=context.requirements_so_far,
            conversation_summary=context.conversation_summary or "Summary unavailable",
            sentiment=context.sentiment,
            recent_messages=recent_messages,
            pending_clarifications=pending,
            ambiguous_terms=ambiguous,
            format_instructions=instructions,
        )

    def _derive_context(self, state: GraphState) -> ExtractionPromptContext:
        recent = _all_messages(state)[-6:]
        convo_ctx = state.conversation_context or {}
        ambiguity_assessment = state.ambiguity_assessment or {}

        return ExtractionPromptContext(
            recent_messages=recent,
            requirements_so_far=len(state.requirements),
            conversation_summary=convo_ctx.get("conversation_summary", ""),
            pending_clarifications=state.pending_clarifications,
            ambiguous_terms=ambiguity_assessment.get("ambiguous_terms", []),
            project_name=state.project_name,
            sentiment=convo_ctx.get("turn_sentiment", state.last_sentiment),
        )
