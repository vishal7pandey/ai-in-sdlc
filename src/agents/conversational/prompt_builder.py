"""Build conversational agent prompts from templates and context."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from src.schemas import Message

TEMPLATE_PATH = Path(__file__).resolve().parents[2] / "templates" / "prompts" / "conversational.txt"


class PromptBuilder:
    """Utility to render the conversational agent system prompt."""

    def __init__(self, template_path: Path | None = None) -> None:
        self.template_path = template_path or TEMPLATE_PATH
        self._template_cache: str | None = None

    def build(
        self,
        *,
        project_name: str,
        current_turn: int,
        requirements_count: int,
        context: dict[str, Any],
        ambiguity_result: dict[str, Any],
        chat_history: list[Message],
    ) -> str:
        template = self._load_template()
        rendered_history = self._render_history(chat_history)
        tone = "supportive" if ambiguity_result.get("is_ambiguous") else "confident"

        template_variables = {
            "project_name": project_name,
            "current_turn": current_turn,
            "requirement_count": requirements_count,
            "chat_history": rendered_history,
            "conversation_summary": context.get("conversation_summary", "No conversation yet."),
            "clarification_gaps": self._format_list(context.get("clarification_gaps")),
            "identified_domain": context.get("identified_domain", "unknown"),
            "tone": tone,
            "mentioned_actors": self._format_list(context.get("mentioned_actors")),
            "mentioned_features": self._format_list(context.get("mentioned_features")),
            "implicit_needs": self._format_list(context.get("implicit_needs")),
            "conversation_momentum": f"{context.get('conversation_momentum', 0.5):.2f}",
            "ambiguity_terms": self._format_list(ambiguity_result.get("ambiguous_terms")),
            "ambiguity_questions": self._format_list(ambiguity_result.get("clarifying_questions")),
            "turn_sentiment": context.get("turn_sentiment", "neutral"),
        }

        return template.format(**template_variables)

    def _load_template(self) -> str:
        if self._template_cache is None:
            self._template_cache = self.template_path.read_text(encoding="utf-8")
        return self._template_cache

    def _render_history(self, chat_history: list[Message]) -> str:
        if not chat_history:
            return "(no prior messages)"

        lines = []
        for msg in chat_history[-8:]:
            lines.append(f"[{msg.role.upper()}] {msg.content}")
        return "\n".join(lines)

    def _format_list(self, items: list[str] | None) -> str:
        if not items:
            return "None"
        return ", ".join(items)
