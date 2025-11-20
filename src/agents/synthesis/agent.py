from __future__ import annotations

from typing import TYPE_CHECKING, Any

from src.agents.base import AgentResult, BaseAgent
from src.utils.logging import get_logger, log_with_context

if TYPE_CHECKING:  # pragma: no cover - type checking only
    from collections.abc import Iterable

    from src.schemas import GraphState, Requirement

logger = get_logger(__name__)


class SynthesisAgent(BaseAgent):
    """Generate a simple markdown Requirements Document from extracted requirements.

    This is an intentionally lightweight, deterministic implementation that does
    not call an LLM. It groups requirements by type and renders a markdown
    document with basic sections, traceability information, and acceptance
    criteria.
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(name="synthesis", temperature=0.0, **kwargs)

    async def execute(self, state: GraphState) -> AgentResult:
        correlation_id = state.correlation_id or "unknown"
        log_with_context(
            logger,
            "info",
            "Synthesis agent started",
            agent=self.name,
            session_id=state.session_id,
            requirements=len(state.requirements),
            correlation_id=correlation_id,
        )

        markdown = self._build_markdown(state.requirements, state.project_name)

        log_with_context(
            logger,
            "info",
            "Synthesis agent completed",
            agent=self.name,
            session_id=state.session_id,
            rd_length=len(markdown),
            correlation_id=correlation_id,
        )

        # rd_version is incremented by the node wrapper to keep concerns
        # separated; here we just return the content.
        return {
            "requirements_update": [],
            "state_updates": {
                "last_agent": self.name,
                "rd_draft": markdown,
            },
        }

    def _build_markdown(self, requirements: Iterable[Requirement], project_name: str) -> str:
        reqs = list(requirements)
        if not reqs:
            return (
                f"# Requirements Document for {project_name or 'Unnamed Project'}\n\n"
                "No requirements have been extracted yet."
            )

        header = f"# Requirements Document for {project_name or 'Unnamed Project'}\n\n"

        sections: dict[str, list[Requirement]] = {}
        for req in reqs:
            sections.setdefault(req.type.value, []).append(req)

        lines: list[str] = [header.rstrip(), ""]

        # Render each requirement type in a separate section, in a stable order.
        for type_key in sorted(sections.keys()):
            pretty_name = type_key.replace("_", " ").title()
            lines.append(f"## {pretty_name} Requirements")
            lines.append("")

            for req in sections[type_key]:
                lines.extend(self._render_requirement(req))
                lines.append("")

        return "\n".join(lines).rstrip() + "\n"

    def _render_requirement(self, req: Requirement) -> list[str]:
        lines: list[str] = []
        lines.append(f"### {req.id}: {req.title}")
        lines.append("")
        lines.append(f"**Actor**: {req.actor}")
        lines.append(f"**Action**: {req.action}")
        if req.condition:
            lines.append(f"**Condition**: {req.condition}")
        lines.append("")

        if req.acceptance_criteria:
            lines.append("**Acceptance Criteria:**")
            for criterion in req.acceptance_criteria:
                lines.append(f"- {criterion}")
            lines.append("")

        if req.source_refs:
            joined_refs = ", ".join(req.source_refs)
            lines.append(f"**Source**: {joined_refs}")
        lines.append(f"**Confidence**: {req.confidence * 100:.0f}%")

        return lines
