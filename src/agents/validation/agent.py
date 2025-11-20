from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Any

from src.agents.base import AgentResult, BaseAgent
from src.utils.logging import get_logger, log_with_context

if TYPE_CHECKING:  # pragma: no cover - type checking only
    from collections.abc import Iterable

    from src.schemas import GraphState
    from src.schemas.requirement import Requirement

logger = get_logger(__name__)


class ValidationAgent(BaseAgent):
    """Agent that validates requirement quality and adjusts confidence.

    This is a pragmatic subset of the full Story 7 design:
    - Runs lightweight structural/content checks on each requirement.
    - Emits structured validation issues (dicts) into ``validation_issues``.
    - Adjusts per-requirement confidence based on issues.
    - Recomputes overall session confidence as a weighted average.
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(name="validation", temperature=0.0, **kwargs)

    async def execute(self, state: GraphState) -> AgentResult:
        correlation_id = state.correlation_id or "unknown"
        log_with_context(
            logger,
            "info",
            "Validation agent started",
            agent=self.name,
            session_id=state.session_id,
            turn=state.current_turn,
            correlation_id=correlation_id,
        )

        issue_counter = 1
        issues: list[dict[str, Any]] = []
        updated_requirements: list[Requirement] = []
        updated_inferred: list[Requirement] = []

        now_iso = datetime.utcnow().isoformat()

        for req in state.requirements:
            req_issues, issue_counter = self._validate_requirement(
                req,
                inferred=False,
                start_index=issue_counter,
                timestamp=now_iso,
            )
            issues.extend(req_issues)
            new_conf = self._adjust_confidence(req, req_issues)
            updated_requirements.append(req.model_copy(update={"confidence": new_conf}))

        for req in state.inferred_requirements:
            req_issues, issue_counter = self._validate_requirement(
                req,
                inferred=True,
                start_index=issue_counter,
                timestamp=now_iso,
            )
            issues.extend(req_issues)
            new_conf = self._adjust_confidence(req, req_issues)
            updated_inferred.append(req.model_copy(update={"confidence": new_conf}))

        session_conf = self._calculate_session_confidence(updated_requirements, updated_inferred)

        log_with_context(
            logger,
            "info",
            "Validation agent completed",
            agent=self.name,
            session_id=state.session_id,
            total_issues=len(issues),
            session_confidence=session_conf,
            correlation_id=correlation_id,
        )

        return {
            "state_updates": {
                "requirements": updated_requirements,
                "inferred_requirements": updated_inferred,
                "validation_issues": issues,
                "confidence": session_conf,
                "last_agent": self.name,
            }
        }

    def _validate_requirement(
        self,
        req: Requirement,
        *,
        inferred: bool,
        start_index: int,
        timestamp: str,
    ) -> tuple[list[dict[str, Any]], int]:
        """Run basic validation checks and return issues + next index.

        This is intentionally lightweight but compatible with the Story 7
        ValidationIssue schema. We currently focus on content quality (title
        length and vague terms); structural fields are already enforced by
        Pydantic and the database schema.
        """

        # Currently explicit and inferred requirements share the same checks;
        # keep the flag around for future divergence while marking it as used
        # for linters.
        _ = inferred

        issues: list[dict[str, Any]] = []
        idx = start_index

        # Title length check (clarity)
        if len(req.title) < 10 or len(req.title) > 80:
            issues.append(
                self._make_issue(
                    index=idx,
                    requirement_id=req.id,
                    severity="warning",
                    category="clarity",
                    message="Title length is outside the recommended 10-80 character range.",
                    suggested_fix="Rephrase the title to be concise and descriptive.",
                    field="title",
                    timestamp=timestamp,
                )
            )
            idx += 1

        # Vague language in title/action
        combined = f"{req.title} {req.action}".lower()
        vague_terms = ["fast", "quick", "user-friendly", "easy to use"]
        if any(term in combined for term in vague_terms):
            issues.append(
                self._make_issue(
                    index=idx,
                    requirement_id=req.id,
                    severity="warning",
                    category="clarity",
                    message="Requirement contains vague language (e.g. 'fast', 'user-friendly').",
                    suggested_fix="Replace vague terms with specific, measurable criteria (e.g. '< 2 seconds').",
                    field="action",
                    timestamp=timestamp,
                )
            )
            idx += 1

        return issues, idx

    def _make_issue(
        self,
        *,
        index: int,
        requirement_id: str,
        severity: str,
        category: str,
        message: str,
        suggested_fix: str | None,
        field: str | None,
        timestamp: str,
    ) -> dict[str, Any]:
        issue_id = f"VAL-{index:03d}"
        return {
            "issue_id": issue_id,
            "requirement_id": requirement_id,
            "severity": severity,
            "category": category,
            "message": message,
            "suggested_fix": suggested_fix,
            "field": field,
            "timestamp": timestamp,
        }

    def _adjust_confidence(self, req: Requirement, issues: Iterable[dict[str, Any]]) -> float:
        """Adjust requirement confidence based on attached issues.

        Mirrors the Story 7 logic:
          - Critical: -0.3
          - Warning:  -0.1
          - Info:     -0.05
        with a minimum confidence floor of 0.1.
        """

        confidence = float(req.confidence)

        for issue in issues:
            severity = issue.get("severity")
            if severity == "critical":
                confidence -= 0.3
            elif severity == "warning":
                confidence -= 0.1
            elif severity == "info":
                confidence -= 0.05

        confidence = max(0.1, min(1.0, confidence))
        return confidence

    def _calculate_session_confidence(
        self,
        explicit: Iterable[Requirement],
        inferred: Iterable[Requirement],
    ) -> float:
        """Weighted session confidence from Story 7.

        - Explicit requirements: weight 1.0
        - Inferred requirements: weight 0.5
        """

        explicit_list = list(explicit)
        inferred_list = list(inferred)

        if not explicit_list and not inferred_list:
            return 0.0

        total_weight = 0.0
        weighted_sum = 0.0

        for req in explicit_list:
            weight = 1.0
            total_weight += weight
            weighted_sum += float(req.confidence) * weight

        for req in inferred_list:
            weight = 0.5
            total_weight += weight
            weighted_sum += float(req.confidence) * weight

        return weighted_sum / total_weight if total_weight > 0 else 0.0
