from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from src.agents.base import AgentResult, BaseAgent
from src.schemas.requirement import Priority, Requirement, RequirementType
from src.utils.logging import get_logger, log_with_context

if TYPE_CHECKING:  # pragma: no cover - type checking only
    from collections.abc import Iterable

    from src.schemas import GraphState

logger = get_logger(__name__)


@dataclass
class InferenceRule:
    """Simple rule for inferring implicit requirements.

    This is a lightweight, deterministic implementation based on Story 7's
    HIGH_CONFIDENCE_RULES concept. Rules are intentionally conservative and
    only cover a few common patterns (e.g. authentication) to avoid noisy
    suggestions.
    """

    trigger_keywords: list[str]
    category: str
    confidence: float
    requirement_template: dict[str, Any]


class InferenceAgent(BaseAgent):
    """Agent that proposes implicit (inferred) requirements.

    For STORY-007 we implement a rule-based agent that looks at existing
    explicit requirements and, when certain high-confidence patterns are
    detected, auto-adds clearly marked inferred requirements (REQ-INF-xxx)
    to the ``inferred_requirements`` collection on GraphState.
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(name="inference", temperature=0.0, **kwargs)
        self.rules: list[InferenceRule] = self._load_inference_rules()

    def _load_inference_rules(self) -> list[InferenceRule]:
        """Load a small set of high-confidence inference rules.

        These rules are intentionally narrow and focused on common
        authentication/login scenarios so we can ship a safe, predictable
        implementation without over-inference.
        """

        return [
            # Password hashing for any login-style requirement.
            InferenceRule(
                trigger_keywords=[
                    "login",
                    "log in",
                    "signin",
                    "sign in",
                    "authenticate",
                    "password",
                ],
                category="security",
                confidence=0.95,
                requirement_template={
                    "title": "User passwords must be securely hashed",
                    "actor": "system",
                    "action": "hash user passwords using a strong one-way algorithm with a unique salt per password",
                    "condition": "when storing user credentials",
                    "acceptance_criteria": [
                        "Passwords are never stored in plaintext",
                        "Each password uses a unique cryptographic salt",
                        "A modern algorithm such as bcrypt or Argon2 is used with an appropriate cost factor",
                    ],
                    "type": RequirementType.SECURITY,
                    "priority": Priority.MUST,
                    "rationale": "Protect user credentials in case of database compromise by never storing raw passwords.",
                },
            ),
            # Rate limiting for failed login attempts.
            InferenceRule(
                trigger_keywords=[
                    "login",
                    "log in",
                    "signin",
                    "sign in",
                    "authenticate",
                    "password",
                ],
                category="security",
                confidence=0.9,
                requirement_template={
                    "title": "Failed login attempts must be rate-limited",
                    "actor": "system",
                    "action": "rate-limit repeated failed authentication attempts from the same account or IP address",
                    "condition": "when a user submits incorrect credentials",
                    "acceptance_criteria": [
                        "After N consecutive failed login attempts, further attempts are temporarily blocked",
                        "Rate-limiting thresholds and lockout durations are configurable",
                        "Rate-limiting events are logged for security monitoring",
                    ],
                    "type": RequirementType.SECURITY,
                    "priority": Priority.HIGH,
                    "rationale": "Reduce the risk of credential stuffing and brute-force attacks on the login endpoint.",
                },
            ),
        ]

    async def execute(self, state: GraphState) -> AgentResult:
        correlation_id = state.correlation_id or "unknown"
        log_with_context(
            logger,
            "info",
            "Inference agent started",
            agent=self.name,
            session_id=state.session_id,
            turn=state.current_turn,
            correlation_id=correlation_id,
        )

        new_inferred = self._infer_requirements(state)

        if not new_inferred:
            log_with_context(
                logger,
                "info",
                "Inference agent completed (no new inferences)",
                agent=self.name,
                session_id=state.session_id,
                inferred_count=0,
                correlation_id=correlation_id,
            )
            return {
                "inferred_requirements_update": [],
                "state_updates": {"last_agent": self.name},
            }

        all_inferred = [*state.inferred_requirements, *new_inferred]
        session_confidence = self._calculate_session_confidence(state.requirements, all_inferred)

        log_with_context(
            logger,
            "info",
            "Inference agent completed",
            agent=self.name,
            session_id=state.session_id,
            inferred_count=len(new_inferred),
            session_confidence=session_confidence,
            correlation_id=correlation_id,
        )

        return {
            "inferred_requirements_update": new_inferred,
            "state_updates": {
                "confidence": session_confidence,
                "last_agent": self.name,
            },
        }

    def _infer_requirements(self, state: GraphState) -> list[Requirement]:
        """Apply inference rules to existing explicit requirements.

        Only requirements with ``inferred == False`` are considered as
        triggers. For each requirement, at most one rule is applied to keep
        behaviour simple and predictable.
        """

        if not state.requirements:
            return []

        existing_ids = {req.id for req in state.requirements}
        existing_ids.update(req.id for req in state.inferred_requirements)

        inferred: list[Requirement] = []

        for req in state.requirements:
            if req.inferred:
                continue

            text = f"{req.title} {req.action}".lower()
            for rule in self.rules:
                if not any(keyword in text for keyword in rule.trigger_keywords):
                    continue

                new_id = self._next_inferred_id(existing_ids)
                existing_ids.add(new_id)

                tmpl = rule.requirement_template
                source_refs = req.source_refs or ["chat:turn:1"]

                payload = {
                    "id": new_id,
                    "title": tmpl["title"],
                    "type": tmpl.get("type", RequirementType.NON_FUNCTIONAL),
                    "actor": tmpl["actor"],
                    "action": tmpl["action"],
                    "condition": tmpl.get("condition"),
                    "acceptance_criteria": list(tmpl.get("acceptance_criteria", [])),
                    "priority": tmpl.get("priority", Priority.MEDIUM),
                    "confidence": rule.confidence,
                    "inferred": True,
                    "rationale": tmpl.get(
                        "rationale",
                        f"Inferred {rule.category} requirement based on explicit requirement {req.id}.",
                    ),
                    "source_refs": source_refs,
                }

                try:
                    inferred_req = Requirement(**payload)
                except Exception as exc:  # pragma: no cover - defensive
                    log_with_context(
                        logger,
                        "warning",
                        "Failed to build inferred requirement",
                        agent=self.name,
                        parent_req_id=req.id,
                        error=str(exc),
                    )
                    continue

                inferred.append(inferred_req)
                # Apply at most one rule per explicit requirement for now.
                break

        return inferred

    def _next_inferred_id(self, existing_ids: set[str]) -> str:
        """Return the next available REQ-INF-xxx identifier."""

        index = 1
        while True:
            candidate = f"REQ-INF-{index:03d}"
            if candidate not in existing_ids:
                return candidate
            index += 1

    def _calculate_session_confidence(
        self,
        explicit: Iterable[Requirement],
        inferred: Iterable[Requirement],
    ) -> float:
        """Weighted average confidence as per Story 7.

        - Explicit requirements weight: 1.0
        - Inferred requirements weight: 0.5
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
