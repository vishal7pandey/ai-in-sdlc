"""Acceptance criteria generation helpers."""

from __future__ import annotations

from typing import ClassVar

from src.schemas.requirement import RequirementType


class CriteriaGenerator:
    """Generate testable acceptance criteria templates."""

    TEMPLATES: ClassVar[dict[str, list[str]]] = {
        "authentication": [
            "{actor} can enter valid {auth_method} credentials",
            "System validates {auth_method} against the directory",
            "Invalid attempts are rejected with an error message",
            "Successful authentication redirects {actor} to the dashboard",
        ],
        "performance": [
            "{action} completes within {threshold} in {percentile} of requests",
            "System measures response time and logs deviations",
            "Load tests at {load} concurrent users meet the threshold",
        ],
        "upload": [
            "{actor} can select a file from {device}",
            "System validates file format is {formats}",
            "Files larger than {size_limit} are rejected with guidance",
            "Upload progress is shown to {actor}",
        ],
        "generic": [
            "{actor} can {action} without errors",
            "System records the outcome of {action}",
            "Failures during {action} provide actionable feedback",
        ],
    }

    def generate(
        self,
        *,
        actor: str,
        action: str,
        req_type: RequirementType,
        context: dict[str, str] | None = None,
    ) -> list[str]:
        ctx = context.copy() if context else {}
        action_lower = action.lower()
        template_key = self._template_for(req_type, action_lower)
        ctx.setdefault("actor", actor)
        ctx.setdefault("action", action)
        ctx.setdefault("auth_method", "email and password")
        ctx.setdefault("threshold", "< 2 seconds")
        ctx.setdefault("percentile", "95%")
        ctx.setdefault("load", "1,000")
        ctx.setdefault("device", "their device")
        ctx.setdefault("formats", "JPEG, PNG")
        ctx.setdefault("size_limit", "5MB")

        templates = self.TEMPLATES.get(template_key, self.TEMPLATES["generic"])
        criteria: list[str] = []
        for template in templates:
            try:
                criteria.append(template.format(**ctx))
            except KeyError:
                continue

        if not criteria:
            criteria.append(f"{actor} can {action}")
        return criteria

    def _template_for(self, req_type: RequirementType, action: str) -> str:
        if req_type == RequirementType.NON_FUNCTIONAL or "load" in action or "response" in action:
            return "performance"
        if "auth" in action or req_type == RequirementType.SECURITY:
            return "authentication"
        if "upload" in action:
            return "upload"
        return "generic"
