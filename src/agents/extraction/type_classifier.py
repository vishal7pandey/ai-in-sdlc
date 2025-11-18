"""Requirement type classification helpers."""

from __future__ import annotations

from typing import ClassVar

from src.schemas import RequirementType


class TypeClassifier:
    """Rule-based requirement classification."""

    INDICATORS: ClassVar[dict[RequirementType, tuple[str, ...]]] = {
        RequirementType.FUNCTIONAL: (
            "user can",
            "system shall",
            "allow",
            "create",
            "update",
            "delete",
            "upload",
            "download",
        ),
        RequirementType.NON_FUNCTIONAL: (
            "performance",
            "latency",
            "throughput",
            "reliability",
            "availability",
            "fast",
            "scalable",
            "response time",
            "load",
            "seconds",
            "uptime",
        ),
        RequirementType.SECURITY: (
            "authenticate",
            "authorize",
            "encryption",
            "token",
            "password",
            "secure",
        ),
        RequirementType.DATA: (
            "database",
            "schema",
            "persist",
            "retention",
            "migration",
        ),
        RequirementType.INTERFACE: (
            "api",
            "endpoint",
            "integration",
            "webhook",
            "ui",
            "screen",
        ),
        RequirementType.BUSINESS: (
            "policy",
            "regulation",
            "compliance",
            "sla",
            "kpi",
            "roi",
        ),
        RequirementType.CONSTRAINT: (
            "budget",
            "timeline",
            "deadline",
            "technology",
            "platform",
            "limit",
        ),
    }

    def classify(self, text: str, *, action: str = "") -> RequirementType:
        """Return the most likely requirement type."""

        haystack = f"{text} {action}".lower()
        scores: dict[RequirementType, int] = dict.fromkeys(RequirementType, 0)
        for rtype, indicators in self.INDICATORS.items():
            for indicator in indicators:
                if indicator.lower() in haystack:
                    scores[rtype] += 1

        best_rtype, best_score = max(scores.items(), key=lambda item: item[1])
        if best_score == 0:
            return RequirementType.FUNCTIONAL
        return best_rtype
