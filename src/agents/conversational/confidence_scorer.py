"""Confidence scoring utilities for conversational responses."""

from __future__ import annotations

from dataclasses import dataclass

VAGUE_TERMS = {
    "fast",
    "quick",
    "easy",
    "scalable",
    "secure",
    "performant",
    "intuitive",
    "robust",
}

REQUIRED_FIELDS = ("message", "next_action", "confidence")


@dataclass(slots=True)
class ConfidenceScorer:
    """Compute final response confidence following STORY-002 formula."""

    def calculate(
        self,
        *,
        llm_confidence: float,
        parsed_fields: dict[str, bool],
        source_text: str,
    ) -> float:
        parse_quality = self._calc_parse_quality(parsed_fields)
        source_clarity = self._calc_source_clarity(source_text)
        score = 0.5 * self._clamp(llm_confidence) + 0.3 * parse_quality + 0.2 * source_clarity
        return self._clamp(score)

    def _calc_parse_quality(self, parsed_fields: dict[str, bool]) -> float:
        present = sum(1 for field in REQUIRED_FIELDS if parsed_fields.get(field))
        return present / len(REQUIRED_FIELDS)

    def _calc_source_clarity(self, text: str) -> float:
        lowered = text.lower()
        matches = sum(1 for term in VAGUE_TERMS if term in lowered)
        if matches == 0:
            return 1.0
        if matches <= 2:
            return 0.7
        return 0.4

    def _clamp(self, value: float) -> float:
        return max(0.0, min(value, 1.0))


__all__ = ["ConfidenceScorer"]
