"""Heuristic ambiguity detection for conversational inputs."""

from __future__ import annotations

from dataclasses import dataclass

VAGUE_ADJECTIVES = {"fast", "easy", "intuitive", "scalable", "secure", "robust"}
MISSING_METRIC_TERMS = {"quick", "soon", "rapid", "performant"}


@dataclass(slots=True)
class ClarificationResult:
    is_ambiguous: bool
    ambiguity_score: float
    ambiguous_terms: list[str]
    clarifying_questions: list[str]
    confidence: float


class ClarificationDetector:
    """Detect whether a user input would benefit from clarification."""

    def detect(self, text: str) -> ClarificationResult:
        lowered = text.lower()
        ambiguous_terms = [word for word in VAGUE_ADJECTIVES if word in lowered]
        ambiguous_terms.extend(word for word in MISSING_METRIC_TERMS if word in lowered)

        needs_metrics = any(word in lowered for word in MISSING_METRIC_TERMS)
        questions: list[str] = []
        if "fast" in lowered or "quick" in lowered:
            questions.append("What response or load time would meet your expectations?")
        if "secure" in lowered:
            questions.append("Are there specific compliance or encryption standards to follow?")
        if "scalable" in lowered:
            questions.append("How many concurrent users or transactions should be supported?")

        is_ambiguous = bool(ambiguous_terms) or needs_metrics
        ambiguity_score = 0.2 + 0.15 * len(set(ambiguous_terms)) if is_ambiguous else 0.0
        ambiguity_score = min(1.0, ambiguity_score)
        confidence = 0.6 + 0.1 * len(questions) if is_ambiguous else 0.4
        confidence = min(0.95, confidence)

        if not questions and is_ambiguous:
            questions.append("Could you provide more specifics or success metrics?")

        return ClarificationResult(
            is_ambiguous=is_ambiguous,
            ambiguity_score=ambiguity_score,
            ambiguous_terms=sorted(set(ambiguous_terms)),
            clarifying_questions=questions,
            confidence=confidence,
        )
