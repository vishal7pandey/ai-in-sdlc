"""Ambiguity detection for extraction stage."""

from __future__ import annotations

from typing import ClassVar


class AmbiguityDetector:
    """Flags vague terms that require clarification."""

    AMBIGUOUS_TERMS: ClassVar[dict[str, list[str]]] = {
        "adjectives": [
            "fast",
            "quick",
            "easy",
            "simple",
            "intuitive",
            "secure",
            "scalable",
            "robust",
            "performant",
        ],
        "quantifiers": [
            "many",
            "few",
            "several",
            "various",
            "multiple",
        ],
        "verbs": [
            "optimize",
            "improve",
            "enhance",
            "maximize",
            "minimize",
        ],
    }

    def detect(self, text: str) -> dict[str, list[str] | float | bool]:
        lowered = text.lower()
        hits: list[str] = []
        for terms in self.AMBIGUOUS_TERMS.values():
            for term in terms:
                if term in lowered:
                    hits.append(term)
        score = min(len(hits) * 0.2, 1.0)
        return {
            "is_ambiguous": bool(hits),
            "ambiguous_terms": hits,
            "ambiguity_score": score,
            "suggestions": [
                f"Clarify what '{term}' means with measurable criteria" for term in hits
            ],
        }
