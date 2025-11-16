"""Tests for the AmbiguityDetector helper."""

from src.agents.extraction import AmbiguityDetector


def test_detects_vague_language_and_suggestions() -> None:
    detector = AmbiguityDetector()
    text = "The app must be fast, intuitive, and highly scalable."

    result = detector.detect(text)

    assert result["is_ambiguous"] is True
    assert any(term in result["ambiguous_terms"] for term in ("fast", "intuitive"))
    assert result["ambiguity_score"] > 0
    assert any("Clarify" in suggestion for suggestion in result["suggestions"])
