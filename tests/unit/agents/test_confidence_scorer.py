"""Unit tests for confidence scoring logic."""

from __future__ import annotations

from src.agents.conversational.confidence_scorer import ConfidenceScorer


def test_confidence_scorer_clear_input() -> None:
    scorer = ConfidenceScorer()
    score = scorer.calculate(
        llm_confidence=0.9,
        parsed_fields={"message": True, "next_action": True, "confidence": True},
        source_text="Users can log in with MFA and SSO",
    )
    assert 0.8 <= score <= 1.0


def test_confidence_scorer_handles_ambiguous_text() -> None:
    scorer = ConfidenceScorer()
    score = scorer.calculate(
        llm_confidence=0.6,
        parsed_fields={"message": True, "next_action": False, "confidence": True},
        source_text="It should be fast, easy, and highly scalable",
    )
    assert score < 0.7
