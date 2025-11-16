"""Tests for the TypeClassifier helper."""

from src.agents.extraction import TypeClassifier
from src.schemas import RequirementType


def test_classifies_functional_requirement() -> None:
    classifier = TypeClassifier()
    text = "Users should be able to create invoices and delete drafts."
    assert classifier.classify(text) == RequirementType.FUNCTIONAL


def test_classifies_non_functional_requirement() -> None:
    classifier = TypeClassifier()
    text = "The response time must stay under 2 seconds under peak load."
    assert classifier.classify(text) == RequirementType.NON_FUNCTIONAL


def test_classifies_security_requirement() -> None:
    classifier = TypeClassifier()
    text = "Passwords must be encrypted and token-based authentication enforced."
    assert classifier.classify(text) == RequirementType.SECURITY
