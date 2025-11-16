"""Tests for the CriteriaGenerator helper."""

from src.agents.extraction import CriteriaGenerator
from src.schemas import RequirementType


def test_generates_authentication_criteria_defaults() -> None:
    generator = CriteriaGenerator()
    criteria = generator.generate(
        actor="user",
        action="authenticate using email and password",
        req_type=RequirementType.FUNCTIONAL,
    )

    assert len(criteria) >= 2
    assert any("validate" in criterion.lower() for criterion in criteria)


def test_generates_performance_criteria_with_context() -> None:
    generator = CriteriaGenerator()
    criteria = generator.generate(
        actor="system",
        action="load dashboard",
        req_type=RequirementType.NON_FUNCTIONAL,
        context={"threshold": "< 3 seconds", "percentile": "99%", "load": "5,000"},
    )

    assert any("3 seconds" in criterion for criterion in criteria)
    assert any("5,000" in criterion for criterion in criteria)
