"""Unit tests for orchestrator routing helpers."""

from __future__ import annotations

from datetime import datetime

from src.orchestrator.routing import (
    decide_next_step,
    review_router,
    should_continue_iteration,
    validation_router,
)
from src.schemas import GraphState, Message, Requirement


def _base_state() -> GraphState:
    return GraphState(
        session_id="sess-1",
        project_name="Demo",
        user_id="user-1",
    )


def test_decide_next_step_error_handler() -> None:
    state = _base_state().with_updates(error_count=3)
    assert decide_next_step(state) == "error_handler"


def test_decide_next_step_explicit_next_action() -> None:
    state = _base_state().with_updates(error_count=0, last_next_action="extract_requirements")
    assert decide_next_step(state) == "extract"


def test_decide_next_step_from_user_intent() -> None:
    msg = Message(
        id="m1",
        role="user",
        content="Please generate a requirements document for me",
        timestamp=datetime.utcnow(),
    )
    state = _base_state().with_updates(chat_history=[msg])

    assert decide_next_step(state) == "extract"


def test_decide_next_step_to_validation_when_requirements_present() -> None:
    req = Requirement(
        id="REQ-001",
        title="User can log in",
        actor="user",
        action="log in",
        acceptance_criteria=["User can log in with email and password"],
        rationale="This is a valid test rationale",
        source_refs=["chat:turn:0"],
    )
    state = _base_state().with_updates(requirements=[req], validation_issues=[])

    assert decide_next_step(state) == "validate"


def test_validation_router_fail_on_low_confidence() -> None:
    state = _base_state().with_updates(confidence=0.4)
    assert validation_router(state) == "fail"


def test_validation_router_fail_on_critical_issue() -> None:
    issues = [{"severity": "critical", "message": "Missing acceptance criteria"}]
    state = _base_state().with_updates(confidence=0.9, validation_issues=issues)

    assert validation_router(state) == "fail"


def test_validation_router_needs_inference_when_no_inferred_requirements() -> None:
    req = Requirement(
        id="REQ-002",
        title="Dashboard loads in under 2 seconds",
        actor="user",
        action="view dashboard",
        acceptance_criteria=["Dashboard loads in under 2 seconds"],
        rationale="Performance requirement",
        source_refs=["chat:turn:1"],
    )
    state = _base_state().with_updates(
        confidence=0.9,
        validation_issues=[],
        requirements=[req],
        inferred_requirements=[],
    )

    assert validation_router(state) == "needs_inference"


def test_validation_router_pass_when_confident_and_inferred_present() -> None:
    req = Requirement(
        id="REQ-003",
        title="Password reset",
        actor="user",
        action="reset password",
        acceptance_criteria=["User can reset password via email"],
        rationale="Security requirement",
        source_refs=["chat:turn:2"],
    )
    state = _base_state().with_updates(
        confidence=0.95,
        validation_issues=[],
        requirements=[req],
        inferred_requirements=[req],
    )

    assert validation_router(state) == "pass"


def test_review_router_routes_based_on_approval_status() -> None:
    assert review_router(_base_state().with_updates(approval_status="approved")) == "approved"
    assert (
        review_router(_base_state().with_updates(approval_status="revision_requested"))
        == "revision"
    )
    assert review_router(_base_state().with_updates(approval_status="pending")) == "pending"


def test_should_continue_iteration_limits_iterations_and_errors() -> None:
    assert should_continue_iteration(_base_state().with_updates(iterations=0, error_count=0))
    assert not should_continue_iteration(_base_state().with_updates(iterations=10, error_count=0))
    assert not should_continue_iteration(_base_state().with_updates(iterations=1, error_count=5))
