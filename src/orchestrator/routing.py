"""Routing functions for the LangGraph orchestrator.

These helpers decide which node should run next based on the current
GraphState. They are used when wiring conditional edges in the graph
definition.
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.schemas import GraphState
else:  # pragma: no cover - runtime alias for type evaluation
    GraphState = import_module("src.schemas").GraphState


def decide_next_step(state: GraphState) -> str:
    """Route from conversational node to the next step.

    Decision logic:
    1. Escalate to an error handler if error count is too high.
    2. Follow the last suggested next action from the conversational agent.
    3. If the last user message explicitly asks for a document, extract.
    4. If we have requirements but they have not been validated, validate.
    5. Otherwise, continue the conversation (END for this graph run).
    """

    # Error escalation
    if state.error_count >= 3:
        return "error_handler"

    # Conversational agent sets "last_next_action"; interpret it here.
    next_action = getattr(state, "last_next_action", "continue_eliciting")
    if next_action == "extract_requirements":
        return "extract"
    if next_action == "validate":
        return "validate"
    if next_action == "synthesize":
        return "synthesis"

    # Check user intent from last message
    if state.chat_history:
        # Look at the last user message, if any
        user_messages = [m for m in state.chat_history if m.role == "user"]
        if user_messages:
            content = user_messages[-1].content.lower()
            # Treat common document- or requirements-focused phrases as a
            # signal to run extraction, even if the conversational agent did
            # not explicitly set nextAction to "extract_requirements".
            intent_keywords = (
                "generate",
                "create document",
                "show me",
                "draft",
                "capture the requirements",
                "capture requirements",
                "extract requirements",
                "requirements we've discussed",
            )
            if any(keyword in content for keyword in intent_keywords):
                return "extract"

    # If we have requirements but no validation status, route to validation.
    if state.requirements and not state.validation_issues:
        return "validate"

    # Default: continue conversation (END this orchestrator run, wait for
    # a new user input to re-invoke the graph).
    return "continue"


def validation_router(state: GraphState) -> str:
    """Route from validation node based on validation results.

    Decision logic:
    1. If confidence < threshold or critical issues exist → "fail".
    2. If we have requirements but no inferred ones yet → "needs_inference".
    3. Otherwise → "pass" (proceed to synthesis).
    """

    confidence = state.confidence
    issues = state.validation_issues
    inferred = state.inferred_requirements
    requirements = state.requirements

    if confidence < 0.60:
        return "fail"

    if any(issue.get("severity") == "critical" for issue in issues):
        return "fail"

    if requirements and not inferred:
        return "needs_inference"

    return "pass"


def review_router(state: GraphState) -> str:
    """Route from review node based on approval status.

    Decision logic:
    1. If approved → "approved" (END).
    2. If revision requested → "revision" (back to conversation).
    3. Otherwise → "pending" (END, waiting for human decision).
    """

    approval_status = state.approval_status

    if approval_status == "approved":
        return "approved"
    if approval_status == "revision_requested":
        return "revision"
    return "pending"


def synthesis_router(state: GraphState) -> str:
    """Route from synthesis node to human review or auto-approval.

    Decision logic (Story 7):
    - If confidence is low, or there are critical validation issues, or
      the session is large/high value → require human review.
    - Otherwise → auto-approve and proceed directly to the review node.
    """

    requires_review = (
        state.confidence < 0.70
        or any(i.get("severity") == "critical" for i in state.validation_issues)
        or len(state.requirements) >= 10
    )

    if requires_review:
        return "human_review"

    return "auto_approve"


def should_continue_iteration(state: GraphState) -> bool:
    """Return True if the orchestrator should continue iterating.

    Safety limits:
    - Max 10 iterations per session.
    - Max 5 errors.
    """

    max_iterations = 10
    max_errors = 5

    if state.iterations >= max_iterations:
        return False

    return state.error_count < max_errors
