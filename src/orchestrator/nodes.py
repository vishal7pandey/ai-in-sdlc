"""LangGraph node wrappers around existing agents.

Each node is a thin async wrapper that:
- Logs start and completion events.
- Delegates to the underlying agent's ``invoke`` method.
- Returns an updated ``GraphState`` instance.

This keeps orchestration concerns separate from agent internals.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from src.agents.conversational.agent import ConversationalAgent
from src.agents.extraction.agent import ExtractionAgent
from src.agents.inference.agent import InferenceAgent
from src.agents.synthesis.agent import SynthesisAgent
from src.agents.validation.agent import ValidationAgent
from src.utils.logging import get_logger, log_with_context

if TYPE_CHECKING:  # pragma: no cover - type checking only
    from src.schemas import GraphState


logger = get_logger(__name__)

# Initialize shared agent instances (singleton-style) so the orchestrator
# does not repeatedly construct ChatOpenAI clients.
_conversational_agent = ConversationalAgent()
_extraction_agent = ExtractionAgent()
_inference_agent = InferenceAgent()
_validation_agent = ValidationAgent()
_synthesis_agent = SynthesisAgent()


async def conversational_node(state: GraphState) -> GraphState:
    """LangGraph node wrapper for the ConversationalAgent."""

    correlation_id = state.correlation_id or "unknown"
    log_with_context(
        logger,
        "info",
        "Conversational node started",
        agent="conversational",
        session_id=state.session_id,
        turn=state.current_turn,
        correlation_id=correlation_id,
    )

    new_state = await _conversational_agent.invoke(state)

    log_with_context(
        logger,
        "info",
        "Conversational node completed",
        agent="conversational",
        confidence=new_state.confidence,
        iterations=new_state.iterations,
        correlation_id=correlation_id,
    )

    return new_state


async def extraction_node(state: GraphState) -> GraphState:
    """LangGraph node wrapper for the ExtractionAgent."""

    correlation_id = state.correlation_id or "unknown"
    log_with_context(
        logger,
        "info",
        "Extraction node started",
        agent="extraction",
        session_id=state.session_id,
        existing_requirements=len(state.requirements),
        correlation_id=correlation_id,
    )

    new_state = await _extraction_agent.invoke(state)

    log_with_context(
        logger,
        "info",
        "Extraction node completed",
        agent="extraction",
        total_requirements=len(new_state.requirements),
        confidence=new_state.confidence,
        correlation_id=correlation_id,
    )

    return new_state


async def inference_node(state: GraphState) -> GraphState:
    """LangGraph node wrapper for the InferenceAgent.

    Applies a small set of high-confidence, rule-based inference rules to
    propose implicit requirements (e.g. security hardening around login
    flows). Results are written into ``inferred_requirements`` on the
    GraphState.
    """

    correlation_id = state.correlation_id or "unknown"
    log_with_context(
        logger,
        "info",
        "Inference node started",
        agent="inference",
        session_id=state.session_id,
        turn=state.current_turn,
        correlation_id=correlation_id,
    )

    new_state = await _inference_agent.invoke(state)

    log_with_context(
        logger,
        "info",
        "Inference node completed",
        agent="inference",
        session_id=new_state.session_id,
        inferred_requirements=len(new_state.inferred_requirements),
        confidence=new_state.confidence,
        correlation_id=correlation_id,
    )

    return new_state


async def validation_node(state: GraphState) -> GraphState:
    """LangGraph node wrapper for the ValidationAgent.

    Runs lightweight structural/content validation over all requirements
    and inferred requirements, updates per-requirement confidence, and
    populates ``validation_issues`` and the overall session confidence on
    the GraphState.
    """

    correlation_id = state.correlation_id or "unknown"
    log_with_context(
        logger,
        "info",
        "Validation node started",
        agent="validation",
        session_id=state.session_id,
        turn=state.current_turn,
        correlation_id=correlation_id,
    )

    new_state = await _validation_agent.invoke(state)

    log_with_context(
        logger,
        "info",
        "Validation node completed",
        agent="validation",
        session_id=new_state.session_id,
        issues=len(new_state.validation_issues),
        confidence=new_state.confidence,
        correlation_id=correlation_id,
    )

    return new_state


async def human_review_node(state: GraphState) -> GraphState:
    """Human review node used as an interrupt point in the workflow.

    The graph is configured to interrupt *before* this node executes, so
    reaching it indicates that a human decision is required before the
    workflow can proceed.
    """

    correlation_id = state.correlation_id or "unknown"
    log_with_context(
        logger,
        "info",
        "Human review interrupt reached",
        agent="human_review",
        session_id=state.session_id,
        approval_status=state.approval_status,
        correlation_id=correlation_id,
    )

    return state


async def synthesis_node(state: GraphState) -> GraphState:
    """Synthesis node that generates an RD draft using SynthesisAgent."""

    correlation_id = state.correlation_id or "unknown"
    log_with_context(
        logger,
        "info",
        "Synthesis node started",
        agent="synthesis",
        session_id=state.session_id,
        turn=state.current_turn,
        correlation_id=correlation_id,
    )

    # Delegate to the SynthesisAgent; it will populate rd_draft in the
    # returned state_updates. We then bump rd_version to reflect a new draft.
    agent_result = await _synthesis_agent.execute(state)
    rd_draft = agent_result["state_updates"].get("rd_draft")

    if rd_draft is None:
        # If, for some reason, no draft was produced, return the state
        # unchanged so downstream nodes are not surprised.
        return state

    new_state = state.with_updates(
        rd_draft=rd_draft,
        rd_version=state.rd_version + 1,
        last_agent="synthesis",
    )

    log_with_context(
        logger,
        "info",
        "Synthesis node completed",
        agent="synthesis",
        session_id=new_state.session_id,
        rd_version=new_state.rd_version,
        correlation_id=correlation_id,
    )

    return new_state


async def review_node(state: GraphState) -> GraphState:
    """Review node that finalizes approval status.

    If the workflow arrived here via the "auto_approve" path after
    synthesis, ``approval_status`` will still be ``"pending"`` and this
    node will mark the session as ``"approved"``. When resuming from a
    human review interrupt, the API sets ``approval_status`` to either
    ``"approved"`` or ``"revision_requested"`` and this node simply
    logs the outcome.
    """

    correlation_id = state.correlation_id or "unknown"

    # Auto-approve high-confidence sessions that skipped human review.
    if state.approval_status == "pending":
        new_state = state.with_updates(approval_status="approved")
    else:
        new_state = state

    log_with_context(
        logger,
        "info",
        "Review node executed",
        agent="review",
        session_id=new_state.session_id,
        approval_status=new_state.approval_status,
        correlation_id=correlation_id,
    )

    return new_state
