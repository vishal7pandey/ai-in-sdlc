"""LangGraph node wrappers around existing agents.

Each node is a thin async wrapper that:
- Logs start and completion events.
- Delegates to the underlying agent's ``invoke`` method.
- Returns an updated ``GraphState`` instance.

This keeps orchestration concerns separate from agent internals.
"""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING

from src.agents.conversational.agent import ConversationalAgent
from src.agents.extraction.agent import ExtractionAgent
from src.utils.logging import get_logger, log_with_context

if TYPE_CHECKING:  # pragma: no cover - type checking only
    from src.schemas import GraphState


logger = get_logger(__name__)

# Initialize shared agent instances (singleton-style) so the orchestrator
# does not repeatedly construct ChatOpenAI clients.
_conversational_agent = ConversationalAgent()
_extraction_agent = ExtractionAgent()


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
    """Placeholder inference node.

    Future stories can implement additional agents that infer implicit
    requirements or derive higher-level specs. For now this node is a
    pass-through that simply logs its execution.
    """

    correlation_id = state.correlation_id or "unknown"
    log_with_context(
        logger,
        "info",
        "Inference node executed (no-op)",
        agent="inference",
        session_id=state.session_id,
        turn=state.current_turn,
        correlation_id=correlation_id,
    )

    # No changes yet; return state unchanged.
    return state


async def validation_node(state: GraphState) -> GraphState:
    """Placeholder validation node.

    This will eventually host a validation/quality agent. For now it
    simply logs and returns the state unchanged so that routing logic can
    still be exercised in tests.
    """

    correlation_id = state.correlation_id or "unknown"
    log_with_context(
        logger,
        "info",
        "Validation node executed (no-op)",
        agent="validation",
        session_id=state.session_id,
        turn=state.current_turn,
        correlation_id=correlation_id,
    )

    return state


async def synthesis_node(state: GraphState) -> GraphState:
    """Placeholder synthesis node that creates a simple RD draft.

    A future story can replace this with a proper RD-generation agent.
    """

    correlation_id = state.correlation_id or "unknown"
    log_with_context(
        logger,
        "info",
        "Synthesis node executed",
        agent="synthesis",
        session_id=state.session_id,
        turn=state.current_turn,
        correlation_id=correlation_id,
    )

    # If an RD draft already exists, leave it alone; otherwise create a
    # minimal placeholder so downstream nodes and tests can assert on it.
    if state.rd_draft is None:
        draft = (
            f"Requirements draft for project '{state.project_name}' generated at "
            f"{datetime.utcnow().isoformat()}"
        )
        return state.with_updates(rd_draft=draft, rd_version=state.rd_version + 1)

    return state


async def review_node(state: GraphState) -> GraphState:
    """Placeholder review node.

    For now this node only logs; a real implementation would coordinate
    human-in-the-loop approval workflows.
    """

    correlation_id = state.correlation_id or "unknown"
    log_with_context(
        logger,
        "info",
        "Review node executed",
        agent="review",
        session_id=state.session_id,
        approval_status=state.approval_status,
        correlation_id=correlation_id,
    )

    return state
