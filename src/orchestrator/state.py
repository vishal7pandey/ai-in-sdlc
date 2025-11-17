"""Orchestrator-level state helpers.

This module provides a thin wrapper around the core GraphState model used
throughout the application so the LangGraph orchestrator can construct and
reason about state in a single place.
"""

from __future__ import annotations

from src.schemas import GraphState as _GraphState

# Re-export the canonical GraphState model for orchestrator callers.
GraphState = _GraphState


def create_initial_state(session_id: str, project_name: str, user_id: str) -> GraphState:
    """Create initial graph state for a new session.

    The underlying GraphState model (src.schemas.state.GraphState) defines
    defaults for all optional fields, so we only need to supply the required
    identifiers here.
    """

    return GraphState(
        session_id=session_id,
        project_name=project_name,
        user_id=user_id,
    )
