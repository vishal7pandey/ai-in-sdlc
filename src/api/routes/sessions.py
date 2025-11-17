"""Session management endpoints (stubbed for STORY-001)."""

from __future__ import annotations

from datetime import datetime
from uuid import uuid4

from fastapi import APIRouter, HTTPException, status

from src.orchestrator.graph import graph
from src.orchestrator.state import create_initial_state
from src.schemas import GraphState, Message
from src.schemas.api import ChatMessageResponse, SendMessageRequest

router = APIRouter()


async def _not_implemented() -> None:
    raise HTTPException(
        status_code=status.HTTP_501_NOT_IMPLEMENTED,
        detail={
            "error": "not_implemented",
            "message": "This endpoint will be implemented in subsequent stories",
        },
    )


@router.post("/")
async def create_session():
    """Stub create session endpoint."""
    await _not_implemented()


@router.get("/")
async def list_sessions():
    """Stub list sessions endpoint."""
    await _not_implemented()


@router.get("/{session_id}")
async def get_session(session_id: str):
    """Stub get session endpoint."""
    await _not_implemented()


@router.post("/{session_id}/messages")
async def post_session_message(
    session_id: str,
    request: SendMessageRequest,
) -> ChatMessageResponse:
    """Send a message into a session and run one orchestrator turn.

    This endpoint:
    - Loads the latest orchestrator state for the given session (if any)
      using LangGraph's checkpointing.
    - Appends the user message to the state and bumps the turn counter.
    - Invokes the LangGraph graph for a single step.
    - Returns the assistant's latest message and the updated state.
    """

    thread_id = f"session-{session_id}"
    config = {"configurable": {"thread_id": thread_id}}

    # Load previous state from checkpoints if available.
    existing_state = await graph.aget_state(config)  # type: ignore[attr-defined]
    if existing_state is None:
        state: GraphState = create_initial_state(
            session_id=session_id,
            project_name=request.project_name or "Untitled Project",
            user_id="anonymous",  # Placeholder until auth is wired in
        )
    else:
        # LangGraph may return a plain dict; coerce to GraphState.
        state = (
            GraphState.model_validate(existing_state)
            if isinstance(existing_state, dict)
            else existing_state
        )

    # Add the new user message.
    user_message = Message(
        id=str(uuid4()),
        role="user",
        content=request.message,
        timestamp=datetime.utcnow(),
        metadata={},
    )

    state = state.with_updates(
        chat_history=[*state.chat_history, user_message],
        current_turn=state.current_turn + 1,
    )

    # Run one graph step.
    result_raw = await graph.ainvoke(state, config=config)
    result_state = (
        GraphState.model_validate(result_raw) if isinstance(result_raw, dict) else result_raw
    )

    # Pick the last assistant message as the primary response.
    assistant_messages = [m for m in result_state.chat_history if m.role == "assistant"]
    if not assistant_messages:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": "no_assistant_message",
                "message": "Orchestrator did not produce an assistant response.",
            },
        )

    return ChatMessageResponse(
        session_id=session_id,
        project_name=result_state.project_name,
        assistant_message=assistant_messages[-1],
        state=result_state,
    )
