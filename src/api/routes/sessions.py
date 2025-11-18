"""Session management endpoints (stubbed for STORY-001)."""

from __future__ import annotations

from datetime import datetime
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException, status
from langgraph.errors import GraphInterrupt
from langgraph.types import StateSnapshot
from sqlalchemy import select

from src.api.dependencies.auth import get_current_user
from src.models.database import SessionModel
from src.orchestrator.graph import graph
from src.orchestrator.state import create_initial_state
from src.schemas import GraphState, Message
from src.schemas.api import (
    CreateSessionRequest,
    HumanReviewDecision,
    OrchestratorTurnResponse,
    SendMessageRequest,
    SessionDetailResponse,
    SessionResponse,
)
from src.storage.postgres import get_session as get_db_session

router = APIRouter()


async def _not_implemented() -> None:
    raise HTTPException(
        status_code=status.HTTP_501_NOT_IMPLEMENTED,
        detail={
            "error": "not_implemented",
            "message": "This endpoint will be implemented in subsequent stories",
        },
    )


@router.post("/", response_model=SessionResponse, status_code=status.HTTP_201_CREATED)
async def create_session(
    payload: CreateSessionRequest,
    user_id: str = Depends(get_current_user),
) -> SessionResponse:
    """Create a new requirements engineering session.

    For STORY-004 we keep this minimal: a SessionModel row is created and
    returned; orchestrator state will be initialized lazily on the first
    message for this session.
    """

    async with get_db_session() as db:
        session = SessionModel(
            project_name=payload.project_name,
            user_id=payload.user_id or user_id,
            metadata_json=payload.metadata or {},
        )
        db.add(session)
        await db.commit()
        await db.refresh(session)

    return SessionResponse(
        id=str(session.id),
        project_name=session.project_name,
        user_id=session.user_id,
        status=session.status,
        created_at=session.created_at,
        updated_at=session.updated_at,
    )


@router.get("/", response_model=list[SessionResponse])
async def list_sessions() -> list[SessionResponse]:
    """List all sessions.

    This is a simple ordered listing by creation time.
    """

    async with get_db_session() as db:
        result = await db.execute(select(SessionModel).order_by(SessionModel.created_at.desc()))
        rows: list[SessionModel] = list(result.scalars().all())

    return [
        SessionResponse(
            id=str(row.id),
            project_name=row.project_name,
            user_id=row.user_id,
            status=row.status,
            created_at=row.created_at,
            updated_at=row.updated_at,
        )
        for row in rows
    ]


@router.get("/{session_id}", response_model=SessionDetailResponse)
async def get_session(session_id: str) -> SessionDetailResponse:
    """Get session metadata plus latest orchestrator state, if available."""

    async with get_db_session() as db:
        result = await db.execute(select(SessionModel).where(SessionModel.id == session_id))
        session_obj: SessionModel | None = result.scalar_one_or_none()

    if session_obj is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"error": "not_found", "message": "Session not found"},
        )

    # Try to load the latest orchestrator state via checkpointing.
    thread_id = f"session-{session_id}"
    config = {"configurable": {"thread_id": thread_id}}
    raw_state = await graph.aget_state(config)
    if isinstance(raw_state, StateSnapshot):
        raw_state = raw_state.values

    state: GraphState | None
    if not raw_state:
        state = None
    else:
        state = GraphState.model_validate(raw_state) if isinstance(raw_state, dict) else raw_state

    return SessionDetailResponse(
        id=str(session_obj.id),
        project_name=session_obj.project_name,
        user_id=session_obj.user_id,
        status=session_obj.status,
        created_at=session_obj.created_at,
        updated_at=session_obj.updated_at,
        state=state,
    )


@router.post("/{session_id}/messages", response_model=OrchestratorTurnResponse)
async def post_session_message(
    session_id: str,
    request: SendMessageRequest,
    user_id: str = Depends(get_current_user),
) -> OrchestratorTurnResponse:
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
    existing_state = await graph.aget_state(config)
    if isinstance(existing_state, StateSnapshot):
        existing_state = existing_state.values

    if not existing_state:
        # No prior orchestrator state: look up the session to derive
        # project_name, falling back to the request override or a
        # generic label.
        async with get_db_session() as db:
            result = await db.execute(select(SessionModel).where(SessionModel.id == session_id))
            session_obj: SessionModel | None = result.scalar_one_or_none()

        project_name = request.project_name or (
            session_obj.project_name if session_obj is not None else "Untitled Project"
        )

        state: GraphState = create_initial_state(
            session_id=session_id,
            project_name=project_name,
            user_id=user_id,
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

    # Run one graph step. If the graph hits a human_review interrupt point,
    # LangGraph will raise GraphInterrupt and we surface that as an
    # interrupt response with the latest checkpointed state.
    try:
        result_raw = await graph.ainvoke(state, config=config)
        result_state = (
            GraphState.model_validate(result_raw) if isinstance(result_raw, dict) else result_raw
        )
        return OrchestratorTurnResponse(
            status="ok",
            interrupt_type=None,
            state=result_state,
        )
    except GraphInterrupt as err:
        # Reload the latest state from the checkpointer; it will be the
        # state right before the human_review node.
        latest_raw = await graph.aget_state(config)
        if isinstance(latest_raw, StateSnapshot):
            latest_raw = latest_raw.values
        if latest_raw is None:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail={
                    "error": "interrupt_without_state",
                    "message": "Graph interrupted but no checkpoint state was found.",
                },
            ) from err

        latest_state = (
            GraphState.model_validate(latest_raw) if isinstance(latest_raw, dict) else latest_raw
        )

        return OrchestratorTurnResponse(
            status="interrupt",
            interrupt_type="human_review",
            state=latest_state,
        )


@router.post("/{session_id}/resume-human-review", response_model=OrchestratorTurnResponse)
async def resume_human_review(
    session_id: str,
    decision: HumanReviewDecision,
    user_id: str = Depends(get_current_user),
) -> OrchestratorTurnResponse:
    """Resume the graph after a human review decision.

    This endpoint expects that the previous run interrupted before the
    ``human_review`` node. It loads the latest state, applies the human's
    approval decision, and resumes the graph from that state.
    """

    thread_id = f"session-{session_id}"
    config = {"configurable": {"thread_id": thread_id}}

    raw_state = await graph.aget_state(config)
    if isinstance(raw_state, StateSnapshot):
        raw_state = raw_state.values

    if not raw_state:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "error": "no_checkpoint",
                "message": "No checkpointed state found to resume from.",
            },
        )

    state = GraphState.model_validate(raw_state) if isinstance(raw_state, dict) else raw_state

    updated_state = state.with_updates(
        approval_status=decision.approval_status,
        review_feedback=decision.review_feedback,
        last_agent="human_reviewer",
    )

    try:
        result_raw = await graph.ainvoke(updated_state, config=config)
        result_state = (
            GraphState.model_validate(result_raw) if isinstance(result_raw, dict) else result_raw
        )
        return OrchestratorTurnResponse(
            status="ok",
            interrupt_type=None,
            state=result_state,
        )
    except GraphInterrupt as err:
        latest_raw = await graph.aget_state(config)
        if isinstance(latest_raw, StateSnapshot):
            latest_raw = latest_raw.values

        if latest_raw is None:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail={
                    "error": "interrupt_without_state",
                    "message": "Graph interrupted but no checkpoint state was found.",
                },
            ) from err

        latest_state = (
            GraphState.model_validate(latest_raw) if isinstance(latest_raw, dict) else latest_raw
        )

        return OrchestratorTurnResponse(
            status="interrupt",
            interrupt_type="human_review",
            state=latest_state,
        )
