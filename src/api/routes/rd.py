from __future__ import annotations

from typing import TYPE_CHECKING
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
from langgraph.types import StateSnapshot
from sqlalchemy import select

from src.api.dependencies.auth import get_current_user
from src.models.database import RDDocumentModel, SessionModel
from src.orchestrator import nodes as orchestrator_nodes
from src.orchestrator.graph import graph
from src.orchestrator.state import create_initial_state
from src.schemas import GraphState
from src.schemas.api import RDResponse
from src.storage.postgres import get_session as get_db_session

if TYPE_CHECKING:  # pragma: no cover - type checking only
    from sqlalchemy.ext.asyncio import AsyncSession

router = APIRouter()


async def _get_session_obj(db: AsyncSession, session_id: str) -> SessionModel:
    result = await db.execute(select(SessionModel).where(SessionModel.id == session_id))
    session_obj: SessionModel | None = result.scalar_one_or_none()
    if session_obj is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"error": "not_found", "message": "Session not found"},
        )
    return session_obj


@router.post("/{session_id}/generate", response_model=RDResponse)
async def generate_rd(session_id: str, user_id: str = Depends(get_current_user)) -> RDResponse:
    """Generate an RD draft for the given session using the orchestrator graph.

    This endpoint:
    - Loads the latest GraphState via LangGraph checkpoints (or initializes one
      if none exists yet).
    - Invokes the graph, allowing the synthesis node to populate rd_draft.
    - Persists the resulting RD into RDDocumentModel.
    - Returns the markdown RD content.
    """

    thread_id = f"session-{session_id}"
    config = {"configurable": {"thread_id": thread_id}}

    async with get_db_session() as db:
        session_obj = await _get_session_obj(db, session_id)

    existing_state = await graph.aget_state(config)
    if isinstance(existing_state, StateSnapshot):
        existing_state = existing_state.values

    if not existing_state:
        state: GraphState = create_initial_state(
            session_id=session_id,
            project_name=session_obj.project_name,
            user_id=user_id,
        )
    else:
        state = (
            GraphState.model_validate(existing_state)
            if isinstance(existing_state, dict)
            else existing_state
        )

    # Rather than invoking the full graph (which may or may not route to the
    # synthesis node depending on conversational state), call the synthesis
    # node directly on the latest GraphState. This keeps RD generation
    # deterministic and avoids extra LLM calls.
    result_state = await orchestrator_nodes.synthesis_node(state)

    async with get_db_session() as db:
        rd_row = RDDocumentModel(
            session_id=UUID(session_id),
            version=result_state.rd_version,
            content=result_state.rd_draft or "",
            format="markdown",
            status="draft",
            metadata_json={"generated_by": user_id},
        )
        db.add(rd_row)
        await db.commit()
        await db.refresh(rd_row)

    return RDResponse(
        session_id=session_id,
        version=result_state.rd_version,
        content=result_state.rd_draft,
        format="markdown",
        status="draft",
    )


@router.get("/{session_id}", response_model=RDResponse)
async def get_rd(session_id: str) -> RDResponse:
    """Return the latest RD document for a session."""

    async with get_db_session() as db:
        await _get_session_obj(db, session_id)

        result = await db.execute(
            select(RDDocumentModel)
            .where(RDDocumentModel.session_id == UUID(session_id))
            .order_by(RDDocumentModel.version.desc())
        )
        rd_row: RDDocumentModel | None = result.scalar_one_or_none()

    if rd_row is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"error": "rd_not_found", "message": "No RD document found for this session"},
        )

    return RDResponse(
        session_id=str(rd_row.session_id),
        version=rd_row.version,
        content=rd_row.content,
        format=rd_row.format,
        status=rd_row.status,
    )


@router.get("/{session_id}/export")
async def export_rd(session_id: str) -> dict[str, str]:
    """Return RD markdown content in a simple JSON envelope.

    A real implementation might stream a file download; for STORY-006 we keep
    this minimal so the frontend can download/save the markdown.
    """

    async with get_db_session() as db:
        await _get_session_obj(db, session_id)

        result = await db.execute(
            select(RDDocumentModel)
            .where(RDDocumentModel.session_id == UUID(session_id))
            .order_by(RDDocumentModel.version.desc())
        )
        rd_row: RDDocumentModel | None = result.scalar_one_or_none()

    if rd_row is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"error": "rd_not_found", "message": "No RD document found for this session"},
        )

    return {"filename": f"requirements-{session_id}.md", "content": rd_row.content}
