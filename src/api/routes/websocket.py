from __future__ import annotations

from datetime import datetime
from typing import Any
from uuid import UUID, uuid4

from fastapi import APIRouter, Query, WebSocket, WebSocketDisconnect
from langgraph.errors import GraphInterrupt
from langgraph.types import StateSnapshot
from sqlalchemy import select

from src.api.websocket.manager import manager
from src.models.database import SessionModel
from src.orchestrator.graph import graph
from src.orchestrator.state import GraphState, create_initial_state
from src.schemas import Message
from src.storage.postgres import get_session as get_db_session

router = APIRouter()


async def _load_existing_state(session_id: str) -> GraphState | None:
    thread_id = f"session-{session_id}"
    config = {"configurable": {"thread_id": thread_id}}
    existing_state = await graph.aget_state(config)
    if isinstance(existing_state, StateSnapshot):
        existing_state = existing_state.values
    if not existing_state:
        return None
    return (
        GraphState.model_validate(existing_state)
        if isinstance(existing_state, dict)
        else existing_state
    )


async def _get_or_create_state(
    session_id: str, user_id: str, project_name_override: str | None
) -> GraphState:
    existing = await _load_existing_state(session_id)
    if existing is not None:
        return existing
    async with get_db_session() as db:
        result = await db.execute(select(SessionModel).where(SessionModel.id == session_id))
        session_obj: SessionModel | None = result.scalar_one_or_none()
    project_name = project_name_override or (
        session_obj.project_name if session_obj is not None else "Untitled Project"
    )
    return create_initial_state(session_id=session_id, project_name=project_name, user_id=user_id)


async def _run_turn_for_message(session_id: str, user_id: str, content: str) -> GraphState:
    state = await _get_or_create_state(
        session_id=session_id, user_id=user_id, project_name_override=None
    )
    user_message = Message(
        id=str(uuid4()),
        role="user",
        content=content,
        timestamp=datetime.utcnow(),
        metadata={},
    )
    state = state.with_updates(
        chat_history=[*state.chat_history, user_message],
        current_turn=state.current_turn + 1,
    )
    thread_id = f"session-{session_id}"
    config = {"configurable": {"thread_id": thread_id}}
    try:
        result_raw: Any = await graph.ainvoke(state, config=config)
        result_state = (
            GraphState.model_validate(result_raw) if isinstance(result_raw, dict) else result_raw
        )
        return result_state
    except GraphInterrupt as err:
        latest_raw: Any = await graph.aget_state(config)
        if isinstance(latest_raw, StateSnapshot):
            latest_raw = latest_raw.values
        if latest_raw is None:
            raise err
        return GraphState.model_validate(latest_raw) if isinstance(latest_raw, dict) else latest_raw


@router.websocket("/ws")
async def websocket_endpoint(
    websocket: WebSocket,
    session_id: str = Query(...),
    user_id: str = Query(default="anonymous"),
) -> None:
    try:
        UUID(session_id)
    except ValueError:
        await websocket.accept()
        await websocket.send_json(
            {
                "type": "error.connection_rejected",
                "code": 4000,
                "reason": "invalid_session_id",
                "message": f"session_id {session_id!r} is not a valid UUID",
            }
        )
        await websocket.close(code=4000, reason="invalid_session_id")
        return

    async with get_db_session() as db:
        result = await db.execute(select(SessionModel).where(SessionModel.id == session_id))
        session_obj: SessionModel | None = result.scalar_one_or_none()

    if session_obj is None:
        await websocket.accept()
        await websocket.send_json(
            {
                "type": "error.connection_rejected",
                "code": 4004,
                "reason": "session_not_found",
                "message": f"Session {session_id} does not exist",
            }
        )
        await websocket.close(code=4004, reason="session_not_found")
        return

    if session_obj.status == "archived":
        await websocket.accept()
        await websocket.send_json(
            {
                "type": "error.connection_rejected",
                "code": 4003,
                "reason": "session_archived",
                "message": f"Session {session_id} is archived",
            }
        )
        await websocket.close(code=4003, reason="session_archived")
        return

    await manager.connect(session_id, websocket)

    await websocket.send_json(
        {
            "type": "connection.established",
            "session_id": session_id,
            "server_time": datetime.utcnow().isoformat(),
            "protocol_version": "1.0",
            "features": {
                "streaming": True,
                "heartbeat_interval": 30,
                "max_message_size": 10485760,
            },
        }
    )

    try:
        while True:
            data = await websocket.receive_json()
            msg_type = data.get("type")
            if msg_type == "ping":
                await websocket.send_json(
                    {
                        "type": "pong",
                        "session_id": session_id,
                        "timestamp": data.get("timestamp") or datetime.utcnow().isoformat(),
                        "server_time": datetime.utcnow().isoformat(),
                    }
                )
            elif msg_type == "chat.message":
                content = data.get("content")
                if not isinstance(content, str) or not content.strip():
                    await websocket.send_json(
                        {
                            "type": "error",
                            "session_id": session_id,
                            "error_code": "invalid_payload",
                            "error_message": "content is required",
                            "severity": "error",
                            "timestamp": datetime.utcnow().isoformat(),
                        }
                    )
                    continue

                # Capture requirements before this turn so we can detect newly extracted items.
                previous_state = await _load_existing_state(session_id)
                previous_requirements = (
                    previous_state.requirements if previous_state is not None else []
                )
                previous_ids = {req.id for req in previous_requirements}

                await websocket.send_json(
                    {
                        "type": "agent.status",
                        "session_id": session_id,
                        "agent": "conversational",
                        "status": "started",
                        "timestamp": datetime.utcnow().isoformat(),
                    }
                )

                turn_started_at = datetime.utcnow()
                result_state = await _run_turn_for_message(
                    session_id=session_id,
                    user_id=user_id,
                    content=content,
                )
                run_duration_ms = int((datetime.utcnow() - turn_started_at).total_seconds() * 1000)

                new_requirements = result_state.requirements
                new_items = [req for req in new_requirements if req.id not in previous_ids]

                assistant_messages = [m for m in result_state.chat_history if m.role == "assistant"]
                if not assistant_messages:
                    await websocket.send_json(
                        {
                            "type": "error",
                            "session_id": session_id,
                            "error_code": "no_assistant_message",
                            "error_message": "No assistant message produced",
                            "severity": "warning",
                            "timestamp": datetime.utcnow().isoformat(),
                        }
                    )
                    await websocket.send_json(
                        {
                            "type": "agent.status",
                            "session_id": session_id,
                            "agent": "conversational",
                            "status": "failed",
                            "timestamp": datetime.utcnow().isoformat(),
                        }
                    )
                    continue

                assistant = assistant_messages[-1]
                message_id = assistant.id
                turn_number = result_state.current_turn
                full_content = assistant.content

                await websocket.send_json(
                    {
                        "type": "message.chunk",
                        "session_id": session_id,
                        "message_id": message_id,
                        "turn_number": turn_number,
                        "delta": full_content,
                        "metadata": {
                            "model": "",
                            "tokens_used": 0,
                        },
                        "timestamp": datetime.utcnow().isoformat(),
                    }
                )

                await websocket.send_json(
                    {
                        "type": "message.complete",
                        "session_id": session_id,
                        "message_id": message_id,
                        "turn_number": turn_number,
                        "full_content": full_content,
                        "metadata": {
                            "model": "",
                            "total_tokens": 0,
                            "duration_ms": 0,
                            "confidence": 1.0,
                            "next_action": result_state.last_next_action,
                        },
                        "timestamp": datetime.utcnow().isoformat(),
                    }
                )

                await websocket.send_json(
                    {
                        "type": "agent.status",
                        "session_id": session_id,
                        "agent": "conversational",
                        "status": "completed",
                        "timestamp": datetime.utcnow().isoformat(),
                    }
                )

                # If new requirements were added in this turn, emit a
                # requirements.extracted event plus extraction agent status
                # updates so the frontend can update the sidebar in real time.
                if new_items:
                    await websocket.send_json(
                        {
                            "type": "agent.status",
                            "session_id": session_id,
                            "agent": "extraction",
                            "status": "started",
                            "timestamp": datetime.utcnow().isoformat(),
                        }
                    )

                    await websocket.send_json(
                        {
                            "type": "requirements.extracted",
                            "session_id": session_id,
                            "extraction_id": f"ext-{uuid4()}",
                            "requirements": [
                                {
                                    "id": req.id,
                                    "title": req.title[:50],
                                    "type": getattr(req.type, "value", str(req.type)),
                                    "confidence": float(req.confidence),
                                    "inferred": bool(req.inferred),
                                }
                                for req in new_items
                            ],
                            "metadata": {
                                "extraction_duration_ms": run_duration_ms,
                                "total_extracted": len(new_items),
                                "total_session_requirements": len(new_requirements),
                            },
                            "timestamp": datetime.utcnow().isoformat(),
                        }
                    )

                    await websocket.send_json(
                        {
                            "type": "agent.status",
                            "session_id": session_id,
                            "agent": "extraction",
                            "status": "completed",
                            "timestamp": datetime.utcnow().isoformat(),
                        }
                    )
            else:
                await websocket.send_json(
                    {
                        "type": "error",
                        "session_id": session_id,
                        "error_code": "unknown_event_type",
                        "error_message": f"Unknown message type: {msg_type}",
                        "severity": "warning",
                        "timestamp": datetime.utcnow().isoformat(),
                    }
                )
    except WebSocketDisconnect:
        manager.disconnect(session_id)
    except Exception as exc:
        await websocket.send_json(
            {
                "type": "error",
                "session_id": session_id,
                "error_code": "internal_error",
                "error_message": str(exc),
                "severity": "error",
                "timestamp": datetime.utcnow().isoformat(),
            }
        )
        manager.disconnect(session_id)
