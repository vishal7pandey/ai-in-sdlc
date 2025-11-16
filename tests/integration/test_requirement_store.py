"""Integration tests for RequirementStore persistence."""

from __future__ import annotations

from datetime import datetime
from uuid import uuid4

import pytest

from src.models.database import SessionModel
from src.schemas import Priority, RequirementItem, RequirementType
from src.storage.postgres import get_session
from src.storage.requirement_store import RequirementStore


@pytest.mark.asyncio
async def test_save_and_retrieve_requirement_round_trip() -> None:
    store = RequirementStore()
    session_id = uuid4()

    requirement = RequirementItem(
        id="REQ-201",
        title="User authentication with email and password",
        type=RequirementType.FUNCTIONAL,
        actor="user",
        action="authenticate using email and password",
        condition="when accessing protected resources",
        acceptance_criteria=[
            "User can enter valid credentials",
            "System validates credentials before granting access",
        ],
        priority=Priority.MUST,
        confidence=0.92,
        inferred=False,
        rationale="Explicit login requirement from conversation",
        source_refs=["chat:turn:0"],
        created_at=datetime.utcnow(),
    )

    async with get_session() as db:
        # Ensure the parent session row exists to satisfy FK constraints
        db_session = SessionModel(
            id=session_id,
            project_name="ReqStoreTest",
            user_id="user-req",
        )
        db.add(db_session)
        await db.commit()
        await db.refresh(db_session)

        saved = await store.save_requirement(db, session_id, requirement)
        assert saved.id == requirement.id
        assert saved.session_id == session_id

    async with get_session() as db:
        loaded = await store.get_requirements(db, session_id)

    assert len(loaded) == 1
    loaded_req = loaded[0]
    assert loaded_req.id == requirement.id
    assert loaded_req.title == requirement.title
    assert loaded_req.actor == requirement.actor
    assert loaded_req.source_refs == requirement.source_refs


@pytest.mark.asyncio
async def test_non_functional_type_mapping_round_trip() -> None:
    store = RequirementStore()
    session_id = uuid4()

    requirement = RequirementItem(
        id="REQ-202",
        title="Dashboard loads in under 2 seconds",
        type=RequirementType.NON_FUNCTIONAL,
        actor="system",
        action="load the dashboard content",
        condition=None,
        acceptance_criteria=[
            "Dashboard loads in under two seconds for 95% of requests",
        ],
        priority=Priority.HIGH,
        confidence=0.85,
        inferred=False,
        rationale="Performance target captured during elicitation",
        source_refs=["chat:turn:1"],
        created_at=datetime.utcnow(),
    )

    async with get_session() as db:
        db_session = SessionModel(
            id=session_id,
            project_name="ReqStoreTestNF",
            user_id="user-req",
        )
        db.add(db_session)
        await db.commit()
        await db.refresh(db_session)

        await store.save_requirement(db, session_id, requirement)

    async with get_session() as db:
        loaded = await store.get_requirements(db, session_id)

    assert len(loaded) == 1
    loaded_req = loaded[0]
    assert loaded_req.type == RequirementType.NON_FUNCTIONAL
    assert loaded_req.priority == Priority.HIGH
