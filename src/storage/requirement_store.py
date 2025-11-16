"""Requirement persistence helpers."""

from __future__ import annotations

from typing import TYPE_CHECKING

from sqlalchemy import select

if TYPE_CHECKING:
    from uuid import UUID

    from sqlalchemy.ext.asyncio import AsyncSession

from src.models.database import RequirementModel
from src.schemas import Priority, RequirementItem, RequirementType


class RequirementStore:
    """Persist and retrieve requirements for a session."""

    async def save_requirement(
        self,
        db: AsyncSession,
        session_id: UUID,
        requirement: RequirementItem,
    ) -> RequirementModel:
        """Save a single requirement to the database.

        The schema and DB model are slightly different (e.g. non-functional naming),
        so this method handles the mapping between them.
        """

        type_value = requirement.type.value
        db_type = "non-functional" if type_value == "non_functional" else type_value

        # Idempotent save: update existing row if the ID already exists.
        db_req = await db.get(RequirementModel, requirement.id)

        if db_req is None:
            db_req = RequirementModel(id=requirement.id)
            db.add(db_req)

        db_req.session_id = session_id
        db_req.title = requirement.title
        db_req.type = db_type
        db_req.actor = requirement.actor
        db_req.action = requirement.action
        db_req.condition = requirement.condition
        db_req.acceptance_criteria = list(requirement.acceptance_criteria)
        db_req.priority = requirement.priority.value
        db_req.confidence = requirement.confidence
        db_req.inferred = requirement.inferred
        db_req.rationale = requirement.rationale
        db_req.source_refs = list(requirement.source_refs)

        await db.commit()
        await db.refresh(db_req)
        return db_req

    async def get_requirements(
        self,
        db: AsyncSession,
        session_id: UUID,
    ) -> list[RequirementItem]:
        """Retrieve all requirements for a given session, ordered by creation time."""

        result = await db.execute(
            select(RequirementModel)
            .where(RequirementModel.session_id == session_id)
            .order_by(RequirementModel.created_at)
        )
        db_reqs = result.scalars().all()
        return [self._to_pydantic(req) for req in db_reqs]

    def _to_pydantic(self, db_req: RequirementModel) -> RequirementItem:
        """Convert a DB row into a RequirementItem model."""

        # Map DB type back to schema enum value
        type_value = db_req.type
        type_value = "non_functional" if type_value == "non-functional" else type_value

        return RequirementItem(
            id=db_req.id,
            title=db_req.title,
            type=RequirementType(type_value),
            actor=db_req.actor,
            action=db_req.action,
            condition=db_req.condition,
            acceptance_criteria=list(db_req.acceptance_criteria or []),
            priority=Priority(db_req.priority),
            confidence=float(db_req.confidence),
            inferred=db_req.inferred,
            rationale=db_req.rationale,
            source_refs=list(db_req.source_refs or []),
            created_at=db_req.created_at,
        )
