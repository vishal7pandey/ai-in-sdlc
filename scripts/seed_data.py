"""Seed script to populate sample sessions for local testing."""

from __future__ import annotations

import asyncio
import logging
import sys
from pathlib import Path
from uuid import uuid4

from sqlalchemy import select

# Ensure project root import
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.database import RequirementModel, SessionModel  # noqa: E402
from src.storage.postgres import get_session  # noqa: E402

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("reqeng.seed")


async def seed() -> None:
    async with get_session() as session:
        demo_session = await session.scalar(
            select(SessionModel).where(SessionModel.project_name == "Demo Requirements")
        )
        if demo_session is None:
            demo_session = SessionModel(
                id=uuid4(),
                project_name="Demo Requirements",
                user_id="demo-user",
                status="active",
                metadata={"source": "seed"},
            )
            session.add(demo_session)
            logger.info("Created demo session")

        requirement = await session.get(RequirementModel, "REQ-001")
        if requirement is None:
            requirement = RequirementModel(
                id="REQ-001",
                session=demo_session,
                title="User can view dashboard",
                type="functional",
                actor="User",
                action="view dashboard",
                condition=None,
                acceptance_criteria=["Dashboard loads within 2 seconds"],
                priority="high",
                confidence=0.9,
                inferred=False,
                rationale="Core functionality",
                source_refs=["seed"],
            )
            session.add(requirement)
            logger.info("Created demo requirement")
        else:
            logger.info("Demo requirement already present; skipping insert")

        await session.commit()
        logger.info("✅ Seed data ensured")


if __name__ == "__main__":
    asyncio.run(seed())
