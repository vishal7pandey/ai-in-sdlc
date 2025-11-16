"""Database migration script for STORY-001."""

from __future__ import annotations

import asyncio
import logging
import sys
from pathlib import Path

from sqlalchemy import text

# Ensure project root is importable when script is executed directly
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.database import Base  # noqa: E402  (import after path tweak)
from src.storage.postgres import get_engine  # noqa: E402

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("reqeng.migrate")


async def run_migrations() -> None:
    """Create database extensions and tables."""
    logger.info("🔧 Starting database migrations...")
    engine = get_engine()

    async with engine.begin() as conn:
        logger.info("Ensuring PostgreSQL extensions exist...")
        await conn.execute(text('CREATE EXTENSION IF NOT EXISTS "uuid-ossp"'))
        await conn.execute(text('CREATE EXTENSION IF NOT EXISTS "pgcrypto"'))

        logger.info("Creating database tables from SQLAlchemy metadata...")
        await conn.run_sync(Base.metadata.create_all)

    logger.info("✅ Database migrations completed successfully")


if __name__ == "__main__":
    asyncio.run(run_migrations())
