"""Integration tests for database connectivity."""

import pytest
from sqlalchemy import text

from src.storage.postgres import get_session


@pytest.mark.asyncio
async def test_database_connection() -> None:
    async with get_session() as session:
        result = await session.execute(text("SELECT 1"))
        assert result.scalar() == 1
