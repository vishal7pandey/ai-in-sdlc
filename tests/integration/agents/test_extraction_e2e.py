"""End-to-end extraction flow including persistence and embeddings."""

from __future__ import annotations

from datetime import datetime
from uuid import uuid4

import pytest

from src.agents.extraction import ExtractionAgent
from src.models.database import SessionModel
from src.schemas import GraphState, Message
from src.services.embedding_service import EmbeddingService
from src.storage.postgres import get_session, init_database
from src.storage.requirement_store import RequirementStore
from src.storage.vectorstore import VectorStoreService

RESPONSE = """
{
  "requirements": [
    {
      "id": "REQ-400",
      "title": "System loads dashboard under 2 seconds",
      "type": "non_functional",
      "actor": "system",
      "action": "load the dashboard content",
      "condition": null,
      "acceptance_criteria": [
        "Dashboard loads under two seconds",
        "Performance metrics are recorded"
      ],
      "priority": "high",
      "confidence": 0.85,
      "inferred": false,
      "rationale": "Derived from user performance statement",
      "source_refs": ["chat:turn:1"]
    }
  ],
  "confidence": 0.82,
  "ambiguous_items": ["Clarify number of concurrent users for 'fast'"] ,
  "metadata": {
    "tokens_used": 900,
    "duration_ms": 210,
    "model": "gpt-4"
  }
}
"""


class StubLLM:
    async def ainvoke(self, *_args, **_kwargs):  # pragma: no cover - simple stub
        class Result:
            def __init__(self, content: str) -> None:
                self.content = content

        return Result(RESPONSE)


@pytest.mark.asyncio
async def test_end_to_end_extraction_persistence_and_search() -> None:
    await init_database()

    agent = ExtractionAgent(openai_api_key="test")
    agent.llm = StubLLM()  # type: ignore[assignment]

    # Use a dedicated vector store so we can inspect embeddings
    vector_store = VectorStoreService()
    agent.vector_store = vector_store

    session_uuid = uuid4()
    state = GraphState(
        session_id=str(session_uuid),
        project_name="Story-003-E2E",
        user_id="user-e2e",
        chat_history=[
            Message(
                id="m1",
                role="user",
                content="Need fast dashboard load times",
                timestamp=datetime.utcnow(),
            ),
            Message(
                id="m2",
                role="user",
                content="Dashboard must load under 2 seconds",
                timestamp=datetime.utcnow(),
            ),
        ],
        current_turn=3,
    )

    new_state = await agent.invoke(state)

    assert new_state.requirements
    requirement = new_state.requirements[0]

    # Persist requirement to the database
    store = RequirementStore()
    async with get_session() as db:
        db_session = SessionModel(
            id=session_uuid,
            project_name="Story-003-E2E",
            user_id="user-e2e",
        )
        db.add(db_session)
        await db.commit()
        await db.refresh(db_session)

        await store.save_requirement(db, session_uuid, requirement)

    async with get_session() as db:
        loaded = await store.get_requirements(db, session_uuid)

    assert loaded
    loaded_req = loaded[0]
    assert loaded_req.id == requirement.id
    assert loaded_req.title == requirement.title

    # Verify embeddings were stored and are searchable
    embedder = EmbeddingService()
    query_embedding = await embedder.get_embedding("dashboard load performance")
    results = await vector_store.semantic_search(query_embedding, threshold=0.1)

    assert results
    top_id, score, metadata = results[0]
    assert top_id == requirement.id
    assert score > 0.1
    assert metadata["session_id"] == str(session_uuid)
