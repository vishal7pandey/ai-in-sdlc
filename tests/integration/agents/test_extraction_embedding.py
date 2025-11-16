"""Integration tests for ExtractionAgent embeddings and semantic search."""

from __future__ import annotations

from datetime import datetime

import pytest

from src.agents.extraction import ExtractionAgent
from src.schemas import GraphState, Message
from src.services.embedding_service import EmbeddingService
from src.storage.vectorstore import VectorStoreService

RESPONSE = """
{
  "requirements": [
    {
      "id": "REQ-310",
      "title": "Users authenticate with email and password",
      "type": "functional",
      "actor": "user",
      "action": "authenticate using email and password credentials",
      "condition": "when accessing protected resources",
      "acceptance_criteria": [
        "User can enter valid email and password",
        "System validates credentials before granting access"
      ],
      "priority": "must",
      "confidence": 0.9,
      "inferred": false,
      "rationale": "Stated explicitly by the user",
      "source_refs": ["chat:turn:0"]
    }
  ],
  "confidence": 0.88,
  "ambiguous_items": ["Clarify maximum allowed login attempts"],
  "metadata": {
    "tokens_used": 256,
    "duration_ms": 100.0,
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
async def test_extraction_agent_stores_embeddings_for_semantic_search() -> None:
    agent = ExtractionAgent(openai_api_key="test")
    agent.llm = StubLLM()  # type: ignore[assignment]

    # Use a dedicated in-memory vector store so we can query it directly.
    vector_store = VectorStoreService()
    agent.vector_store = vector_store

    state = GraphState(
        session_id="sess-embedding",
        project_name="Story-003",
        user_id="user-embedding",
        chat_history=[
            Message(
                id="m1",
                role="user",
                content="Users must be able to log in with email and password.",
                timestamp=datetime.utcnow(),
            ),
        ],
        current_turn=1,
    )

    new_state = await agent.invoke(state)

    assert new_state.requirements
    requirement = new_state.requirements[0]

    # Build a query embedding similar to the requirement text
    embedder = EmbeddingService()
    query_embedding = await embedder.get_embedding("email login password")

    results = await vector_store.semantic_search(query_embedding, threshold=0.1)

    assert results
    top_id, score, metadata = results[0]
    assert top_id == requirement.id
    assert score > 0.1
    assert metadata["title"] == requirement.title
