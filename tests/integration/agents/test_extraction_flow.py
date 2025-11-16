"""Integration-level tests for ExtractionAgent."""

from __future__ import annotations

from datetime import datetime

import pytest

from src.agents.extraction import ExtractionAgent
from src.schemas import GraphState, Message

RESPONSE = """
{
  "requirements": [
    {
      "id": "REQ-002",
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
    def __init__(self, content: str) -> None:
        self._content = content

    async def ainvoke(self, *_args, **_kwargs):  # pragma: no cover - trivial stub
        class Result:
            def __init__(self, content: str) -> None:
                self.content = content

        return Result(self._content)


@pytest.mark.asyncio
async def test_extraction_agent_flow_enriches_state() -> None:
    agent = ExtractionAgent(openai_api_key="test")
    agent.llm = StubLLM(RESPONSE)  # type: ignore[assignment]

    state = GraphState(
        session_id="sess-002",
        project_name="Story-003",
        user_id="user-2",
        chat_history=[
            Message(id="m1", role="user", content="Need fast login", timestamp=datetime.utcnow()),
            Message(
                id="m2",
                role="user",
                content="Dashboard must load under 2 seconds",
                timestamp=datetime.utcnow(),
            ),
        ],
        current_turn=3,
        requirements=[],
        pending_clarifications=["Define 'fast'"],
    )

    new_state = await agent.invoke(state)

    assert len(new_state.requirements) == 1
    assert any(ref.startswith("chat:turn:") for ref in new_state.requirements[0].source_refs)
    assert new_state.last_agent == "extraction"
    assert (
        new_state.extraction_metadata is not None
        and new_state.extraction_metadata.get("tokens_used") == 900
    )
    assert new_state.ambiguous_items
