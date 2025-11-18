"""Unit tests for ExtractionAgent orchestration."""

from __future__ import annotations

from datetime import datetime

import pytest

from src.agents.extraction import ExtractionAgent
from src.schemas import GraphState, Message

RESPONSE = """
{
  "requirements": [
    {
      "id": "REQ-001",
      "title": "Users authenticate with email and password",
      "type": "functional",
      "actor": "user",
      "action": "authenticate using email and password credentials",
      "condition": "when accessing protected resources",
      "acceptance_criteria": [
        "User can enter valid email and password credentials",
        "System validates credentials before granting access"
      ],
      "priority": "must",
      "confidence": 0.92,
      "inferred": false,
      "rationale": "Stated explicitly by the user",
      "source_refs": ["chat:turn:0"]
    }
  ],
  "confidence": 0.9,
  "ambiguous_items": ["Clarify target response time for login"],
  "metadata": {
    "tokens_used": 1200,
    "duration_ms": 350.5,
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
async def test_extraction_agent_updates_state_with_requirements() -> None:
    agent = ExtractionAgent(openai_api_key="test")
    agent.llm = StubLLM(RESPONSE)

    state = GraphState(
        session_id="sess-001",
        project_name="Demo",
        user_id="user-1",
        chat_history=[
            Message(id="m1", role="user", content="Need email login", timestamp=datetime.utcnow())
        ],
        current_turn=2,
    )

    new_state = await agent.invoke(state)

    assert len(new_state.requirements) == 1
    requirement = new_state.requirements[0]
    assert requirement.id == "REQ-001"
    assert new_state.last_agent == "extraction"
    assert "Clarify" in new_state.ambiguous_items[0]
    assert new_state.extraction_metadata is not None
