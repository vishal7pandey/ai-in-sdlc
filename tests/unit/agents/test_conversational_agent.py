"""Integration-style tests for ConversationalAgent."""

from __future__ import annotations

from datetime import datetime
from typing import Any

import pytest

from src.agents.conversational.agent import ConversationalAgent
from src.schemas import GraphState, Message


class StubLLM:
    """Minimal async-compatible LLM stub."""

    def __init__(self, response_text: str) -> None:
        self.response_text = response_text

    async def ainvoke(self, *_args: Any, **_kwargs: Any) -> Any:  # pragma: no cover - trivial stub
        class _Result:
            def __init__(self, content: str) -> None:
                self.content = content

        return _Result(self.response_text)


@pytest.mark.asyncio
async def test_conversational_agent_updates_state_fields(monkeypatch: pytest.MonkeyPatch) -> None:
    agent = ConversationalAgent(openai_api_key="test")
    stub_response = """
response: Thanks for sharing the details! I'll capture that thoroughly so we can keep moving with clarity.
nextAction: clarify
clarifyingQuestions:
  - Should the checkout support guest users?
confidence: 0.83
extractedTopics:
  - checkout
  - performance
sentiment: neutral
"""
    agent.llm = StubLLM(stub_response)

    base_state = GraphState(
        session_id="sess-1",
        project_name="Demo Shop",
        user_id="user-123",
        chat_history=[
            Message(
                id="m1", role="user", content="Need fast checkout", timestamp=datetime.utcnow()
            ),
        ],
        current_turn=1,
    )

    new_state = await agent.invoke(base_state)

    assert len(new_state.chat_history) == 2
    assert new_state.pending_clarifications == ["Should the checkout support guest users?"]
    assert new_state.extracted_topics == ["checkout", "performance"]
    assert new_state.last_next_action == "clarify"
    assert new_state.last_sentiment == "neutral"
    assert new_state.conversation_context is not None
    assert new_state.ambiguity_assessment is not None


@pytest.mark.asyncio
async def test_conversational_agent_handles_non_clarify(monkeypatch: pytest.MonkeyPatch) -> None:
    agent = ConversationalAgent(openai_api_key="test")
    stub_response = """
response: Looks good and comprehensive. Based on the information, we can proceed to extraction with confidence.
nextAction: extract_requirements
confidence: 0.9
extractedTopics:
  - authentication
sentiment: positive
"""
    agent.llm = StubLLM(stub_response)

    base_state = GraphState(
        session_id="sess-2",
        project_name="Demo",
        user_id="user-456",
        chat_history=[
            Message(id="m1", role="user", content="We need login", timestamp=datetime.utcnow()),
        ],
        current_turn=1,
    )

    new_state = await agent.invoke(base_state)

    assert new_state.pending_clarifications == []
    assert new_state.last_next_action == "extract_requirements"
    assert new_state.extracted_topics == ["authentication"]
    assert new_state.confidence <= 1.0
