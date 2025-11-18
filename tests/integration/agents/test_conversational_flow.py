"""Integration tests covering multi-turn conversational agent flow."""

from __future__ import annotations

from datetime import datetime

import pytest

from src.agents.conversational.agent import ConversationalAgent
from src.schemas import GraphState, Message


class StubLLM:
    def __init__(self, responses: list[str]) -> None:
        self._responses = responses

    async def ainvoke(self, *_args, **_kwargs):  # pragma: no cover - trivial stub
        class _Result:
            content: str

        result = _Result()
        result.content = self._responses.pop(0)
        return result


@pytest.mark.asyncio
async def test_three_turn_conversation_flow() -> None:
    responses = [
        """
response: Hello! Let's discuss your mobile app. Are you targeting both iOS and Android?
nextAction: continue_eliciting
confidence: 0.9
extractedTopics:
  - greeting
sentiment: positive
""",
        """
response: Performance matters. What load time is acceptable for users in India?
nextAction: clarify
clarifyingQuestions:
  - What is the target load time on 4G?
confidence: 0.82
extractedTopics:
  - performance
sentiment: neutral
""",
        """
response: Great, I'll move to extraction with the 2.5 second goal documented.
nextAction: extract_requirements
confidence: 0.88
extractedTopics:
  - performance target
sentiment: positive
""",
    ]
    agent = ConversationalAgent(openai_api_key="test")
    agent.llm = StubLLM(responses)

    state = GraphState(
        session_id="sess-test",
        project_name="E-comm",
        user_id="user-123",
        chat_history=[
            Message(
                id="m1", role="user", content="We need a mobile app", timestamp=datetime.utcnow()
            ),
        ],
        current_turn=1,
    )

    # Turn 1
    state = await agent.invoke(state)
    assert state.last_next_action == "continue_eliciting"
    assert len(state.chat_history) == 2

    # Add new user reply
    state = state.with_updates(
        chat_history=[
            *state.chat_history,
            Message(
                id="m2", role="user", content="Yes, both platforms", timestamp=datetime.utcnow()
            ),
        ],
        current_turn=2,
    )

    # Turn 2
    state = await agent.invoke(state)
    assert state.last_next_action == "clarify"
    assert state.pending_clarifications

    # User provides clarification
    state = state.with_updates(
        chat_history=[
            *state.chat_history,
            Message(
                id="m3", role="user", content="2.5 seconds cold start", timestamp=datetime.utcnow()
            ),
        ],
        current_turn=3,
    )

    # Turn 3
    state = await agent.invoke(state)
    assert state.last_next_action == "extract_requirements"
    assert "performance target" in state.extracted_topics
    assert state.pending_clarifications == []
