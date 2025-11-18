"""Tests covering conversational agent error handling scenarios."""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING

import pytest

from src.agents.conversational.agent import ConversationalAgent
from src.agents.conversational.response_formatter import ResponseFormatter
from src.schemas import GraphState, Message

if TYPE_CHECKING:
    from src.agents.conversational.schemas import ConversationalResponse


class ExplodingLLM:
    async def ainvoke(self, *_args, **_kwargs):  # pragma: no cover - simple stub
        raise RuntimeError("LLM unavailable")


def _base_state() -> GraphState:
    return GraphState(
        session_id="sess-err",
        project_name="Demo",
        user_id="user-err",
        chat_history=[
            Message(
                id="m1", role="user", content="Need fast checkout", timestamp=datetime.utcnow()
            ),
        ],
        current_turn=1,
    )


@pytest.mark.asyncio
async def test_conversational_agent_fallback_on_llm_error() -> None:
    agent = ConversationalAgent(openai_api_key="test")
    agent.llm = ExplodingLLM()

    result = await agent.execute(_base_state())
    assert result["state_updates"]["confidence"] == 0.3
    assert result["state_updates"]["last_next_action"] == "wait_for_input"
    assert result["state_updates"]["pending_clarifications"] == []


class BadFormatter(ResponseFormatter):
    def parse_and_validate(
        self, _content: str
    ) -> ConversationalResponse:  # pragma: no cover - stub
        raise ValueError("parse error")


class StubLLM:
    async def ainvoke(self, *_args, **_kwargs):  # pragma: no cover - stub
        class _Result:
            content = "response: hi\nnextAction: continue_eliciting\nconfidence: 0.5"

        return _Result()


@pytest.mark.asyncio
async def test_conversational_agent_fallback_on_parse_error() -> None:
    agent = ConversationalAgent(openai_api_key="test")
    agent.llm = StubLLM()
    agent.response_formatter = BadFormatter()

    result = await agent.execute(_base_state())

    assert result["state_updates"]["confidence"] == 0.3
    assert result["state_updates"]["last_sentiment"] == "neutral"
    assert result["state_updates"]["last_next_action"] == "wait_for_input"
