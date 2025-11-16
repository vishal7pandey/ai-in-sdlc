"""Unit tests for conversational agent helper components."""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

import pytest

from src.agents.conversational.clarification_detector import ClarificationDetector
from src.agents.conversational.context_manager import ContextManager
from src.agents.conversational.prompt_builder import PromptBuilder
from src.agents.conversational.response_formatter import ResponseFormatter
from src.schemas import GraphState, Message


@pytest.mark.asyncio
async def test_context_manager_returns_expected_keys() -> None:
    manager = ContextManager()
    state = GraphState(
        session_id="sess-1",
        project_name="Demo",
        user_id="user-1",
        chat_history=[
            Message(
                id="m1",
                role="user",
                content="We need a fast secure portal",
                timestamp=datetime.utcnow(),
            ),
            Message(
                id="m2",
                role="user",
                content="It should support iOS and Android",
                timestamp=datetime.utcnow(),
            ),
        ],
        current_turn=2,
    )

    context = await manager.extract_context(state)

    assert "conversation_summary" in context
    assert "identified_domain" in context
    assert "clarification_gaps" in context
    assert context["identified_domain"] in {"mobile", "unknown"}


def test_clarification_detector_identifies_ambiguity() -> None:
    detector = ClarificationDetector()
    result = detector.detect("It must be fast and scalable")

    assert result.is_ambiguous is True
    assert {"fast", "scalable"}.issubset(set(result.ambiguous_terms))
    assert result.clarifying_questions
    assert 0.0 <= result.ambiguity_score <= 1.0


def test_response_formatter_parses_block() -> None:
    raw = """
response: Thanks for the details! I'll capture that and ensure we track login flows carefully.
nextAction: clarify
clarifyingQuestions:
  - Should we support password reset?
confidence: 0.82
extractedTopics:
  - authentication
sentiment: positive
"""
    formatter = ResponseFormatter()

    parsed = formatter.parse_and_validate(raw)

    assert parsed.next_action == "clarify"
    assert parsed.clarifying_questions is not None
    assert parsed.extracted_topics == ["authentication"]


def test_prompt_builder_includes_context(tmp_path: Path) -> None:
    template = (
        "Project: {project_name}\n"
        "Actors: {mentioned_actors}\n"
        "Ambiguity: {ambiguity_terms}\n"
        "History:\n{chat_history}\n"
    )
    template_path = tmp_path / "conversational.txt"
    template_path.write_text(template, encoding="utf-8")

    builder = PromptBuilder(template_path=template_path)
    msg = Message(id="m1", role="user", content="Need fast checkout", timestamp=datetime.utcnow())
    context = {
        "conversation_summary": "Need fast checkout",
        "clarification_gaps": ["Define fast"],
        "identified_domain": "e-commerce",
        "mentioned_actors": ["user"],
        "mentioned_features": ["checkout"],
        "implicit_needs": ["speed"],
        "conversation_momentum": 0.6,
        "turn_sentiment": "neutral",
    }
    ambiguity = {"ambiguous_terms": ["fast"], "clarifying_questions": ["What load time?"]}

    prompt = builder.build(
        project_name="Shop",
        current_turn=1,
        requirements_count=0,
        context=context,
        ambiguity_result=ambiguity,
        chat_history=[msg],
    )

    assert "Project: Shop" in prompt
    assert "Actors: user" in prompt
    assert "Ambiguity: fast" in prompt
    assert "Need fast checkout" in prompt
