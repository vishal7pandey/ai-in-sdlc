"""Tests for the ExtractionPromptBuilder."""

from datetime import datetime

from src.agents.extraction import ExtractionPromptBuilder
from src.schemas import GraphState, Message


def test_prompt_builder_inserts_context(tmp_path) -> None:
    template = tmp_path / "template.txt"
    template.write_text(
        "Project: {project_name}\nPending:\n{pending_clarifications}\nFormat: {format_instructions}"
    )

    builder = ExtractionPromptBuilder(template_path=template)
    state = GraphState(
        session_id="s1",
        project_name="Demo",
        user_id="u1",
        chat_history=[
            Message(id="m1", role="user", content="Need fast login", timestamp=datetime.utcnow()),
        ],
        current_turn=1,
        pending_clarifications=["Define fast"],
    )

    prompt = builder.build(state, format_instructions="Return JSON")

    assert "Project: Demo" in prompt
    assert "- Define fast" in prompt
    assert "Return JSON" in prompt
