"""Tests for the TraceabilityLinker helper."""

from datetime import datetime

from src.agents.extraction import TraceabilityLinker
from src.schemas import Message


def test_links_to_user_turns_based_on_overlap() -> None:
    linker = TraceabilityLinker()
    chat = [
        Message(
            id="1", role="user", content="Users log in with email", timestamp=datetime.utcnow()
        ),
        Message(id="2", role="assistant", content="ok", timestamp=datetime.utcnow()),
        Message(id="3", role="user", content="Login must be fast", timestamp=datetime.utcnow()),
    ]

    refs = linker.link("User authentication using email", chat)

    assert refs, "Expected at least one reference"
    assert any(ref.startswith("chat:turn:") for ref in refs)
