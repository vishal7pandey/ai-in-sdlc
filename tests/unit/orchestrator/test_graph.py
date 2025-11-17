"""Basic tests for the orchestrator graph wiring."""

from __future__ import annotations

from src.orchestrator.graph import graph


def test_graph_contains_expected_nodes() -> None:
    ascii_graph = graph.get_graph().draw_ascii()

    assert "conversational" in ascii_graph
    assert "extraction" in ascii_graph
    assert "validation" in ascii_graph
    assert "synthesis" in ascii_graph
    assert "review" in ascii_graph


def test_graph_has_start_and_end() -> None:
    ascii_graph = graph.get_graph().draw_ascii()

    assert "START" in ascii_graph
    assert "END" in ascii_graph
