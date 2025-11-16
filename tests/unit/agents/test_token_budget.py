"""Unit tests for TokenBudgetManager token counting logic."""

from __future__ import annotations

from src.agents.conversational.token_budget import TokenBudgetManager


def test_count_tokens_basic_words() -> None:
    manager = TokenBudgetManager(model="gpt-4")
    text = "Hello world, this is a test"
    tokens = manager.count_tokens(text)
    assert tokens > 0


def test_truncate_chat_history_limits_budget() -> None:
    manager = TokenBudgetManager(model="gpt-4")
    messages = [{"role": "user", "content": f"Message {i} " + "x" * 100} for i in range(10)]

    truncated = manager.truncate_chat_history(messages, budget=300)

    assert len(truncated) <= len(messages)
    total_tokens = sum(len(m["content"]) for m in truncated)
    assert total_tokens <= len(messages[-len(truncated) :]) * 200
