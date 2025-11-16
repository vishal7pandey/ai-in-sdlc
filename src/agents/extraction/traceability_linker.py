"""Traceability helper linking requirements to chat turns."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

    from src.schemas import Message


class TraceabilityLinker:
    """Link requirement text back to source conversation turns."""

    def link(self, requirement_text: str, chat_history: Sequence[Message]) -> list[str]:
        keywords = self._keywords(requirement_text)
        matched: list[str] = []

        for idx, message in enumerate(chat_history):
            if message.role != "user":
                continue
            overlap = self._overlap(keywords, message.content)
            if overlap >= 0.3:
                matched.append(f"chat:turn:{idx}")

        if not matched:
            for idx in range(len(chat_history) - 1, -1, -1):
                if chat_history[idx].role == "user":
                    matched.append(f"chat:turn:{idx}")
                    break

        return matched or ["chat:turn:0"]

    def _keywords(self, text: str) -> set[str]:
        stop_words = {
            "the",
            "and",
            "with",
            "that",
            "this",
            "from",
            "will",
            "shall",
            "should",
        }
        tokens = [token.strip(".,!?:").lower() for token in text.split()]
        return {token for token in tokens if len(token) > 3 and token not in stop_words}

    def _overlap(self, keywords: set[str], text: str) -> float:
        if not keywords:
            return 0.0
        haystack = {token.strip(".,!?:").lower() for token in text.split()}
        matches = keywords.intersection(haystack)
        return len(matches) / len(keywords)
