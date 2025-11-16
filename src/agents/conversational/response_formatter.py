"""Parse and validate ConversationalAgent raw outputs."""

from __future__ import annotations

import re
from typing import Any

from pydantic import ValidationError

from .schemas import ConversationalResponse


class ResponseFormatter:
    """Parse LLM responses into structured ConversationalResponse objects."""

    def parse_and_validate(self, raw_output: str) -> ConversationalResponse:
        """Parse raw LLM output formatted as simple key:value blocks."""

        sections = self._extract_sections(raw_output)
        try:
            return ConversationalResponse(**sections)
        except ValidationError as exc:  # pragma: no cover - re-raising message
            raise ValueError(f"Invalid conversational response: {exc}") from exc

    def _extract_sections(self, raw_output: str) -> dict[str, Any]:
        pattern = re.compile(r"^(\w+):\s*(.*)$", re.MULTILINE)
        data: dict[str, Any] = {}
        current_key: str | None = None
        lines: list[str] = []

        for line in raw_output.strip().splitlines():
            match = pattern.match(line)
            if match:
                if current_key and lines:
                    data[current_key] = "\n".join(lines).strip()
                current_key = match.group(1).strip()
                lines = [match.group(2).strip()]
            else:
                lines.append(line)

        if current_key and lines:
            data[current_key] = "\n".join(lines).strip()

        return {
            "message": data.get("response", ""),
            "next_action": data.get("nextAction", "continue_eliciting"),
            "clarifying_questions": self._split_list(data.get("clarifyingQuestions")),
            "confidence": float(data.get("confidence", 0.7)),
            "extracted_topics": self._split_list(data.get("extractedTopics")) or [],
            "sentiment": data.get("sentiment", "neutral"),
        }

    def _split_list(self, value: str | None) -> list[str] | None:
        if value is None or not value.strip():
            return None
        items = [item.strip("- ") for item in value.splitlines() if item.strip()]
        return items or None
