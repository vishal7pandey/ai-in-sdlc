"""Parse and validate ConversationalAgent raw outputs."""

from __future__ import annotations

import json
import re
from typing import Any, cast

from pydantic import ValidationError

from .schemas import ConversationalResponse


class ResponseFormatter:
    """Parse LLM responses into structured ConversationalResponse objects."""

    def parse_and_validate(self, raw_output: str) -> ConversationalResponse:
        """Parse raw LLM output into a ConversationalResponse model."""

        sections = self._extract_sections(raw_output)
        try:
            return ConversationalResponse(**sections)
        except ValidationError as exc:  # pragma: no cover - re-raising message
            raise ValueError(f"Invalid conversational response: {exc}") from exc

    def _extract_sections(self, raw_output: str) -> dict[str, Any]:
        # First try to handle JSON-style outputs where the model returns a
        # structured object like {"conversational_agent_response": "..."}.
        stripped = raw_output.strip()

        # If the model returned a mix of natural language text followed by a
        # JSON block (e.g. analysis metadata), keep only the human-readable
        # prefix so that raw JSON does not leak into the chat UI. When the
        # output is pure JSON (starts with "{"), we keep it for structured
        # parsing below.
        json_start = stripped.find("{")
        if json_start > 0:
            candidate = stripped[json_start:]
            try:
                json.loads(candidate)
            except json.JSONDecodeError:
                pass
            else:
                prefix = stripped[:json_start].rstrip()
                if prefix:
                    stripped = prefix
        if stripped.startswith("{") and stripped.endswith("}"):
            try:
                obj = json.loads(stripped)
            except json.JSONDecodeError:
                obj = None
            if isinstance(obj, dict):
                # Prefer a nested response.text field when present, which is
                # how the current prompts structure conversational answers.
                nested_response = obj.get("response")
                message: str | None = None
                if isinstance(nested_response, dict):
                    message = nested_response.get("text") or nested_response.get("message")

                # Support envelopes like {"conversational_agent": {"response": "..."}}.
                source_for_meta: dict[str, Any] = obj
                conv_agent = obj.get("conversational_agent")
                if isinstance(conv_agent, dict):
                    source_for_meta = conv_agent
                    if not message:
                        nested_msg = conv_agent.get("response") or conv_agent.get("message")
                        if isinstance(nested_msg, str):
                            message = nested_msg

                # Fall back to common top-level keys.
                if not message:
                    message = obj.get("conversational_agent_response") or obj.get("message")

                # If the model returned a document object, surface a short
                # human-readable summary instead of the raw JSON.
                if not message and isinstance(obj.get("document"), dict):
                    doc = obj["document"]
                    title = doc.get("title") or "Generated requirements document"
                    sections = doc.get("sections") or []
                    section_titles: list[str] = [
                        cast("str", s.get("title"))
                        for s in sections
                        if isinstance(s, dict) and isinstance(s.get("title"), str)
                    ]
                    summary_lines = [title]
                    if section_titles:
                        summary_lines.append("")
                        summary_lines.append("Sections: " + ", ".join(section_titles))
                    message = "\n".join(summary_lines)

                if not message:
                    message = stripped

                return {
                    "message": str(message).strip(),
                    "next_action": source_for_meta.get("next_action")
                    or source_for_meta.get("nextAction")
                    or "continue_eliciting",
                    "clarifying_questions": source_for_meta.get("clarifying_questions")
                    or source_for_meta.get("clarifyingQuestions"),
                    "confidence": float(source_for_meta.get("confidence", 0.7)),
                    "extracted_topics": source_for_meta.get("extracted_topics")
                    or source_for_meta.get("extractedTopics")
                    or [],
                    "sentiment": source_for_meta.get("sentiment", "neutral"),
                }

        # Fallback: parse simple key:value block format.
        pattern = re.compile(r"^(\w+):\s*(.*)$", re.MULTILINE)
        data: dict[str, Any] = {}
        current_key: str | None = None
        lines: list[str] = []

        for line in stripped.splitlines():
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

        # Prefer the structured `response:` block, but if the model didn't
        # follow the expected format, fall back to the raw output so the
        # user still sees a meaningful assistant message instead of the
        # generic hiccup fallback.
        message = (data.get("response") or "").strip()
        if not message and stripped:
            message = stripped

        return {
            "message": message,
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
