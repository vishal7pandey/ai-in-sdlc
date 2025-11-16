"""Entity extraction utilities for requirements parsing."""

from __future__ import annotations

import re
from typing import ClassVar


class EntityExtractor:
    """Extract actors, actions, and conditions from natural language."""

    ACTOR_PATTERNS: ClassVar[list[str]] = [
        r"\busers?\b",
        r"\badmins?\b",
        r"\bcustomers?\b",
        r"\bstakeholders?\b",
        r"\bsystem\b",
        r"\bplatform\b",
        r"\bapplication\b",
        r"\bservice\b",
    ]

    ACTION_VERBS: ClassVar[set[str]] = {
        "authenticate",
        "log in",
        "sign in",
        "reset",
        "view",
        "create",
        "update",
        "delete",
        "upload",
        "download",
        "generate",
        "approve",
        "reject",
        "submit",
        "validate",
        "verify",
    }

    CONDITION_MARKERS: ClassVar[list[str]] = ["when", "if", "while", "before", "after", "during"]

    def extract_actors(self, text: str) -> list[str]:
        lowered = text.lower()
        actors: list[str] = []
        for pattern in self.ACTOR_PATTERNS:
            match = re.search(pattern, lowered)
            if match:
                actor = match.group(0)
                if actor not in actors:
                    actors.append(actor)
        return actors or ["system"]

    def extract_action(self, text: str) -> str:
        lowered = text.lower()
        for verb in self.ACTION_VERBS:
            if verb in lowered:
                idx = lowered.index(verb)
                span = text[idx : idx + 120]
                return span.strip()
        # fallback to first sentence fragment
        return text.strip().split(".")[0][:120]

    def extract_condition(self, text: str) -> str | None:
        lowered = text.lower()
        for marker in self.CONDITION_MARKERS:
            if marker in lowered:
                idx = lowered.index(marker)
                return text[idx : idx + 120].strip()
        return None
