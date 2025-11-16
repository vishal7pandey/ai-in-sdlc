"""Conversation context extraction utilities."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, ClassVar

from src.utils.logging import get_logger

if TYPE_CHECKING:
    from src.schemas import GraphState

logger = get_logger(__name__)


@dataclass
class ConversationContext:
    conversation_summary: str = ""
    identified_domain: str = "unknown"
    mentioned_actors: list[str] = field(default_factory=list)
    mentioned_features: list[str] = field(default_factory=list)
    implicit_needs: list[str] = field(default_factory=list)
    clarification_gaps: list[str] = field(default_factory=list)
    turn_sentiment: str = "neutral"
    conversation_momentum: float = 0.5

    def as_dict(self) -> dict[str, Any]:
        return {
            "conversation_summary": self.conversation_summary,
            "identified_domain": self.identified_domain,
            "mentioned_actors": self.mentioned_actors,
            "mentioned_features": self.mentioned_features,
            "implicit_needs": self.implicit_needs,
            "clarification_gaps": self.clarification_gaps,
            "turn_sentiment": self.turn_sentiment,
            "conversation_momentum": self.conversation_momentum,
        }


class ContextManager:
    """Extract actionable context from conversation history."""

    ACTOR_KEYWORDS: ClassVar[set[str]] = {"user", "admin", "customer", "agent", "manager", "system"}
    DOMAIN_INDICATORS: ClassVar[dict[str, list[str]]] = {
        "e-commerce": ["checkout", "cart", "product", "order", "payment"],
        "saas": ["subscription", "billing", "tenant", "license", "api"],
        "mobile": ["ios", "android", "mobile", "app", "smartphone"],
        "social": ["post", "follow", "comment", "share", "profile"],
    }

    async def extract_context(self, state: GraphState) -> dict[str, Any]:
        if not state.chat_history:
            return ConversationContext().as_dict()

        latest_messages = state.chat_history[-5:]
        summary = latest_messages[-1].content

        context = ConversationContext(
            conversation_summary=summary,
            identified_domain=self._infer_domain(latest_messages),
            mentioned_actors=self._find_keywords(latest_messages, self.ACTOR_KEYWORDS),
            mentioned_features=self._collect_features(latest_messages),
            implicit_needs=self._infer_needs(latest_messages),
            clarification_gaps=self._detect_gaps(latest_messages),
            turn_sentiment=self._estimate_sentiment(latest_messages[-1].content),
            conversation_momentum=min(1.0, max(0.1, state.current_turn / 10)),
        )

        logger.debug("Context extracted", extra={"context": context.as_dict()})
        return context.as_dict()

    def _infer_domain(self, messages: list[Any]) -> str:
        text = " ".join(msg.content.lower() for msg in messages)
        for domain, cues in self.DOMAIN_INDICATORS.items():
            if any(cue in text for cue in cues):
                return domain
        return "unknown"

    def _find_keywords(self, messages: list[Any], keywords: set[str]) -> list[str]:
        found = set()
        for msg in messages:
            for word in keywords:
                if word in msg.content.lower():
                    found.add(word)
        return sorted(found)

    def _collect_features(self, messages: list[Any]) -> list[str]:
        features = []
        for msg in messages:
            tokens = [t.strip(".,?!") for t in msg.content.split()]
            features.extend(token for token in tokens if len(token) > 4)
        return features[:10]

    def _infer_needs(self, messages: list[Any]) -> list[str]:
        needs = []
        for msg in messages:
            if "need" in msg.content.lower():
                needs.append(msg.content)
        return needs[-3:]

    def _detect_gaps(self, messages: list[Any]) -> list[str]:
        gaps = []
        for msg in messages:
            if "fast" in msg.content.lower():
                gaps.append("Define performance metrics")
            if "secure" in msg.content.lower():
                gaps.append("Clarify security expectations")
        return gaps[:5]

    def _estimate_sentiment(self, text: str) -> str:
        lowered = text.lower()
        if any(word in lowered for word in ["great", "thanks", "awesome", "love"]):
            return "positive"
        if any(word in lowered for word in ["frustrated", "angry", "upset"]):
            return "negative"
        return "neutral"
