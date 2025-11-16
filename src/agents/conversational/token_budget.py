"""Token budget helper for conversational agent prompts."""

from __future__ import annotations

from dataclasses import dataclass

try:
    import tiktoken
except ImportError:  # pragma: no cover - fallback when tiktoken missing
    tiktoken = None


@dataclass(slots=True)
class TokenBudgetManager:
    """Track prompt/response token limits with simple heuristics."""

    model: str
    max_prompt_tokens: int = 3500
    max_response_tokens: int = 500

    def remaining_prompt_tokens(self, used_tokens: int) -> int:
        """Return how many tokens remain for the prompt."""

        return max(0, self.max_prompt_tokens - used_tokens)

    def clamp_response_tokens(self, requested: int | None) -> int:
        """Clamp requested response tokens into allowed range."""

        if requested is None:
            return self.max_response_tokens
        return max(50, min(requested, self.max_response_tokens))

    def truncate_chat_history(self, messages: list[dict], budget: int) -> list[dict]:
        """Return latest subset of messages whose token count stays under budget."""

        encoder = self._get_encoder()
        accumulated: list[dict] = []
        tokens_used = 0

        for message in reversed(messages):
            content = message.get("content", "")
            tokens = len(encoder.encode(content)) if encoder else len(content.split()) * 2
            if tokens_used + tokens > budget:
                break
            tokens_used += tokens
            accumulated.append(message)

        return list(reversed(accumulated))

    def count_tokens(self, text: str) -> int:
        encoder = self._get_encoder()
        if encoder:
            return len(encoder.encode(text))
        return len(text.split()) * 2

    def _get_encoder(self):
        if tiktoken is None:
            return None
        try:
            return tiktoken.encoding_for_model(self.model)
        except KeyError:  # pragma: no cover - fallback when model not recognized
            return tiktoken.get_encoding("cl100k_base")
