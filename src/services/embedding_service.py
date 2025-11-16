"""Embedding service used by agents for semantic search.

This implementation intentionally avoids external network calls so tests can run
without OpenAI credentials. It provides deterministic, low-dimensional
embeddings based on token hashing.
"""

from __future__ import annotations

import hashlib
import math


class EmbeddingService:
    """Simple, deterministic text embedding service.

    The interface mirrors what a real embedding client (e.g. OpenAI) would
    expose, so it can be swapped out later without changing callers.
    """

    def __init__(self, dimension: int = 32) -> None:
        self.dimension = dimension

    async def get_embedding(self, text: str) -> list[float]:
        """Return a normalized embedding vector for the given text."""

        tokens = [t for t in text.lower().split() if t]
        if not tokens:
            return [0.0] * self.dimension

        vector = [0.0] * self.dimension

        for token in tokens:
            # Stable hash → bucket index
            digest = hashlib.md5(token.encode("utf-8")).hexdigest()
            bucket = int(digest[:8], 16) % self.dimension
            vector[bucket] += 1.0

        norm = math.sqrt(sum(v * v for v in vector)) or 1.0
        return [v / norm for v in vector]
