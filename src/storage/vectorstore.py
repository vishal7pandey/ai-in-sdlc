"""ChromaDB client helpers and in-memory vector store service.

The health-check functions keep the original Story-1 behavior of pinging the
ChromaDB HTTP endpoint. For Story-3 semantic search, we provide an
in-memory ``VectorStoreService`` that exposes the same API expected by the
design documents, but does not depend on an external vector database. This
keeps tests fast and deterministic while matching the intended interface.
"""

from __future__ import annotations

import math
from typing import Any

import httpx

from src.config import settings


async def init_vector_store() -> None:
    # For now just ping health endpoint; future stories may instantiate
    # a real Chroma client here.
    await check_chromadb_health()


async def check_chromadb_health() -> bool:
    url = f"http://{settings.CHROMA_HOST}:{settings.CHROMA_PORT}/api/v1/heartbeat"
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(url)
            response.raise_for_status()
        return True
    except Exception:
        return False


class VectorStoreService:
    """Minimal in-memory vector store used for semantic search tests.

    The public API mirrors what a Chroma-backed implementation would expose,
    so it can be swapped out in a future story without changing callers.
    """

    def __init__(self) -> None:
        # requirement_id -> {"embedding": [...], "metadata": {...}}
        self._store: dict[str, dict[str, Any]] = {}

    async def add_requirement(
        self,
        *,
        requirement_id: str,
        title: str,
        action: str,
        embedding: list[float],
        metadata: dict[str, Any],
    ) -> None:
        """Add or update a requirement vector in the store."""

        self._store[requirement_id] = {
            "embedding": list(embedding),
            "metadata": {**metadata, "title": title, "action": action},
        }

    async def semantic_search(
        self,
        query_embedding: list[float],
        *,
        limit: int = 5,
        threshold: float = 0.0,
    ) -> list[tuple[str, float, dict[str, Any]]]:
        """Return matching requirement IDs ordered by cosine similarity.

        Results below ``threshold`` are filtered out.
        """

        scores: list[tuple[str, float, dict[str, Any]]] = []
        for req_id, payload in self._store.items():
            score = _cosine_similarity(query_embedding, payload["embedding"])
            if score >= threshold:
                scores.append((req_id, score, payload["metadata"]))

        scores.sort(key=lambda tup: tup[1], reverse=True)
        return scores[:limit]


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b, strict=False))
    norm_a = math.sqrt(sum(x * x for x in a)) or 1.0
    norm_b = math.sqrt(sum(y * y for y in b)) or 1.0
    return dot / (norm_a * norm_b)
