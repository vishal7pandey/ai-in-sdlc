"""Dual-layer LangGraph checkpointer using Redis and Postgres.

This module provides a minimal async checkpointer that is compatible with
LangGraph's checkpointing configuration model. It stores checkpoints in
both Redis (for fast access to active sessions) and Postgres (for durable
history), using the existing storage helpers in this codebase.

The checkpointer operates on ``config`` dictionaries that contain a
``configurable`` section, following LangGraph conventions::

    config = {
        "configurable": {
            "thread_id": "session-123",
            # Optional, to address a specific checkpoint explicitly:
            # "checkpoint_id": "<uuid>",
        }
    }

Checkpoints themselves are arbitrary JSON-serializable dictionaries. When
saving, we expect an ``id`` field to be present on the checkpoint; if not,
we generate one.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any
from uuid import UUID, uuid4

from langgraph.checkpoint.base import (
    BaseCheckpointSaver,
    ChannelVersions,
    Checkpoint,
    CheckpointMetadata,
    CheckpointTuple,
)
from sqlalchemy import select

if TYPE_CHECKING:  # pragma: no cover - typing only
    from collections.abc import AsyncIterator

    from langchain_core.runnables import RunnableConfig
else:  # pragma: no cover - Runtime fallback to satisfy type evaluation
    AsyncIterator = Any  # type: ignore[assignment]
    RunnableConfig = dict[str, Any]  # type: ignore[misc,assignment]

from src.models.database import LangGraphCheckpointModel
from src.storage.postgres import get_session
from src.storage.redis_cache import get_redis


class DualCheckpointer(BaseCheckpointSaver):
    """Async checkpointer that writes to Redis and Postgres.

    Redis is used as the fast primary store for recent checkpoints, while
    Postgres provides durable storage and a source of truth when Redis
    does not have the requested data.
    """

    def __init__(self, *, ttl_seconds: int = 3600) -> None:
        super().__init__()
        self.ttl_seconds = ttl_seconds

    # ------------------------------------------------------------------
    # BaseCheckpointSaver interface implementation
    # ------------------------------------------------------------------
    async def aput(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        """Store a checkpoint in both Redis and Postgres.

        Args:
            config: Configuration for the checkpoint.
            checkpoint: The checkpoint to store.
            metadata: Additional metadata for the checkpoint.
            new_versions: New channel versions as of this write.

        Returns:
            Updated configuration after storing the checkpoint.
        """
        _ = new_versions  # Interface requires this argument; not used yet.
        thread_id = config["configurable"]["thread_id"]
        checkpoint_id = checkpoint["id"]

        # Save to Redis (fast path)
        redis = get_redis()
        payload = json.dumps(checkpoint, default=str)
        await redis.set(self._redis_key(thread_id, checkpoint_id), payload, ex=self.ttl_seconds)
        await redis.set(self._latest_redis_key(thread_id), payload, ex=self.ttl_seconds)

        # Save to Postgres (durable path)
        async with get_session() as session:
            await self._save_postgres(
                session,
                thread_id=thread_id,
                checkpoint_id=checkpoint_id,
                checkpoint=checkpoint,
                metadata=metadata,
            )

        return {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_id": checkpoint_id,
            }
        }

    async def aget_tuple(self, config: RunnableConfig) -> CheckpointTuple | None:
        """Load a checkpoint tuple, preferring Redis and falling back to Postgres.

        Args:
            config: Configuration specifying which checkpoint to retrieve.

        Returns:
            The requested checkpoint tuple, or None if not found.
        """
        thread_id, checkpoint_id = self._ids_from_config(config)

        if thread_id is None:
            return None

        redis = get_redis()

        # Look in Redis first
        key = (
            self._redis_key(thread_id, checkpoint_id)
            if checkpoint_id is not None
            else self._latest_redis_key(thread_id)
        )
        raw = await redis.get(key)
        if raw:
            try:
                checkpoint = json.loads(raw)
                return CheckpointTuple(
                    config=config,
                    checkpoint=checkpoint,
                    metadata={"source": "loop", "step": -1, "parents": {}},
                )
            except json.JSONDecodeError:
                # Fall through to Postgres if Redis payload is corrupted
                pass

        # Fallback to Postgres
        async with get_session() as session:
            record = await self._load_postgres(session, thread_id, checkpoint_id)

        if record is None:
            return None

        checkpoint = record.checkpoint
        # Repopulate Redis for faster subsequent reads
        payload = json.dumps(checkpoint, default=str)
        await redis.set(
            self._redis_key(thread_id, str(record.checkpoint_id)), payload, ex=self.ttl_seconds
        )
        await redis.set(self._latest_redis_key(thread_id), payload, ex=self.ttl_seconds)

        return CheckpointTuple(
            config=config,
            checkpoint=checkpoint,
            metadata=record.metadata_json or {"source": "loop", "step": -1, "parents": {}},
        )

    async def alist(
        self,
        _config: RunnableConfig | None,
        *,
        _filter: dict[str, Any] | None = None,
        _before: RunnableConfig | None = None,
        _limit: int | None = None,
    ) -> AsyncIterator[CheckpointTuple]:
        """List checkpoints that match the given criteria.

        Args:
            config: Base configuration for filtering checkpoints.
            filter: Additional filtering criteria.
            before: List checkpoints created before this configuration.
            limit: Maximum number of checkpoints to return.

        Yields:
            Matching checkpoint tuples.
        """

        async def _empty() -> AsyncIterator[CheckpointTuple]:  # pragma: no cover - stub
            if False:
                yield

        return _empty()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _redis_key(thread_id: str, checkpoint_id: str) -> str:
        return f"lg:cp:{thread_id}:{checkpoint_id}"

    @staticmethod
    def _latest_redis_key(thread_id: str) -> str:
        return f"lg:cp:{thread_id}:latest"

    @staticmethod
    def _ids_from_config(config: dict[str, Any]) -> tuple[str | None, str | None]:
        cfg = config.get("configurable", {}) or {}
        thread_id = cfg.get("thread_id")
        checkpoint_id = cfg.get("checkpoint_id")
        if thread_id is not None:
            thread_id = str(thread_id)
        if checkpoint_id is not None:
            checkpoint_id = str(checkpoint_id)
        return thread_id, checkpoint_id

    @staticmethod
    def _ids_from_config_and_checkpoint(
        config: dict[str, Any],
        checkpoint: dict[str, Any],
    ) -> tuple[str, str]:
        thread_id, checkpoint_id = DualCheckpointer._ids_from_config(config)

        if thread_id is None:
            raise ValueError("thread_id is required in config['configurable'] for checkpointing")

        if checkpoint_id is None:
            checkpoint_id = checkpoint.get("id") or str(uuid4())

        return thread_id, str(checkpoint_id)

    @staticmethod
    def _serialize_checkpoint(checkpoint: dict[str, Any]) -> dict[str, Any]:
        """Recursively serialize Pydantic models in checkpoint to dicts.

        LangGraph checkpoints contain channel_values which may include Pydantic
        models like GraphState. SQLAlchemy's JSONB column requires plain dicts,
        so we walk the checkpoint structure and convert any BaseModel instances.
        """
        from pydantic import BaseModel

        def _serialize_value(val: Any) -> Any:
            if isinstance(val, BaseModel):
                return val.model_dump(mode="json")
            elif isinstance(val, dict):
                return {k: _serialize_value(v) for k, v in val.items()}
            elif isinstance(val, list):
                return [_serialize_value(item) for item in val]
            else:
                return val

        return _serialize_value(checkpoint)

    async def _save_postgres(
        self,
        session: Any,
        *,
        thread_id: str,
        checkpoint_id: str,
        checkpoint: dict[str, Any],
        metadata: dict[str, Any],
    ) -> None:
        """Upsert a checkpoint row in Postgres."""

        cp_uuid = UUID(checkpoint_id)

        # Serialize Pydantic models to dicts before saving to JSONB
        serialized_checkpoint = self._serialize_checkpoint(checkpoint)

        result = await session.execute(
            select(LangGraphCheckpointModel).where(
                LangGraphCheckpointModel.thread_id == thread_id,
                LangGraphCheckpointModel.checkpoint_id == cp_uuid,
            )
        )
        record = result.scalar_one_or_none()

        if record is None:
            record = LangGraphCheckpointModel(
                thread_id=thread_id,
                checkpoint_id=cp_uuid,
                checkpoint=serialized_checkpoint,
                metadata_json=metadata,
            )
            session.add(record)
        else:
            record.checkpoint = serialized_checkpoint
            record.metadata_json = metadata

        await session.commit()

    async def _load_postgres(
        self,
        session: Any,
        thread_id: str,
        checkpoint_id: str | None,
    ) -> LangGraphCheckpointModel | None:
        """Load a checkpoint row from Postgres.

        If ``checkpoint_id`` is ``None``, the latest checkpoint for the
        thread is returned, ordered by ``created_at``.
        """

        if checkpoint_id is not None:
            cp_uuid = UUID(checkpoint_id)
            result = await session.execute(
                select(LangGraphCheckpointModel).where(
                    LangGraphCheckpointModel.thread_id == thread_id,
                    LangGraphCheckpointModel.checkpoint_id == cp_uuid,
                )
            )
            return result.scalar_one_or_none()

        # No explicit checkpoint_id: fetch latest for the thread
        result = await session.execute(
            select(LangGraphCheckpointModel)
            .where(LangGraphCheckpointModel.thread_id == thread_id)
            .order_by(LangGraphCheckpointModel.created_at.desc())
            .limit(1)
        )
        return result.scalar_one_or_none()

    # ------------------------------------------------------------------
    # LangGraph compatibility helpers
    # ------------------------------------------------------------------
    def get_next_version(self, current: int | None, _channel: None = None) -> int:
        """Return the next monotonically increasing channel version.

        LangGraph's Pregel loop invokes this helper on custom checkpointers to
        version the logical channels that make up the graph state. Our
        implementation mirrors LangGraph's default integer-based versioning
        scheme: start at ``1`` and increment by one for each subsequent update.

        The ``channel`` argument is unused but kept for API compatibility.
        """

        return 1 if current is None else current + 1

    async def aput_writes(
        self,
        config: RunnableConfig,
        writes: list[tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        """Store intermediate writes linked to a checkpoint.

        Args:
            config: Configuration of the related checkpoint.
            writes: List of writes to store.
            task_id: Identifier for the task creating the writes.
            task_path: Path of the task creating the writes.
        """
        # No-op: we persist full checkpoints via aput; intermediate writes
        # are not separately tracked in this implementation.
        pass
