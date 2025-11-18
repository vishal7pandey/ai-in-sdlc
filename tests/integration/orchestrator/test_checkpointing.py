"""Integration tests for LangGraph orchestrator checkpointing.

These tests exercise the compiled graph together with the DualCheckpointer,
verifying that:

- Running the graph stores checkpoints in Postgres.
- Checkpoints can be loaded again via the checkpointer API.
"""

from __future__ import annotations

from datetime import datetime

import pytest
from sqlalchemy import select

from src.models.database import LangGraphCheckpointModel
from src.orchestrator import nodes as orchestrator_nodes
from src.orchestrator.checkpointer import DualCheckpointer
from src.orchestrator.graph import graph
from src.orchestrator.state import create_initial_state
from src.schemas import GraphState, Message
from src.storage.postgres import get_session, init_database


@pytest.mark.asyncio
async def test_checkpoint_saved_and_loaded_round_trip() -> None:
    """Run the graph for two turns and verify checkpoint persistence.

    We patch the conversational agent's execute method to avoid real LLM
    calls and ensure deterministic behavior, then:
    - Invoke the graph twice for the same thread.
    - Assert that at least one checkpoint row exists in Postgres.
    - Assert that DualCheckpointer.aget can restore a checkpoint.
    """

    await init_database()

    # Patch conversational agent execute to avoid LLM usage
    original_execute = orchestrator_nodes._conversational_agent.execute

    async def fake_execute(state):
        msg = Message(
            id="msg-stub",
            role="assistant",
            content="Stub response from conversational agent",
            timestamp=datetime.utcnow(),
        )
        return {
            "chat_history_update": [msg],
            "state_updates": {
                "confidence": state.confidence,
                # Keep the orchestrator on the conversational path by
                # default; routing logic can still inspect user intent
                # and validation state if needed.
                "last_next_action": "continue_eliciting",
            },
        }

    orchestrator_nodes._conversational_agent.execute = fake_execute  # type: ignore[method-assign]

    try:
        session_id = "orch-sess-1"
        thread_id = f"session-{session_id}"

        state = create_initial_state(
            session_id=session_id,
            project_name="OrchestratorCheckpointTest",
            user_id="user-orch",
        )

        config = {"configurable": {"thread_id": thread_id}}

        # First run: START -> conversational -> END
        result_raw = await graph.ainvoke(state, config=config)
        # LangGraph may return a plain dict; coerce to GraphState for convenience.
        if isinstance(result_raw, dict):
            result_state = GraphState.model_validate(result_raw)
        else:
            result_state = result_raw
        assert result_state.session_id == session_id
        assert result_state.chat_history  # stub assistant message added

        # Second run: simulate another user turn and invoke again
        user_msg = Message(
            id="msg-user-2",
            role="user",
            content="Please continue",
            timestamp=datetime.utcnow(),
        )
        state_turn2 = result_state.with_updates(
            chat_history=[*result_state.chat_history, user_msg],
            current_turn=result_state.current_turn + 1,
        )

        result_raw_2 = await graph.ainvoke(state_turn2, config=config)
        if isinstance(result_raw_2, dict):
            result_state_2 = GraphState.model_validate(result_raw_2)
        else:
            result_state_2 = result_raw_2

        assert result_state_2.current_turn == state_turn2.current_turn

        # Verify that at least one checkpoint row exists for this thread
        async with get_session() as db:
            db_result = await db.execute(
                select(LangGraphCheckpointModel).where(
                    LangGraphCheckpointModel.thread_id == thread_id,
                )
            )
            rows = db_result.scalars().all()

        assert rows, "Expected at least one checkpoint row in Postgres"

        # Verify that DualCheckpointer can load the latest checkpoint
        checkpointer = DualCheckpointer()
        loaded = await checkpointer.aget(config)

        assert loaded is not None
        assert isinstance(loaded, dict)

    finally:
        # Restore original behavior to avoid side effects on other tests
        orchestrator_nodes._conversational_agent.execute = original_execute  # type: ignore[method-assign]
