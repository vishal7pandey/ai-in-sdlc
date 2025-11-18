"""API integration tests for the orchestrator sessions endpoints.

These tests exercise the FastAPI layer around the LangGraph orchestrator,
verifying that:

- A session can be created via POST /api/v1/sessions.
- A message can be sent to the orchestrator via
  POST /api/v1/sessions/{session_id}/messages.

The tests focus on the basic happy path and assume that database and Redis
infrastructure are available (as in other integration tests).
"""

from __future__ import annotations

from datetime import datetime
from uuid import UUID

import pytest
from fastapi.testclient import TestClient

from src.main import app

client = TestClient(app)


@pytest.mark.integration
@pytest.mark.api
def test_send_message_happy_path() -> None:
    """Create a session and send a single message through the orchestrator API.

    This verifies that:
    - The session creation endpoint responds with valid metadata.
    - The message endpoint returns an OrchestratorTurnResponse-like payload
      with status and state fields.
    """

    # 1. Create a new session
    create_resp = client.post(
        "/api/v1/sessions",
        json={"project_name": "API Orchestrator Test"},
        headers={"X-User-Id": "user-api-test"},
    )
    assert create_resp.status_code == 201
    body = create_resp.json()
    session_id = body["id"]

    # Basic shape checks on session response
    assert UUID(session_id)  # valid UUID
    assert body["project_name"] == "API Orchestrator Test"
    assert body["user_id"] == "user-api-test"
    assert body["status"] in {"active", "reviewing", "approved", "archived"}
    # created_at should be a valid ISO timestamp
    datetime.fromisoformat(body["created_at"])

    # 2. Send a message to the orchestrator for this session
    msg_resp = client.post(
        f"/api/v1/sessions/{session_id}/messages",
        json={"message": "We need a login feature"},
        headers={"X-User-Id": "user-api-test"},
    )
    assert msg_resp.status_code == 200
    msg_body = msg_resp.json()

    # Check unified orchestrator response shape
    assert msg_body["status"] in {"ok", "interrupt"}
    assert "state" in msg_body
    state = msg_body["state"]

    # Minimal sanity checks on returned state
    assert state["session_id"] == session_id
    assert state["project_name"] == "API Orchestrator Test"
    assert state["user_id"] == "user-api-test"
    assert isinstance(state["current_turn"], int)
    assert isinstance(state["chat_history"], list)

    # If the graph completed a normal turn, we expect at least one assistant message
    if msg_body["status"] == "ok":
        assistant_msgs = [m for m in state["chat_history"] if m["role"] == "assistant"]
        assert assistant_msgs, "expected at least one assistant message in chat_history"
