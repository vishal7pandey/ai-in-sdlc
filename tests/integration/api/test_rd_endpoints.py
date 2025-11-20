from __future__ import annotations

from datetime import datetime
from uuid import UUID

import pytest
from fastapi.testclient import TestClient

from src.main import app

client = TestClient(app)


@pytest.mark.integration
@pytest.mark.api
def test_rd_generate_and_retrieve_happy_path() -> None:
    """Create a session, send a message, generate RD, and retrieve it.

    This exercises the new RD API endpoints end-to-end at the backend level,
    without involving the frontend. It assumes that extraction and synthesis
    are wired into the orchestrator graph.
    """

    # 1. Create a new session
    create_resp = client.post(
        "/api/v1/sessions",
        json={"project_name": "RD API Test"},
        headers={"X-User-Id": "user-rd-test"},
    )
    assert create_resp.status_code == 201
    body = create_resp.json()
    session_id = body["id"]

    assert UUID(session_id)
    assert body["project_name"] == "RD API Test"
    datetime.fromisoformat(body["created_at"])

    # 2. Send at least one message to produce requirements
    msg_resp = client.post(
        f"/api/v1/sessions/{session_id}/messages",
        json={"message": "Users must be able to log in with email and password."},
        headers={"X-User-Id": "user-rd-test"},
    )
    assert msg_resp.status_code == 200

    # 3. Generate RD via the new endpoint
    rd_gen_resp = client.post(
        f"/api/v1/rd/{session_id}/generate",
        headers={"X-User-Id": "user-rd-test"},
    )
    assert rd_gen_resp.status_code == 200
    rd_body = rd_gen_resp.json()

    assert rd_body["session_id"] == session_id
    assert rd_body["version"] >= 1
    assert rd_body["format"] == "markdown"
    assert rd_body["status"] == "draft"
    assert "Requirements Document" in rd_body["content"]

    # 4. Retrieve RD via GET /rd/{session_id}
    rd_get_resp = client.get(f"/api/v1/rd/{session_id}")
    assert rd_get_resp.status_code == 200
    rd_get_body = rd_get_resp.json()

    assert rd_get_body["session_id"] == session_id
    assert rd_get_body["version"] == rd_body["version"]
    assert rd_get_body["content"] == rd_body["content"]

    # 5. Export RD via GET /rd/{session_id}/export
    rd_export_resp = client.get(f"/api/v1/rd/{session_id}/export")
    assert rd_export_resp.status_code == 200
    export_body = rd_export_resp.json()

    assert export_body["filename"].endswith(f"{session_id}.md")
    assert export_body["content"] == rd_body["content"]
