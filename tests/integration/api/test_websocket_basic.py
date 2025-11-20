from __future__ import annotations

from datetime import datetime

import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from src.main import app

client = TestClient(app)


def _create_session() -> str:
    resp = client.post(
        "/api/v1/sessions",
        json={"project_name": "WS Test Session"},
        headers={"X-User-Id": "user-ws-test"},
    )
    assert resp.status_code == 201
    body = resp.json()
    return body["id"]


@pytest.mark.integration
@pytest.mark.api
def test_websocket_connection_and_heartbeat() -> None:
    session_id = _create_session()

    with client.websocket_connect(f"/ws?session_id={session_id}&user_id=user-ws-test") as ws:
        # First message should be connection.established
        msg = ws.receive_json()
        assert msg["type"] == "connection.established"
        assert msg["session_id"] == session_id
        assert msg["protocol_version"] == "1.0"

        # Send a ping and expect a pong response.
        timestamp = datetime.utcnow().isoformat()
        ws.send_json(
            {
                "type": "ping",
                "session_id": session_id,
                "timestamp": timestamp,
            }
        )

        pong = ws.receive_json()
        assert pong["type"] == "pong"
        assert pong["session_id"] == session_id
        assert "server_time" in pong


@pytest.mark.integration
@pytest.mark.api
def test_websocket_rejects_invalid_session_id_format() -> None:
    with client.websocket_connect("/ws?session_id=not-a-uuid&user_id=user-ws-test") as ws:
        error = ws.receive_json()
        assert error["type"] == "error.connection_rejected"
        assert error["reason"] == "invalid_session_id"

        # Connection should be closed after the rejection message.
        with pytest.raises(WebSocketDisconnect):
            ws.receive_json()


@pytest.mark.integration
@pytest.mark.api
def test_websocket_rejects_unknown_session() -> None:
    # Use a syntactically valid UUID that does not correspond to any session.
    unknown_session = "00000000-0000-0000-0000-000000000000"

    with client.websocket_connect(f"/ws?session_id={unknown_session}&user_id=user-ws-test") as ws:
        error = ws.receive_json()
        assert error["type"] == "error.connection_rejected"
        assert error["reason"] == "session_not_found"

        with pytest.raises(WebSocketDisconnect):
            ws.receive_json()


@pytest.mark.integration
@pytest.mark.api
def test_websocket_allows_reconnect_after_close() -> None:
    session_id = _create_session()

    # First connection
    with client.websocket_connect(f"/ws?session_id={session_id}&user_id=user-ws-test") as ws1:
        msg = ws1.receive_json()
        assert msg["type"] == "connection.established"

    # After the context manager exits, the connection is closed. A new
    # connection for the same session should succeed and emit another
    # connection.established event.
    with client.websocket_connect(f"/ws?session_id={session_id}&user_id=user-ws-test") as ws2:
        msg2 = ws2.receive_json()
        assert msg2["type"] == "connection.established"
        assert msg2["session_id"] == session_id
