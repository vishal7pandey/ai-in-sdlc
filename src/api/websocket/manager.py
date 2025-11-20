from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - type checking only
    from fastapi import WebSocket


class ConnectionManager:
    def __init__(self) -> None:
        self.active_connections: dict[str, WebSocket] = {}

    async def connect(self, session_id: str, websocket: WebSocket) -> None:
        await websocket.accept()
        self.active_connections[session_id] = websocket

    def disconnect(self, session_id: str) -> None:
        if session_id in self.active_connections:
            del self.active_connections[session_id]

    async def send_message(self, session_id: str, message: dict) -> None:
        websocket = self.active_connections.get(session_id)
        if websocket is not None:
            await websocket.send_json(message)

    async def broadcast(self, message: dict) -> None:
        for connection in self.active_connections.values():
            await connection.send_json(message)


manager = ConnectionManager()
