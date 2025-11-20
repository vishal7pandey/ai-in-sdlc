# WebSocket Protocol Specification v1.0

**Document Version**: 1.0
**Last Updated**: 2025-11-19
**Status**: Implementation Ready
**Scope**: Real-time bidirectional communication between frontend and backend

---

## Table of Contents

1. [Overview](#overview)
2. [Connection Management](#connection-management)
3. [Message Formats](#message-formats)
4. [Client-to-Server Events](#client-to-server-events)
5. [Server-to-Client Events](#server-to-client-events)
6. [Heartbeat & Keep-Alive](#heartbeat--keep-alive)
7. [Error Handling](#error-handling)
8. [Reconnection Strategy](#reconnection-strategy)
9. [Authentication & Authorization](#authentication--authorization)
10. [Performance Requirements](#performance-requirements)
11. [Implementation Checklist](#implementation-checklist)

---

## 1. Overview

### 1.1 Purpose

The WebSocket protocol enables:
- **Real-time AI response streaming** (character-by-character)
- **Live requirement extraction notifications**
- **Agent status updates** (conversational, extraction, synthesis, etc.)
- **Bidirectional communication** without HTTP polling overhead

### 1.2 Architecture Decision

**Single WebSocket per session** approach:
- One WebSocket connection per active session
- Connection tied to `session_id`
- Multiple browser tabs for same session = multiple connections (future optimization: shared connection)

**Rationale**: Simplifies state management, easier to debug, clear session isolation.

---

## 2. Connection Management

### 2.1 WebSocket Endpoint

**Endpoint URL**:
```
ws://localhost:8000/ws?session_id={uuid}&user_id={uuid}
```

**Production**:
```
wss://api.yourplatform.com/ws?session_id={uuid}&user_id={uuid}
```

**Query Parameters**:

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `session_id` | UUID v4 | Yes | Active session identifier |
| `user_id` | UUID v4 | Yes (when auth enabled) | Authenticated user ID |
| `token` | JWT string | Yes (when auth enabled) | Authentication token |

**Example Connection**:
```javascript
// Development (no auth)
const ws = new WebSocket(
  `ws://localhost:8000/ws?session_id=${sessionId}&user_id=anonymous`
);

// Production (with auth)
const ws = new WebSocket(
  `wss://api.platform.com/ws?session_id=${sessionId}&user_id=${userId}&token=${jwtToken}`
);
```

### 2.2 Connection Lifecycle

```
[CONNECTING] → [OPEN] → [CLOSING] → [CLOSED]
                 ↓
            [Message Exchange]
                 ↓
          [Heartbeat Loop]
```

**State Transitions**:

1. **CONNECTING** (0):
   - Client initiates connection
   - Server validates `session_id` exists
   - Server validates `user_id` (if auth enabled)
   - Server validates session belongs to user (if auth enabled)

2. **OPEN** (1):
   - Connection established
   - Server sends `connection.established` event
   - Client starts heartbeat timer (30s interval)
   - Ready for message exchange

3. **CLOSING** (2):
   - Graceful shutdown initiated
   - Server sends `connection.closing` event
   - Pending messages flushed

4. **CLOSED** (3):
   - Connection terminated
   - Client initiates reconnection (if not intentional close)

### 2.3 Connection Acceptance Criteria

Server **MUST** accept connection if:
- ✅ `session_id` is valid UUID format
- ✅ Session exists in database
- ✅ Session is not archived
- ✅ `user_id` matches session owner (if auth enabled)
- ✅ Token is valid and not expired (if auth enabled)

Server **MUST** reject connection if:
- ❌ `session_id` missing or invalid format
- ❌ Session not found (404)
- ❌ Session archived/deleted
- ❌ User not authorized for session (403)
- ❌ Token expired or invalid (401)

**Rejection Response**:
```json
{
  "type": "error.connection_rejected",
  "code": 4000,
  "reason": "session_not_found",
  "message": "Session abc123 does not exist"
}
```

Then server closes WebSocket with code `4000-4099`.

---

## 3. Message Formats

### 3.1 Base Message Structure

**All messages are JSON** with this structure:

```typescript
interface BaseMessage {
  type: string;           // Event type (e.g., "chat.message")
  session_id: string;     // UUID of session
  message_id?: string;    // UUID of message (if applicable)
  timestamp: string;      // ISO 8601 timestamp
  [key: string]: any;     // Event-specific fields
}
```

### 3.2 Message Types Overview

| Direction | Type | Purpose |
|-----------|------|---------|
| Client → Server | `chat.message` | User sends chat message |
| Client → Server | `ping` | Heartbeat ping |
| Server → Client | `pong` | Heartbeat pong |
| Server → Client | `connection.established` | Connection ready |
| Server → Client | `message.chunk` | AI response streaming |
| Server → Client | `message.complete` | AI response finished |
| Server → Client | `requirements.extracted` | Requirements extracted |
| Server → Client | `agent.status` | Agent status update |
| Server → Client | `error` | Error occurred |

### 3.3 TypeScript Definitions

```typescript
// Client-to-Server Events
type ClientMessage =
  | ChatMessageEvent
  | PingEvent;

// Server-to-Client Events
type ServerMessage =
  | ConnectionEstablishedEvent
  | MessageChunkEvent
  | MessageCompleteEvent
  | RequirementsExtractedEvent
  | AgentStatusEvent
  | PongEvent
  | ErrorEvent;
```

---

## 4. Client-to-Server Events

### 4.1 `chat.message` - User Sends Message

**Purpose**: User submits a chat message to the AI.

**Schema**:
```typescript
interface ChatMessageEvent {
  type: "chat.message";
  session_id: string;
  message_id: string;        // Client-generated UUID
  content: string;           // User's message text
  timestamp: string;         // ISO 8601 (client time)
  metadata?: {
    turn_number?: number;    // Optional: conversation turn
    parent_message_id?: string; // Optional: for threading
  };
}
```

**Example**:
```json
{
  "type": "chat.message",
  "session_id": "550e8400-e29b-41d4-a716-446655440000",
  "message_id": "7c9e6679-7425-40de-944b-e07fc1f90ae7",
  "content": "Users should be able to reset their password",
  "timestamp": "2025-11-19T14:30:00.000Z",
  "metadata": {
    "turn_number": 3
  }
}
```

**Server Processing**:
1. Validate session exists and is active
2. Save user message to `chat_messages` table
3. Invoke LangGraph orchestrator with updated state
4. Stream AI response via `message.chunk` events
5. If requirements extracted, send `requirements.extracted` event

**Server Response Flow**:
```
User sends chat.message
    ↓
Server sends agent.status (conversational, started)
    ↓
Server sends message.chunk (streaming response)
    ↓ (multiple chunks)
Server sends message.complete
    ↓
Server sends agent.status (conversational, completed)
    ↓ (if extraction triggered)
Server sends agent.status (extraction, started)
    ↓
Server sends requirements.extracted
    ↓
Server sends agent.status (extraction, completed)
```

### 4.2 `ping` - Heartbeat Ping

**Purpose**: Client keeps connection alive and verifies server responsiveness.

**Schema**:
```typescript
interface PingEvent {
  type: "ping";
  session_id: string;
  timestamp: string;
}
```

**Example**:
```json
{
  "type": "ping",
  "session_id": "550e8400-e29b-41d4-a716-446655440000",
  "timestamp": "2025-11-19T14:30:30.000Z"
}
```

**Server Response**:
- Immediately reply with `pong` event
- If no `pong` received within 5 seconds → client considers connection unhealthy

---

## 5. Server-to-Client Events

### 5.1 `connection.established` - Connection Ready

**Purpose**: Confirms connection is established and ready.

**Schema**:
```typescript
interface ConnectionEstablishedEvent {
  type: "connection.established";
  session_id: string;
  server_time: string;       // ISO 8601 server timestamp
  protocol_version: string;  // "1.0"
  features: {
    streaming: boolean;
    heartbeat_interval: number; // seconds
    max_message_size: number;   // bytes
  };
}
```

**Example**:
```json
{
  "type": "connection.established",
  "session_id": "550e8400-e29b-41d4-a716-446655440000",
  "server_time": "2025-11-19T14:30:00.000Z",
  "protocol_version": "1.0",
  "features": {
    "streaming": true,
    "heartbeat_interval": 30,
    "max_message_size": 10485760
  }
}
```

**Client Action**:
- Start heartbeat timer
- Update UI connection status to "Connected"
- Flush any queued messages from offline mode

### 5.2 `message.chunk` - AI Response Streaming

**Purpose**: Stream AI response character-by-character.

**Schema**:
```typescript
interface MessageChunkEvent {
  type: "message.chunk";
  session_id: string;
  message_id: string;        // AI message UUID
  turn_number: number;       // Conversation turn
  delta: string;             // Incremental text chunk
  accumulated?: string;      // Full text so far (optional, for recovery)
  metadata: {
    model: string;           // "gpt-4" etc.
    tokens_used: number;     // Running token count
    confidence?: number;     // 0.0-1.0
  };
  timestamp: string;
}
```

**Example**:
```json
{
  "type": "message.chunk",
  "session_id": "550e8400-e29b-41d4-a716-446655440000",
  "message_id": "9b1deb4d-3b7d-4bad-9bdd-2b0d7b3dcb6d",
  "turn_number": 4,
  "delta": "That's a great",
  "metadata": {
    "model": "gpt-4",
    "tokens_used": 45
  },
  "timestamp": "2025-11-19T14:30:01.123Z"
}
```

**Client Behavior**:
- Append `delta` to existing message text
- Display text with typing animation
- Show "AI is typing..." indicator

**Streaming Rate**:
- Target: 10-50 chunks per second
- Latency: < 100ms per chunk
- Buffer size: 10-50 characters per chunk (adjustable)

### 5.3 `message.complete` - AI Response Finished

**Purpose**: Signals AI response is complete.

**Schema**:
```typescript
interface MessageCompleteEvent {
  type: "message.complete";
  session_id: string;
  message_id: string;
  turn_number: number;
  full_content: string;      // Complete message text
  metadata: {
    model: string;
    total_tokens: number;
    duration_ms: number;
    confidence: number;       // 0.0-1.0
    next_action?: string;     // "extract_requirements" | "continue" | "clarify"
  };
  timestamp: string;
}
```

**Example**:
```json
{
  "type": "message.complete",
  "session_id": "550e8400-e29b-41d4-a716-446655440000",
  "message_id": "9b1deb4d-3b7d-4bad-9bdd-2b0d7b3dcb6d",
  "turn_number": 4,
  "full_content": "That's a great requirement. To clarify: should password reset...",
  "metadata": {
    "model": "gpt-4",
    "total_tokens": 234,
    "duration_ms": 1850,
    "confidence": 0.89,
    "next_action": "extract_requirements"
  },
  "timestamp": "2025-11-19T14:30:03.500Z"
}
```

**Client Action**:
- Stop "typing" indicator
- Save complete message to state
- Show confidence badge
- If `next_action === "extract_requirements"`, show "Extracting requirements..." indicator

### 5.4 `requirements.extracted` - Requirements Extracted

**Purpose**: Notify frontend that requirements have been extracted.

**Schema**:
```typescript
interface RequirementsExtractedEvent {
  type: "requirements.extracted";
  session_id: string;
  extraction_id: string;     // UUID of extraction batch
  requirements: RequirementSummary[];
  metadata: {
    extraction_duration_ms: number;
    total_extracted: number;
    total_session_requirements: number;
  };
  timestamp: string;
}

interface RequirementSummary {
  id: string;                // REQ-001, REQ-002, etc.
  title: string;             // First 50 chars
  type: string;              // "functional" | "non_functional" | ...
  confidence: number;        // 0.0-1.0
  inferred: boolean;
}
```

**Example**:
```json
{
  "type": "requirements.extracted",
  "session_id": "550e8400-e29b-41d4-a716-446655440000",
  "extraction_id": "ext-7c9e6679-7425-40de-944b",
  "requirements": [
    {
      "id": "REQ-003",
      "title": "User password reset via email",
      "type": "functional",
      "confidence": 0.92,
      "inferred": false
    },
    {
      "id": "REQ-004",
      "title": "Password reset link expires in 1 hour",
      "type": "non_functional",
      "confidence": 0.78,
      "inferred": true
    }
  ],
  "metadata": {
    "extraction_duration_ms": 450,
    "total_extracted": 2,
    "total_session_requirements": 4
  },
  "timestamp": "2025-11-19T14:30:04.000Z"
}
```

**Client Action**:
- Add new requirement cards to right panel
- Animate card appearance (fade in)
- Update session requirement count
- Show notification: "2 requirements extracted"

### 5.5 `agent.status` - Agent Status Update

**Purpose**: Inform frontend which agent is running.

**Schema**:
```typescript
interface AgentStatusEvent {
  type: "agent.status";
  session_id: string;
  agent: "conversational" | "extraction" | "inference" | "validation" | "synthesis" | "review";
  status: "started" | "running" | "completed" | "failed";
  metadata?: {
    duration_ms?: number;    // If completed/failed
    error?: string;          // If failed
    progress?: number;       // 0-100 (optional)
  };
  timestamp: string;
}
```

**Example**:
```json
{
  "type": "agent.status",
  "session_id": "550e8400-e29b-41d4-a716-446655440000",
  "agent": "extraction",
  "status": "started",
  "timestamp": "2025-11-19T14:30:03.600Z"
}
```

**Client Action**:
- Update agent status pill in UI
- Show "Extracting requirements..." message
- If failed, show error notification

**Agent Status Pills UI**:
```
Conversational ✓ → Extraction ⏳ → Validation ○ → Synthesis ○
```

### 5.6 `pong` - Heartbeat Response

**Purpose**: Server confirms connection alive.

**Schema**:
```typescript
interface PongEvent {
  type: "pong";
  session_id: string;
  timestamp: string;
  server_time: string;
}
```

**Example**:
```json
{
  "type": "pong",
  "session_id": "550e8400-e29b-41d4-a716-446655440000",
  "timestamp": "2025-11-19T14:30:30.123Z",
  "server_time": "2025-11-19T14:30:30.125Z"
}
```

**Client Action**:
- Reset heartbeat timeout timer
- Calculate latency: `server_time - timestamp`
- Update connection quality indicator

### 5.7 `error` - Error Occurred

**Purpose**: Notify client of errors.

**Schema**:
```typescript
interface ErrorEvent {
  type: "error";
  session_id: string;
  error_code: string;        // "agent_timeout" | "llm_rate_limit" | "session_not_found" | ...
  error_message: string;     // Human-readable message
  severity: "warning" | "error" | "fatal";
  metadata?: {
    agent?: string;          // Which agent failed
    retry_after?: number;    // Seconds to wait before retry
  };
  timestamp: string;
}
```

**Example**:
```json
{
  "type": "error",
  "session_id": "550e8400-e29b-41d4-a716-446655440000",
  "error_code": "llm_rate_limit",
  "error_message": "OpenAI rate limit exceeded. Please wait 60 seconds.",
  "severity": "error",
  "metadata": {
    "agent": "conversational",
    "retry_after": 60
  },
  "timestamp": "2025-11-19T14:30:05.000Z"
}
```

**Client Action**:
- Show error toast notification
- If `retry_after` present, disable send button with countdown
- If `severity === "fatal"`, close connection and show reconnect button

**Error Codes**:

| Code | Severity | Description |
|------|----------|-------------|
| `session_not_found` | fatal | Session doesn't exist |
| `session_archived` | fatal | Session is archived |
| `unauthorized` | fatal | User not authorized |
| `agent_timeout` | error | Agent took too long (>30s) |
| `llm_rate_limit` | error | OpenAI rate limit hit |
| `llm_error` | error | OpenAI API error |
| `extraction_failed` | warning | Extraction completed with errors |
| `validation_failed` | warning | Validation found issues |

---

## 6. Heartbeat & Keep-Alive

### 6.1 Heartbeat Interval

**Interval**: 30 seconds

**Client Behavior**:
```javascript
const HEARTBEAT_INTERVAL = 30000; // 30 seconds
const HEARTBEAT_TIMEOUT = 5000;   // 5 seconds

let heartbeatTimer;
let heartbeatTimeoutTimer;

function startHeartbeat() {
  heartbeatTimer = setInterval(() => {
    ws.send(JSON.stringify({
      type: 'ping',
      session_id: sessionId,
      timestamp: new Date().toISOString()
    }));

    // Expect pong within 5 seconds
    heartbeatTimeoutTimer = setTimeout(() => {
      console.error('Heartbeat timeout - connection unhealthy');
      ws.close(1006, 'Heartbeat timeout');
    }, HEARTBEAT_TIMEOUT);
  }, HEARTBEAT_INTERVAL);
}

function handlePong() {
  clearTimeout(heartbeatTimeoutTimer);
}
```

### 6.2 Connection Health States

| State | Condition | Action |
|-------|-----------|--------|
| **Healthy** | Pong received within 5s | Continue normal operation |
| **Degraded** | Pong delayed 5-10s | Show warning indicator |
| **Unhealthy** | No pong after 10s | Close connection, reconnect |

---

## 7. Error Handling

### 7.1 Client-Side Error Handling

**WebSocket Error Events**:

```javascript
ws.onerror = (error) => {
  console.error('WebSocket error:', error);
  // Show error notification
  showToast('Connection error. Reconnecting...', 'error');
};

ws.onclose = (event) => {
  console.log('WebSocket closed:', event.code, event.reason);

  // Categorize close codes
  if (event.code === 1000) {
    // Normal closure - don't reconnect
    showToast('Connection closed', 'info');
  } else if (event.code >= 4000 && event.code < 4100) {
    // Application error - don't reconnect automatically
    showToast(`Connection rejected: ${event.reason}`, 'error');
  } else {
    // Network error or server crash - reconnect
    scheduleReconnect();
  }
};
```

### 7.2 Server-Side Error Handling

**Server should send `error` event before closing for application errors**:

```python
async def handle_message(websocket, message):
    try:
        # Process message
        await process_chat_message(message)
    except SessionNotFoundError:
        await websocket.send_json({
            "type": "error",
            "session_id": message["session_id"],
            "error_code": "session_not_found",
            "error_message": "Session not found",
            "severity": "fatal"
        })
        await websocket.close(code=4004, reason="session_not_found")
    except RateLimitError as e:
        await websocket.send_json({
            "type": "error",
            "session_id": message["session_id"],
            "error_code": "llm_rate_limit",
            "error_message": str(e),
            "severity": "error",
            "metadata": {"retry_after": e.retry_after}
        })
```

---

## 8. Reconnection Strategy

### 8.1 Exponential Backoff

**Algorithm**:
```javascript
const INITIAL_RETRY_DELAY = 1000;     // 1 second
const MAX_RETRY_DELAY = 30000;        // 30 seconds
const MAX_RECONNECT_ATTEMPTS = 10;

let reconnectAttempts = 0;

function scheduleReconnect() {
  if (reconnectAttempts >= MAX_RECONNECT_ATTEMPTS) {
    showToast('Unable to reconnect. Please refresh the page.', 'error');
    return;
  }

  const delay = Math.min(
    INITIAL_RETRY_DELAY * Math.pow(2, reconnectAttempts),
    MAX_RETRY_DELAY
  );

  console.log(`Reconnecting in ${delay}ms (attempt ${reconnectAttempts + 1})`);

  setTimeout(() => {
    reconnectAttempts++;
    connect();
  }, delay);
}

function connect() {
  ws = new WebSocket(wsUrl);

  ws.onopen = () => {
    console.log('WebSocket connected');
    reconnectAttempts = 0; // Reset on successful connection
    startHeartbeat();
  };

  // ... error handlers
}
```

### 8.2 Message Queue During Reconnection

**Client should queue messages while disconnected**:

```javascript
let messageQueue = [];

function sendMessage(message) {
  if (ws.readyState === WebSocket.OPEN) {
    ws.send(JSON.stringify(message));
  } else {
    // Queue message for later
    messageQueue.push(message);
    showToast('Message queued. Will send when reconnected.', 'info');
  }
}

ws.onopen = () => {
  // Flush queued messages
  while (messageQueue.length > 0) {
    const message = messageQueue.shift();
    ws.send(JSON.stringify(message));
  }
};
```

---

## 9. Authentication & Authorization

### 9.1 Current State (No Auth)

**Development Mode**:
- No authentication required
- `user_id=anonymous` accepted
- All sessions accessible

### 9.2 Future State (With Auth)

**Connection with JWT**:
```
wss://api.platform.com/ws?session_id={uuid}&user_id={uuid}&token={jwt}
```

**Server Validation**:
1. Verify JWT signature and expiration
2. Extract `user_id` from JWT claims
3. Verify `user_id` matches query param
4. Check user has access to `session_id`
5. Reject with code `4001` if unauthorized

**Token Refresh**:
- Client should refresh JWT before expiration
- If token expires during connection, server sends `error` with `error_code: "token_expired"`
- Client must reconnect with new token

---

## 10. Performance Requirements

### 10.1 Latency Targets

| Metric | Target | Critical Threshold |
|--------|--------|-------------------|
| Connection establishment | < 500ms | < 1000ms |
| Message send (client → server) | < 50ms | < 100ms |
| Message chunk latency | < 100ms | < 200ms |
| Heartbeat round-trip | < 100ms | < 500ms |
| Reconnection time | < 2s | < 5s |

### 10.2 Throughput Targets

| Metric | Target |
|--------|--------|
| Streaming rate | 10-50 chunks/second |
| Concurrent connections per server | 1000+ |
| Message size limit | 10 MB |
| Max messages per second per connection | 10 |

### 10.3 Reliability Targets

| Metric | Target |
|--------|--------|
| Message delivery | 99.9% (at-least-once) |
| Connection uptime | 99.5% |
| Reconnection success rate | 95% |

---

## 11. Implementation Checklist

### 11.1 Backend Implementation

**FastAPI WebSocket Endpoint**:

- [ ] Create `src/api/routes/websocket.py`
- [ ] Add `/ws` endpoint with `WebSocket` dependency
- [ ] Validate `session_id` query parameter
- [ ] Validate session exists and is active
- [ ] Implement connection manager (track active connections)
- [ ] Handle `chat.message` event → invoke orchestrator
- [ ] Stream `message.chunk` events during LLM response
- [ ] Send `message.complete` when response finished
- [ ] Send `requirements.extracted` after extraction
- [ ] Send `agent.status` events for agent transitions
- [ ] Implement heartbeat: receive `ping`, send `pong`
- [ ] Handle errors gracefully, send `error` events
- [ ] Close connection properly with appropriate codes
- [ ] Add logging for all WebSocket events
- [ ] Add metrics (active connections, messages/sec, latency)

**Connection Manager**:

```python
# src/api/websocket/manager.py
from typing import Dict
from fastapi import WebSocket

class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}

    async def connect(self, session_id: str, websocket: WebSocket):
        await websocket.accept()
        self.active_connections[session_id] = websocket

    def disconnect(self, session_id: str):
        if session_id in self.active_connections:
            del self.active_connections[session_id]

    async def send_message(self, session_id: str, message: dict):
        if session_id in self.active_connections:
            await self.active_connections[session_id].send_json(message)

    async def broadcast(self, message: dict):
        for connection in self.active_connections.values():
            await connection.send_json(message)

manager = ConnectionManager()
```

**WebSocket Endpoint**:

```python
# src/api/routes/websocket.py
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Query
from src.api.websocket.manager import manager

router = APIRouter()

@router.websocket("/ws")
async def websocket_endpoint(
    websocket: WebSocket,
    session_id: str = Query(...),
    user_id: str = Query(default="anonymous")
):
    # Validate session exists
    session = await get_session(session_id)
    if not session:
        await websocket.close(code=4004, reason="session_not_found")
        return

    # Accept connection
    await manager.connect(session_id, websocket)

    # Send connection established
    await websocket.send_json({
        "type": "connection.established",
        "session_id": session_id,
        "server_time": datetime.utcnow().isoformat(),
        "protocol_version": "1.0",
        "features": {
            "streaming": True,
            "heartbeat_interval": 30,
            "max_message_size": 10485760
        }
    })

    try:
        while True:
            # Receive message
            data = await websocket.receive_json()

            # Handle message types
            if data["type"] == "chat.message":
                await handle_chat_message(websocket, session_id, data)
            elif data["type"] == "ping":
                await websocket.send_json({
                    "type": "pong",
                    "session_id": session_id,
                    "timestamp": data["timestamp"],
                    "server_time": datetime.utcnow().isoformat()
                })

    except WebSocketDisconnect:
        manager.disconnect(session_id)
        print(f"Client disconnected: {session_id}")
    except Exception as e:
        await websocket.send_json({
            "type": "error",
            "session_id": session_id,
            "error_code": "internal_error",
            "error_message": str(e),
            "severity": "error"
        })
        manager.disconnect(session_id)
```

### 11.2 Frontend Implementation

**useWebSocket Hook**:

- [ ] Create `src/hooks/useWebSocket.ts`
- [ ] Implement connection logic
- [ ] Implement reconnection with exponential backoff
- [ ] Implement heartbeat (ping every 30s)
- [ ] Handle all server event types
- [ ] Queue messages during disconnection
- [ ] Expose connection state and methods
- [ ] Add TypeScript types for all events

**Hook Implementation**:

```typescript
// frontend/src/hooks/useWebSocket.ts
import { useEffect, useRef, useState, useCallback } from 'react';

interface UseWebSocketProps {
  sessionId: string;
  userId?: string;
  onMessageChunk?: (chunk: MessageChunkEvent) => void;
  onMessageComplete?: (message: MessageCompleteEvent) => void;
  onRequirementsExtracted?: (data: RequirementsExtractedEvent) => void;
  onAgentStatus?: (status: AgentStatusEvent) => void;
  onError?: (error: ErrorEvent) => void;
}

export function useWebSocket({
  sessionId,
  userId = 'anonymous',
  onMessageChunk,
  onMessageComplete,
  onRequirementsExtracted,
  onAgentStatus,
  onError
}: UseWebSocketProps) {
  const [isConnected, setIsConnected] = useState(false);
  const [connectionQuality, setConnectionQuality] = useState<'good' | 'degraded' | 'poor'>('good');
  const wsRef = useRef<WebSocket | null>(null);
  const reconnectAttemptsRef = useRef(0);
  const heartbeatTimerRef = useRef<NodeJS.Timeout>();
  const heartbeatTimeoutRef = useRef<NodeJS.Timeout>();
  const messageQueueRef = useRef<any[]>([]);

  const connect = useCallback(() => {
    const wsUrl = `${import.meta.env.VITE_WS_URL}/ws?session_id=${sessionId}&user_id=${userId}`;
    const ws = new WebSocket(wsUrl);

    ws.onopen = () => {
      console.log('WebSocket connected');
      setIsConnected(true);
      setConnectionQuality('good');
      reconnectAttemptsRef.current = 0;

      // Start heartbeat
      startHeartbeat();

      // Flush queued messages
      while (messageQueueRef.current.length > 0) {
        const message = messageQueueRef.current.shift();
        ws.send(JSON.stringify(message));
      }
    };

    ws.onmessage = (event) => {
      const data = JSON.parse(event.data);

      switch (data.type) {
        case 'connection.established':
          console.log('Connection established:', data);
          break;
        case 'message.chunk':
          onMessageChunk?.(data);
          break;
        case 'message.complete':
          onMessageComplete?.(data);
          break;
        case 'requirements.extracted':
          onRequirementsExtracted?.(data);
          break;
        case 'agent.status':
          onAgentStatus?.(data);
          break;
        case 'pong':
          handlePong();
          break;
        case 'error':
          onError?.(data);
          break;
      }
    };

    ws.onerror = (error) => {
      console.error('WebSocket error:', error);
      setConnectionQuality('poor');
    };

    ws.onclose = (event) => {
      console.log('WebSocket closed:', event.code, event.reason);
      setIsConnected(false);
      stopHeartbeat();

      if (event.code !== 1000 && event.code < 4000) {
        // Reconnect on unexpected closure
        scheduleReconnect();
      }
    };

    wsRef.current = ws;
  }, [sessionId, userId]);

  const startHeartbeat = () => {
    heartbeatTimerRef.current = setInterval(() => {
      sendMessage({
        type: 'ping',
        session_id: sessionId,
        timestamp: new Date().toISOString()
      });

      heartbeatTimeoutRef.current = setTimeout(() => {
        console.warn('Heartbeat timeout');
        setConnectionQuality('degraded');
        wsRef.current?.close(1006, 'Heartbeat timeout');
      }, 5000);
    }, 30000);
  };

  const stopHeartbeat = () => {
    if (heartbeatTimerRef.current) {
      clearInterval(heartbeatTimerRef.current);
    }
    if (heartbeatTimeoutRef.current) {
      clearTimeout(heartbeatTimeoutRef.current);
    }
  };

  const handlePong = () => {
    if (heartbeatTimeoutRef.current) {
      clearTimeout(heartbeatTimeoutRef.current);
    }
    setConnectionQuality('good');
  };

  const scheduleReconnect = () => {
    const maxAttempts = 10;
    if (reconnectAttemptsRef.current >= maxAttempts) {
      console.error('Max reconnection attempts reached');
      return;
    }

    const delay = Math.min(1000 * Math.pow(2, reconnectAttemptsRef.current), 30000);

    setTimeout(() => {
      reconnectAttemptsRef.current++;
      connect();
    }, delay);
  };

  const sendMessage = useCallback((message: any) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify(message));
    } else {
      messageQueueRef.current.push(message);
    }
  }, []);

  useEffect(() => {
    connect();
    return () => {
      stopHeartbeat();
      wsRef.current?.close(1000, 'Component unmounted');
    };
  }, [connect]);

  return {
    isConnected,
    connectionQuality,
    sendMessage
  };
}
```

### 11.3 Testing Checklist

**Unit Tests**:
- [ ] Test connection manager (connect, disconnect, send)
- [ ] Test message serialization/deserialization
- [ ] Test heartbeat logic
- [ ] Test reconnection backoff calculation

**Integration Tests**:
- [ ] Test WebSocket connection establishment
- [ ] Test chat message → AI response flow
- [ ] Test requirements extraction notification
- [ ] Test agent status updates
- [ ] Test heartbeat ping/pong
- [ ] Test error handling
- [ ] Test reconnection after disconnect

**End-to-End Tests**:
- [ ] Test full user flow with WebSocket
- [ ] Test streaming response display
- [ ] Test offline → online reconnection
- [ ] Test concurrent users (load test)

---

## 12. Future Enhancements

### 12.1 Compression

Enable WebSocket compression for reduced bandwidth:

```python
# Backend
@router.websocket("/ws", compression="deflate")

# Benefits: 50-70% reduction in message size
```

### 12.2 Binary Frames

Use binary frames for large payloads (embeddings, images):

```javascript
// Send binary data
ws.send(new Uint8Array(embedding));
```

### 12.3 Sub-protocols

Support multiple sub-protocols for versioning:

```javascript
const ws = new WebSocket(url, ['reqeng-v1', 'reqeng-v2']);
```

### 12.4 Multi-tab Synchronization

Share one WebSocket across multiple tabs using SharedWorker:

```javascript
const worker = new SharedWorker('/ws-worker.js');
worker.port.postMessage({type: 'connect', sessionId});
```

---

## Appendix A: Error Codes Reference

| Code | Meaning | Action |
|------|---------|--------|
| 1000 | Normal closure | Don't reconnect |
| 1001 | Going away | Reconnect |
| 1006 | Abnormal closure | Reconnect |
| 4000 | Application error | Check `reason` |
| 4001 | Unauthorized | Re-authenticate |
| 4004 | Session not found | Show error, go to home |
| 4008 | Policy violation | Show error, don't reconnect |
| 4009 | Message too large | Show error, reduce size |

---

## Appendix B: Example Full Flow

**User sends message → AI responds → Requirements extracted**:

```
1. Client → Server:
{
  "type": "chat.message",
  "session_id": "abc123",
  "message_id": "msg001",
  "content": "Users need login",
  "timestamp": "2025-11-19T14:00:00Z"
}

2. Server → Client:
{
  "type": "agent.status",
  "session_id": "abc123",
  "agent": "conversational",
  "status": "started"
}

3. Server → Client (multiple):
{
  "type": "message.chunk",
  "session_id": "abc123",
  "message_id": "ai001",
  "delta": "Great! ",
  "timestamp": "2025-11-19T14:00:01Z"
}
{
  "type": "message.chunk",
  "delta": "Can you clarify..."
}

4. Server → Client:
{
  "type": "message.complete",
  "session_id": "abc123",
  "message_id": "ai001",
  "full_content": "Great! Can you clarify the authentication method?",
  "metadata": {
    "next_action": "extract_requirements"
  }
}

5. Server → Client:
{
  "type": "agent.status",
  "agent": "extraction",
  "status": "started"
}

6. Server → Client:
{
  "type": "requirements.extracted",
  "session_id": "abc123",
  "requirements": [
    {
      "id": "REQ-001",
      "title": "User authentication via email/password",
      "confidence": 0.89
    }
  ]
}

7. Server → Client:
{
  "type": "agent.status",
  "agent": "extraction",
  "status": "completed"
}
```

---

**End of WebSocket Protocol Specification v1.0**
