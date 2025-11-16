# WebSocket Protocol Specification
## Requirements Engineering Platform

**Version:** 1.0
**Last Updated:** November 16, 2025

---

## Connection Establishment

```typescript
// Client-side connection
const WS_URL = import.meta.env.VITE_WS_URL || 'ws://localhost:8000/ws';

const socket = new WebSocket(`${WS_URL}?token=${authToken}&session_id=${sessionId}`);

socket.onopen = () => {
  console.log('WebSocket connected');
  // Send heartbeat every 30 seconds
  setInterval(() => {
    socket.send(JSON.stringify({ type: 'ping', timestamp: Date.now() }));
  }, 30000);
};
```

---

## Message Formats

### Client → Server Events

#### 1. Chat Message
```typescript
interface ChatMessageEvent {
  type: 'chat_message';
  payload: {
    session_id: string;
    message: string;
    timestamp: string;
  };
}

// Example
{
  "type": "chat_message",
  "payload": {
    "session_id": "550e8400-e29b-41d4-a716-446655440000",
    "message": "Users should be able to reset their password",
    "timestamp": "2025-11-16T10:30:00Z"
  }
}
```

#### 2. Requirement Update
```typescript
interface RequirementUpdateEvent {
  type: 'requirement_update';
  payload: {
    requirement_id: string;
    updates: Partial<Requirement>;
  };
}

// Example
{
  "type": "requirement_update",
  "payload": {
    "requirement_id": "REQ-001",
    "updates": {
      "priority": "high",
      "acceptance_criteria": [
        "User enters email",
        "System sends reset link",
        "Link expires in 1 hour"
      ]
    }
  }
}
```

#### 3. Accept Inferred Requirement
```typescript
interface AcceptInferredEvent {
  type: 'accept_inferred';
  payload: {
    requirement_id: string;
    accepted: boolean;
    edits?: Partial<Requirement>;
  };
}
```

#### 4. Request State Update
```typescript
interface StateUpdateRequestEvent {
  type: 'request_state';
  payload: {
    session_id: string;
  };
}
```

---

### Server → Client Events

#### 1. Message Chunk (Streaming)
```typescript
interface MessageChunkEvent {
  type: 'message_chunk';
  payload: {
    message_id: string;
    content: string;
    is_final: boolean;
  };
}

// Example (streaming response)
{
  "type": "message_chunk",
  "payload": {
    "message_id": "msg_123",
    "content": "I've captured that",
    "is_final": false
  }
}
// ... followed by more chunks ...
{
  "type": "message_chunk",
  "payload": {
    "message_id": "msg_123",
    "content": " as REQ-001: Password Reset.",
    "is_final": true
  }
}
```

#### 2. Agent Status Update
```typescript
interface AgentUpdateEvent {
  type: 'agent_update';
  payload: {
    agent: 'conversational' | 'extraction' | 'inference' | 'validation' | 'synthesis';
    status: 'idle' | 'running' | 'complete' | 'error';
    progress?: number; // 0-100
    message?: string;
  };
}

// Example
{
  "type": "agent_update",
  "payload": {
    "agent": "extraction",
    "status": "running",
    "progress": 45,
    "message": "Analyzing message for requirements..."
  }
}
```

#### 3. Requirements Extracted
```typescript
interface RequirementsExtractedEvent {
  type: 'requirements_extracted';
  payload: {
    requirements: Requirement[];
    metadata: {
      extraction_time_ms: number;
      confidence: number;
    };
  };
}

// Example
{
  "type": "requirements_extracted",
  "payload": {
    "requirements": [
      {
        "id": "REQ-001",
        "title": "Password Reset Functionality",
        "type": "functional",
        "actor": "User",
        "action": "Reset password via email link",
        "acceptance_criteria": ["..."],
        "confidence": 0.89,
        "inferred": false,
        "source_refs": ["chat_turn_5"]
      }
    ],
    "metadata": {
      "extraction_time_ms": 1250,
      "confidence": 0.89
    }
  }
}
```

#### 4. Inferred Requirements
```typescript
interface InferredRequirementsEvent {
  type: 'inferred_requirements';
  payload: {
    requirements: Requirement[];
    rationale: string;
  };
}

// Example
{
  "type": "inferred_requirements",
  "payload": {
    "requirements": [
      {
        "id": "REQ-INF-001",
        "title": "Password Strength Validation",
        "confidence": 0.72,
        "inferred": true,
        "rationale": "Password reset typically requires strong passwords"
      }
    ],
    "rationale": "Based on security best practices"
  }
}
```

#### 5. Validation Results
```typescript
interface ValidationResultsEvent {
  type: 'validation_results';
  payload: {
    issues: Array<{
      requirement_id: string;
      severity: 'error' | 'warning' | 'info';
      message: string;
      field?: string;
    }>;
    overall_quality: number; // 0-1
  };
}

// Example
{
  "type": "validation_results",
  "payload": {
    "issues": [
      {
        "requirement_id": "REQ-002",
        "severity": "warning",
        "message": "Acceptance criteria contains ambiguous verb 'optimize'",
        "field": "acceptance_criteria"
      }
    ],
    "overall_quality": 0.85
  }
}
```

#### 6. RD Update
```typescript
interface RDUpdateEvent {
  type: 'rd_update';
  payload: {
    version: number;
    content: string;
    format: 'markdown' | 'json';
    status: 'draft' | 'under_review' | 'approved';
  };
}
```

#### 7. Error Event
```typescript
interface ErrorEvent {
  type: 'error';
  payload: {
    error_code: string;
    message: string;
    recoverable: boolean;
    retry_after?: number; // seconds
  };
}

// Example
{
  "type": "error",
  "payload": {
    "error_code": "LLM_TIMEOUT",
    "message": "AI assistant timed out. Retrying...",
    "recoverable": true,
    "retry_after": 5
  }
}
```

#### 8. State Sync
```typescript
interface StateSyncEvent {
  type: 'state_sync';
  payload: {
    session_id: string;
    requirements_count: number;
    rd_version: number;
    approval_status: string;
    last_updated: string;
  };
}
```

---

## Heartbeat & Reconnection

### Heartbeat Protocol
```typescript
// Client sends ping every 30 seconds
setInterval(() => {
  if (socket.readyState === WebSocket.OPEN) {
    socket.send(JSON.stringify({ type: 'ping', timestamp: Date.now() }));
  }
}, 30000);

// Server responds with pong
socket.onmessage = (event) => {
  const data = JSON.parse(event.data);
  if (data.type === 'pong') {
    lastPongTime = Date.now();
  }
};

// Check for stale connection (no pong in 60 seconds)
setInterval(() => {
  if (Date.now() - lastPongTime > 60000) {
    console.warn('Connection stale, reconnecting...');
    socket.close();
    reconnect();
  }
}, 10000);
```

### Reconnection Strategy[100][103][106]
```typescript
// Exponential backoff reconnection
let reconnectAttempts = 0;
const MAX_RECONNECT_ATTEMPTS = 10;

function reconnect() {
  if (reconnectAttempts >= MAX_RECONNECT_ATTEMPTS) {
    console.error('Max reconnection attempts reached');
    showOfflineBanner();
    return;
  }

  const delay = Math.min(1000 * Math.pow(2, reconnectAttempts), 30000);
  console.log(`Reconnecting in ${delay}ms (attempt ${reconnectAttempts + 1})`);

  setTimeout(() => {
    reconnectAttempts++;
    socket = new WebSocket(WS_URL);

    socket.onopen = () => {
      console.log('Reconnected successfully');
      reconnectAttempts = 0;
      hideOfflineBanner();

      // Request state sync
      socket.send(JSON.stringify({
        type: 'request_state',
        payload: { session_id: currentSessionId }
      }));
    };

    socket.onerror = () => {
      reconnect();
    };
  }, delay);
}

socket.onclose = (event) => {
  if (event.code !== 1000) { // Not a normal close
    reconnect();
  }
};
```

---

## React Hook Implementation

```typescript
// src/hooks/useWebSocket.ts
import { useEffect, useRef, useState, useCallback } from 'react';
import { useAuthStore } from '@/store/authStore';

interface UseWebSocketOptions {
  sessionId: string;
  onMessage?: (event: any) => void;
  onError?: (error: Error) => void;
  onReconnect?: () => void;
}

export function useWebSocket({
  sessionId,
  onMessage,
  onError,
  onReconnect
}: UseWebSocketOptions) {
  const socketRef = useRef<WebSocket | null>(null);
  const [isConnected, setIsConnected] = useState(false);
  const [lastMessage, setLastMessage] = useState<any>(null);
  const { token } = useAuthStore();

  const connect = useCallback(() => {
    const ws = new WebSocket(
      `${import.meta.env.VITE_WS_URL}?token=${token}&session_id=${sessionId}`
    );

    ws.onopen = () => {
      console.log('[WS] Connected');
      setIsConnected(true);
    };

    ws.onmessage = (event) => {
      const data = JSON.parse(event.data);
      setLastMessage(data);
      onMessage?.(data);
    };

    ws.onerror = (error) => {
      console.error('[WS] Error:', error);
      onError?.(new Error('WebSocket error'));
    };

    ws.onclose = (event) => {
      console.log('[WS] Disconnected');
      setIsConnected(false);

      if (event.code !== 1000) {
        // Abnormal close, attempt reconnect
        setTimeout(() => {
          connect();
          onReconnect?.();
        }, 3000);
      }
    };

    socketRef.current = ws;
  }, [sessionId, token, onMessage, onError, onReconnect]);

  useEffect(() => {
    connect();

    return () => {
      socketRef.current?.close(1000);
    };
  }, [connect]);

  const sendMessage = useCallback((type: string, payload: any) => {
    if (socketRef.current?.readyState === WebSocket.OPEN) {
      socketRef.current.send(JSON.stringify({ type, payload }));
    } else {
      console.warn('[WS] Not connected, message queued');
      // Queue for when connection restored
    }
  }, []);

  return {
    isConnected,
    lastMessage,
    sendMessage,
    socket: socketRef.current
  };
}
```

---

## Event Handling in Components

```typescript
// Example: ChatPanel component
function ChatPanel({ sessionId }: { sessionId: string }) {
  const [messages, setMessages] = useState<Message[]>([]);
  const [streamingMessage, setStreamingMessage] = useState('');

  const handleWebSocketMessage = useCallback((event: any) => {
    switch (event.type) {
      case 'message_chunk':
        if (event.payload.is_final) {
          setMessages(prev => [...prev, {
            id: event.payload.message_id,
            role: 'assistant',
            content: streamingMessage + event.payload.content,
            timestamp: new Date().toISOString()
          }]);
          setStreamingMessage('');
        } else {
          setStreamingMessage(prev => prev + event.payload.content);
        }
        break;

      case 'agent_update':
        // Update agent status indicator
        setAgentStatus(event.payload.agent, event.payload.status);
        break;

      case 'requirements_extracted':
        // Add requirements to list with animation
        addRequirements(event.payload.requirements);
        break;

      case 'error':
        toast.error(event.payload.message);
        if (event.payload.recoverable) {
          // Show retry option
        }
        break;
    }
  }, [streamingMessage]);

  const { sendMessage, isConnected } = useWebSocket({
    sessionId,
    onMessage: handleWebSocketMessage
  });

  const handleSendMessage = (text: string) => {
    // Optimistic update
    setMessages(prev => [...prev, {
      id: generateId(),
      role: 'user',
      content: text,
      timestamp: new Date().toISOString()
    }]);

    // Send via WebSocket
    sendMessage('chat_message', {
      session_id: sessionId,
      message: text,
      timestamp: new Date().toISOString()
    });
  };

  return (
    <div>
      {!isConnected && <OfflineBanner />}

      {messages.map(msg => (
        <MessageBubble key={msg.id} {...msg} />
      ))}

      {streamingMessage && (
        <MessageBubble
          role="assistant"
          content={streamingMessage}
          isStreaming
          timestamp={new Date().toISOString()}
        />
      )}

      <ChatInput onSend={handleSendMessage} disabled={!isConnected} />
    </div>
  );
}
```

---

## Error Handling

### Connection Errors
```typescript
{
  "type": "error",
  "payload": {
    "error_code": "CONNECTION_LOST",
    "message": "Connection to server lost",
    "recoverable": true
  }
}
```

### LLM Errors
```typescript
{
  "type": "error",
  "payload": {
    "error_code": "LLM_RATE_LIMIT",
    "message": "AI rate limit exceeded. Please wait.",
    "recoverable": true,
    "retry_after": 60
  }
}
```

### Validation Errors
```typescript
{
  "type": "error",
  "payload": {
    "error_code": "VALIDATION_FAILED",
    "message": "Message validation failed",
    "recoverable": false,
    "details": {
      "field": "message",
      "reason": "exceeds_max_length"
    }
  }
}
```

---

## Performance Considerations

### Message Batching
For high-frequency updates (e.g., agent progress), batch messages:

```python
# Backend (FastAPI)
async def broadcast_agent_updates(session_id: str):
    batch = []

    async for update in agent_stream:
        batch.append(update)

        # Send batch every 100ms or when batch reaches 10 items
        if len(batch) >= 10 or time.time() - last_send > 0.1:
            await manager.send_json({
                "type": "agent_updates_batch",
                "payload": {"updates": batch}
            })
            batch = []
            last_send = time.time()
```

### Compression
Enable WebSocket compression for large messages:

```typescript
// Client-side (browser automatically supports)
const socket = new WebSocket(url, {
  perMessageDeflate: true
});
```

### Message Size Limits
- Max message size: 1MB
- If RD content > 1MB, use pagination or streaming

---

## Security

### Authentication
- JWT token passed in WebSocket connection URL
- Token validated on connection and periodically

### Authorization
- Each message validates user has access to session
- Rate limiting per user (100 messages/minute)

### Message Validation
All incoming messages validated against schema before processing.
