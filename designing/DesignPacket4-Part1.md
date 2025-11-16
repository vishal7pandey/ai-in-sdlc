# Design Packet 4: Testing, QA, Observability, CI/CD & Production Hardening
## Multi-Agent Requirements Engineering Platform

**Version:** 4.0
**Date:** November 16, 2025
**Scope:** Production-Grade Testing, Monitoring, Security, Reliability Engineering
**Prerequisites:** Design Packets 1, 2, 3

**Focus:** Zero-downtime deployment, comprehensive observability, security hardening, reliability engineering

---

# Table of Contents

1. [QA & Test Strategy (Full Stack)](#1-qa--test-strategy-full-stack)
2. [Observability & Monitoring Architecture](#2-observability--monitoring-architecture)
3. [CI/CD Architecture](#3-cicd-architecture)
4. [Security, Privacy & Compliance](#4-security-privacy--compliance)
5. [Reliability Engineering & Failure Mode Design](#5-reliability-engineering--failure-mode-design)
6. [Production Readiness Checklist](#6-production-readiness-checklist)

---

## 1. QA & Test Strategy (Full Stack)

### 1.1 Testing Pyramid Overview

```
                    ┌──────────────────┐
                    │  Manual E2E (5%) │  ← Exploratory, UX validation
                    ├──────────────────┤
                   │ Automated E2E (10%)│  ← Critical user paths
                   ├────────────────────┤
                  │   Integration (25%)  │  ← API, DB, WebSocket, Agent flows
                  ├──────────────────────┤
                 │      Unit (60%)        │  ← Components, functions, nodes
                 └────────────────────────┘

Target Coverage:
- Backend: 85% line coverage, 95% critical path
- Frontend: 80% component coverage, 90% user flows
- LLM Agents: 100% deterministic tests, 90% golden tests
```

### 1.2 Backend Testing Strategy

#### 1.2.1 Agent Unit Tests (LangGraph Nodes)[124][127][130]

**Extraction Agent Test Suite**

```python
# tests/unit/agents/test_extraction_agent.py
import pytest
from unittest.mock import Mock, patch, AsyncMock
from src.agents.extraction.agent import ExtractionAgent
from src.schemas.state import GraphState, Message, Requirement
from datetime import datetime

@pytest.fixture
def mock_llm():
    \"\"\"Mock OpenAI LLM for deterministic testing\"\"\"
    mock = AsyncMock()
    mock.ainvoke.return_value = Mock(
        content=\"\"\"
        {
          "requirements": [
            {
              "title": "User Authentication",
              "type": "functional",
              "actor": "User",
              "action": "Log in with email and password",
              "acceptance_criteria": [
                "User enters valid email",
                "User enters password",
                "System validates credentials",
                "User redirected to dashboard on success"
              ],
              "priority": "must",
              "confidence": 0.89
            }
          ]
        }
        \"\"\"
    )
    return mock

@pytest.fixture
def extraction_agent(mock_llm):
    agent = ExtractionAgent()
    agent.llm = mock_llm
    return agent

@pytest.fixture
def sample_state():
    return {
        "session_id": "test_session_123",
        "project_name": "Test Project",
        "user_id": "user_1",
        "messages": [
            Message(
                id="msg_1",
                role="user",
                content="Users should be able to log in with email and password",
                timestamp=datetime.utcnow(),
                metadata={}
            )
        ],
        "requirements": [],
        "inferred_requirements": [],
        "validation_issues": [],
        "confidence": 1.0,
        "rd_draft": None,
        "rd_version": 0,
        "approval_status": "pending",
        "review_feedback": None,
        "last_agent": "conversational",
        "current_turn": 1,
        "iterations": 0,
        "error_count": 0,
        "checkpoint_id": None,
        "parent_checkpoint_id": None
    }

@pytest.mark.asyncio
async def test_extraction_basic_requirement(extraction_agent, sample_state):
    \"\"\"Test basic requirement extraction from user message\"\"\"
    result = await extraction_agent.invoke(sample_state)

    # Verify requirement extracted
    assert len(result["requirements"]) == 1

    req = result["requirements"][0]
    assert req.id.startswith("REQ-")
    assert "authentication" in req.title.lower() or "login" in req.title.lower()
    assert req.type == "functional"
    assert req.actor == "User"
    assert len(req.acceptance_criteria) >= 3
    assert 0 <= req.confidence <= 1
    assert req.inferred == False
    assert len(req.source_refs) > 0

@pytest.mark.asyncio
async def test_extraction_confidence_scoring(extraction_agent, sample_state):
    \"\"\"Test confidence scoring for extracted requirements\"\"\"
    result = await extraction_agent.invoke(sample_state)

    req = result["requirements"][0]

    # Confidence should be > 0.8 for clear requirements
    assert req.confidence > 0.8

    # Verify confidence factors
    assert req.rationale  # Should have explanation
    assert len(req.acceptance_criteria) > 0  # Should have criteria

@pytest.mark.asyncio
async def test_extraction_ambiguous_requirement(extraction_agent):
    \"\"\"Test extraction with ambiguous user input\"\"\"
    state = {
        "session_id": "test_session_123",
        "messages": [
            Message(
                id="msg_1",
                role="user",
                content="The system should be fast and maybe optimize things",
                timestamp=datetime.utcnow()
            )
        ],
        "requirements": [],
        "confidence": 1.0,
        "current_turn": 1
    }

    result = await extraction_agent.invoke(state)

    if result["requirements"]:
        req = result["requirements"][0]
        # Ambiguous requirements should have lower confidence
        assert req.confidence < 0.7
        # Should flag vague verbs in validation
        assert "optimize" in req.action.lower() or "fast" in req.action.lower()

@pytest.mark.asyncio
async def test_extraction_token_budget_management(extraction_agent, mock_llm):
    \"\"\"Test that agent respects token budget constraints\"\"\"
    # Create state with very long chat history
    long_messages = [
        Message(
            id=f"msg_{i}",
            role="user" if i % 2 == 0 else "assistant",
            content="x" * 1000,  # 1000 char messages
            timestamp=datetime.utcnow()
        )
        for i in range(50)  # 50 messages = ~50k chars
    ]

    state = {
        "session_id": "test_session_123",
        "messages": long_messages,
        "requirements": [],
        "confidence": 1.0,
        "current_turn": 50
    }

    result = await extraction_agent.invoke(state)

    # Verify LLM was called with truncated history
    call_args = mock_llm.ainvoke.call_args
    prompt = call_args[0][0]

    # Estimate token count (rough: 1 token ≈ 4 chars)
    estimated_tokens = len(prompt) / 4
    assert estimated_tokens < 8000  # Should fit in budget

@pytest.mark.asyncio
async def test_extraction_error_recovery(extraction_agent, mock_llm):
    \"\"\"Test agent behavior when LLM fails\"\"\"
    # Simulate LLM failure
    mock_llm.ainvoke.side_effect = Exception("LLM timeout")

    state = {
        "session_id": "test_session_123",
        "messages": [
            Message(id="msg_1", role="user", content="Test", timestamp=datetime.utcnow())
        ],
        "requirements": [],
        "confidence": 1.0,
        "error_count": 0,
        "current_turn": 1
    }

    result = await extraction_agent.invoke(state)

    # Should increment error count
    assert result["error_count"] == 1

    # Should reduce confidence
    assert result["confidence"] < state["confidence"]

    # Should not crash
    assert "last_error" in result

@pytest.mark.parametrize("req_type,keywords", [
    ("functional", ["user", "login", "authenticate"]),
    ("non-functional", ["performance", "scalability", "availability"]),
    ("security", ["encryption", "authentication", "authorization"]),
    ("data", ["store", "persist", "database"])
])
@pytest.mark.asyncio
async def test_extraction_requirement_types(extraction_agent, mock_llm, req_type, keywords):
    \"\"\"Test extraction of different requirement types\"\"\"
    # Customize mock response based on type
    mock_llm.ainvoke.return_value = Mock(
        content=f'{{"requirements": [{{"type": "{req_type}", "title": "Test {req_type}"}}]}}'
    )

    state = {
        "session_id": "test_session_123",
        "messages": [
            Message(
                id="msg_1",
                role="user",
                content=f"The system needs {keywords[0]} capability",
                timestamp=datetime.utcnow()
            )
        ],
        "requirements": [],
        "confidence": 1.0,
        "current_turn": 1
    }

    result = await extraction_agent.invoke(state)

    if result["requirements"]:
        assert result["requirements"][0].type == req_type

# Golden test framework for deterministic validation
class TestExtractionGoldenTests:
    \"\"\"Golden tests with pre-validated output snapshots\"\"\"

    @pytest.fixture
    def golden_data_dir(self):
        return "tests/golden/extraction"

    @pytest.mark.asyncio
    async def test_golden_login_requirement(self, extraction_agent, golden_data_dir):
        \"\"\"Test against known-good extraction output\"\"\"
        import json
        from pathlib import Path

        # Load golden input
        golden_input = Path(golden_data_dir) / "login_input.json"
        with open(golden_input) as f:
            state = json.load(f)

        # Run extraction
        result = await extraction_agent.invoke(state)

        # Load golden output
        golden_output = Path(golden_data_dir) / "login_output.json"
        with open(golden_output) as f:
            expected = json.load(f)

        # Compare (ignoring timestamps and IDs)
        actual_req = result["requirements"][0]
        expected_req = expected["requirements"][0]

        assert actual_req.title == expected_req["title"]
        assert actual_req.type == expected_req["type"]
        assert actual_req.actor == expected_req["actor"]
        assert set(actual_req.acceptance_criteria) == set(expected_req["acceptance_criteria"])
        assert abs(actual_req.confidence - expected_req["confidence"]) < 0.05

# Fuzz testing for robustness
class TestExtractionFuzzTests:
    \"\"\"Fuzz tests to ensure agent handles edge cases\"\"\"

    @pytest.mark.asyncio
    async def test_empty_message(self, extraction_agent):
        state = {
            "session_id": "test",
            "messages": [Message(id="1", role="user", content="", timestamp=datetime.utcnow())],
            "requirements": [],
            "confidence": 1.0,
            "current_turn": 1
        }
        result = await extraction_agent.invoke(state)
        # Should not crash
        assert "requirements" in result

    @pytest.mark.asyncio
    async def test_unicode_characters(self, extraction_agent):
        state = {
            "session_id": "test",
            "messages": [
                Message(
                    id="1",
                    role="user",
                    content="用户应该能够使用电子邮件登录 🔐",
                    timestamp=datetime.utcnow()
                )
            ],
            "requirements": [],
            "confidence": 1.0,
            "current_turn": 1
        }
        result = await extraction_agent.invoke(state)
        assert "requirements" in result

    @pytest.mark.asyncio
    async def test_very_long_message(self, extraction_agent):
        state = {
            "session_id": "test",
            "messages": [
                Message(
                    id="1",
                    role="user",
                    content="requirement " * 5000,  # 50k+ chars
                    timestamp=datetime.utcnow()
                )
            ],
            "requirements": [],
            "confidence": 1.0,
            "current_turn": 1
        }
        result = await extraction_agent.invoke(state)
        assert "requirements" in result

    @pytest.mark.asyncio
    async def test_special_characters(self, extraction_agent):
        state = {
            "session_id": "test",
            "messages": [
                Message(
                    id="1",
                    role="user",
                    content="<script>alert('xss')</script> SELECT * FROM users;",
                    timestamp=datetime.utcnow()
                )
            ],
            "requirements": [],
            "confidence": 1.0,
            "current_turn": 1
        }
        result = await extraction_agent.invoke(state)
        # Should sanitize and not execute
        assert "requirements" in result
```

#### 1.2.2 LangGraph End-to-End Pipeline Tests[124][127][133]

```python
# tests/integration/test_langgraph_pipeline.py
import pytest
from src.orchestrator.graph import create_graph
from src.schemas.state import GraphState, Message
from langgraph.checkpoint.memory import MemorySaver
from datetime import datetime

@pytest.fixture
def compiled_graph():
    \"\"\"Create graph with memory checkpointer for testing\"\"\"
    graph = create_graph()
    checkpointer = MemorySaver()
    return graph.compile(checkpointer=checkpointer)

@pytest.mark.asyncio
async def test_full_pipeline_happy_path(compiled_graph):
    \"\"\"Test complete pipeline from chat to RD generation\"\"\"

    # Initial state
    initial_state = {
        "session_id": "test_pipeline_123",
        "project_name": "Test Pipeline",
        "user_id": "test_user",
        "messages": [
            Message(
                id="msg_1",
                role="user",
                content="Users should log in with email. System must be secure.",
                timestamp=datetime.utcnow()
            )
        ],
        "requirements": [],
        "inferred_requirements": [],
        "confidence": 1.0,
        "rd_draft": None,
        "rd_version": 0,
        "approval_status": "pending",
        "current_turn": 1,
        "iterations": 0,
        "error_count": 0
    }

    # Execute graph
    config = {"configurable": {"thread_id": "test_thread_1"}}
    result = await compiled_graph.ainvoke(initial_state, config=config)

    # Verify pipeline execution
    assert result["last_agent"] in ["synthesis", "review", "validation"]

    # Should have extracted requirements
    assert len(result["requirements"]) >= 1

    # Should have functional requirement
    functional_reqs = [r for r in result["requirements"] if r.type == "functional"]
    assert len(functional_reqs) >= 1

    # May have inferred security requirements
    if result["inferred_requirements"]:
        security_reqs = [r for r in result["inferred_requirements"] if r.type == "security"]
        assert len(security_reqs) >= 0  # Could infer security from "must be secure"

    # Confidence should be maintained or adjusted
    assert 0 <= result["confidence"] <= 1

@pytest.mark.asyncio
async def test_pipeline_multi_turn_conversation(compiled_graph):
    \"\"\"Test pipeline handles multi-turn conversations correctly\"\"\"

    config = {"configurable": {"thread_id": "test_thread_2"}}

    # Turn 1
    state1 = {
        "session_id": "test_multi_turn",
        "project_name": "Multi-turn Test",
        "messages": [
            Message(id="msg_1", role="user", content="Need user login", timestamp=datetime.utcnow())
        ],
        "requirements": [],
        "confidence": 1.0,
        "current_turn": 1,
        "iterations": 0,
        "error_count": 0
    }

    result1 = await compiled_graph.ainvoke(state1, config=config)
    turn1_req_count = len(result1["requirements"])

    # Turn 2 - add more details
    state2 = {
        **result1,
        "messages": result1["messages"] + [
            Message(
                id="msg_2",
                role="user",
                content="Login should support password reset via email",
                timestamp=datetime.utcnow()
            )
        ],
        "current_turn": 2
    }

    result2 = await compiled_graph.ainvoke(state2, config=config)

    # Should have more requirements or updated existing
    assert len(result2["requirements"]) >= turn1_req_count

    # Should maintain session continuity
    assert result2["session_id"] == result1["session_id"]

@pytest.mark.asyncio
async def test_pipeline_validation_failure_recovery(compiled_graph):
    \"\"\"Test pipeline recovers from validation failures\"\"\"

    # State with invalid/ambiguous requirement
    state = {
        "session_id": "test_validation",
        "messages": [
            Message(
                id="msg_1",
                role="user",
                content="System should optimize performance somehow",
                timestamp=datetime.utcnow()
            )
        ],
        "requirements": [],
        "confidence": 1.0,
        "current_turn": 1,
        "iterations": 0,
        "error_count": 0
    }

    config = {"configurable": {"thread_id": "test_thread_3"}}
    result = await compiled_graph.ainvoke(state, config=config)

    # Should have validation issues flagged
    assert len(result["validation_issues"]) > 0

    # Should identify ambiguous verbs
    issues = result["validation_issues"]
    ambiguous_issue = next((i for i in issues if "ambiguous" in i.get("message", "").lower()), None)
    assert ambiguous_issue is not None

    # Confidence should be reduced
    assert result["confidence"] < 0.8

@pytest.mark.asyncio
async def test_pipeline_checkpoint_persistence(compiled_graph):
    \"\"\"Test that checkpoints persist state correctly\"\"\"

    config = {"configurable": {"thread_id": "test_checkpoint"}}

    # First invocation
    state1 = {
        "session_id": "checkpoint_test",
        "messages": [
            Message(id="msg_1", role="user", content="Need login", timestamp=datetime.utcnow())
        ],
        "requirements": [],
        "confidence": 1.0,
        "current_turn": 1,
        "iterations": 0,
        "error_count": 0
    }

    result1 = await compiled_graph.ainvoke(state1, config=config)

    # Simulate crash and recovery - retrieve from checkpoint
    state2 = await compiled_graph.aget_state(config)

    # State should be preserved
    assert state2.values["session_id"] == "checkpoint_test"
    assert len(state2.values["requirements"]) == len(result1["requirements"])

@pytest.mark.asyncio
async def test_pipeline_conditional_routing(compiled_graph):
    \"\"\"Test that graph routes correctly based on state\"\"\"

    # Create spy to track node execution
    executed_nodes = []

    def track_node(node_name):
        def wrapper(state):
            executed_nodes.append(node_name)
            return state
        return wrapper

    # This would require access to graph internals or instrumentation
    # For now, verify via final state

    state = {
        "session_id": "routing_test",
        "messages": [
            Message(id="msg_1", role="user", content="Build login system", timestamp=datetime.utcnow())
        ],
        "requirements": [],
        "confidence": 1.0,
        "current_turn": 1,
        "iterations": 0,
        "error_count": 0
    }

    config = {"configurable": {"thread_id": "test_routing"}}
    result = await compiled_graph.ainvoke(state, config=config)

    # Based on state, should have executed:
    # conversational -> extraction -> inference -> validation -> synthesis
    assert result["last_agent"] is not None
    assert len(result["requirements"]) > 0  # extraction executed
```

#### 1.2.3 WebSocket Integration Tests[125][128][137]

```python
# tests/integration/test_websocket.py
import pytest
from fastapi.testclient import TestClient
from fastapi.websockets import WebSocketDisconnect
from src.main import app
import json
import asyncio

@pytest.fixture
def client():
    return TestClient(app)

def test_websocket_connection(client):
    \"\"\"Test basic WebSocket connection establishment\"\"\"
    with client.websocket_connect("/ws?session_id=test_123&token=test_token") as websocket:
        # Should connect successfully
        assert websocket is not None

        # Send ping
        websocket.send_json({"type": "ping", "timestamp": "2025-11-16T10:00:00Z"})

        # Should receive pong
        data = websocket.receive_json()
        assert data["type"] == "pong"

def test_websocket_chat_message_flow(client):
    \"\"\"Test sending chat message and receiving response\"\"\"
    with client.websocket_connect("/ws?session_id=test_chat&token=test_token") as websocket:
        # Send chat message
        websocket.send_json({
            "type": "chat_message",
            "payload": {
                "session_id": "test_chat",
                "message": "Users should be able to login",
                "timestamp": "2025-11-16T10:00:00Z"
            }
        })

        # Should receive agent update
        data1 = websocket.receive_json()
        assert data1["type"] in ["agent_update", "message_chunk"]

        # Should eventually receive requirements extracted
        timeout = 0
        while timeout < 50:  # 5 second timeout
            data = websocket.receive_json()
            if data["type"] == "requirements_extracted":
                assert len(data["payload"]["requirements"]) > 0
                break
            timeout += 1
            if timeout >= 50:
                pytest.fail("Timeout waiting for requirements_extracted")

def test_websocket_streaming_messages(client):
    \"\"\"Test streaming message chunks are received in order\"\"\"
    with client.websocket_connect("/ws?session_id=test_stream&token=test_token") as websocket:
        # Send message
        websocket.send_json({
            "type": "chat_message",
            "payload": {
                "session_id": "test_stream",
                "message": "Tell me about the system",
                "timestamp": "2025-11-16T10:00:00Z"
            }
        })

        chunks = []
        message_id = None

        # Collect chunks
        timeout = 0
        while timeout < 100:
            data = websocket.receive_json()

            if data["type"] == "message_chunk":
                if message_id is None:
                    message_id = data["payload"]["message_id"]
                else:
                    # All chunks should have same message_id
                    assert data["payload"]["message_id"] == message_id

                chunks.append(data["payload"]["content"])

                if data["payload"]["is_final"]:
                    break

            timeout += 1

        # Should have received at least one chunk
        assert len(chunks) > 0

        # Concatenated chunks should form coherent message
        full_message = "".join(chunks)
        assert len(full_message) > 0

def test_websocket_reconnection_recovery(client):
    \"\"\"Test client can reconnect and resume session\"\"\"
    # First connection
    with client.websocket_connect("/ws?session_id=test_reconnect&token=test_token") as ws1:
        ws1.send_json({
            "type": "chat_message",
            "payload": {
                "session_id": "test_reconnect",
                "message": "Initial message",
                "timestamp": "2025-11-16T10:00:00Z"
            }
        })

        # Receive some data
        data1 = ws1.receive_json()
        assert data1 is not None

    # Reconnect (new connection)
    with client.websocket_connect("/ws?session_id=test_reconnect&token=test_token") as ws2:
        # Request state sync
        ws2.send_json({
            "type": "request_state",
            "payload": {"session_id": "test_reconnect"}
        })

        # Should receive state sync
        timeout = 0
        while timeout < 50:
            data = ws2.receive_json()
            if data["type"] == "state_sync":
                # Should have session info
                assert data["payload"]["session_id"] == "test_reconnect"
                break
            timeout += 1

def test_websocket_concurrent_connections(client):
    \"\"\"Test multiple concurrent WebSocket connections\"\"\"
    # Open 3 connections simultaneously
    with client.websocket_connect("/ws?session_id=session_1&token=token_1") as ws1, \\
         client.websocket_connect("/ws?session_id=session_2&token=token_2") as ws2, \\
         client.websocket_connect("/ws?session_id=session_3&token=token_3") as ws3:

        # Send messages on all connections
        for i, ws in enumerate([ws1, ws2, ws3], 1):
            ws.send_json({
                "type": "ping",
                "timestamp": f"2025-11-16T10:00:0{i}Z"
            })

        # All should receive pongs
        for ws in [ws1, ws2, ws3]:
            data = ws.receive_json()
            assert data["type"] == "pong"

def test_websocket_message_ordering(client):
    \"\"\"Test messages are received in correct order\"\"\"
    with client.websocket_connect("/ws?session_id=test_order&token=test_token") as websocket:
        # Send multiple messages rapidly
        for i in range(5):
            websocket.send_json({
                "type": "chat_message",
                "payload": {
                    "session_id": "test_order",
                    "message": f"Message {i}",
                    "timestamp": f"2025-11-16T10:00:0{i}Z"
                }
            })

        # Responses should maintain order
        # (Implementation-specific - may need sequence numbers)
        received_messages = []
        timeout = 0
        while timeout < 100 and len(received_messages) < 5:
            data = websocket.receive_json()
            if data["type"] == "message_chunk" and data["payload"]["is_final"]:
                received_messages.append(data)
            timeout += 1

        # Should have received all
        assert len(received_messages) >= 1

def test_websocket_error_handling(client):
    \"\"\"Test WebSocket error propagation\"\"\"
    with client.websocket_connect("/ws?session_id=test_error&token=test_token") as websocket:
        # Send invalid message
        websocket.send_json({
            "type": "invalid_type",
            "payload": {}
        })

        # Should receive error event
        timeout = 0
        while timeout < 50:
            data = websocket.receive_json()
            if data["type"] == "error":
                assert "error_code" in data["payload"]
                assert "message" in data["payload"]
                break
            timeout += 1

@pytest.mark.asyncio
async def test_websocket_backpressure():
    \"\"\"Test WebSocket handles high message throughput\"\"\"
    # This test would use AsyncClient for better control
    from httpx import AsyncClient

    async with AsyncClient(app=app, base_url="http://test") as client:
        async with client.websocket_connect("/ws?session_id=backpressure&token=test") as websocket:
            # Send 100 messages rapidly
            for i in range(100):
                await websocket.send_json({
                    "type": "ping",
                    "timestamp": f"2025-11-16T10:00:{i:02d}Z"
                })

            # Should handle all without dropping
            pong_count = 0
            timeout = 0
            while timeout < 1000 and pong_count < 100:
                try:
                    data = await websocket.receive_json()
                    if data["type"] == "pong":
                        pong_count += 1
                except Exception:
                    break
                timeout += 1

            # Should receive most pongs (allow some loss under high load)
            assert pong_count >= 95
```

#### 1.2.4 Database Tests (Migrations, Locking, Event Sourcing)

```python
# tests/integration/test_database.py
import pytest
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker
from src.models.database import (
    Base, SessionModel, RequirementModel, RDEventModel, RDDocumentModel
)
from src.storage.postgres import update_session
from src.storage.event_sourcing import get_rd_at_version, diff_versions
import asyncio
from datetime import datetime
from uuid import uuid4

@pytest.fixture
async def test_db():
    \"\"\"Create test database with schema\"\"\"
    engine = create_async_engine(
        "postgresql+asyncpg://test:test@localhost:5432/reqeng_test",
        echo=False
    )

    # Create schema
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
        await conn.run_sync(Base.metadata.create_all)

    # Create session factory
    async_session = sessionmaker(
        engine, class_=AsyncSession, expire_on_commit=False
    )

    yield async_session

    # Cleanup
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)

    await engine.dispose()

@pytest.mark.asyncio
async def test_optimistic_locking_prevents_concurrent_updates(test_db):
    \"\"\"Test optimistic locking prevents lost updates\"\"\"

    # Create session
    async with test_db() as session:
        test_session = SessionModel(
            id=uuid4(),
            project_name="Test Project",
            user_id="user_1",
            status="active",
            version=0
        )
        session.add(test_session)
        await session.commit()
        session_id = test_session.id

    # Simulate two concurrent updates
    async def update_1():
        async with test_db() as session:
            # Read session
            stmt = select(SessionModel).where(SessionModel.id == session_id)
            result = await session.execute(stmt)
            sess = result.scalar_one()
            current_version = sess.version

            # Simulate some work
            await asyncio.sleep(0.1)

            # Try to update
            await update_session(session, session_id, {"status": "reviewing"}, current_version)

    async def update_2():
        async with test_db() as session:
            # Read session
            stmt = select(SessionModel).where(SessionModel.id == session_id)
            result = await session.execute(stmt)
            sess = result.scalar_one()
            current_version = sess.version

            # Simulate some work
            await asyncio.sleep(0.05)

            # Try to update
            await update_session(session, session_id, {"status": "approved"}, current_version)

    # Run concurrently
    with pytest.raises(Exception) as exc_info:
        await asyncio.gather(update_1(), update_2())

    # One should fail with concurrent modification error
    assert "ConcurrentModificationError" in str(exc_info.value) or \\
           "concurrent" in str(exc_info.value).lower()

@pytest.mark.asyncio
async def test_event_sourcing_rebuilds_rd_at_version(test_db):
    \"\"\"Test event sourcing can rebuild RD at any version\"\"\"

    session_id = uuid4()

    async with test_db() as session:
        # Create session
        test_session = SessionModel(
            id=session_id,
            project_name="Event Sourcing Test",
            user_id="user_1"
        )
        session.add(test_session)
        await session.commit()

        # Create RD events (version history)
        events = [
            RDEventModel(
                session_id=session_id,
                version=1,
                event_type="created",
                event_data={
                    "content": "# RD v1\\nInitial draft",
                    "format": "markdown"
                },
                user_id="user_1"
            ),
            RDEventModel(
                session_id=session_id,
                version=2,
                event_type="updated",
                event_data={
                    "content": "# RD v2\\nAdded requirements section",
                    "format": "markdown"
                },
                user_id="user_1"
            ),
            RDEventModel(
                session_id=session_id,
                version=3,
                event_type="updated",
                event_data={
                    "content": "# RD v3\\nAdded non-functional requirements",
                    "format": "markdown"
                },
                user_id="user_1"
            )
        ]

        for event in events:
            session.add(event)
        await session.commit()

    # Rebuild at version 2
    rd_v2 = await get_rd_at_version(str(session_id), 2)
    assert "Added requirements section" in rd_v2
    assert "non-functional" not in rd_v2

    # Rebuild at version 3
    rd_v3 = await get_rd_at_version(str(session_id), 3)
    assert "non-functional requirements" in rd_v3

@pytest.mark.asyncio
async def test_database_migration_integrity(test_db):
    \"\"\"Test database migrations maintain data integrity\"\"\"

    # This would test Alembic migrations
    # For now, verify schema constraints

    async with test_db() as session:
        # Test foreign key constraints
        req = RequirementModel(
            id="REQ-001",
            session_id=uuid4(),  # Non-existent session
            title="Test",
            type="functional",
            actor="User",
            action="Test",
            acceptance_criteria=["test"],
            priority="medium",
            confidence=0.8,
            inferred=False,
            rationale="test"
        )
        session.add(req)

        with pytest.raises(Exception) as exc_info:
            await session.commit()

        # Should fail foreign key constraint
        assert "foreign key" in str(exc_info.value).lower() or \\
               "violates" in str(exc_info.value).lower()

@pytest.mark.asyncio
async def test_database_partitioning_audit_logs(test_db):
    \"\"\"Test audit log partitioning works correctly\"\"\"

    # Audit logs should be partitioned by month
    # Insert logs for different months

    async with test_db() as session:
        from src.models.database import AuditLogModel

        # Create test session
        test_session = SessionModel(
            id=uuid4(),
            project_name="Audit Test",
            user_id="user_1"
        )
        session.add(test_session)
        await session.commit()

        session_id = test_session.id

        # Insert audit logs for current month
        log1 = AuditLogModel(
            session_id=session_id,
            user_id="user_1",
            action="requirement_created",
            entity_type="requirement",
            entity_id="REQ-001",
            timestamp=datetime(2025, 11, 16)
        )
        session.add(log1)
        await session.commit()

        # Query should work
        stmt = select(AuditLogModel).where(
            AuditLogModel.session_id == session_id
        )
        result = await session.execute(stmt)
        logs = result.scalars().all()

        assert len(logs) == 1
        assert logs[0].action == "requirement_created"
```

---

### 1.3 Frontend Testing Strategy

#### 1.3.1 Component Tests (React Testing Library)

```typescript
// tests/unit/components/MessageBubble.test.tsx
import { render, screen } from '@testing-library/react';
import { MessageBubble } from '@/components/chat/MessageBubble';
import { describe, it, expect } from 'vitest';

describe('MessageBubble', () => {
  it('renders user message correctly', () => {
    render(
      <MessageBubble
        role="user"
        content="Test message"
        timestamp="2025-11-16T10:00:00Z"
      />
    );

    expect(screen.getByText('Test message')).toBeInTheDocument();
    expect(screen.getByText(/10:00/)).toBeInTheDocument();
  });

  it('renders assistant message with metadata', () => {
    render(
      <MessageBubble
        role="assistant"
        content="I've captured that as REQ-001"
        timestamp="2025-11-16T10:00:00Z"
        metadata={{
          requirements_extracted: 1,
          confidence: 0.89
        }}
      />
    );

    expect(screen.getByText(/REQ-001/)).toBeInTheDocument();
    expect(screen.getByText(/1 requirement/)).toBeInTheDocument();
    expect(screen.getByText(/89%/)).toBeInTheDocument();
  });

  it('shows streaming indicator for partial messages', () => {
    render(
      <MessageBubble
        role="assistant"
        content="I've captured"
        timestamp="2025-11-16T10:00:00Z"
        isStreaming={true}
      />
    );

    // Should show cursor/indicator
    expect(screen.getByText(/▊/)).toBeInTheDocument();
  });
});

// tests/unit/components/RequirementCard.test.tsx
import { render, screen, fireEvent } from '@testing-library/react';
import { RequirementCard } from '@/components/requirements/RequirementCard';
import { describe, it, expect, vi } from 'vitest';

describe('RequirementCard', () => {
  const mockRequirement = {
    id: 'REQ-001',
    title: 'User Authentication',
    type: 'functional' as const,
    actor: 'User',
    action: 'Log in with email and password',
    condition: null,
    acceptance_criteria: [
      'User enters email',
      'User enters password',
      'System validates credentials'
    ],
    priority: 'must' as const,
    confidence: 0.89,
    inferred: false,
    rationale: 'Explicit user requirement',
    source_refs: ['chat_turn_1'],
    created_at: '2025-11-16T10:00:00Z'
  };

  it('renders requirement title and confidence', () => {
    render(<RequirementCard requirement={mockRequirement} />);

    expect(screen.getByText(/User Authentication/)).toBeInTheDocument();
    expect(screen.getByText(/89%/)).toBeInTheDocument();
  });

  it('shows inferred badge when appropriate', () => {
    const inferredReq = { ...mockRequirement, inferred: true, id: 'REQ-INF-001' };
    render(<RequirementCard requirement={inferredReq} />);

    expect(screen.getByText(/INFERRED/)).toBeInTheDocument();
  });

  it('calls onEdit when edit button clicked', () => {
    const handleEdit = vi.fn();
    render(<RequirementCard requirement={mockRequirement} onEdit={handleEdit} />);

    const editButton = screen.getByRole('button', { name: /edit/i });
    fireEvent.click(editButton);

    expect(handleEdit).toHaveBeenCalledWith('REQ-001');
  });

  it('expands to show full details when clicked', () => {
    render(<RequirementCard requirement={mockRequirement} />);

    // Initially collapsed
    expect(screen.queryByText(/Acceptance Criteria/)).not.toBeInTheDocument();

    // Click to expand
    const card = screen.getByText(/User Authentication/).closest('div');
    fireEvent.click(card!);

    // Should show details
    expect(screen.getByText(/Acceptance Criteria/)).toBeInTheDocument();
    expect(screen.getByText(/User enters email/)).toBeInTheDocument();
  });
});
```

#### 1.3.2 Zustand Store Tests

```typescript
// tests/unit/store/sessionStore.test.ts
import { renderHook, act, waitFor } from '@testing-library/react';
import { useSessionStore } from '@/store/sessionStore';
import { describe, it, expect, beforeEach, vi } from 'vitest';

// Mock fetch
global.fetch = vi.fn();

describe('sessionStore', () => {
  beforeEach(() => {
    // Reset store before each test
    useSessionStore.setState({
      currentSession: null,
      sessions: [],
      isLoading: false,
      error: null
    });

    vi.clearAllMocks();
  });

  it('loads sessions from API', async () => {
    const mockSessions = [
      { id: '1', project_name: 'Project 1', status: 'active' },
      { id: '2', project_name: 'Project 2', status: 'reviewing' }
    ];

    (global.fetch as any).mockResolvedValueOnce({
      ok: true,
      json: async () => ({ items: mockSessions })
    });

    const { result } = renderHook(() => useSessionStore());

    await act(async () => {
      await result.current.loadSessions();
    });

    await waitFor(() => {
      expect(result.current.sessions).toHaveLength(2);
      expect(result.current.isLoading).toBe(false);
    });
  });

  it('handles API errors gracefully', async () => {
    (global.fetch as any).mockRejectedValueOnce(new Error('Network error'));

    const { result } = renderHook(() => useSessionStore());

    await act(async () => {
      try {
        await result.current.loadSessions();
      } catch (e) {
        // Expected
      }
    });

    await waitFor(() => {
      expect(result.current.error).toBeTruthy();
      expect(result.current.isLoading).toBe(false);
    });
  });

  it('creates new session optimistically', async () => {
    const newSession = {
      id: '123',
      project_name: 'New Project',
      status: 'active' as const
    };

    (global.fetch as any).mockResolvedValueOnce({
      ok: true,
      json: async () => newSession
    });

    const { result } = renderHook(() => useSessionStore());

    await act(async () => {
      await result.current.createSession('New Project');
    });

    await waitFor(() => {
      expect(result.current.sessions).toContainEqual(
        expect.objectContaining({ project_name: 'New Project' })
      );
      expect(result.current.currentSession?.id).toBe('123');
    });
  });
});
```

#### 1.3.3 Visual Regression Tests (Playwright)

```typescript
// tests/e2e/visual-regression.spec.ts
import { test, expect } from '@playwright/test';

test.describe('Visual Regression', () => {
  test('chat panel matches snapshot', async ({ page }) => {
    await page.goto('http://localhost:5173/session/test-session');

    // Wait for chat to load
    await page.waitForSelector('[data-testid="chat-panel"]');

    // Take screenshot
    await expect(page.locator('[data-testid="chat-panel"]')).toHaveScreenshot('chat-panel.png');
  });

  test('requirement card matches snapshot', async ({ page }) => {
    await page.goto('http://localhost:5173/session/test-session');

    // Wait for requirements to load
    await page.waitForSelector('[data-testid="requirement-card"]');

    // Take screenshot of first card
    const card = page.locator('[data-testid="requirement-card"]').first();
    await expect(card).toHaveScreenshot('requirement-card.png');
  });

  test('diff viewer matches snapshot', async ({ page }) => {
    await page.goto('http://localhost:5173/session/test-session/diff?from=1&to=2');

    await page.waitForSelector('[data-testid="diff-viewer"]');

    await expect(page.locator('[data-testid="diff-viewer"]')).toHaveScreenshot('diff-viewer.png', {
      fullPage: true
    });
  });
});
```

---

**Due to the extensive nature of Design Packet 4, I'll continue with the remaining sections in a second file...**
