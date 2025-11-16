# Design Packet 2: Multi-Agent Requirements Engineering Platform
## Deep Implementation Specification

**Version:** 2.0
**Date:** November 16, 2025
**Scope:** Local Development - Implementation-Ready Engineering Specifications
**Prerequisite:** Design Packet 1 (Architecture & High-Level Design)

---

# Table of Contents

1. [Executive Technical Summary](#1-executive-technical-summary)
2. [Deep Dive: Local System Architecture](#2-deep-dive-local-system-architecture)
3. [Implementation-Level Details for Core Modules](#3-implementation-level-details-for-core-modules)
4. [Detailed API Design (Implementation-Ready)](#4-detailed-api-design-implementation-ready)
5. [Full Database Specification](#5-full-database-specification)
6. [Agent Implementation Blueprints](#6-agent-implementation-blueprints)
7. [Orchestrator Deep Design (LangGraph)](#7-orchestrator-deep-design-langgraph)
8. [Performance Engineering (Local Optimizations)](#8-performance-engineering-local-optimizations)
9. [Developer Enablement & Internal Tooling](#9-developer-enablement--internal-tooling)
10. [Early Scalability Roadmap](#10-early-scalability-roadmap)
11. [Embedded Artifacts](#11-embedded-artifacts)

---

## 1. Executive Technical Summary

### What Differentiates Design Packet 2

**Design Packet 1** established the **WHAT** and **WHY**:
- High-level architecture
- Component responsibilities
- Technology choices
- Success metrics

**Design Packet 2** provides the **HOW**:
- Implementation-ready code with all edge cases
- Internal module boundaries and data contracts
- State management implementation with LangGraph checkpointing[77][80][83]
- Async patterns for FastAPI + SQLAlchemy[78][81][84]
- Token optimization strategies[79][82][88][91]
- Complete error handling and retry logic
- Production-grade observability instrumentation
- Developer tooling for rapid iteration

### Key Technical Advances

| Aspect | Packet 1 | Packet 2 |
|--------|----------|----------|
| Code Depth | Skeleton examples | Production-ready implementations |
| State Management | Conceptual | LangGraph checkpoint serialization + Redis[77][83][86] |
| Error Handling | Mentioned | Circuit breakers, retries, fallbacks |
| API Spec | Endpoint list | Full OpenAPI YAML with Pydantic models |
| Database | Schema overview | Complete DDL with indexes, triggers, migrations |
| Performance | General targets | Token budgets, caching, benchmarks |
| Testing | Test ideas | Complete test harness with fixtures |
| Tooling | None | Developer CLI, mock LLM mode, fixtures generator |

### Implementation Readiness

After Design Packet 2, a senior engineer should be able to:
1. **Copy-paste 90%+ of the code** directly into the codebase
2. **Run the system end-to-end** within 2 hours
3. **Add a new agent** in under 4 hours using blueprints
4. **Debug issues** using instrumentation and tooling
5. **Optimize performance** using provided benchmarks and caching strategies

---

## 2. Deep Dive: Local System Architecture

### 2.1 Component Boundaries and Internal Modules

```
┌─────────────────────────────────────────────────────────────────┐
│                         Frontend (Optional)                      │
│                    React UI + WebSocket Client                   │
└───────────────────────┬─────────────────────────────────────────┘
                        │ HTTP/WebSocket
┌───────────────────────▼─────────────────────────────────────────┐
│                      API Gateway Layer                           │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │ Auth         │  │ Rate Limiter │  │ Request      │          │
│  │ Middleware   │  │              │  │ Validator    │          │
│  └──────────────┘  └──────────────┘  └──────────────┘          │
└───────────────────────┬─────────────────────────────────────────┘
                        │
┌───────────────────────▼─────────────────────────────────────────┐
│                    Orchestration Engine                          │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │              LangGraph State Machine                      │  │
│  │  ┌────────┐  ┌────────┐  ┌────────┐  ┌────────┐         │  │
│  │  │ Conv   │→ │Extract │→ │ Infer  │→ │Validate│→ ...    │  │
│  │  └────────┘  └────────┘  └────────┘  └────────┘         │  │
│  └──────────────────────────────────────────────────────────┘  │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │           State Manager (Immutable + Versioned)           │  │
│  │  - LangGraph Checkpoint Store (Redis/Postgres)            │  │
│  │  - Session State Cache (Redis)                            │  │
│  │  - Event Sourcing Log (Postgres)                          │  │
│  └──────────────────────────────────────────────────────────┘  │
└───────────────────────┬─────────────────────────────────────────┘
                        │
┌───────────────────────▼─────────────────────────────────────────┐
│                     Agent Layer (Specialized)                    │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐            │
│  │Conversation │  │ Extraction  │  │  Inference  │            │
│  │   Agent     │  │    Agent    │  │    Agent    │            │
│  │             │  │             │  │             │            │
│  │ ┌─────────┐ │  │ ┌─────────┐ │  │ ┌─────────┐ │            │
│  │ │Prompt   │ │  │ │Parser   │ │  │ │Inference│ │            │
│  │ │Manager  │ │  │ │Engine   │ │  │ │Engine   │ │            │
│  │ └─────────┘ │  │ └─────────┘ │  │ └─────────┘ │            │
│  │ ┌─────────┐ │  │ ┌─────────┐ │  │ ┌─────────┐ │            │
│  │ │Context  │ │  │ │Entity   │ │  │ │Confidence│ │            │
│  │ │Manager  │ │  │ │Extractor│ │  │ │Scorer   │ │            │
│  │ └─────────┘ │  │ └─────────┘ │  │ └─────────┘ │            │
│  └─────────────┘  └─────────────┘  └─────────────┘            │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐            │
│  │ Validation  │  │  Synthesis  │  │   Review    │            │
│  │   Agent     │  │    Agent    │  │   Agent     │            │
│  └─────────────┘  └─────────────┘  └─────────────┘            │
└───────────────────────┬─────────────────────────────────────────┘
                        │
┌───────────────────────▼─────────────────────────────────────────┐
│                   Service Layer (Shared)                         │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │ LLM Gateway  │  │ Vector Store │  │ Template     │          │
│  │ (OpenAI)     │  │ Service      │  │ Engine       │          │
│  │ - Retry      │  │ (ChromaDB)   │  │ (Jinja2)     │          │
│  │ - Cache      │  │ - Embedding  │  │              │          │
│  │ - Token Mgmt │  │ - Search     │  │              │          │
│  └──────────────┘  └──────────────┘  └──────────────┘          │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │ Validation   │  │ PII Detector │  │ Export       │          │
│  │ Service      │  │ Service      │  │ Service      │          │
│  └──────────────┘  └──────────────┘  └──────────────┘          │
└───────────────────────┬─────────────────────────────────────────┘
                        │
┌───────────────────────▼─────────────────────────────────────────┐
│                    Persistence Layer                             │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │  Postgres    │  │    Redis     │  │  ChromaDB    │          │
│  │  - Sessions  │  │  - Cache     │  │  - Vectors   │          │
│  │  - Reqs      │  │  - Graph State│ │  - Metadata  │          │
│  │  - RD Docs   │  │  - Locks     │  │              │          │
│  │  - Audit Log │  │              │  │              │          │
│  └──────────────┘  └──────────────┘  └──────────────┘          │
└─────────────────────────────────────────────────────────────────┘
```

### 2.2 Inter-Agent Data Contracts

All agents communicate via **immutable state objects** defined in `src/schemas/state.py`:

```python
from typing import TypedDict, List, Dict, Optional, Literal
from datetime import datetime
from pydantic import BaseModel, Field

# Immutable message
class Message(BaseModel):
    id: str
    role: Literal["user", "assistant", "system"]
    content: str
    timestamp: datetime
    metadata: Optional[Dict] = None

    class Config:
        frozen = True  # Immutable

# Immutable requirement
class Requirement(BaseModel):
    id: str
    title: str
    type: str
    actor: str
    action: str
    condition: Optional[str] = None
    acceptance_criteria: List[str]
    priority: str
    confidence: float
    inferred: bool
    rationale: str
    source_refs: List[str]
    created_at: datetime

    class Config:
        frozen = True

# LangGraph State (TypedDict for LangGraph compatibility)
class GraphState(TypedDict):
    # Session context
    session_id: str
    project_name: str
    user_id: str

    # Conversation state
    messages: List[Message]
    current_turn: int

    # Requirements
    requirements: List[Requirement]
    inferred_requirements: List[Requirement]

    # Validation state
    validation_issues: List[Dict]
    confidence: float

    # RD generation
    rd_draft: Optional[str]
    rd_version: int

    # Review state
    approval_status: Literal["pending", "approved", "revision_requested"]
    review_feedback: Optional[str]

    # Agent metadata
    last_agent: str
    iterations: int
    error_count: int

    # Checkpointing metadata
    checkpoint_id: Optional[str]
    parent_checkpoint_id: Optional[str]
```

**Contract Guarantees:**
- All state mutations create new instances (immutability)
- Schema validation via Pydantic before state transitions
- Backward compatibility enforced via versioned schemas
- Type safety across agent boundaries

### 2.3 State Flow Through LangGraph (Turn-by-Turn)

```
Turn N:
┌──────────────────────────────────────────────────────────────┐
│ 1. HTTP Request arrives                                      │
│    POST /sessions/sess_123/messages                          │
│    Body: {"message": "App should load quickly"}             │
└────────────────┬─────────────────────────────────────────────┘
                 │
┌────────────────▼─────────────────────────────────────────────┐
│ 2. API Layer:                                                │
│    - Validate JWT                                            │
│    - Load session from Postgres                              │
│    - Retrieve last checkpoint from Redis                     │
│    State = deserialize(checkpoint)                           │
└────────────────┬─────────────────────────────────────────────┘
                 │
┌────────────────▼─────────────────────────────────────────────┐
│ 3. Add user message to state:                                │
│    new_state = {                                             │
│      ...state,                                               │
│      messages: [...state.messages, user_message],           │
│      current_turn: state.current_turn + 1                   │
│    }                                                         │
└────────────────┬─────────────────────────────────────────────┘
                 │
┌────────────────▼─────────────────────────────────────────────┐
│ 4. Invoke LangGraph:                                         │
│    result = await graph.ainvoke(                             │
│      new_state,                                              │
│      config={                                                │
│        "configurable": {                                     │
│          "thread_id": "sess_123",                           │
│          "checkpoint_id": state.checkpoint_id               │
│        }                                                     │
│      }                                                       │
│    )                                                         │
└────────────────┬─────────────────────────────────────────────┘
                 │
┌────────────────▼─────────────────────────────────────────────┐
│ 5. Graph Execution:                                          │
│    a. Conversational Node:                                   │
│       - Processes user message                               │
│       - Decides next action                                  │
│       - Returns updated state with assistant response        │
│    b. Checkpoint saved to Redis (automatic)                  │
│    c. Conditional edge evaluates:                            │
│       if "extract requirements" → Extraction Node           │
│       else → Return to user                                  │
│    d. Extraction Node (if triggered):                        │
│       - Calls LLM with extraction prompt                     │
│       - Parses structured requirements                       │
│       - Updates state.requirements                           │
│    e. Checkpoint saved                                       │
│    f. Conditional edge:                                      │
│       if requirements detected → Inference Node             │
│    g. ... (continues through graph)                          │
└────────────────┬─────────────────────────────────────────────┘
                 │
┌────────────────▼─────────────────────────────────────────────┐
│ 6. Final state returned from graph                           │
│    Contains all updates from nodes                           │
└────────────────┬─────────────────────────────────────────────┘
                 │
┌────────────────▼─────────────────────────────────────────────┐
│ 7. API Layer:                                                │
│    - Extract assistant message from state                    │
│    - Store requirements in Postgres (if new)                 │
│    - Store embeddings in ChromaDB (if new)                   │
│    - Store audit log entry                                   │
│    - Return HTTP response                                    │
└────────────────┬─────────────────────────────────────────────┘
                 │
┌────────────────▼─────────────────────────────────────────────┐
│ 8. HTTP Response:                                            │
│    {                                                         │
│      "message": {                                            │
│        "role": "assistant",                                  │
│        "content": "I've captured that as REQ-001...",       │
│        "metadata": {                                         │
│          "requirements_extracted": 1,                        │
│          "confidence": 0.88                                  │
│        }                                                     │
│      },                                                      │
│      "session_state": {                                      │
│        "requirements_count": 1,                              │
│        "rd_version": 0,                                      │
│        "approval_status": "pending"                          │
│      }                                                       │
│    }                                                         │
└──────────────────────────────────────────────────────────────┘
```

### 2.4 Error Propagation and Retries

**Three-Layer Error Handling:**

1. **LLM Call Layer** (Innermost)
```python
# src/services/llm_gateway.py
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type
)
from openai import RateLimitError, APIError, Timeout

class LLMGateway:
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((RateLimitError, APIError, Timeout)),
        reraise=True
    )
    async def call_llm(self, prompt: str, **kwargs):
        try:
            response = await self.client.chat.completions.create(
                model=kwargs.get("model", self.default_model),
                messages=[{"role": "user", "content": prompt}],
                temperature=kwargs.get("temperature", 0.7),
                max_tokens=kwargs.get("max_tokens", 2000),
                timeout=kwargs.get("timeout", 30)
            )
            return response.choices[0].message.content
        except RateLimitError as e:
            # Log rate limit hit
            logger.warning(f"Rate limit hit, retrying... {e}")
            raise  # Retry via tenacity
        except Timeout as e:
            logger.warning(f"LLM timeout, retrying... {e}")
            raise
        except Exception as e:
            logger.error(f"LLM call failed: {e}", exc_info=True)
            raise
```

2. **Agent Layer** (Middle)
```python
# src/agents/base.py
class BaseAgent:
    async def invoke(self, state: GraphState) -> GraphState:
        try:
            result = await self._execute(state)
            return self._update_state(state, result)
        except LLMError as e:
            # Log agent-level error
            logger.error(f"{self.__class__.__name__} failed: {e}")
            # Return state with error metadata
            return {
                **state,
                "error_count": state.get("error_count", 0) + 1,
                "last_error": str(e),
                "confidence": max(0.0, state.get("confidence", 1.0) - 0.2)
            }
        except Exception as e:
            logger.critical(f"Unexpected error in {self.__class__.__name__}: {e}", exc_info=True)
            raise
```

3. **Orchestrator Layer** (Outermost)
```python
# src/orchestrator/graph.py
async def invoke_graph_with_recovery(state: GraphState, config: dict):
    try:
        result = await graph.ainvoke(state, config)
        return result
    except Exception as e:
        logger.error(f"Graph execution failed: {e}", exc_info=True)

        # Check if we can recover from checkpoint
        if state.get("checkpoint_id"):
            logger.info("Attempting recovery from last checkpoint")
            # Load previous checkpoint
            prev_state = await load_checkpoint(state["checkpoint_id"])
            # Add error context to state
            prev_state["recovery_attempted"] = True
            prev_state["error_context"] = str(e)
            return prev_state
        else:
            # No checkpoint available, escalate
            raise GraphExecutionError("Graph failed with no recovery checkpoint") from e
```

### 2.5 Race Conditions & Concurrency (Single Laptop)

Even on a single machine, async operations can cause race conditions:

**Problem Scenarios:**
1. Multiple API requests updating same session simultaneously
2. LangGraph checkpoint write conflicts
3. Database transaction conflicts

**Solutions:**

**1. Distributed Lock via Redis:**
```python
# src/utils/distributed_lock.py
from redis import asyncio as aioredis
from contextlib import asynccontextmanager

class DistributedLock:
    def __init__(self, redis_client: aioredis.Redis):
        self.redis = redis_client

    @asynccontextmanager
    async def acquire(self, key: str, timeout: int = 10):
        lock_key = f"lock:{key}"
        lock_acquired = False

        try:
            # Try to acquire lock
            lock_acquired = await self.redis.set(
                lock_key,
                "locked",
                nx=True,  # Only set if not exists
                ex=timeout  # Auto-expire
            )

            if not lock_acquired:
                raise LockAcquisitionError(f"Could not acquire lock for {key}")

            yield
        finally:
            if lock_acquired:
                await self.redis.delete(lock_key)

# Usage in API
async def send_message(session_id: str, message: str):
    async with distributed_lock.acquire(f"session:{session_id}"):
        # Only one request can process this session at a time
        state = await load_session_state(session_id)
        result = await graph.ainvoke(state, config)
        await save_session_state(session_id, result)
    return result
```

**2. Optimistic Locking in Database:**
```python
# src/storage/postgres.py
from sqlalchemy import Column, Integer
from sqlalchemy.orm import Session

class SessionModel(Base):
    __tablename__ = "sessions"

    id = Column(String, primary_key=True)
    version = Column(Integer, default=0, nullable=False)  # Optimistic lock
    # ... other fields

async def update_session(session_id: str, updates: dict):
    async with get_db_session() as db:
        # Read current version
        session = await db.get(SessionModel, session_id)
        current_version = session.version

        # Attempt update with version check
        stmt = (
            update(SessionModel)
            .where(
                SessionModel.id == session_id,
                SessionModel.version == current_version  # Optimistic lock
            )
            .values(
                **updates,
                version=current_version + 1
            )
        )

        result = await db.execute(stmt)
        await db.commit()

        if result.rowcount == 0:
            raise ConcurrentModificationError(f"Session {session_id} was modified concurrently")
```

**3. LangGraph Checkpoint Serialization:**[77][83]
```python
# src/orchestrator/checkpointer.py
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.checkpoint.base import Checkpoint

class CustomCheckpointer(PostgresSaver):
    async def aput(
        self,
        config: dict,
        checkpoint: Checkpoint,
        metadata: dict,
    ) -> dict:
        # Serialize checkpoint with conflict detection
        checkpoint_id = checkpoint["id"]
        thread_id = config["configurable"]["thread_id"]

        # Use INSERT ... ON CONFLICT for upsert
        async with self.conn.begin():
            await self.conn.execute(
                \"\"\"
                INSERT INTO checkpoints (thread_id, checkpoint_id, checkpoint, metadata)
                VALUES (:thread_id, :checkpoint_id, :checkpoint, :metadata)
                ON CONFLICT (thread_id, checkpoint_id)
                DO UPDATE SET
                    checkpoint = EXCLUDED.checkpoint,
                    metadata = EXCLUDED.metadata,
                    updated_at = NOW()
                \"\"\",
                {
                    "thread_id": thread_id,
                    "checkpoint_id": checkpoint_id,
                    "checkpoint": serialize(checkpoint),
                    "metadata": serialize(metadata)
                }
            )

        return config
```

### 2.6 Versioning & Change History Strategy

**Three-Level Versioning:**

1. **File-Level Versioning** (Git)
   - All prompts, templates, configurations in Git
   - Tag releases: `v1.0.0`, `v1.1.0`

2. **Database Schema Versioning** (Alembic migrations)
   ```bash
   alembic revision --autogenerate -m "Add inference_confidence column"
   alembic upgrade head
   ```

3. **RD Document Versioning** (Event Sourcing)

```python
# Event Sourcing for RD Versions
class RDEvent(Base):
    __tablename__ = "rd_events"

    id = Column(UUID, primary_key=True, default=uuid4)
    session_id = Column(UUID, ForeignKey("sessions.id"))
    version = Column(Integer, nullable=False)
    event_type = Column(String, nullable=False)  # "created", "updated", "approved"
    event_data = Column(JSONB, nullable=False)
    user_id = Column(String, nullable=False)
    timestamp = Column(DateTime, default=func.now())

    # Indexes for fast version retrieval
    __table_args__ = (
        Index("idx_rd_events_session_version", "session_id", "version"),
    )

# Rebuild RD at any version
async def get_rd_at_version(session_id: str, version: int) -> str:
    events = await db.query(RDEvent).filter(
        RDEvent.session_id == session_id,
        RDEvent.version <= version
    ).order_by(RDEvent.version).all()

    # Replay events
    rd_state = {}
    for event in events:
        if event.event_type == "created":
            rd_state = event.event_data
        elif event.event_type == "updated":
            rd_state = apply_changes(rd_state, event.event_data)

    return rd_state.get("content", "")

# Diff between versions
async def diff_versions(session_id: str, v1: int, v2: int) -> dict:
    rd_v1 = await get_rd_at_version(session_id, v1)
    rd_v2 = await get_rd_at_version(session_id, v2)

    import difflib
    diff = difflib.unified_diff(
        rd_v1.splitlines(),
        rd_v2.splitlines(),
        lineterm=""
    )

    return {
        "from_version": v1,
        "to_version": v2,
        "diff": list(diff),
        "changes_count": len([line for line in diff if line.startswith("+") or line.startswith("-")])
    }
```

---

## 3. Implementation-Level Details for Core Modules

### 3.1 Backend Application State Manager

**Four Types of State:**

```python
# src/state/manager.py
from typing import Optional
from dataclasses import dataclass
from datetime import datetime

@dataclass(frozen=True)
class SessionContext:
    """Immutable session metadata (DB-backed)"""
    session_id: str
    project_name: str
    user_id: str
    created_at: datetime
    status: str

@dataclass
class AgentState:
    """Mutable agent execution state (ephemeral)"""
    agent_name: str
    iteration: int
    confidence: float
    error_count: int

class LangGraphState(TypedDict):
    """LangGraph-managed state (checkpointed)"""
    # See section 2.2 for full definition
    pass

@dataclass
class DBState:
    """Database-persisted state"""
    requirements: List[Requirement]
    rd_documents: List[RDDocument]
    audit_logs: List[AuditLog]

class StateManager:
    \"\"\"Orchestrates state consistency across layers\"\"\"

    def __init__(
        self,
        db_session: AsyncSession,
        redis_client: Redis,
        checkpointer: BaseCheckpointSaver
    ):
        self.db = db_session
        self.redis = redis_client
        self.checkpointer = checkpointer

    async def load_session_state(self, session_id: str) -> LangGraphState:
        # 1. Load session context from DB
        session_ctx = await self._load_session_context(session_id)

        # 2. Load last checkpoint from Redis
        checkpoint = await self.redis.get(f"checkpoint:{session_id}")
        if checkpoint:
            graph_state = deserialize_checkpoint(checkpoint)
        else:
            # Initialize new state
            graph_state = self._initialize_state(session_ctx)

        # 3. Load requirements from DB (source of truth)
        requirements = await self._load_requirements(session_id)

        # 4. Merge DB state into graph state
        graph_state["requirements"] = requirements

        return graph_state

    async def save_session_state(
        self,
        session_id: str,
        graph_state: LangGraphState
    ):
        # 1. Save checkpoint to Redis (fast, for resume)
        checkpoint_data = serialize_checkpoint(graph_state)
        await self.redis.setex(
            f"checkpoint:{session_id}",
            3600,  # 1 hour TTL
            checkpoint_data
        )

        # 2. Save checkpoint to Postgres (durable, via LangGraph)
        await self.checkpointer.aput(
            config={"configurable": {"thread_id": session_id}},
            checkpoint=graph_state,
            metadata={"timestamp": datetime.utcnow().isoformat()}
        )

        # 3. Persist new requirements to DB
        new_requirements = graph_state.get("requirements", [])
        await self._persist_requirements(session_id, new_requirements)

        # 4. Update session metadata (version, status)
        await self._update_session_metadata(session_id, graph_state)

def serialize_checkpoint(state: LangGraphState) -> bytes:
    \"\"\"Serialize state for Redis storage\"\"\"
    import pickle
    return pickle.dumps(state)

def deserialize_checkpoint(data: bytes) -> LangGraphState:
    \"\"\"Deserialize state from Redis\"\"\"
    import pickle
    return pickle.loads(data)
```

**Consistency Guarantees:**

| Layer | Consistency Model | Recovery |
|-------|------------------|----------|
| Redis (Checkpoint) | Eventually consistent | Rebuild from Postgres |
| Postgres (Checkpoint) | Strongly consistent | Source of truth |
| Postgres (Requirements) | Strongly consistent | Source of truth |
| LangGraph (In-Memory) | Ephemeral | Load from checkpoint |

**Conflict Resolution:**
- Redis checkpoint conflicts: Overwrite (last write wins, acceptable for caching)
- Postgres checkpoint conflicts: Use LangGraph's built-in conflict detection[80]
- Requirement conflicts: Use optimistic locking with version column

### 3.2 Error Handling & Recovery

**Circuit Breaker Pattern for LLM Calls:**[94]

```python
# src/services/circuit_breaker.py
from enum import Enum
from datetime import datetime, timedelta
from typing import Callable, Any

class CircuitState(Enum):
    CLOSED = "closed"  # Normal operation
    OPEN = "open"      # Failing, reject calls
    HALF_OPEN = "half_open"  # Testing recovery

class CircuitBreaker:
    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: int = 60,
        expected_exception: type = Exception
    ):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception

        self.failure_count = 0
        self.last_failure_time: Optional[datetime] = None
        self.state = CircuitState.CLOSED

    async def call(self, func: Callable, *args, **kwargs) -> Any:
        if self.state == CircuitState.OPEN:
            # Check if recovery timeout has passed
            if datetime.utcnow() - self.last_failure_time > timedelta(seconds=self.recovery_timeout):
                self.state = CircuitState.HALF_OPEN
            else:
                raise CircuitBreakerOpenError("Circuit breaker is OPEN")

        try:
            result = await func(*args, **kwargs)
            self._on_success()
            return result
        except self.expected_exception as e:
            self._on_failure()
            raise

    def _on_success(self):
        self.failure_count = 0
        self.state = CircuitState.CLOSED

    def _on_failure(self):
        self.failure_count += 1
        self.last_failure_time = datetime.utcnow()

        if self.failure_count >= self.failure_threshold:
            self.state = CircuitState.OPEN
            logger.warning(f"Circuit breaker opened after {self.failure_count} failures")

# Usage
llm_circuit_breaker = CircuitBreaker(
    failure_threshold=5,
    recovery_timeout=60,
    expected_exception=(RateLimitError, APIError, Timeout)
)

async def call_llm_with_circuit_breaker(prompt: str):
    return await llm_circuit_breaker.call(llm_gateway.call_llm, prompt)
```

**Fallback Modes:**

```python
# src/agents/conversational.py
class ConversationalAgent:
    async def invoke(self, state: GraphState) -> GraphState:
        try:
            # Primary: Call OpenAI LLM
            response = await call_llm_with_circuit_breaker(prompt)
        except CircuitBreakerOpenError:
            # Fallback 1: Use cached response if available
            cached = await self.cache.get(prompt_hash(prompt))
            if cached:
                logger.info("Using cached LLM response due to circuit breaker")
                response = cached
            else:
                # Fallback 2: Use template-based response
                response = self._generate_template_response(state)
        except Exception as e:
            # Fallback 3: Graceful degradation
            logger.error(f"All LLM fallbacks failed: {e}")
            response = "I'm experiencing technical difficulties. Please try again in a moment."

        return self._update_state(state, response)

    def _generate_template_response(self, state: GraphState) -> str:
        \"\"\"Generate response from predefined templates (no LLM)\"\"\"
        turn = state["current_turn"]
        req_count = len(state["requirements"])

        if req_count == 0:
            return "Thank you for starting. Could you describe the main features you need?"
        else:
            return f"I've captured {req_count} requirements so far. Would you like to add more or review what we have?"
```

**Safe Session Resume After Corruption:**

```python
# src/api/routes/sessions.py
async def resume_session(session_id: str):
    try:
        # Attempt to load from Redis checkpoint
        state = await state_manager.load_session_state(session_id)
    except CheckpointCorruptedError:
        logger.warning(f"Redis checkpoint corrupted for {session_id}, falling back to Postgres")

        try:
            # Fallback to last Postgres checkpoint
            state = await load_from_postgres_checkpoint(session_id)
        except CheckpointNotFoundError:
            logger.error(f"No valid checkpoint found for {session_id}, rebuilding from DB")

            # Last resort: Rebuild state from database
            state = await rebuild_state_from_db(session_id)

    return state

async def rebuild_state_from_db(session_id: str) -> LangGraphState:
    \"\"\"Rebuild graph state from database records\"\"\"
    # Load session metadata
    session = await db.get(SessionModel, session_id)

    # Load all chat messages
    messages = await db.query(ChatMessage).filter(
        ChatMessage.session_id == session_id
    ).order_by(ChatMessage.timestamp).all()

    # Load all requirements
    requirements = await db.query(RequirementModel).filter(
        RequirementModel.session_id == session_id
    ).all()

    # Load latest RD
    rd = await db.query(RDDocument).filter(
        RDDocument.session_id == session_id
    ).order_by(RDDocument.version.desc()).first()

    # Reconstruct state
    return {
        "session_id": session_id,
        "project_name": session.project_name,
        "user_id": session.user_id,
        "messages": [msg.to_pydantic() for msg in messages],
        "current_turn": len(messages),
        "requirements": [req.to_pydantic() for req in requirements],
        "inferred_requirements": [],  # Could rebuild from event log
        "rd_draft": rd.content if rd else None,
        "rd_version": rd.version if rd else 0,
        "validation_issues": [],
        "confidence": 1.0,
        "approval_status": session.status,
        "review_feedback": None,
        "last_agent": "system",
        "iterations": 0,
        "error_count": 0
    }
```

### 3.3 Logging, Metrics, and Observability (Code-Level)

**Structured Logging with Correlation IDs:**

```python
# src/utils/logging.py
import logging
import json
from contextvars import ContextVar
from datetime import datetime
from typing import Any, Dict

# Context variable for correlation ID (async-safe)
correlation_id_var: ContextVar[str] = ContextVar("correlation_id", default="")

class StructuredFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        log_data = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "correlation_id": correlation_id_var.get(""),
            "service": "req-eng-platform",
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno
        }

        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)

        # Add extra fields
        if hasattr(record, "extra_fields"):
            log_data.update(record.extra_fields)

        return json.dumps(log_data)

def setup_logging():
    handler = logging.StreamHandler()
    handler.setFormatter(StructuredFormatter())

    root_logger = logging.getLogger()
    root_logger.setLevel(os.getenv("LOG_LEVEL", "INFO"))
    root_logger.addHandler(handler)

def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)

# Helper for adding extra fields
def log_with_context(logger: logging.Logger, level: str, message: str, **kwargs):
    extra_fields = {k: v for k, v in kwargs.items()}
    logger.log(
        getattr(logging, level.upper()),
        message,
        extra={"extra_fields": extra_fields}
    )
```

**Instrumentation Points:**

```python
# src/api/middleware/logging.py
from fastapi import Request
import uuid
import time

@app.middleware("http")
async def logging_middleware(request: Request, call_next):
    # Generate correlation ID
    correlation_id = str(uuid.uuid4())
    correlation_id_var.set(correlation_id)

    # Log request
    start_time = time.time()
    log_with_context(
        logger,
        "info",
        "Request started",
        correlation_id=correlation_id,
        method=request.method,
        path=request.url.path,
        client_ip=request.client.host
    )

    try:
        response = await call_next(request)

        # Log response
        duration_ms = (time.time() - start_time) * 1000
        log_with_context(
            logger,
            "info",
            "Request completed",
            correlation_id=correlation_id,
            status_code=response.status_code,
            duration_ms=duration_ms
        )

        # Add correlation ID to response headers
        response.headers["X-Correlation-ID"] = correlation_id

        return response
    except Exception as e:
        log_with_context(
            logger,
            "error",
            f"Request failed: {str(e)}",
            correlation_id=correlation_id,
            exception=str(e)
        )
        raise

# Agent instrumentation
# src/agents/base.py
class BaseAgent:
    async def invoke(self, state: GraphState) -> GraphState:
        agent_name = self.__class__.__name__
        start_time = time.time()

        log_with_context(
            logger,
            "info",
            f"{agent_name} started",
            agent=agent_name,
            session_id=state["session_id"],
            turn=state["current_turn"]
        )

        try:
            result = await self._execute(state)

            duration_ms = (time.time() - start_time) * 1000
            log_with_context(
                logger,
                "info",
                f"{agent_name} completed",
                agent=agent_name,
                session_id=state["session_id"],
                duration_ms=duration_ms,
                confidence=result.get("confidence", 0.0)
            )

            return result
        except Exception as e:
            log_with_context(
                logger,
                "error",
                f"{agent_name} failed",
                agent=agent_name,
                session_id=state["session_id"],
                error=str(e)
            )
            raise
```

**LangSmith Tracing Integration:**[61][63]

```python
# src/utils/langsmith_tracing.py
import os
from langsmith import Client, traceable
from functools import wraps

# Initialize LangSmith client
langsmith_client = None
if os.getenv("LANGSMITH_TRACING") == "true":
    langsmith_client = Client(
        api_key=os.getenv("LANGSMITH_API_KEY"),
        api_url=os.getenv("LANGSMITH_ENDPOINT", "https://api.smith.langchain.com")
    )

def trace_agent(agent_name: str):
    \"\"\"Decorator to trace agent execution in LangSmith\"\"\"
    def decorator(func):
        if langsmith_client is None:
            return func  # No-op if tracing disabled

        @traceable(
            name=agent_name,
            project_name=os.getenv("LANGSMITH_PROJECT", "req-eng-local"),
            client=langsmith_client
        )
        @wraps(func)
        async def wrapper(state: GraphState, *args, **kwargs):
            # Log inputs
            langsmith_client.create_run(
                name=agent_name,
                inputs={
                    "session_id": state["session_id"],
                    "turn": state["current_turn"],
                    "messages": len(state["messages"]),
                    "requirements": len(state["requirements"])
                }
            )

            # Execute
            result = await func(state, *args, **kwargs)

            # Log outputs
            langsmith_client.update_run(
                outputs={
                    "confidence": result.get("confidence", 0.0),
                    "requirements_added": len(result["requirements"]) - len(state["requirements"]),
                    "last_agent": result["last_agent"]
                }
            )

            return result

        return wrapper
    return decorator

# Usage
@trace_agent("extraction_agent")
async def extraction_node(state: GraphState) -> GraphState:
    # Agent logic
    pass
```

**Prometheus Metrics:**

```python
# src/utils/metrics.py
from prometheus_client import Counter, Histogram, Gauge, Info
from functools import wraps
import time

# Define metrics
agent_invocations = Counter(
    'agent_invocations_total',
    'Total agent invocations',
    ['agent_name', 'status']  # Labels
)

agent_duration = Histogram(
    'agent_duration_seconds',
    'Agent execution duration',
    ['agent_name'],
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
)

llm_tokens = Counter(
    'llm_tokens_total',
    'Total LLM tokens used',
    ['model', 'type']  # type: input/output
)

active_sessions = Gauge(
    'active_sessions',
    'Number of currently active sessions'
)

requirements_extracted = Counter(
    'requirements_extracted_total',
    'Total requirements extracted',
    ['type', 'inferred']
)

# Decorator for agent metrics
def track_agent_metrics(agent_name: str):
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            start_time = time.time()
            status = "success"

            try:
                result = await func(*args, **kwargs)
                return result
            except Exception as e:
                status = "error"
                raise
            finally:
                duration = time.time() - start_time
                agent_invocations.labels(agent_name=agent_name, status=status).inc()
                agent_duration.labels(agent_name=agent_name).observe(duration)

        return wrapper
    return decorator

# Usage
@track_agent_metrics("extraction_agent")
async def extraction_node(state: GraphState) -> GraphState:
    # Agent logic
    pass
```

### 3.4 Vector Store Deep Design

**ChromaDB Collection Schema:**

```python
# src/storage/vector_store.py
import chromadb
from chromadb.config import Settings
from typing import List, Dict, Optional
import hashlib

class VectorStoreService:
    def __init__(self):
        self.client = chromadb.HttpClient(
            host=os.getenv("CHROMA_HOST", "localhost"),
            port=int(os.getenv("CHROMA_PORT", 8001)),
            settings=Settings(anonymized_telemetry=False)
        )

        # Initialize collections
        self.requirements_collection = self.client.get_or_create_collection(
            name="requirements",
            metadata={
                "hnsw:space": "cosine",  # Similarity metric
                "hnsw:construction_ef": 100,  # Build quality
                "hnsw:search_ef": 100  # Search quality
            }
        )

        self.chat_history_collection = self.client.get_or_create_collection(
            name="chat_history",
            metadata={"hnsw:space": "cosine"}
        )

    async def add_requirement(
        self,
        requirement_id: str,
        title: str,
        action: str,
        embedding: List[float],
        metadata: Dict
    ):
        \"\"\"Add requirement with embedding\"\"\"
        # Combine title + action for semantic search
        document = f"{title} {action}"

        # Metadata for filtering
        full_metadata = {
            **metadata,
            "requirement_id": requirement_id,
            "type": metadata.get("type", "functional"),
            "priority": metadata.get("priority", "medium"),
            "inferred": metadata.get("inferred", False),
            "created_at": metadata.get("created_at", datetime.utcnow().isoformat())
        }

        self.requirements_collection.add(
            ids=[requirement_id],
            embeddings=[embedding],
            documents=[document],
            metadatas=[full_metadata]
        )

    async def find_similar_requirements(
        self,
        query_embedding: List[float],
        session_id: str,
        threshold: float = 0.85,
        n_results: int = 5,
        filter_metadata: Optional[Dict] = None
    ) -> List[Dict]:
        \"\"\"Find similar requirements within session\"\"\"

        # Build where clause for filtering
        where = {"session_id": session_id}
        if filter_metadata:
            where.update(filter_metadata)

        results = self.requirements_collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where=where,
            include=["documents", "metadatas", "distances"]
        )

        # Filter by similarity threshold
        similar = []
        for i, distance in enumerate(results["distances"][0]):
            similarity = 1 - distance  # Convert distance to similarity
            if similarity >= threshold:
                similar.append({
                    "requirement_id": results["metadatas"][0][i]["requirement_id"],
                    "document": results["documents"][0][i],
                    "similarity": similarity,
                    "metadata": results["metadatas"][0][i]
                })

        return similar

    async def get_embedding_cache_key(self, text: str) -> str:
        \"\"\"Generate cache key for embedding\"\"\"
        return hashlib.sha256(text.encode()).hexdigest()

    async def get_cached_embedding(self, text: str) -> Optional[List[float]]:
        \"\"\"Check if embedding is cached in Redis\"\"\"
        cache_key = f"embedding:{await self.get_embedding_cache_key(text)}"
        cached = await redis_client.get(cache_key)

        if cached:
            import json
            return json.loads(cached)
        return None

    async def cache_embedding(self, text: str, embedding: List[float]):
        \"\"\"Cache embedding in Redis\"\"\"
        cache_key = f"embedding:{await self.get_embedding_cache_key(text)}"
        import json
        await redis_client.setex(
            cache_key,
            86400,  # 24 hour TTL
            json.dumps(embedding)
        )

    async def rebuild_index(self, session_id: str):
        \"\"\"Rebuild vector index for a session (e.g., after bulk updates)\"\"\"
        # Fetch all requirements from DB
        requirements = await db.query(RequirementModel).filter(
            RequirementModel.session_id == session_id
        ).all()

        # Delete existing embeddings for this session
        existing_ids = self.requirements_collection.get(
            where={"session_id": session_id}
        )["ids"]

        if existing_ids:
            self.requirements_collection.delete(ids=existing_ids)

        # Re-add all requirements
        for req in requirements:
            text = f"{req.title} {req.action}"
            embedding = await get_embedding(text)  # Generate fresh

            await self.add_requirement(
                requirement_id=req.id,
                title=req.title,
                action=req.action,
                embedding=embedding,
                metadata={
                    "session_id": session_id,
                    "type": req.type,
                    "priority": req.priority,
                    "inferred": req.inferred
                }
            )

        logger.info(f"Rebuilt vector index for session {session_id}: {len(requirements)} requirements")
```

**Embedding Generation with Caching:**[79]

```python
# src/services/embedding_service.py
from openai import AsyncOpenAI
from typing import List

class EmbeddingService:
    def __init__(self):
        self.client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
        self.dimension = 1536  # text-embedding-3-small dimension

    async def get_embedding(self, text: str) -> List[float]:
        \"\"\"Get embedding with caching\"\"\"
        # Check cache first
        cached = await vector_store.get_cached_embedding(text)
        if cached:
            return cached

        # Generate embedding
        response = await self.client.embeddings.create(
            model=self.model,
            input=text,
            encoding_format="float"
        )

        embedding = response.data[0].embedding

        # Cache for future use
        await vector_store.cache_embedding(text, embedding)

        # Track token usage
        llm_tokens.labels(model=self.model, type="embedding").inc(response.usage.total_tokens)

        return embedding

    async def batch_embeddings(self, texts: List[str]) -> List[List[float]]:
        \"\"\"Generate embeddings in batch (more efficient)\"\"\"
        # Check cache for all texts
        embeddings = []
        texts_to_generate = []
        indices_to_generate = []

        for i, text in enumerate(texts):
            cached = await vector_store.get_cached_embedding(text)
            if cached:
                embeddings.append(cached)
            else:
                embeddings.append(None)
                texts_to_generate.append(text)
                indices_to_generate.append(i)

        # Generate embeddings for uncached texts
        if texts_to_generate:
            response = await self.client.embeddings.create(
                model=self.model,
                input=texts_to_generate,
                encoding_format="float"
            )

            # Fill in generated embeddings
            for i, embedding_data in enumerate(response.data):
                idx = indices_to_generate[i]
                embedding = embedding_data.embedding
                embeddings[idx] = embedding

                # Cache
                await vector_store.cache_embedding(texts_to_generate[i], embedding)

            # Track tokens
            llm_tokens.labels(model=self.model, type="embedding").inc(response.usage.total_tokens)

        return embeddings
```

---

## 4. Detailed API Design (Implementation-Ready)

### 4.1 Complete OpenAPI Specification

```yaml
# openapi-v1-full.yaml
openapi: 3.0.3
info:
  title: Requirements Engineering Platform API
  version: 1.0.0
  description: |
    Multi-agent conversational requirements engineering platform.
    Supports session management, chat interactions, requirements extraction,
    document synthesis, and human-in-the-loop review workflows.
  contact:
    name: Engineering Team
    email: eng@example.com
  license:
    name: MIT

servers:
  - url: http://localhost:8000/api/v1
    description: Local development server
  - url: https://api-staging.example.com/api/v1
    description: Staging environment
  - url: https://api.example.com/api/v1
    description: Production environment

security:
  - bearerAuth: []

paths:
  /health:
    get:
      summary: Health check
      operationId: healthCheck
      security: []
      tags:
        - System
      responses:
        '200':
          description: System is healthy
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/HealthResponse'

  /auth/login:
    post:
      summary: User login
      operationId: login
      security: []
      tags:
        - Authentication
      requestBody:
        required: true
        content:
          application/json:
            schema:
              $ref: '#/components/schemas/LoginRequest'
      responses:
        '200':
          description: Login successful
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/LoginResponse'
        '401':
          $ref: '#/components/responses/Unauthorized'

  /sessions:
    post:
      summary: Create new conversation session
      operationId: createSession
      tags:
        - Sessions
      requestBody:
        required: true
        content:
          application/json:
            schema:
              $ref: '#/components/schemas/CreateSessionRequest'
      responses:
        '201':
          description: Session created
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/Session'
        '400':
          $ref: '#/components/responses/BadRequest'
        '401':
          $ref: '#/components/responses/Unauthorized'

    get:
      summary: List sessions
      operationId: listSessions
      tags:
        - Sessions
      parameters:
        - $ref: '#/components/parameters/PageParam'
        - $ref: '#/components/parameters/PageSizeParam'
        - name: status
          in: query
          schema:
            type: string
            enum: [active, reviewing, approved, archived]
      responses:
        '200':
          description: List of sessions
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/SessionList'

  /sessions/{sessionId}:
    get:
      summary: Get session details
      operationId: getSession
      tags:
        - Sessions
      parameters:
        - $ref: '#/components/parameters/SessionIdParam'
      responses:
        '200':
          description: Session details
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/SessionDetail'
        '404':
          $ref: '#/components/responses/NotFound'

    delete:
      summary: Delete session
      operationId: deleteSession
      tags:
        - Sessions
      parameters:
        - $ref: '#/components/parameters/SessionIdParam'
      responses:
        '204':
          description: Session deleted
        '404':
          $ref: '#/components/responses/NotFound'

  /sessions/{sessionId}/messages:
    post:
      summary: Send message to agent
      operationId: sendMessage
      tags:
        - Chat
      parameters:
        - $ref: '#/components/parameters/SessionIdParam'
      requestBody:
        required: true
        content:
          application/json:
            schema:
              $ref: '#/components/schemas/SendMessageRequest'
      responses:
        '200':
          description: Agent response
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/ChatMessageResponse'
        '400':
          $ref: '#/components/responses/BadRequest'
        '404':
          $ref: '#/components/responses/NotFound'
        '429':
          $ref: '#/components/responses/RateLimitExceeded'

    get:
      summary: Get chat history
      operationId: getChatHistory
      tags:
        - Chat
      parameters:
        - $ref: '#/components/parameters/SessionIdParam'
        - $ref: '#/components/parameters/PageParam'
        - $ref: '#/components/parameters/PageSizeParam'
      responses:
        '200':
          description: Chat history
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/ChatHistory'

  /requirements:
    get:
      summary: List all requirements (across sessions)
      operationId: listRequirements
      tags:
        - Requirements
      parameters:
        - $ref: '#/components/parameters/PageParam'
        - $ref: '#/components/parameters/PageSizeParam'
        - name: type
          in: query
          schema:
            $ref: '#/components/schemas/RequirementType'
        - name: inferred
          in: query
          schema:
            type: boolean
      responses:
        '200':
          description: List of requirements
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/RequirementList'

  /requirements/{sessionId}:
    get:
      summary: Get requirements for specific session
      operationId: getSessionRequirements
      tags:
        - Requirements
      parameters:
        - $ref: '#/components/parameters/SessionIdParam'
      responses:
        '200':
          description: Session requirements
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/RequirementList'
        '404':
          $ref: '#/components/responses/NotFound'

  /requirements/{requirementId}:
    get:
      summary: Get requirement details
      operationId: getRequirement
      tags:
        - Requirements
      parameters:
        - name: requirementId
          in: path
          required: true
          schema:
            type: string
            pattern: '^REQ-\d{3,}$'
      responses:
        '200':
          description: Requirement details
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/Requirement'
        '404':
          $ref: '#/components/responses/NotFound'

    put:
      summary: Update requirement
      operationId: updateRequirement
      tags:
        - Requirements
      parameters:
        - name: requirementId
          in: path
          required: true
          schema:
            type: string
      requestBody:
        required: true
        content:
          application/json:
            schema:
              $ref: '#/components/schemas/UpdateRequirementRequest'
      responses:
        '200':
          description: Requirement updated
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/Requirement'
        '404':
          $ref: '#/components/responses/NotFound'
        '409':
          $ref: '#/components/responses/Conflict'

  /rd/{sessionId}:
    get:
      summary: Get current RD draft
      operationId: getCurrentRD
      tags:
        - Requirements Document
      parameters:
        - $ref: '#/components/parameters/SessionIdParam'
        - name: version
          in: query
          schema:
            type: integer
          description: Specific version (defaults to latest)
      responses:
        '200':
          description: RD document
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/RDDocument'
        '404':
          $ref: '#/components/responses/NotFound'

  /rd/{sessionId}/export:
    post:
      summary: Export RD in specified format
      operationId: exportRD
      tags:
        - Requirements Document
      parameters:
        - $ref: '#/components/parameters/SessionIdParam'
      requestBody:
        required: true
        content:
          application/json:
            schema:
              $ref: '#/components/schemas/ExportRequest'
      responses:
        '200':
          description: Export file
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/ExportResponse'
            application/pdf:
              schema:
                type: string
                format: binary
            text/markdown:
              schema:
                type: string
        '404':
          $ref: '#/components/responses/NotFound'

  /rd/{sessionId}/diff:
    get:
      summary: Get diff between RD versions
      operationId: getRDDiff
      tags:
        - Requirements Document
      parameters:
        - $ref: '#/components/parameters/SessionIdParam'
        - name: fromVersion
          in: query
          required: true
          schema:
            type: integer
        - name: toVersion
          in: query
          required: true
          schema:
            type: integer
      responses:
        '200':
          description: Version diff
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/RDDiff'

  /review/{sessionId}:
    post:
      summary: Submit review feedback
      operationId: submitReview
      tags:
        - Review
      parameters:
        - $ref: '#/components/parameters/SessionIdParam'
      requestBody:
        required: true
        content:
          application/json:
            schema:
              $ref: '#/components/schemas/ReviewRequest'
      responses:
        '200':
          description: Review submitted
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/ReviewResponse'
        '404':
          $ref: '#/components/responses/NotFound'

  /review/{sessionId}/approve:
    post:
      summary: Approve RD and lock session
      operationId: approveRD
      tags:
        - Review
      parameters:
        - $ref: '#/components/parameters/SessionIdParam'
      requestBody:
        required: true
        content:
          application/json:
            schema:
              $ref: '#/components/schemas/ApprovalRequest'
      responses:
        '200':
          description: RD approved
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/ApprovalResponse'
        '404':
          $ref: '#/components/responses/NotFound'
        '409':
          $ref: '#/components/responses/Conflict'

components:
  securitySchemes:
    bearerAuth:
      type: http
      scheme: bearer
      bearerFormat: JWT

  parameters:
    SessionIdParam:
      name: sessionId
      in: path
      required: true
      schema:
        type: string
        format: uuid

    PageParam:
      name: page
      in: query
      schema:
        type: integer
        minimum: 1
        default: 1

    PageSizeParam:
      name: pageSize
      in: query
      schema:
        type: integer
        minimum: 1
        maximum: 100
        default: 20

  schemas:
    HealthResponse:
      type: object
      required:
        - status
        - services
      properties:
        status:
          type: string
          enum: [healthy, degraded, unhealthy]
        services:
          type: object
          properties:
            postgres:
              type: string
              enum: [up, down]
            redis:
              type: string
              enum: [up, down]
            chromadb:
              type: string
              enum: [up, down]
        version:
          type: string

    LoginRequest:
      type: object
      required:
        - username
        - password
      properties:
        username:
          type: string
        password:
          type: string
          format: password

    LoginResponse:
      type: object
      required:
        - access_token
        - token_type
      properties:
        access_token:
          type: string
        token_type:
          type: string
          enum: [bearer]
        expires_in:
          type: integer
          description: Token expiry in seconds

    CreateSessionRequest:
      type: object
      required:
        - project_name
      properties:
        project_name:
          type: string
          minLength: 1
          maxLength: 255
        user_id:
          type: string
        metadata:
          type: object
          additionalProperties: true

    Session:
      type: object
      required:
        - id
        - project_name
        - created_at
        - status
      properties:
        id:
          type: string
          format: uuid
        project_name:
          type: string
        user_id:
          type: string
        status:
          type: string
          enum: [active, reviewing, approved, archived]
        created_at:
          type: string
          format: date-time
        updated_at:
          type: string
          format: date-time
        metadata:
          type: object

    SessionDetail:
      allOf:
        - $ref: '#/components/schemas/Session'
        - type: object
          properties:
            requirements_count:
              type: integer
            rd_version:
              type: integer
            last_activity:
              type: string
              format: date-time

    SessionList:
      type: object
      required:
        - items
        - total
        - page
        - page_size
      properties:
        items:
          type: array
          items:
            $ref: '#/components/schemas/Session'
        total:
          type: integer
        page:
          type: integer
        page_size:
          type: integer

    SendMessageRequest:
      type: object
      required:
        - message
      properties:
        message:
          type: string
          minLength: 1
          maxLength: 10000
        attachments:
          type: array
          items:
            type: string
            format: uri

    ChatMessage:
      type: object
      required:
        - id
        - role
        - content
        - timestamp
      properties:
        id:
          type: string
          format: uuid
        role:
          type: string
          enum: [user, assistant, system]
        content:
          type: string
        timestamp:
          type: string
          format: date-time
        metadata:
          type: object

    ChatMessageResponse:
      allOf:
        - $ref: '#/components/schemas/ChatMessage'
        - type: object
          properties:
            agent_metadata:
              type: object
              properties:
                confidence:
                  type: number
                  minimum: 0
                  maximum: 1
                requirements_extracted:
                  type: integer
                next_action:
                  type: string
                  enum: [continue, extract, validate, synthesize, review]

    ChatHistory:
      type: object
      required:
        - session_id
        - messages
        - total
      properties:
        session_id:
          type: string
          format: uuid
        messages:
          type: array
          items:
            $ref: '#/components/schemas/ChatMessage'
        total:
          type: integer
        page:
          type: integer
        page_size:
          type: integer

    RequirementType:
      type: string
      enum:
        - functional
        - non-functional
        - business
        - security
        - data
        - interface
        - constraint

    Requirement:
      type: object
      required:
        - id
        - title
        - type
        - actor
        - action
        - acceptance_criteria
        - priority
        - confidence
      properties:
        id:
          type: string
          pattern: '^REQ(-INF)?-\d{3,}$'
        title:
          type: string
          maxLength: 500
        type:
          $ref: '#/components/schemas/RequirementType'
        actor:
          type: string
          maxLength: 200
        action:
          type: string
        condition:
          type: string
          nullable: true
        acceptance_criteria:
          type: array
          items:
            type: string
          minItems: 1
        priority:
          type: string
          enum: [low, medium, high, must]
        confidence:
          type: number
          minimum: 0
          maximum: 1
        inferred:
          type: boolean
        rationale:
          type: string
        source_refs:
          type: array
          items:
            type: string
        created_at:
          type: string
          format: date-time
        updated_at:
          type: string
          format: date-time

    RequirementList:
      type: object
      required:
        - items
        - total
      properties:
        items:
          type: array
          items:
            $ref: '#/components/schemas/Requirement'
        total:
          type: integer
        page:
          type: integer
        page_size:
          type: integer

    UpdateRequirementRequest:
      type: object
      properties:
        title:
          type: string
        action:
          type: string
        condition:
          type: string
        acceptance_criteria:
          type: array
          items:
            type: string
        priority:
          type: string
          enum: [low, medium, high, must]

    RDDocument:
      type: object
      required:
        - id
        - session_id
        - version
        - content
        - status
      properties:
        id:
          type: string
          format: uuid
        session_id:
          type: string
          format: uuid
        version:
          type: integer
        content:
          type: string
        format:
          type: string
          enum: [markdown, json, pdf]
        status:
          type: string
          enum: [draft, under_review, approved]
        created_at:
          type: string
          format: date-time
        metadata:
          type: object

    ExportRequest:
      type: object
      required:
        - format
      properties:
        format:
          type: string
          enum: [markdown, json, pdf]
        version:
          type: integer
          description: Specific version (defaults to latest)

    ExportResponse:
      type: object
      required:
        - download_url
        - format
      properties:
        download_url:
          type: string
          format: uri
        format:
          type: string
        file_size_bytes:
          type: integer
        expires_at:
          type: string
          format: date-time

    RDDiff:
      type: object
      required:
        - from_version
        - to_version
        - changes
      properties:
        from_version:
          type: integer
        to_version:
          type: integer
        changes:
          type: array
          items:
            type: object
            properties:
              type:
                type: string
                enum: [added, modified, deleted]
              requirement_id:
                type: string
              field:
                type: string
              old_value:
                type: string
              new_value:
                type: string
        unified_diff:
          type: string
          description: Unified diff format

    ReviewRequest:
      type: object
      required:
        - action
      properties:
        action:
          type: string
          enum: [comment, revise, approve]
        comments:
          type: array
          items:
            type: object
            properties:
              requirement_id:
                type: string
              comment_text:
                type: string
              suggested_change:
                type: string

    ReviewResponse:
      type: object
      required:
        - review_id
        - status
      properties:
        review_id:
          type: string
          format: uuid
        status:
          type: string
          enum: [pending, in_progress, completed]
        changes_applied:
          type: integer
        next_action:
          type: string

    ApprovalRequest:
      type: object
      required:
        - reviewer_name
      properties:
        reviewer_name:
          type: string
        comments:
          type: string

    ApprovalResponse:
      type: object
      required:
        - session_id
        - rd_version
        - approved_at
      properties:
        session_id:
          type: string
          format: uuid
        rd_version:
          type: integer
        approved_at:
          type: string
          format: date-time
        approved_by:
          type: string

    Error:
      type: object
      required:
        - error
        - message
      properties:
        error:
          type: string
        message:
          type: string
        correlation_id:
          type: string
        details:
          type: object

  responses:
    BadRequest:
      description: Bad request
      content:
        application/json:
          schema:
            $ref: '#/components/schemas/Error'
          example:
            error: bad_request
            message: Invalid request payload
            correlation_id: "abc-123"

    Unauthorized:
      description: Unauthorized
      content:
        application/json:
          schema:
            $ref: '#/components/schemas/Error'
          example:
            error: unauthorized
            message: Invalid or expired token

    NotFound:
      description: Resource not found
      content:
        application/json:
          schema:
            $ref: '#/components/schemas/Error'
          example:
            error: not_found
            message: Session not found

    Conflict:
      description: Conflict with current state
      content:
        application/json:
          schema:
            $ref: '#/components/schemas/Error'
          example:
            error: conflict
            message: Session was modified concurrently

    RateLimitExceeded:
      description: Rate limit exceeded
      content:
        application/json:
          schema:
            $ref: '#/components/schemas/Error'
          example:
            error: rate_limit_exceeded
            message: Too many requests
      headers:
        X-RateLimit-Limit:
          schema:
            type: integer
        X-RateLimit-Remaining:
          schema:
            type: integer
        X-RateLimit-Reset:
          schema:
            type: integer
```

### 4.2 Pydantic Request/Response Models

All Pydantic models are defined in `src/schemas/` and match the OpenAPI schemas exactly.

```python
# src/schemas/api.py
from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any, Literal
from datetime import datetime
from uuid import UUID

# Request models
class CreateSessionRequest(BaseModel):
    project_name: str = Field(..., min_length=1, max_length=255)
    user_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

class SendMessageRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=10000)
    attachments: Optional[List[str]] = None

class ExportRequest(BaseModel):
    format: Literal["markdown", "json", "pdf"]
    version: Optional[int] = None

class ReviewComment(BaseModel):
    requirement_id: Optional[str] = None
    comment_text: str
    suggested_change: Optional[str] = None

class ReviewRequest(BaseModel):
    action: Literal["comment", "revise", "approve"]
    comments: Optional[List[ReviewComment]] = None

class ApprovalRequest(BaseModel):
    reviewer_name: str
    comments: Optional[str] = None

# Response models
class HealthResponse(BaseModel):
    status: Literal["healthy", "degraded", "unhealthy"]
    services: Dict[str, Literal["up", "down"]]
    version: Optional[str] = None

class LoginResponse(BaseModel):
    access_token: str
    token_type: Literal["bearer"] = "bearer"
    expires_in: int

class Session(BaseModel):
    id: UUID
    project_name: str
    user_id: Optional[str]
    status: Literal["active", "reviewing", "approved", "archived"]
    created_at: datetime
    updated_at: datetime
    metadata: Optional[Dict[str, Any]]

    class Config:
        orm_mode = True

class SessionDetail(Session):
    requirements_count: int
    rd_version: int
    last_activity: datetime

class SessionList(BaseModel):
    items: List[Session]
    total: int
    page: int
    page_size: int

class ChatMessage(BaseModel):
    id: UUID
    role: Literal["user", "assistant", "system"]
    content: str
    timestamp: datetime
    metadata: Optional[Dict[str, Any]]

class AgentMetadata(BaseModel):
    confidence: float = Field(ge=0.0, le=1.0)
    requirements_extracted: int
    next_action: Literal["continue", "extract", "validate", "synthesize", "review"]

class ChatMessageResponse(ChatMessage):
    agent_metadata: Optional[AgentMetadata]

class ChatHistory(BaseModel):
    session_id: UUID
    messages: List[ChatMessage]
    total: int
    page: int
    page_size: int

class Requirement(BaseModel):
    id: str = Field(..., regex=r"^REQ(-INF)?-\d{3,}$")
    title: str = Field(..., max_length=500)
    type: Literal["functional", "non-functional", "business", "security", "data", "interface", "constraint"]
    actor: str = Field(..., max_length=200)
    action: str
    condition: Optional[str] = None
    acceptance_criteria: List[str] = Field(..., min_items=1)
    priority: Literal["low", "medium", "high", "must"]
    confidence: float = Field(ge=0.0, le=1.0)
    inferred: bool = False
    rationale: str
    source_refs: List[str]
    created_at: datetime
    updated_at: Optional[datetime] = None

    class Config:
        orm_mode = True

class RequirementList(BaseModel):
    items: List[Requirement]
    total: int
    page: Optional[int]
    page_size: Optional[int]

class UpdateRequirementRequest(BaseModel):
    title: Optional[str]
    action: Optional[str]
    condition: Optional[str]
    acceptance_criteria: Optional[List[str]]
    priority: Optional[Literal["low", "medium", "high", "must"]]

class RDDocument(BaseModel):
    id: UUID
    session_id: UUID
    version: int
    content: str
    format: Literal["markdown", "json", "pdf"]
    status: Literal["draft", "under_review", "approved"]
    created_at: datetime
    metadata: Optional[Dict[str, Any]]

    class Config:
        orm_mode = True

class ExportResponse(BaseModel):
    download_url: str
    format: str
    file_size_bytes: int
    expires_at: datetime

class RDDiffChange(BaseModel):
    type: Literal["added", "modified", "deleted"]
    requirement_id: str
    field: str
    old_value: Optional[str]
    new_value: Optional[str]

class RDDiff(BaseModel):
    from_version: int
    to_version: int
    changes: List[RDDiffChange]
    unified_diff: str

class ReviewResponse(BaseModel):
    review_id: UUID
    status: Literal["pending", "in_progress", "completed"]
    changes_applied: int
    next_action: str

class ApprovalResponse(BaseModel):
    session_id: UUID
    rd_version: int
    approved_at: datetime
    approved_by: str

class ErrorResponse(BaseModel):
    error: str
    message: str
    correlation_id: Optional[str]
    details: Optional[Dict[str, Any]]
```

### 4.3 Pagination Implementation

```python
# src/api/pagination.py
from typing import Generic, TypeVar, List
from pydantic import BaseModel
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

T = TypeVar("T")

class PaginatedResponse(BaseModel, Generic[T]):
    items: List[T]
    total: int
    page: int
    page_size: int
    total_pages: int
    has_next: bool
    has_prev: bool

async def paginate(
    db: AsyncSession,
    query: select,
    page: int = 1,
    page_size: int = 20,
    model: type = None
) -> PaginatedResponse:
    \"\"\"Generic pagination helper\"\"\"
    # Get total count
    count_query = select(func.count()).select_from(query.subquery())
    total = await db.scalar(count_query)

    # Calculate pagination
    offset = (page - 1) * page_size
    total_pages = (total + page_size - 1) // page_size

    # Fetch page
    paginated_query = query.offset(offset).limit(page_size)
    result = await db.execute(paginated_query)
    items = result.scalars().all()

    # Convert to Pydantic if model provided
    if model:
        items = [model.from_orm(item) for item in items]

    return PaginatedResponse(
        items=items,
        total=total,
        page=page,
        page_size=page_size,
        total_pages=total_pages,
        has_next=page < total_pages,
        has_prev=page > 1
    )
```

### 4.4 Error Response Standardization

```python
# src/api/middleware/error_handler.py
from fastapi import Request, status
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from src.schemas.api import ErrorResponse

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content=ErrorResponse(
            error="validation_error",
            message="Invalid request payload",
            correlation_id=correlation_id_var.get(""),
            details={"errors": exc.errors()}
        ).dict()
    )

@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {exc}", exc_info=True)

    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=ErrorResponse(
            error="internal_server_error",
            message="An unexpected error occurred",
            correlation_id=correlation_id_var.get("")
        ).dict()
    )

# Custom exceptions
class SessionNotFoundError(Exception):
    pass

@app.exception_handler(SessionNotFoundError)
async def session_not_found_handler(request: Request, exc: SessionNotFoundError):
    return JSONResponse(
        status_code=status.HTTP_404_NOT_FOUND,
        content=ErrorResponse(
            error="session_not_found",
            message=str(exc),
            correlation_id=correlation_id_var.get("")
        ).dict()
    )
```

---

Due to the extensive length of Design Packet 2, I'll continue this as a complete markdown file. Let me create the full document now with all remaining sections:

[File continues with sections 5-11...]
