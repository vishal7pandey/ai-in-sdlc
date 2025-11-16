# Design Packet 2: Part 2 - Database, Agents, and Performance

## 5. Full Database Specification (DDL + Low-Level Models)

### 5.1 Complete SQL DDL

```sql
-- database-ddl.sql
-- Requirements Engineering Platform Database Schema
-- PostgreSQL 15+

-- Enable extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pgcrypto";

-- Sessions table
CREATE TABLE sessions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    project_name VARCHAR(255) NOT NULL,
    user_id VARCHAR(100) NOT NULL,
    status VARCHAR(50) DEFAULT 'active' CHECK (status IN ('active', 'reviewing', 'approved', 'archived')),
    version INTEGER DEFAULT 0 NOT NULL,  -- Optimistic locking
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    metadata JSONB DEFAULT '{}'::jsonb,
    CONSTRAINT project_name_not_empty CHECK (char_length(project_name) > 0)
);

-- Chat messages table
CREATE TABLE chat_messages (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    session_id UUID NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    role VARCHAR(20) NOT NULL CHECK (role IN ('user', 'assistant', 'system')),
    content TEXT NOT NULL,
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    metadata JSONB DEFAULT '{}'::jsonb,
    CONSTRAINT content_not_empty CHECK (char_length(content) > 0)
);

-- Requirements table
CREATE TABLE requirements (
    id VARCHAR(50) PRIMARY KEY CHECK (id ~ '^REQ(-INF)?-\d{3,}$'),
    session_id UUID NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    title VARCHAR(500) NOT NULL,
    type VARCHAR(50) NOT NULL CHECK (type IN ('functional', 'non-functional', 'business', 'security', 'data', 'interface', 'constraint')),
    actor VARCHAR(200) NOT NULL,
    action TEXT NOT NULL,
    condition TEXT,
    acceptance_criteria JSONB NOT NULL,
    priority VARCHAR(20) NOT NULL CHECK (priority IN ('low', 'medium', 'high', 'must')),
    confidence DECIMAL(3,2) NOT NULL CHECK (confidence >= 0 AND confidence <= 1),
    inferred BOOLEAN DEFAULT FALSE,
    rationale TEXT NOT NULL,
    source_refs JSONB NOT NULL DEFAULT '[]'::jsonb,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    version INTEGER DEFAULT 0 NOT NULL,
    CONSTRAINT acceptance_criteria_not_empty CHECK (jsonb_array_length(acceptance_criteria) > 0)
);

-- RD documents table (event sourcing)
CREATE TABLE rd_events (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    session_id UUID NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    version INTEGER NOT NULL,
    event_type VARCHAR(50) NOT NULL CHECK (event_type IN ('created', 'updated', 'approved', 'exported')),
    event_data JSONB NOT NULL,
    user_id VARCHAR(100) NOT NULL,
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    CONSTRAINT version_positive CHECK (version > 0)
);

-- Materialized view of latest RD for fast access
CREATE TABLE rd_documents (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    session_id UUID NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    version INTEGER NOT NULL,
    content TEXT NOT NULL,
    format VARCHAR(20) DEFAULT 'markdown' CHECK (format IN ('markdown', 'json', 'pdf')),
    status VARCHAR(20) DEFAULT 'draft' CHECK (status IN ('draft', 'under_review', 'approved')),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    metadata JSONB DEFAULT '{}'::jsonb,
    CONSTRAINT unique_session_version UNIQUE (session_id, version)
);

-- Audit logs table
CREATE TABLE audit_logs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    session_id UUID REFERENCES sessions(id) ON DELETE SET NULL,
    user_id VARCHAR(100) NOT NULL,
    action VARCHAR(100) NOT NULL,
    entity_type VARCHAR(50),
    entity_id VARCHAR(100),
    changes JSONB,
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    ip_address INET,
    user_agent TEXT
);

-- LangGraph checkpoints table
CREATE TABLE langgraph_checkpoints (
    thread_id VARCHAR(255) NOT NULL,
    checkpoint_id UUID NOT NULL,
    parent_checkpoint_id UUID,
    checkpoint JSONB NOT NULL,
    metadata JSONB DEFAULT '{}'::jsonb,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    PRIMARY KEY (thread_id, checkpoint_id)
);

-- LangGraph checkpoint writes (pending operations)
CREATE TABLE langgraph_checkpoint_writes (
    thread_id VARCHAR(255) NOT NULL,
    checkpoint_id UUID NOT NULL,
    task_id VARCHAR(255) NOT NULL,
    idx INTEGER NOT NULL,
    channel VARCHAR(255) NOT NULL,
    value JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    PRIMARY KEY (thread_id, checkpoint_id, task_id, idx),
    FOREIGN KEY (thread_id, checkpoint_id) REFERENCES langgraph_checkpoints(thread_id, checkpoint_id) ON DELETE CASCADE
);

-- Indexes for performance
CREATE INDEX idx_sessions_user_id ON sessions(user_id);
CREATE INDEX idx_sessions_status ON sessions(status);
CREATE INDEX idx_sessions_created_at ON sessions(created_at DESC);

CREATE INDEX idx_chat_messages_session ON chat_messages(session_id, timestamp DESC);
CREATE INDEX idx_chat_messages_role ON chat_messages(role);

CREATE INDEX idx_requirements_session ON requirements(session_id);
CREATE INDEX idx_requirements_type ON requirements(type);
CREATE INDEX idx_requirements_priority ON requirements(priority);
CREATE INDEX idx_requirements_inferred ON requirements(inferred);
CREATE INDEX idx_requirements_created_at ON requirements(created_at DESC);

CREATE INDEX idx_rd_events_session_version ON rd_events(session_id, version DESC);
CREATE INDEX idx_rd_events_session_timestamp ON rd_events(session_id, timestamp DESC);

CREATE INDEX idx_rd_documents_session ON rd_documents(session_id);
CREATE INDEX idx_rd_documents_version ON rd_documents(version DESC);

CREATE INDEX idx_audit_logs_session ON audit_logs(session_id, timestamp DESC);
CREATE INDEX idx_audit_logs_user ON audit_logs(user_id, timestamp DESC);
CREATE INDEX idx_audit_logs_entity ON audit_logs(entity_type, entity_id);

CREATE INDEX idx_checkpoint_thread ON langgraph_checkpoints(thread_id, created_at DESC);

-- Triggers for automatic updated_at
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER update_sessions_updated_at
    BEFORE UPDATE ON sessions
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_requirements_updated_at
    BEFORE UPDATE ON requirements
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- Trigger to maintain RD materialized view
CREATE OR REPLACE FUNCTION update_rd_document_from_event()
RETURNS TRIGGER AS $$
BEGIN
    -- Insert or update latest RD document
    INSERT INTO rd_documents (session_id, version, content, format, status, metadata)
    VALUES (
        NEW.session_id,
        NEW.version,
        NEW.event_data->>'content',
        COALESCE(NEW.event_data->>'format', 'markdown'),
        CASE
            WHEN NEW.event_type = 'approved' THEN 'approved'
            ELSE 'draft'
        END,
        NEW.event_data
    )
    ON CONFLICT (session_id, version) DO UPDATE
    SET
        content = EXCLUDED.content,
        status = EXCLUDED.status,
        metadata = EXCLUDED.metadata;

    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER rd_event_to_document
    AFTER INSERT ON rd_events
    FOR EACH ROW
    EXECUTE FUNCTION update_rd_document_from_event();

-- Function to get latest checkpoint
CREATE OR REPLACE FUNCTION get_latest_checkpoint(p_thread_id VARCHAR)
RETURNS TABLE (
    checkpoint_id UUID,
    checkpoint JSONB,
    metadata JSONB
) AS $$
BEGIN
    RETURN QUERY
    SELECT
        c.checkpoint_id,
        c.checkpoint,
        c.metadata
    FROM langgraph_checkpoints c
    WHERE c.thread_id = p_thread_id
    ORDER BY c.created_at DESC
    LIMIT 1;
END;
$$ LANGUAGE plpgsql;

-- Partitioning for audit logs (by month)
CREATE TABLE audit_logs_template (
    LIKE audit_logs INCLUDING ALL
);

-- Create partitions for current and next 3 months
CREATE TABLE audit_logs_2025_11 PARTITION OF audit_logs
    FOR VALUES FROM ('2025-11-01') TO ('2025-12-01');

CREATE TABLE audit_logs_2025_12 PARTITION OF audit_logs
    FOR VALUES FROM ('2025-12-01') TO ('2026-01-01');

-- Views for common queries
CREATE VIEW active_sessions AS
SELECT * FROM sessions
WHERE status = 'active'
ORDER BY updated_at DESC;

CREATE VIEW session_summary AS
SELECT
    s.id,
    s.project_name,
    s.status,
    s.created_at,
    COUNT(DISTINCT cm.id) as message_count,
    COUNT(DISTINCT r.id) as requirement_count,
    MAX(rd.version) as latest_rd_version
FROM sessions s
LEFT JOIN chat_messages cm ON s.id = cm.session_id
LEFT JOIN requirements r ON s.id = r.session_id
LEFT JOIN rd_documents rd ON s.id = rd.session_id
GROUP BY s.id, s.project_name, s.status, s.created_at;

-- Constraints for data integrity
ALTER TABLE requirements
ADD CONSTRAINT fk_requirement_session
FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE DEFERRABLE INITIALLY DEFERRED;

-- Comments for documentation
COMMENT ON TABLE sessions IS 'Main conversation sessions';
COMMENT ON COLUMN sessions.version IS 'Optimistic locking version counter';
COMMENT ON TABLE requirements IS 'Extracted software requirements';
COMMENT ON TABLE rd_events IS 'Event sourcing log for RD versions';
COMMENT ON TABLE langgraph_checkpoints IS 'LangGraph state persistence';
```

### 5.2 SQLAlchemy Models

```python
# src/models/database.py
from sqlalchemy import (
    Column, String, Integer, Boolean, DateTime, Text,
    ForeignKey, DECIMAL, CheckConstraint, Index, func,
    JSON as SAJSON
)
from sqlalchemy.dialects.postgresql import UUID, INET, JSONB
from sqlalchemy.orm import relationship, declarative_base
from sqlalchemy.ext.asyncio import AsyncAttrs
from datetime import datetime
from uuid import uuid4

Base = declarative_base()

class SessionModel(AsyncAttrs, Base):
    __tablename__ = "sessions"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    project_name = Column(String(255), nullable=False)
    user_id = Column(String(100), nullable=False)
    status = Column(String(50), default="active", nullable=False)
    version = Column(Integer, default=0, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    metadata = Column(JSONB, default={})

    # Relationships
    chat_messages = relationship("ChatMessageModel", back_populates="session", cascade="all, delete-orphan")
    requirements = relationship("RequirementModel", back_populates="session", cascade="all, delete-orphan")
    rd_documents = relationship("RDDocumentModel", back_populates="session")

    # Indexes
    __table_args__ = (
        CheckConstraint("status IN ('active', 'reviewing', 'approved', 'archived')", name="check_status"),
        Index("idx_sessions_user_id", "user_id"),
        Index("idx_sessions_status", "status"),
    )

class ChatMessageModel(AsyncAttrs, Base):
    __tablename__ = "chat_messages"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    session_id = Column(UUID(as_uuid=True), ForeignKey("sessions.id", ondelete="CASCADE"), nullable=False)
    role = Column(String(20), nullable=False)
    content = Column(Text, nullable=False)
    timestamp = Column(DateTime(timezone=True), server_default=func.now())
    metadata = Column(JSONB, default={})

    # Relationships
    session = relationship("SessionModel", back_populates="chat_messages")

    __table_args__ = (
        CheckConstraint("role IN ('user', 'assistant', 'system')", name="check_role"),
        Index("idx_chat_messages_session", "session_id", "timestamp"),
    )

class RequirementModel(AsyncAttrs, Base):
    __tablename__ = "requirements"

    id = Column(String(50), primary_key=True)
    session_id = Column(UUID(as_uuid=True), ForeignKey("sessions.id", ondelete="CASCADE"), nullable=False)
    title = Column(String(500), nullable=False)
    type = Column(String(50), nullable=False)
    actor = Column(String(200), nullable=False)
    action = Column(Text, nullable=False)
    condition = Column(Text)
    acceptance_criteria = Column(JSONB, nullable=False)
    priority = Column(String(20), nullable=False)
    confidence = Column(DECIMAL(3, 2), nullable=False)
    inferred = Column(Boolean, default=False)
    rationale = Column(Text, nullable=False)
    source_refs = Column(JSONB, default=[])
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    version = Column(Integer, default=0)

    # Relationships
    session = relationship("SessionModel", back_populates="requirements")

    __table_args__ = (
        CheckConstraint("type IN ('functional', 'non-functional', 'business', 'security', 'data', 'interface', 'constraint')"),
        CheckConstraint("priority IN ('low', 'medium', 'high', 'must')"),
        CheckConstraint("confidence >= 0 AND confidence <= 1"),
        Index("idx_requirements_session", "session_id"),
        Index("idx_requirements_type", "type"),
    )

    def to_pydantic(self):
        from src.schemas.requirement import RequirementItem
        return RequirementItem(
            id=self.id,
            title=self.title,
            type=self.type,
            actor=self.actor,
            action=self.action,
            condition=self.condition,
            acceptance_criteria=self.acceptance_criteria,
            priority=self.priority,
            confidence=float(self.confidence),
            inferred=self.inferred,
            rationale=self.rationale,
            source_refs=self.source_refs,
            created_at=self.created_at
        )

class RDEventModel(AsyncAttrs, Base):
    __tablename__ = "rd_events"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    session_id = Column(UUID(as_uuid=True), ForeignKey("sessions.id", ondelete="CASCADE"), nullable=False)
    version = Column(Integer, nullable=False)
    event_type = Column(String(50), nullable=False)
    event_data = Column(JSONB, nullable=False)
    user_id = Column(String(100), nullable=False)
    timestamp = Column(DateTime(timezone=True), server_default=func.now())

    __table_args__ = (
        CheckConstraint("event_type IN ('created', 'updated', 'approved', 'exported')"),
        Index("idx_rd_events_session_version", "session_id", "version"),
    )

class RDDocumentModel(AsyncAttrs, Base):
    __tablename__ = "rd_documents"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    session_id = Column(UUID(as_uuid=True), ForeignKey("sessions.id", ondelete="CASCADE"), nullable=False)
    version = Column(Integer, nullable=False)
    content = Column(Text, nullable=False)
    format = Column(String(20), default="markdown")
    status = Column(String(20), default="draft")
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    metadata = Column(JSONB, default={})

    # Relationships
    session = relationship("SessionModel", back_populates="rd_documents")

    __table_args__ = (
        CheckConstraint("format IN ('markdown', 'json', 'pdf')"),
        CheckConstraint("status IN ('draft', 'under_review', 'approved')"),
    )

class AuditLogModel(AsyncAttrs, Base):
    __tablename__ = "audit_logs"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    session_id = Column(UUID(as_uuid=True), ForeignKey("sessions.id", ondelete="SET NULL"))
    user_id = Column(String(100), nullable=False)
    action = Column(String(100), nullable=False)
    entity_type = Column(String(50))
    entity_id = Column(String(100))
    changes = Column(JSONB)
    timestamp = Column(DateTime(timezone=True), server_default=func.now())
    ip_address = Column(INET)
    user_agent = Column(Text)

    __table_args__ = (
        Index("idx_audit_logs_session", "session_id", "timestamp"),
    )

class LangGraphCheckpointModel(AsyncAttrs, Base):
    __tablename__ = "langgraph_checkpoints"

    thread_id = Column(String(255), primary_key=True)
    checkpoint_id = Column(UUID(as_uuid=True), primary_key=True)
    parent_checkpoint_id = Column(UUID(as_uuid=True))
    checkpoint = Column(JSONB, nullable=False)
    metadata = Column(JSONB, default={})
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
```

### 5.3 Alembic Migration

```python
# alembic/versions/001_initial_schema.py
\"\"\"Initial database schema

Revision ID: 001
Revises:
Create Date: 2025-11-16
\"\"\"
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers
revision = '001'
down_revision = None
branch_labels = None
depends_on = None

def upgrade():
    # Create extensions
    op.execute('CREATE EXTENSION IF NOT EXISTS "uuid-ossp"')
    op.execute('CREATE EXTENSION IF NOT EXISTS "pgcrypto"')

    # Create sessions table
    op.create_table(
        'sessions',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.text('uuid_generate_v4()')),
        sa.Column('project_name', sa.String(255), nullable=False),
        sa.Column('user_id', sa.String(100), nullable=False),
        sa.Column('status', sa.String(50), server_default='active', nullable=False),
        sa.Column('version', sa.Integer, server_default='0', nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column('metadata', postgresql.JSONB, server_default='{}'),
        sa.CheckConstraint("status IN ('active', 'reviewing', 'approved', 'archived')", name='check_status')
    )

    op.create_index('idx_sessions_user_id', 'sessions', ['user_id'])
    op.create_index('idx_sessions_status', 'sessions', ['status'])

    # [Continue with all other tables...]
    # (Full migration script follows the DDL above)

def downgrade():
    op.drop_table('audit_logs')
    op.drop_table('rd_documents')
    op.drop_table('rd_events')
    op.drop_table('requirements')
    op.drop_table('chat_messages')
    op.drop_table('sessions')
    op.execute('DROP EXTENSION IF EXISTS "pgcrypto"')
    op.execute('DROP EXTENSION IF EXISTS "uuid-ossp"')
```

---

## 6. Agent Implementation Blueprints

### 6.1 Extraction Agent (Complete Implementation)

```markdown
# extraction-agent-blueprint.md

## Internal Module Structure

```
src/agents/extraction/
├── __init__.py
├── agent.py              # Main agent class
├── parser.py             # LLM response parser
├── entity_extractor.py   # Extract actors, actions, conditions
├── criteria_generator.py # Generate acceptance criteria
├── confidence_scorer.py  # Calculate confidence scores
├── prompts/
│   ├── extraction_base.txt
│   ├── extraction_functional.txt
│   └── extraction_nonfunctional.txt
└── tests/
    ├── test_parser.py
    ├── test_extractor.py
    └── fixtures/
        └── sample_chats.json
```

## Class Diagram

```
┌─────────────────────────────────────┐
│       ExtractionAgent               │
├─────────────────────────────────────┤
│ - llm: ChatOpenAI                   │
│ - parser: ResponseParser            │
│ - entity_extractor: EntityExtractor │
│ - criteria_gen: CriteriaGenerator   │
│ - scorer: ConfidenceScorer          │
├─────────────────────────────────────┤
│ + invoke(state) → GraphState        │
│ - _execute(state) → List[Req]       │
│ - _build_prompt(state) → str        │
│ - _select_prompt_template(type)     │
│ - _post_process(reqs) → List[Req]   │
│ - _assign_ids(reqs) → List[Req]     │
│ - _link_sources(reqs, msgs)         │
└─────────────────────────────────────┘
```

## Token Budgeting Strategy

**Budget Allocation (max 8000 tokens):**
- System prompt: 800 tokens (10%)
- Chat history (last 10 messages): 3000 tokens (37.5%)
- Examples: 1200 tokens (15%)
- Response buffer: 3000 tokens (37.5%)

**Token Optimization:**[79][82][88]
1. Truncate old messages beyond context window
2. Summarize long messages (>500 tokens) before including
3. Cache system prompt + examples (static content)
4. Use `max_tokens` parameter to cap response length

```python
# src/agents/extraction/token_budget.py
import tiktoken

class TokenBudgetManager:
    def __init__(self, model="gpt-4"):
        self.encoding = tiktoken.encoding_for_model(model)
        self.max_context = 8000
        self.system_prompt_budget = 800
        self.history_budget = 3000
        self.examples_budget = 1200
        self.response_budget = 3000

    def count_tokens(self, text: str) -> int:
        return len(self.encoding.encode(text))

    def truncate_chat_history(self, messages: List[Dict]) -> List[Dict]:
        \"\"\"Keep only recent messages that fit in budget\"\"\"
        truncated = []
        token_count = 0

        # Iterate from most recent
        for msg in reversed(messages):
            msg_tokens = self.count_tokens(msg["content"])

            if token_count + msg_tokens > self.history_budget:
                # Summarize this message if it's important
                if msg["role"] == "user":
                    summary = self._summarize_message(msg["content"])
                    msg_tokens = self.count_tokens(summary)
                    msg = {**msg, "content": summary}
                else:
                    break  # Skip old assistant messages

            truncated.insert(0, msg)
            token_count += msg_tokens

        return truncated

    def _summarize_message(self, text: str) -> str:
        \"\"\"Quick summarization for long messages\"\"\"
        if len(text) < 500:
            return text

        # Simple extractive summary (first + last sentences)
        sentences = text.split('. ')
        if len(sentences) <= 3:
            return text

        return f"{sentences[0]}. [...] {sentences[-1]}"
```

## Confidence Scoring Formula

**Confidence Calculation:**

\\[
\\text{confidence} = w_1 \\cdot \\text{llm\\_confidence} + w_2 \\cdot \\text{parse\\_quality} + w_3 \\cdot \\text{source\\_clarity}
\\]

Where:
- \\( w_1 = 0.5 \\) (LLM self-reported confidence)
- \\( w_2 = 0.3 \\) (Parsing success quality)
- \\( w_3 = 0.2 \\) (Source text clarity score)

```python
# src/agents/extraction/confidence_scorer.py
from typing import Dict, List

class ConfidenceScorer:
    def __init__(self):
        self.weights = {
            "llm_confidence": 0.5,
            "parse_quality": 0.3,
            "source_clarity": 0.2
        }

    def calculate(
        self,
        llm_confidence: float,
        parsed_fields: Dict,
        source_text: str
    ) -> float:
        \"\"\"Calculate overall confidence score\"\"\"

        # LLM confidence (0-1)
        llm_score = llm_confidence

        # Parse quality (how many required fields present)
        required_fields = ["title", "actor", "action", "acceptance_criteria"]
        present_fields = sum(1 for f in required_fields if parsed_fields.get(f))
        parse_score = present_fields / len(required_fields)

        # Source clarity (inverse of ambiguity indicators)
        clarity_score = self._assess_clarity(source_text)

        # Weighted sum
        final_confidence = (
            self.weights["llm_confidence"] * llm_score +
            self.weights["parse_quality"] * parse_score +
            self.weights["source_clarity"] * clarity_score
        )

        return round(final_confidence, 2)

    def _assess_clarity(self, text: str) -> float:
        \"\"\"Score text clarity (1.0 = very clear, 0.0 = very ambiguous)\"\"\"
        ambiguous_terms = ["maybe", "possibly", "perhaps", "might", "could be", "not sure"]
        vague_verbs = ["optimize", "improve", "enhance", "better"]

        text_lower = text.lower()

        # Penalize ambiguity
        ambiguity_penalty = sum(0.1 for term in ambiguous_terms if term in text_lower)
        vagueness_penalty = sum(0.05 for verb in vague_verbs if verb in text_lower)

        clarity = max(0.0, 1.0 - ambiguity_penalty - vagueness_penalty)
        return clarity
```

## Complete Implementation

```python
# src/agents/extraction/agent.py
from typing import List, Dict
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from src.schemas.state import GraphState, Requirement
from src.agents.extraction.parser import ResponseParser
from src.agents.extraction.entity_extractor import EntityExtractor
from src.agents.extraction.confidence_scorer import ConfidenceScorer
from src.agents.extraction.token_budget import TokenBudgetManager
import os

class ExtractionAgent:
    def __init__(self):
        self.llm = ChatOpenAI(
            model=os.getenv("OPENAI_MODEL", "gpt-4-turbo-preview"),
            temperature=0.2,
            max_tokens=3000,
            openai_api_key=os.getenv("OPENAI_API_KEY")
        )

        self.parser = ResponseParser()
        self.entity_extractor = EntityExtractor()
        self.scorer = ConfidenceScorer()
        self.token_manager = TokenBudgetManager()

        # Load prompts
        self.prompts = self._load_prompts()

    def _load_prompts(self) -> Dict[str, PromptTemplate]:
        \"\"\"Load prompt templates from files\"\"\"
        prompts = {}
        prompt_dir = "src/agents/extraction/prompts"

        for prompt_file in ["extraction_base.txt", "extraction_functional.txt"]:
            with open(f"{prompt_dir}/{prompt_file}") as f:
                content = f.read()
                name = prompt_file.replace(".txt", "")
                prompts[name] = PromptTemplate(
                    template=content,
                    input_variables=["chat_history", "session_context"]
                )

        return prompts

    async def invoke(self, state: GraphState) -> GraphState:
        \"\"\"Main entry point for extraction agent\"\"\"
        try:
            # Execute extraction
            extracted_reqs = await self._execute(state)

            # Post-process
            extracted_reqs = self._assign_ids(extracted_reqs, state)
            extracted_reqs = await self._link_sources(extracted_reqs, state["messages"])

            # Calculate overall confidence
            avg_confidence = sum(r.confidence for r in extracted_reqs) / len(extracted_reqs) if extracted_reqs else 1.0

            # Update state
            return {
                **state,
                "requirements": state["requirements"] + extracted_reqs,
                "confidence": min(state["confidence"], avg_confidence),
                "last_agent": "extraction"
            }

        except Exception as e:
            logger.error(f"Extraction agent failed: {e}", exc_info=True)
            return {
                **state,
                "error_count": state.get("error_count", 0) + 1,
                "last_error": str(e),
                "confidence": max(0.0, state.get("confidence", 1.0) - 0.2)
            }

    async def _execute(self, state: GraphState) -> List[Requirement]:
        \"\"\"Core extraction logic\"\"\"
        # Build prompt
        prompt = self._build_prompt(state)

        # Call LLM
        response = await self.llm.ainvoke(prompt)

        # Parse response
        raw_reqs = self.parser.parse(response.content)

        # Convert to Requirement objects with confidence scoring
        requirements = []
        for raw_req in raw_reqs:
            confidence = self.scorer.calculate(
                llm_confidence=raw_req.get("confidence", 0.8),
                parsed_fields=raw_req,
                source_text=raw_req.get("source_text", "")
            )

            req = Requirement(
                id=raw_req["id"],  # Temporary, will be reassigned
                title=raw_req["title"],
                type=raw_req["type"],
                actor=raw_req["actor"],
                action=raw_req["action"],
                condition=raw_req.get("condition"),
                acceptance_criteria=raw_req["acceptance_criteria"],
                priority=raw_req["priority"],
                confidence=confidence,
                inferred=False,
                rationale=raw_req["rationale"],
                source_refs=[]  # Will be populated by _link_sources
            )
            requirements.append(req)

        return requirements

    def _build_prompt(self, state: GraphState) -> str:
        \"\"\"Construct optimized prompt\"\"\"
        # Truncate chat history to fit token budget
        truncated_messages = self.token_manager.truncate_chat_history(state["messages"])

        # Format chat history
        chat_text = "\\n".join([
            f"[Turn {i}] {msg.role.upper()}: {msg.content}"
            for i, msg in enumerate(truncated_messages)
        ])

        # Select appropriate prompt template
        template = self.prompts["extraction_base"]

        # Render prompt
        return template.format(
            chat_history=chat_text,
            session_context=f"Project: {state['project_name']}"
        )

    def _assign_ids(self, requirements: List[Requirement], state: GraphState) -> List[Requirement]:
        \"\"\"Assign sequential requirement IDs\"\"\"
        existing_count = len(state["requirements"])

        for i, req in enumerate(requirements):
            req_num = existing_count + i + 1
            req.id = f"REQ-{req_num:03d}"

        return requirements

    async def _link_sources(self, requirements: List[Requirement], messages: List) -> List[Requirement]:
        \"\"\"Link requirements to source chat turns\"\"\"
        for req in requirements:
            # Find message indices that mention keywords from requirement
            keywords = req.title.lower().split()
            source_turns = []

            for i, msg in enumerate(messages):
                if any(kw in msg.content.lower() for kw in keywords):
                    source_turns.append(f"chat_turn_{i}")

            req.source_refs = source_turns if source_turns else ["chat_turn_0"]

        return requirements
```

## Unit Test Example

```python
# tests/unit/test_extraction.py
import pytest
from src.agents.extraction.agent import ExtractionAgent
from src.schemas.state import GraphState, Message
from datetime import datetime

@pytest.mark.asyncio
async def test_extraction_login_requirement():
    \"\"\"Test extraction of login requirement\"\"\"
    agent = ExtractionAgent()

    state: GraphState = {
        "session_id": "test_123",
        "project_name": "Test App",
        "messages": [
            Message(
                id="msg_1",
                role="user",
                content="Users should be able to log in with email and password",
                timestamp=datetime.utcnow()
            )
        ],
        "requirements": [],
        "confidence": 1.0,
        "current_turn": 1
    }

    result = await agent.invoke(state)

    # Assertions
    assert len(result["requirements"]) >= 1

    req = result["requirements"][0]
    assert req.id.startswith("REQ-")
    assert "login" in req.title.lower() or "authentication" in req.title.lower()
    assert req.type in ["functional", "security"]
    assert len(req.acceptance_criteria) > 0
    assert 0 <= req.confidence <= 1
    assert len(req.source_refs) > 0
```

---

**Due to length constraints, the complete Design Packet 2 continues with:**
- Section 6.2-6.6: Other agent blueprints (Inference, Validation, Synthesis, etc.)
- Section 7: Complete LangGraph orchestrator implementation
- Section 8: Performance optimization strategies
- Section 9: Developer CLI and tooling
- Section 10: Scalability roadmap
- Section 11: All embedded artifacts (full code files)

[File Created: DesignPacket2-Part1.md and DesignPacket2-Part2.md]
