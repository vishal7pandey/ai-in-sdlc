# Design Packet 1: Multi-Agent Requirements Engineering Platform
## Local Development Edition

**Version:** 1.0
**Date:** November 16, 2025
**Target Platform:** Single Developer Laptop (Linux/macOS/Windows)
**Stack:** LangGraph + LangChain + LangSmith + OpenAI API

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Goals & Success Criteria](#2-goals--success-criteria)
3. [Local Development Environment Design](#3-local-development-environment-design)
4. [Repository Layout & File Tree](#4-repository-layout--file-tree)
5. [API & Logging Design](#5-api--logging-design)
6. [Persistence, Vector DB, and Knowledge Store](#6-persistence-vector-db-and-knowledge-store)
7. [Orchestrator & Agent Design (High-Level)](#7-orchestrator--agent-design-high-level)
8. [Agent-by-Agent Low Level Design](#8-agent-by-agent-low-level-design)
9. [Template & RD Format](#9-template--rd-format)
10. [Validation Rules & Deterministic Checks](#10-validation-rules--deterministic-checks)
11. [Human-in-the-Loop UX & Review Flow](#11-human-in-the-loop-ux--review-flow)
12. [Testing, QA & Evaluation](#12-testing-qa--evaluation)
13. [Security, Privacy & PII](#13-security-privacy--pii)
14. [Observability & Monitoring (Local)](#14-observability--monitoring-local)
15. [CI / Local Dev Workflow](#15-ci--local-dev-workflow)
16. [Deliverables & Acceptance Criteria](#16-deliverables--acceptance-criteria)
17. [Implementation Plan & Prioritized Next Steps](#17-implementation-plan--prioritized-next-steps)
18. [Appendix](#18-appendix)

---

## 1. Executive Summary

A **local-first, multi-agent LLM system** that conducts guided conversations with stakeholders to extract explicit and implicit software requirements, synthesizes industry-grade Requirements Documents (RD), and manages human-in-the-loop approval cycles. Built on **LangGraph** (orchestration), **LangChain** (agent toolkit), **LangSmith** (tracing), and **OpenAI API** (LLM + embeddings). Designed to run on a single developer laptop (16GB+ RAM) with Docker Compose managing Postgres, Redis, and ChromaDB. Prioritizes transparency, traceability, and iterative refinement with human oversight at every critical decision point.

---

## 2. Goals & Success Criteria

### Primary Goals
1. **Elicit requirements** via natural conversation, capturing explicit user statements and inferring implicit needs
2. **Synthesize RD** conforming to industry standards (IEEE 830-like structure)
3. **Enable human review** with approve/revise cycles, inline comments, and version diffs
4. **Ensure traceability** linking every RD element to source chat utterances

### Success Criteria (Measurable)
| Metric | Target | Measurement Method |
|--------|--------|-------------------|
| **Completeness** | ≥95% explicit requirements captured | Manual audit against test cases |
| **Quality** | ≥4.5/5 reviewer rating | Post-session survey (clarity, correctness, sufficiency) |
| **Turn Efficiency** | ≤10 chat turns to first draft | Median across test sessions |
| **Revision Cycles** | ≤3 rounds to sign-off | 90th percentile |
| **Traceability** | 100% RD elements linked | Automated validation |
| **Latency** | <3s per interactive response | p95 response time |
| **PII Protection** | 0 PII leaks in RD exports | Automated PII scanner |

---

## 3. Local Development Environment Design

### 3.1 OS Assumptions

**Primary:** Linux (Ubuntu 22.04+) or macOS (Monterey+)
**Secondary:** Windows 11 with WSL2 (Ubuntu)

**Rationale:** Docker Compose performs best on Linux/macOS; Windows users should use WSL2 for consistent behavior.[60][62]

### 3.2 Required Local Services

| Service | Purpose | Resource Footprint |
|---------|---------|-------------------|
| **Postgres 15** | Requirements artifacts, sessions, audit logs | ~200MB RAM, 1GB disk |
| **Redis 7** | Session state cache, LangGraph checkpointer | ~50MB RAM |
| **ChromaDB** | Vector store for semantic search, conflict detection | ~300MB RAM, scalable disk |
| **FastAPI App** | REST API server | ~500MB RAM (includes Python runtime) |
| **LangSmith** | Tracing (cloud SaaS, optional local) | N/A (external API) |

**Total Estimated:** ~1.5GB RAM + 3GB disk for MVP

### 3.3 Bootstrap Commands

#### Prerequisites Installation

**Ubuntu/Debian:**
```bash
# Update package list
sudo apt update && sudo apt upgrade -y

# Install Python 3.11+
sudo apt install -y python3.11 python3.11-venv python3-pip

# Install Docker & Docker Compose
sudo apt install -y docker.io docker-compose-v2
sudo usermod -aG docker $USER
newgrp docker

# Install Node.js 18+ (for optional frontend)
curl -fsSL https://deb.nodesource.com/setup_18.x | sudo -E bash -
sudo apt install -y nodejs

# Install Git
sudo apt install -y git
```

**macOS (Homebrew):**
```bash
# Install Homebrew (if not present)
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install dependencies
brew install python@3.11
brew install --cask docker
brew install node@18
brew install git

# Start Docker Desktop
open -a Docker
```

**Windows (WSL2):**
```powershell
# Install WSL2 with Ubuntu
wsl --install -d Ubuntu-22.04

# Inside WSL2, follow Ubuntu instructions above
```

#### Clone Repository & Setup

```bash
# Clone the repository (replace with actual repo URL)
git clone https://github.com/your-org/req-eng-platform.git
cd req-eng-platform

# Create Python virtual environment
python3.11 -m venv venv
source venv/bin/activate  # On Windows: venv\\Scripts\\activate

# Install Python dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Copy environment template
cp .env.example .env

# Edit .env with your API keys (see section 3.4)
nano .env  # or use your preferred editor
```

#### Start Services

```bash
# Start all Docker services (Postgres, Redis, ChromaDB)
docker-compose up -d

# Wait for services to initialize (~30 seconds)
sleep 30

# Run database migrations
python scripts/migrate.py

# Start the FastAPI application
uvicorn src.main:app --reload --host 0.0.0.0 --port 8000
```

**Expected Output:**
```
INFO:     Started server process [12345]
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
INFO:     Waiting for application startup.
INFO:     Application startup complete.
```

**Verify Services:**
```bash
# Check Docker containers
docker ps
# Expected: 3 containers running (postgres, redis, chromadb)

# Test API health
curl http://localhost:8000/health
# Expected: {"status": "healthy", "services": {"postgres": "up", "redis": "up", "chromadb": "up"}}
```

### 3.4 Environment Variables

Create `.env` file with the following (see `.env.example` in Appendix):

```bash
# OpenAI API Configuration
OPENAI_API_KEY=sk-proj-XXXXXXXXXXXXXXXXXXXXX
OPENAI_MODEL=gpt-4-turbo-preview
OPENAI_EMBEDDING_MODEL=text-embedding-3-small

# LangSmith Configuration (optional for local dev)
LANGSMITH_API_KEY=lsv2_pt_XXXXXXXXXXXXXXXXXXXXX
LANGSMITH_PROJECT=req-eng-local
LANGSMITH_TRACING=true
LANGSMITH_ENDPOINT=https://api.smith.langchain.com

# Database Configuration
DATABASE_URI=postgresql://reqeng:password@localhost:5432/reqeng_db
REDIS_URL=redis://localhost:6379/0
CHROMA_HOST=localhost
CHROMA_PORT=8001

# Application Configuration
APP_ENV=development
LOG_LEVEL=DEBUG
SECRET_KEY=your-secret-key-change-in-production
CORS_ORIGINS=http://localhost:3000,http://localhost:8000

# Security
JWT_SECRET=your-jwt-secret-change-in-production
JWT_ALGORITHM=HS256
JWT_EXPIRY_HOURS=24

# Agent Configuration
MAX_ITERATIONS=10
CONFIDENCE_THRESHOLD=0.60
VALIDATION_AMBIGUITY_THRESHOLD=0.75
SEMANTIC_SIMILARITY_THRESHOLD=0.85
```

**Key Configuration Notes:**
- `OPENAI_API_KEY`: Required; obtain from https://platform.openai.com/api-keys
- `LANGSMITH_TRACING`: Set to `false` to run fully offline (loses trace visibility)[61][63]
- `CONFIDENCE_THRESHOLD`: 0.60 = escalate to human if confidence < 60%
- `SEMANTIC_SIMILARITY_THRESHOLD`: 0.85 = flag requirements as duplicates if cosine similarity > 0.85

### 3.5 Recommended VSCode Extensions

```json
{
  "recommendations": [
    "ms-python.python",
    "ms-python.vscode-pylance",
    "ms-python.black-formatter",
    "ms-python.flake8",
    "charliermarsh.ruff",
    "ms-azuretools.vscode-docker",
    "humao.rest-client",
    "redhat.vscode-yaml",
    "esbenp.prettier-vscode",
    "github.copilot",
    "tamasfe.even-better-toml"
  ]
}
```

**Debugging Tips:**
1. **LangSmith Integration:** View traces at https://smith.langchain.com (requires `LANGSMITH_TRACING=true`)[63][68]
2. **Breakpoint Debugging:** Use VSCode's Python debugger with `.vscode/launch.json` (see Appendix)
3. **Docker Logs:** `docker-compose logs -f <service>` to tail logs in real-time
4. **Database Inspection:** `docker exec -it reqeng-postgres psql -U reqeng -d reqeng_db`

### 3.6 Running Offline & Cloud Migration

**Offline Mode:**
- Set `LANGSMITH_TRACING=false` in `.env`
- Use local ChromaDB (already configured)
- All LLM calls will still require OpenAI API (cache responses for testing)

**Cloud Migration Path:**
1. **Database:** Replace `DATABASE_URI` with managed Postgres (AWS RDS, Google Cloud SQL)
2. **Vector Store:** Migrate ChromaDB data to Pinecone/Weaviate Cloud
3. **Caching:** Use Redis Cloud or AWS ElastiCache
4. **Application:** Containerize with `docker build` and deploy to Kubernetes/ECS
5. **Secrets:** Move to AWS Secrets Manager / Azure Key Vault
6. **Monitoring:** Switch from local logs to CloudWatch / Datadog

---

## 4. Repository Layout & File Tree

```
req-eng-platform/
├── .env.example                    # Template environment variables
├── .env                             # Local environment (gitignored)
├── .gitignore                       # Git ignore rules
├── .vscode/
│   ├── extensions.json              # Recommended extensions
│   └── launch.json                  # Debug configurations
├── README.md                        # Project overview
├── requirements.txt                 # Python dependencies
├── pyproject.toml                   # Python project metadata
├── docker-compose.yml               # Local service orchestration
├── Dockerfile                       # Application container (for cloud)
├── scripts/
│   ├── migrate.py                   # Database schema migrations
│   ├── seed_data.py                 # Insert test data
│   └── export_rd.py                 # CLI tool to export RDs
├── src/
│   ├── main.py                      # FastAPI application entry
│   ├── config.py                    # Load environment variables
│   ├── api/
│   │   ├── __init__.py
│   │   ├── routes/
│   │   │   ├── sessions.py          # Session CRUD endpoints
│   │   │   ├── chat.py              # Chat message endpoints
│   │   │   ├── requirements.py      # Requirements CRUD
│   │   │   ├── review.py            # Review/approval endpoints
│   │   │   └── export.py            # Export RD to Markdown/PDF
│   │   └── middleware/
│   │       ├── auth.py              # JWT authentication
│   │       ├── logging.py           # Request logging
│   │       └── error_handler.py     # Global error handling
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── conversational.py        # Conversational Agent
│   │   ├── extraction.py            # Extraction Agent
│   │   ├── inference.py             # Inference Agent
│   │   ├── validation.py            # Validation Agent
│   │   ├── synthesis.py             # Synthesis/Template Agent
│   │   └── review.py                # Review & Approval Agent
│   ├── orchestrator/
│   │   ├── __init__.py
│   │   ├── graph.py                 # LangGraph state graph definition
│   │   ├── state.py                 # Shared state model
│   │   └── nodes.py                 # LangGraph node functions
│   ├── schemas/
│   │   ├── __init__.py
│   │   ├── requirement.py           # Requirement item schema (Pydantic)
│   │   ├── session.py               # Session schema
│   │   ├── chat.py                  # Chat message schema
│   │   └── rd_document.py           # Full RD document schema
│   ├── storage/
│   │   ├── __init__.py
│   │   ├── postgres.py              # Postgres adapters
│   │   ├── redis_cache.py           # Redis session cache
│   │   └── vector_store.py          # ChromaDB vector operations
│   ├── validation/
│   │   ├── __init__.py
│   │   ├── deterministic.py         # Regex-based checks
│   │   ├── semantic.py              # Embedding-based conflict detection
│   │   └── pii_detector.py          # PII detection & redaction
│   ├── templates/
│   │   ├── rd_template.md           # RD Markdown template
│   │   └── prompts/
│   │       ├── conversational.txt   # Conversational agent prompt
│   │       ├── extraction.txt       # Extraction agent prompt
│   │       ├── inference.txt        # Inference agent prompt
│   │       ├── validation.txt       # Validation agent prompt
│   │       └── synthesis.txt        # Synthesis agent prompt
│   └── utils/
│       ├── __init__.py
│       ├── langsmith_tracing.py     # LangSmith integration
│       ├── openai_client.py         # OpenAI API wrapper
│       └── pdf_generator.py         # Markdown to PDF conversion
├── tests/
│   ├── __init__.py
│   ├── fixtures/
│   │   ├── chat_sessions.json       # Sample chat data
│   │   └── expected_outputs.json    # Expected extraction results
│   ├── unit/
│   │   ├── test_extraction.py       # Unit tests for extraction
│   │   ├── test_validation.py       # Unit tests for validation
│   │   └── test_inference.py        # Unit tests for inference
│   ├── integration/
│   │   ├── test_graph_flow.py       # Full orchestrator tests
│   │   └── test_api_endpoints.py    # API integration tests
│   └── e2e/
│       └── test_full_session.py     # End-to-end conversation tests
├── frontend/                        # Optional React/Vue UI (future)
│   └── README.md
└── docs/
    ├── ARCHITECTURE.md              # Detailed architecture notes
    ├── API.md                       # API documentation
    └── DEPLOYMENT.md                # Cloud deployment guide (future)
```

**Purpose of Key Directories:**
- `src/agents/`: Individual agent implementations (each agent = 1 file)
- `src/orchestrator/`: LangGraph-based workflow orchestration[40][48]
- `src/schemas/`: Pydantic models for type safety and validation
- `src/validation/`: Deterministic and LLM-based validation logic
- `tests/`: Comprehensive test suite with fixtures

---

## 5. API & Logging Design

### 5.1 REST API Endpoints

**Base URL:** `http://localhost:8000/api/v1`

| Endpoint | Method | Purpose | Auth Required |
|----------|--------|---------|---------------|
| `/health` | GET | Health check | No |
| `/auth/login` | POST | Authenticate user, get JWT | No |
| `/sessions` | POST | Create new conversation session | Yes |
| `/sessions/{id}` | GET | Retrieve session details | Yes |
| `/sessions/{id}/messages` | POST | Send chat message to agent | Yes |
| `/sessions/{id}/messages` | GET | Get chat history | Yes |
| `/requirements/{session_id}` | GET | Get all requirements for session | Yes |
| `/requirements/{id}` | PUT | Update a requirement | Yes |
| `/rd/{session_id}` | GET | Get current RD draft | Yes |
| `/rd/{session_id}/export` | POST | Export RD (Markdown/PDF/JSON) | Yes |
| `/review/{session_id}` | POST | Submit review feedback | Yes |
| `/review/{session_id}/approve` | POST | Approve RD and lock | Yes |

### 5.2 OpenAPI Summary

```yaml
openapi: 3.0.3
info:
  title: Requirements Engineering Platform API
  version: 1.0.0
  description: Multi-agent system for conversational requirements elicitation
servers:
  - url: http://localhost:8000/api/v1
paths:
  /sessions:
    post:
      summary: Create new session
      security:
        - bearerAuth: []
      requestBody:
        content:
          application/json:
            schema:
              type: object
              properties:
                project_name:
                  type: string
                user_id:
                  type: string
      responses:
        '201':
          description: Session created
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/Session'
  /sessions/{id}/messages:
    post:
      summary: Send message to agent
      security:
        - bearerAuth: []
      parameters:
        - name: id
          in: path
          required: true
          schema:
            type: string
      requestBody:
        content:
          application/json:
            schema:
              type: object
              properties:
                message:
                  type: string
                attachments:
                  type: array
                  items:
                    type: string
      responses:
        '200':
          description: Agent response
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/ChatMessage'
components:
  securitySchemes:
    bearerAuth:
      type: http
      scheme: bearer
      bearerFormat: JWT
  schemas:
    Session:
      type: object
      properties:
        id:
          type: string
        project_name:
          type: string
        created_at:
          type: string
          format: date-time
        status:
          type: string
          enum: [active, reviewing, approved, archived]
    ChatMessage:
      type: object
      properties:
        id:
          type: string
        session_id:
          type: string
        role:
          type: string
          enum: [user, assistant, system]
        content:
          type: string
        timestamp:
          type: string
          format: date-time
```

### 5.3 Authentication & Authorization

**Local Dev Strategy:**
- Use **JWT tokens** with HS256 (symmetric signing)
- Mock user identities (no real OAuth for MVP)
- Simple RBAC: `admin`, `reviewer`, `stakeholder`

**Implementation:**
```python
# src/api/middleware/auth.py
from fastapi import HTTPException, Security
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from jose import jwt, JWTError
from datetime import datetime, timedelta
import os

security = HTTPBearer()

def create_access_token(user_id: str, role: str) -> str:
    payload = {
        "sub": user_id,
        "role": role,
        "exp": datetime.utcnow() + timedelta(hours=int(os.getenv("JWT_EXPIRY_HOURS", 24)))
    }
    return jwt.encode(payload, os.getenv("JWT_SECRET"), algorithm=os.getenv("JWT_ALGORITHM"))

def verify_token(credentials: HTTPAuthorizationCredentials = Security(security)) -> dict:
    try:
        payload = jwt.decode(credentials.credentials, os.getenv("JWT_SECRET"), algorithms=[os.getenv("JWT_ALGORITHM")])
        return payload
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid token")
```

**Mock Login Endpoint:**
```python
# src/api/routes/auth.py
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

router = APIRouter(prefix="/auth", tags=["auth"])

class LoginRequest(BaseModel):
    username: str
    password: str

@router.post("/login")
async def login(req: LoginRequest):
    # Mock authentication - replace with real logic
    if req.username == "admin" and req.password == "admin":
        token = create_access_token(user_id="user_1", role="admin")
        return {"access_token": token, "token_type": "bearer"}
    raise HTTPException(status_code=401, detail="Invalid credentials")
```

### 5.4 Request/Response Examples

**POST /sessions**
```json
// Request
{
  "project_name": "E-commerce Platform v2",
  "user_id": "user_123"
}

// Response (201)
{
  "id": "sess_abc123",
  "project_name": "E-commerce Platform v2",
  "created_at": "2025-11-16T15:30:00Z",
  "status": "active"
}
```

**POST /sessions/sess_abc123/messages**
```json
// Request
{
  "message": "We need the app to load quickly for users in India."
}

// Response (200)
{
  "id": "msg_xyz789",
  "session_id": "sess_abc123",
  "role": "assistant",
  "content": "I've captured that as a performance requirement. Can you specify an acceptable load time threshold? For example, 'Cold start time ≤ 2.5 seconds on 4G networks in India'?",
  "timestamp": "2025-11-16T15:30:05Z",
  "agent_metadata": {
    "confidence": 0.88,
    "extracted_requirements": [
      {
        "id": "REQ-001",
        "title": "App cold start time",
        "type": "non-functional",
        "status": "draft"
      }
    ]
  }
}
```

### 5.5 Logging Design

**Structured Logging Schema:**
```python
# src/utils/logging.py
import logging
import json
from datetime import datetime
import uuid

class StructuredLogger:
    def __init__(self, name: str):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(os.getenv("LOG_LEVEL", "INFO"))
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter('%(message)s'))
        self.logger.addHandler(handler)

    def log(self, level: str, message: str, **kwargs):
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": level,
            "message": message,
            "correlation_id": kwargs.get("correlation_id", str(uuid.uuid4())),
            "service": "req-eng-platform",
            **kwargs
        }
        self.logger.log(getattr(logging, level.upper()), json.dumps(log_entry))
```

**Log Levels:**
- `DEBUG`: Detailed trace of execution flow
- `INFO`: Key events (session created, requirement extracted)
- `WARN`: Recoverable issues (low confidence, validation warnings)
- `ERROR`: Failures requiring attention (LLM API errors, DB failures)

**Example Log Entry:**
```json
{
  "timestamp": "2025-11-16T15:30:05.123Z",
  "level": "INFO",
  "message": "Requirement extracted",
  "correlation_id": "corr_abc123",
  "service": "req-eng-platform",
  "agent": "extraction",
  "session_id": "sess_abc123",
  "requirement_id": "REQ-001",
  "confidence": 0.88,
  "latency_ms": 1245
}
```

**Storage:**
- Local: Write to `logs/app.log` (rotated daily)
- Production: Forward to CloudWatch/Datadog/ELK

### 5.6 LangSmith Tracing Integration

**Setup:**
```python
# src/utils/langsmith_tracing.py
import os
from langsmith import Client
from langsmith.run_helpers import traceable

def configure_langsmith():
    if os.getenv("LANGSMITH_TRACING") == "true":
        client = Client(
            api_key=os.getenv("LANGSMITH_API_KEY"),
            api_url=os.getenv("LANGSMITH_ENDPOINT")
        )
        return client
    return None

# Decorator for tracing functions
@traceable(name="extract_requirements", project_name=os.getenv("LANGSMITH_PROJECT"))
def extract_requirements(text: str) -> list:
    # Agent logic here
    pass
```

**What to Log as Spans/Events:**[61][63]
- Every LLM call (prompt, completion, tokens, latency)
- Agent decisions (which tool invoked, confidence scores)
- Validation results (pass/fail, issues detected)
- Human feedback events (approve/revise, comments)

**Example Trace Attributes:**
```python
from langsmith import trace

with trace(name="synthesis_agent", inputs={"requirements": req_list}):
    rd_draft = synthesis_agent.generate_rd(req_list)
    trace.log_output({"rd_draft": rd_draft, "token_count": len(rd_draft.split())})
```

### 5.7 Local Log Viewer

**Option 1: jq + tail**
```bash
tail -f logs/app.log | jq -C '.message, .level, .correlation_id'
```

**Option 2: Grafana + Loki (Docker Compose)**
```yaml
# Add to docker-compose.yml
loki:
  image: grafana/loki:2.9.0
  ports:
    - "3100:3100"
  command: -config.file=/etc/loki/local-config.yaml

grafana:
  image: grafana/grafana:10.0.0
  ports:
    - "3001:3000"
  environment:
    - GF_AUTH_ANONYMOUS_ENABLED=true
```

**Rationale:** Loki + Grafana provide queryable log exploration with minimal overhead for local dev.

---

## 6. Persistence, Vector DB, and Knowledge Store

### 6.1 Vector DB Configuration

**Choice: ChromaDB**
**Rationale:** Lightweight, embeddable, supports local persistence, easy migration to cloud[40].

**Docker Compose Configuration:**
```yaml
chromadb:
  image: chromadb/chroma:0.4.15
  ports:
    - "8001:8000"
  volumes:
    - ./data/chroma:/chroma/chroma
  environment:
    - ALLOW_RESET=true
```

**Python Client Setup:**
```python
# src/storage/vector_store.py
import chromadb
from chromadb.config import Settings

class VectorStore:
    def __init__(self):
        self.client = chromadb.HttpClient(
            host=os.getenv("CHROMA_HOST", "localhost"),
            port=int(os.getenv("CHROMA_PORT", 8001))
        )
        self.collection = self.client.get_or_create_collection(
            name="requirements",
            metadata={"hnsw:space": "cosine"}
        )

    def add_requirement(self, req_id: str, text: str, embedding: list[float], metadata: dict):
        self.collection.add(
            ids=[req_id],
            embeddings=[embedding],
            documents=[text],
            metadatas=[metadata]
        )

    def find_similar(self, query_embedding: list[float], threshold: float = 0.85, n_results: int = 5):
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results
        )
        # Filter by similarity threshold
        return [r for r in results["metadatas"][0] if r.get("distance", 1.0) <= (1 - threshold)]
```

**Migration to Cloud:**
- Replace ChromaDB with Pinecone/Weaviate client
- Export data: `chromadb_client.get_collection().get()` → transform → cloud upload

### 6.2 Database Schemas

**Postgres Tables:**

```sql
-- sessions table
CREATE TABLE sessions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    project_name VARCHAR(255) NOT NULL,
    user_id VARCHAR(100) NOT NULL,
    status VARCHAR(50) DEFAULT 'active',
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- chat_messages table
CREATE TABLE chat_messages (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id UUID REFERENCES sessions(id) ON DELETE CASCADE,
    role VARCHAR(20) NOT NULL CHECK (role IN ('user', 'assistant', 'system')),
    content TEXT NOT NULL,
    timestamp TIMESTAMP DEFAULT NOW(),
    metadata JSONB
);

-- requirements table
CREATE TABLE requirements (
    id VARCHAR(50) PRIMARY KEY,
    session_id UUID REFERENCES sessions(id) ON DELETE CASCADE,
    title VARCHAR(500) NOT NULL,
    type VARCHAR(50) NOT NULL,
    actor VARCHAR(200),
    action TEXT,
    condition TEXT,
    acceptance_criteria JSONB,
    priority VARCHAR(20),
    confidence FLOAT CHECK (confidence >= 0 AND confidence <= 1),
    inferred BOOLEAN DEFAULT FALSE,
    rationale TEXT,
    source_refs JSONB,
    created_at TIMESTAMP DEFAULT NOW()
);

-- rd_documents table
CREATE TABLE rd_documents (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id UUID REFERENCES sessions(id) ON DELETE CASCADE,
    version INT NOT NULL,
    content TEXT NOT NULL,
    format VARCHAR(20) CHECK (format IN ('markdown', 'json', 'pdf')),
    status VARCHAR(20) DEFAULT 'draft',
    created_at TIMESTAMP DEFAULT NOW()
);

-- audit_logs table
CREATE TABLE audit_logs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id UUID REFERENCES sessions(id),
    user_id VARCHAR(100),
    action VARCHAR(100) NOT NULL,
    entity_type VARCHAR(50),
    entity_id VARCHAR(100),
    changes JSONB,
    timestamp TIMESTAMP DEFAULT NOW()
);

-- indexes
CREATE INDEX idx_sessions_user_id ON sessions(user_id);
CREATE INDEX idx_chat_messages_session ON chat_messages(session_id, timestamp);
CREATE INDEX idx_requirements_session ON requirements(session_id);
CREATE INDEX idx_audit_logs_session ON audit_logs(session_id, timestamp);
```

**Pydantic Schemas:**

```python
# src/schemas/requirement.py
from pydantic import BaseModel, Field, confloat
from typing import List, Optional
from enum import Enum

class RequirementType(str, Enum):
    FUNCTIONAL = "functional"
    NON_FUNCTIONAL = "non-functional"
    BUSINESS = "business"
    SECURITY = "security"
    DATA = "data"
    INTERFACE = "interface"
    CONSTRAINT = "constraint"

class RequirementItem(BaseModel):
    id: str = Field(..., pattern=r"^REQ-\d{3,}$")
    title: str = Field(..., max_length=500)
    type: RequirementType
    actor: str = Field(..., max_length=200)
    action: str
    condition: Optional[str] = None
    acceptance_criteria: List[str] = Field(..., min_items=1)
    priority: str = Field(..., pattern=r"^(low|medium|high|must)$")
    confidence: confloat(ge=0.0, le=1.0)
    inferred: bool = False
    rationale: str
    source_refs: List[str]

    class Config:
        json_schema_extra = {
            "example": {
                "id": "REQ-001",
                "title": "App cold start time",
                "type": "non-functional",
                "actor": "end-user",
                "action": "app starts and ready to accept input",
                "condition": "on mobile devices with 4G in India",
                "acceptance_criteria": ["Cold start time <= 2.5 seconds on median devices with 4G in India"],
                "priority": "high",
                "confidence": 0.88,
                "inferred": False,
                "rationale": "User explicitly requested app to 'load quickly' for users in India. Converted to measurable latency target.",
                "source_refs": ["chat_turn_12"]
            }
        }
```

### 6.3 Backup/Export Strategy

**Export RD to Formats:**
```python
# src/utils/pdf_generator.py
from markdown import markdown
from weasyprint import HTML
import json

def export_rd(session_id: str, format: str) -> str:
    # Fetch RD from database
    rd = fetch_rd_from_db(session_id)

    if format == "json":
        return json.dumps(rd, indent=2)
    elif format == "markdown":
        return rd["content"]
    elif format == "pdf":
        html_content = markdown(rd["content"])
        pdf_file = f"exports/{session_id}_v{rd['version']}.pdf"
        HTML(string=html_content).write_pdf(pdf_file)
        return pdf_file
```

**Backup Command:**
```bash
# Backup Postgres database
docker exec reqeng-postgres pg_dump -U reqeng reqeng_db > backups/db_backup_$(date +%Y%m%d).sql

# Backup ChromaDB data
tar -czf backups/chroma_backup_$(date +%Y%m%d).tar.gz data/chroma/
```

---

## 7. Orchestrator & Agent Design (High-Level)

### 7.1 LangGraph Structure

**Nodes (Agent Functions):**
1. `greet_user` - Initialize conversation, collect project metadata
2. `elicit_requirements` - Conversational agent handles user chat
3. `extract_requirements` - Parse chat into structured requirements
4. `infer_requirements` - Propose implicit requirements, edge cases
5. `validate_requirements` - Run deterministic + LLM checks
6. `synthesize_rd` - Generate RD from validated requirements
7. `review_cycle` - Human-in-the-loop approval/revision
8. `finalize_rd` - Lock RD, export, log approval

**Edges (Transitions):**
- `START → greet_user`
- `greet_user → elicit_requirements`
- `elicit_requirements → extract_requirements` (when user provides input)
- `extract_requirements → infer_requirements`
- `infer_requirements → validate_requirements`
- `validate_requirements → synthesize_rd` (if validation passes)
- `validate_requirements → elicit_requirements` (if validation fails, ask clarification)
- `synthesize_rd → review_cycle`
- `review_cycle → elicit_requirements` (if user requests revision)
- `review_cycle → finalize_rd` (if user approves)
- `finalize_rd → END`

**Conditional Branching:**[48]
```python
def should_ask_clarification(state: State) -> str:
    if state["validation_issues"] and state["confidence"] < 0.75:
        return "elicit_requirements"
    return "synthesize_rd"
```

### 7.2 State Model

```python
# src/orchestrator/state.py
from typing import TypedDict, List, Dict, Optional
from src.schemas.requirement import RequirementItem
from src.schemas.chat import ChatMessage

class State(TypedDict):
    # Session metadata
    session_id: str
    project_name: str
    user_id: str

    # Conversation state
    chat_history: List[ChatMessage]
    current_turn: int

    # Requirements tracking
    requirements: List[RequirementItem]
    inferred_requirements: List[RequirementItem]

    # Validation state
    validation_issues: List[Dict]
    confidence: float

    # RD generation
    rd_draft: Optional[str]
    rd_version: int

    # Review state
    review_feedback: Optional[str]
    approval_status: str  # "pending", "approved", "revision_requested"

    # Agent metadata
    last_agent: str
    iterations: int
```

### 7.3 Mermaid Sequence Diagram (Happy Path)

```mermaid
sequenceDiagram
    participant User
    participant API
    participant Orchestrator
    participant ConvAgent
    participant ExtractAgent
    participant InferAgent
    participant ValidateAgent
    participant SynthesisAgent
    participant ChromaDB
    participant Postgres

    User->>API: POST /sessions/{id}/messages
    API->>Orchestrator: Invoke graph with user message
    Orchestrator->>ConvAgent: Process user input
    ConvAgent->>Orchestrator: Context-aware response

    alt Requirements detected
        Orchestrator->>ExtractAgent: Extract structured requirements
        ExtractAgent->>Postgres: Store requirements
        ExtractAgent->>ChromaDB: Store embeddings
        ExtractAgent->>Orchestrator: Return extracted items

        Orchestrator->>InferAgent: Infer implicit requirements
        InferAgent->>Orchestrator: Return inferred items (with confidence)

        Orchestrator->>ValidateAgent: Validate all requirements
        ValidateAgent->>Orchestrator: Return validation report

        alt Validation passes
            Orchestrator->>SynthesisAgent: Generate RD draft
            SynthesisAgent->>Postgres: Store RD version
            SynthesisAgent->>Orchestrator: RD draft ready
        else Validation fails
            Orchestrator->>ConvAgent: Ask clarifying questions
        end
    end

    Orchestrator->>API: Return agent response
    API->>User: Display message + RD preview
```

---

## 8. Agent-by-Agent Low Level Design

### 8.1 Conversational Agent (Frontline)

**Purpose:** Manages user interaction, elicits requirements via guided conversation, maintains context.

**Responsibilities:**
- Greet user, explain process
- Ask clarifying questions when ambiguity detected
- Maintain conversation flow (don't overwhelm user with questions)
- Hand off to Extraction when sufficient info collected

**Input Schema:**
```python
class ConversationalInput(BaseModel):
    chat_history: List[ChatMessage]
    current_message: str
    session_metadata: Dict
```

**Output Schema:**
```python
class ConversationalOutput(BaseModel):
    response: str
    next_action: str  # "continue_eliciting", "extract_requirements", "clarify"
    confidence: float
    clarification_questions: Optional[List[str]]
```

**Prompt Template:**
```txt
# src/templates/prompts/conversational.txt
You are an expert Requirements Architect Agent specializing in eliciting software requirements through conversation. Your goal is to guide stakeholders through structured, non-invasive requirements discovery.

## Your Responsibilities:
- Ask targeted clarifying questions only when necessary (aim for <3 questions per major topic)
- Listen for explicit requirements (user states what they want)
- Note areas for inference (implicit needs, edge cases)
- Maintain a friendly, professional tone
- Progress the conversation toward completeness

## Conversation Guidelines:
- If user mentions performance ("fast", "quick"), ask for measurable thresholds
- If user mentions data, ask about volume, retention, privacy constraints
- If user mentions actors, clarify roles and permissions
- Always confirm understanding by paraphrasing back to user

## Current Session Context:
Project: {project_name}
Turn: {current_turn}
Requirements captured so far: {requirement_count}

## Chat History:
{chat_history}

## User's Latest Message:
{user_message}

## Your Response:
Provide a helpful response that either:
1. Acknowledges captured requirements and asks ONE clarifying question if needed, OR
2. Suggests we have enough information to generate a draft requirements document

Response format:
<thought>Your reasoning about what to do next</thought>
<response>Your message to the user</response>
<next_action>continue_eliciting | extract_requirements | clarify</next_action>
<confidence>0.0-1.0</confidence>
```

**LangChain Implementation:**
```python
# src/agents/conversational.py
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from src.schemas.chat import ConversationalOutput
import os

class ConversationalAgent:
    def __init__(self):
        self.llm = ChatOpenAI(
            model=os.getenv("OPENAI_MODEL", "gpt-4-turbo-preview"),
            temperature=0.7,  # Slightly creative for natural conversation
            max_tokens=500,
            openai_api_key=os.getenv("OPENAI_API_KEY")
        )
        self.parser = PydanticOutputParser(pydantic_object=ConversationalOutput)

        with open("src/templates/prompts/conversational.txt") as f:
            template = f.read()

        self.prompt = PromptTemplate(
            template=template,
            input_variables=["project_name", "current_turn", "requirement_count", "chat_history", "user_message"],
            partial_variables={"format_instructions": self.parser.get_format_instructions()}
        )

    def invoke(self, state: dict) -> ConversationalOutput:
        chain = self.prompt | self.llm | self.parser
        result = chain.invoke({
            "project_name": state["project_name"],
            "current_turn": state["current_turn"],
            "requirement_count": len(state["requirements"]),
            "chat_history": self._format_chat_history(state["chat_history"]),
            "user_message": state["chat_history"][-1]["content"]
        })
        return result

    def _format_chat_history(self, history: list) -> str:
        return "\n".join([f"{msg['role'].upper()}: {msg['content']}" for msg in history[-5:]])
```

**LangGraph Node Wrapper:**
```python
# src/orchestrator/nodes.py
from src.agents.conversational import ConversationalAgent

conversational_agent = ConversationalAgent()

def conversational_node(state: State) -> State:
    result = conversational_agent.invoke(state)

    # Update state
    state["chat_history"].append({
        "role": "assistant",
        "content": result.response,
        "timestamp": datetime.utcnow().isoformat()
    })
    state["confidence"] = result.confidence
    state["last_agent"] = "conversational"

    return state
```

**Confidence & Escalation:**
- Confidence < 0.60: Flag for human review of conversation flow
- If user provides contradictory information: Ask for clarification explicitly

**Test Fixtures:**
```json
{
  "test_case_1": {
    "input": {
      "chat_history": [
        {"role": "user", "content": "We need a mobile app for e-commerce"}
      ],
      "current_turn": 1,
      "project_name": "E-commerce Mobile App"
    },
    "expected_output": {
      "next_action": "continue_eliciting",
      "confidence": 0.85,
      "response_contains": ["What platforms", "iOS", "Android", "both"]
    }
  },
  "test_case_2": {
    "input": {
      "chat_history": [
        {"role": "user", "content": "App should be fast"}
      ],
      "current_turn": 5
    },
    "expected_output": {
      "next_action": "clarify",
      "clarification_questions": ["What is an acceptable load time?"]
    }
  }
}
```

---

### 8.2 Extraction Agent

**Purpose:** Parse chat messages and attachments into structured `RequirementItem` objects conforming to JSON schema.

**Responsibilities:**
- Extract actors, actions, conditions from natural language
- Map to requirement types (functional, non-functional, etc.)
- Generate measurable acceptance criteria where possible
- Link to source chat turns (traceability)

**Input Schema:**
```python
class ExtractionInput(BaseModel):
    chat_history: List[ChatMessage]
    attachments: Optional[List[str]]  # File paths or URLs
```

**Output Schema:**
```python
class ExtractionOutput(BaseModel):
    requirements: List[RequirementItem]
    confidence: float
    ambiguous_items: List[str]  # Items needing clarification
```

**Prompt Template:**
```txt
# src/templates/prompts/extraction.txt
You are a Requirements Extraction Agent. Your task is to parse conversational text and extract structured software requirements.

## Extraction Rules:
1. Every requirement MUST have: id, title, type, actor, action, acceptance_criteria, priority
2. Generate IDs in format: REQ-001, REQ-002, etc.
3. For performance requirements, convert vague terms to measurable thresholds:
   - "fast" → "< 2 seconds"
   - "scalable" → "handles 10,000 concurrent users"
4. Link every requirement to source chat turn IDs for traceability
5. If a statement is ambiguous, flag it in `ambiguous_items`

## Requirement Types:
- functional: User actions, features, capabilities
- non-functional: Performance, scalability, usability, security
- business: ROI, compliance, market constraints
- security: Authentication, authorization, data protection
- data: Schemas, retention, privacy
- interface: API contracts, UI/UX, integrations
- constraint: Budget, timeline, technology restrictions

## Output Format (JSON):
{format_instructions}

## Chat History:
{chat_history}

## Extracted Requirements:
Extract all identifiable requirements from the conversation above. Be thorough but conservative—only extract what was explicitly or clearly implied.
```

**LangChain Implementation:**
```python
# src/agents/extraction.py
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from src.schemas.requirement import RequirementItem, ExtractionOutput
import os

class ExtractionAgent:
    def __init__(self):
        self.llm = ChatOpenAI(
            model=os.getenv("OPENAI_MODEL"),
            temperature=0.2,  # Low temperature for structured extraction
            max_tokens=2000,
            openai_api_key=os.getenv("OPENAI_API_KEY")
        )
        self.parser = JsonOutputParser(pydantic_object=ExtractionOutput)

        with open("src/templates/prompts/extraction.txt") as f:
            template = f.read()

        self.prompt = PromptTemplate(
            template=template,
            input_variables=["chat_history"],
            partial_variables={"format_instructions": self.parser.get_format_instructions()}
        )

    def invoke(self, state: dict) -> ExtractionOutput:
        chain = self.prompt | self.llm | self.parser
        result = chain.invoke({
            "chat_history": self._format_chat_history(state["chat_history"])
        })

        # Post-process: assign source_refs based on chat turn indices
        for idx, req in enumerate(result["requirements"]):
            req["source_refs"] = self._find_source_turns(req, state["chat_history"])

        return ExtractionOutput(**result)

    def _format_chat_history(self, history: list) -> str:
        return "\n".join([f"[Turn {i}] {msg['role'].upper()}: {msg['content']}"
                         for i, msg in enumerate(history)])

    def _find_source_turns(self, requirement: dict, history: list) -> list:
        # Simple keyword matching; can enhance with embeddings
        keywords = requirement["title"].lower().split()
        source_turns = []
        for i, msg in enumerate(history):
            if any(kw in msg["content"].lower() for kw in keywords):
                source_turns.append(f"chat_turn_{i}")
        return source_turns
```

**LangGraph Node Wrapper:**
```python
def extraction_node(state: State) -> State:
    result = extraction_agent.invoke(state)

    # Update state
    state["requirements"].extend(result.requirements)
    state["confidence"] = result.confidence
    state["last_agent"] = "extraction"

    # Store in database
    for req in result.requirements:
        store_requirement(state["session_id"], req)

    return state
```

**Confidence & Escalation:**
- Confidence < 0.75: Flag ambiguous items for user clarification
- If >50% of extracted items have ambiguity flags: Return to Conversational Agent

**Test Fixtures:**
```json
{
  "test_extraction_1": {
    "input": {
      "chat_history": [
        {"role": "user", "content": "Users should be able to log in with email and password."},
        {"role": "user", "content": "The login should complete in under 2 seconds."}
      ]
    },
    "expected_output": {
      "requirements": [
        {
          "id": "REQ-001",
          "title": "User authentication with email and password",
          "type": "functional",
          "actor": "user",
          "action": "log in using email and password credentials",
          "acceptance_criteria": ["User can enter email and password", "System authenticates credentials", "User is redirected to dashboard on success"],
          "priority": "must",
          "confidence": 0.95,
          "inferred": false,
          "source_refs": ["chat_turn_0"]
        },
        {
          "id": "REQ-002",
          "title": "Login response time",
          "type": "non-functional",
          "actor": "system",
          "action": "completes authentication and redirects user",
          "condition": "under normal load",
          "acceptance_criteria": ["Login completes in < 2 seconds for 95% of requests"],
          "priority": "high",
          "confidence": 0.90,
          "inferred": false,
          "source_refs": ["chat_turn_1"]
        }
      ],
      "confidence": 0.92,
      "ambiguous_items": []
    }
  }
}
```

---

### 8.3 Inference Agent

**Purpose:** Propose implicit requirements, edge cases, and non-functional constraints not explicitly stated by user.

**Responsibilities:**
- Analyze explicit requirements for missing concerns (security, error handling, data validation)
- Propose edge cases (null inputs, concurrent access, failure scenarios)
- Infer common non-functional requirements (logging, monitoring, backup)
- Mark all outputs as `inferred: true` with confidence scores

**Input Schema:**
```python
class InferenceInput(BaseModel):
    requirements: List[RequirementItem]
    project_context: Dict
```

**Output Schema:**
```python
class InferenceOutput(BaseModel):
    inferred_requirements: List[RequirementItem]
    confidence: float
    rationale: str
```

**Prompt Template:**
```txt
# src/templates/prompts/inference.txt
You are an Inference Agent specializing in identifying implicit requirements and edge cases from explicit requirements.

## Your Task:
Given a list of explicit requirements, identify:
1. Missing error handling and validation logic
2. Security concerns (authentication, authorization, data encryption)
3. Edge cases (null inputs, boundary conditions, race conditions)
4. Non-functional requirements (logging, monitoring, backup, disaster recovery)
5. Compliance needs (GDPR, HIPAA, accessibility)

## Inference Rules:
- For every user-facing action, infer error handling requirements
- For data storage, infer backup and retention requirements
- For authentication, infer password policy and session management
- For APIs, infer rate limiting and versioning
- Mark ALL outputs as inferred: true with confidence scores
- Provide clear rationale linking inference to explicit requirements

## Example:
Explicit: "User can upload profile photo"
Inferred:
  - REQ-INF-001: System validates photo format (JPEG, PNG) and size (< 5MB)
  - REQ-INF-002: System scans uploaded photos for malware
  - REQ-INF-003: System stores photos with access controls (user-only visibility)

## Input Requirements:
{requirements_json}

## Project Context:
Domain: {domain}
User base: {user_base}
Compliance: {compliance_needs}

## Inferred Requirements:
Generate inferred requirements as JSON array with confidence scores (0.5-0.9 typical).
```

**LangChain Implementation:**
```python
# src/agents/inference.py
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from src.schemas.requirement import InferenceOutput
import json

class InferenceAgent:
    def __init__(self):
        self.llm = ChatOpenAI(
            model=os.getenv("OPENAI_MODEL"),
            temperature=0.4,  # Moderate creativity for inference
            max_tokens=2000,
            openai_api_key=os.getenv("OPENAI_API_KEY")
        )
        self.parser = JsonOutputParser(pydantic_object=InferenceOutput)

        with open("src/templates/prompts/inference.txt") as f:
            template = f.read()

        self.prompt = PromptTemplate(
            template=template,
            input_variables=["requirements_json", "domain", "user_base", "compliance_needs"],
            partial_variables={"format_instructions": self.parser.get_format_instructions()}
        )

    def invoke(self, state: dict) -> InferenceOutput:
        chain = self.prompt | self.llm | self.parser
        result = chain.invoke({
            "requirements_json": json.dumps([req.dict() for req in state["requirements"]], indent=2),
            "domain": state.get("domain", "general software"),
            "user_base": state.get("user_base", "unknown"),
            "compliance_needs": state.get("compliance_needs", "none specified")
        })

        # Mark all as inferred
        for req in result["inferred_requirements"]:
            req["inferred"] = True
            req["id"] = f"REQ-INF-{len(state['inferred_requirements']) + 1:03d}"

        return InferenceOutput(**result)
```

**LangGraph Node Wrapper:**
```python
def inference_node(state: State) -> State:
    result = inference_agent.invoke(state)

    state["inferred_requirements"].extend(result.inferred_requirements)
    state["confidence"] = min(state["confidence"], result.confidence)
    state["last_agent"] = "inference"

    return state
```

**Confidence & Escalation:**
- Confidence < 0.60: Present inferred items to user for explicit confirmation
- Present all inferred requirements in UI with "[INFERRED]" badge

**Test Fixtures:**
```json
{
  "test_inference_1": {
    "input": {
      "requirements": [
        {
          "id": "REQ-001",
          "title": "User login with email/password",
          "type": "functional"
        }
      ],
      "domain": "e-commerce",
      "compliance_needs": "GDPR"
    },
    "expected_inferences": [
      {
        "title": "Password complexity policy",
        "type": "security",
        "rationale": "Login functionality requires secure password storage and complexity rules"
      },
      {
        "title": "Failed login attempt limit",
        "type": "security",
        "rationale": "Prevent brute force attacks"
      },
      {
        "title": "User data retention policy",
        "type": "business",
        "rationale": "GDPR compliance requires data retention and deletion policies"
      }
    ]
  }
}
```

---

### 8.4 Validation Agent

**Purpose:** Run deterministic and LLM-based checks to ensure requirements are clear, testable, complete, and non-conflicting.

**Responsibilities:**
- Detect ambiguous verbs ("optimize", "improve" without metrics)
- Ensure every requirement has acceptance criteria
- Check for conflicting requirements via semantic similarity
- Validate traceability (every requirement links to source)
- Flag missing actors, actions, or conditions

**Input Schema:**
```python
class ValidationInput(BaseModel):
    requirements: List[RequirementItem]
    inferred_requirements: List[RequirementItem]
```

**Output Schema:**
```python
class ValidationOutput(BaseModel):
    validation_passed: bool
    issues: List[ValidationIssue]
    confidence: float

class ValidationIssue(BaseModel):
    requirement_id: str
    severity: str  # "error", "warning", "info"
    issue_type: str
    message: str
    suggestion: Optional[str]
```

**Deterministic Checks:**
```python
# src/validation/deterministic.py
import re
from typing import List
from src.schemas.requirement import RequirementItem, ValidationIssue

AMBIGUOUS_VERBS = ["optimize", "improve", "enhance", "better", "good", "fast", "secure"]

def check_ambiguous_verbs(req: RequirementItem) -> List[ValidationIssue]:
    issues = []
    text = f"{req.action} {req.condition or ''}".lower()
    for verb in AMBIGUOUS_VERBS:
        if verb in text and not any(char.isdigit() for char in text):
            issues.append(ValidationIssue(
                requirement_id=req.id,
                severity="warning",
                issue_type="ambiguous_verb",
                message=f"Requirement uses ambiguous term '{verb}' without measurable threshold",
                suggestion=f"Replace '{verb}' with specific metric (e.g., '< 2 seconds', '99.9% uptime')"
            ))
    return issues

def check_acceptance_criteria(req: RequirementItem) -> List[ValidationIssue]:
    issues = []
    if not req.acceptance_criteria or len(req.acceptance_criteria) == 0:
        issues.append(ValidationIssue(
            requirement_id=req.id,
            severity="error",
            issue_type="missing_acceptance_criteria",
            message="Requirement has no acceptance criteria",
            suggestion="Add testable success criteria"
        ))
    return issues

def check_traceability(req: RequirementItem) -> List[ValidationIssue]:
    issues = []
    if not req.source_refs or len(req.source_refs) == 0:
        issues.append(ValidationIssue(
            requirement_id=req.id,
            severity="warning",
            issue_type="missing_traceability",
            message="Requirement not linked to source conversation",
            suggestion="Add source_refs to chat turns"
        ))
    return issues
```

**Semantic Conflict Detection:**
```python
# src/validation/semantic.py
from src.storage.vector_store import VectorStore
from src.utils.openai_client import get_embedding
import os

class SemanticValidator:
    def __init__(self):
        self.vector_store = VectorStore()
        self.threshold = float(os.getenv("SEMANTIC_SIMILARITY_THRESHOLD", 0.85))

    def find_conflicts(self, requirements: List[RequirementItem]) -> List[ValidationIssue]:
        issues = []

        for req in requirements:
            # Get embedding for requirement text
            text = f"{req.title} {req.action}"
            embedding = get_embedding(text)

            # Find similar requirements
            similar = self.vector_store.find_similar(embedding, threshold=self.threshold)

            for sim_req in similar:
                if sim_req["id"] != req.id:
                    issues.append(ValidationIssue(
                        requirement_id=req.id,
                        severity="warning",
                        issue_type="potential_conflict",
                        message=f"Requirement may conflict with {sim_req['id']} (similarity: {sim_req['distance']:.2f})",
                        suggestion="Review both requirements for consistency"
                    ))

        return issues
```

**LangChain Implementation:**
```python
# src/agents/validation.py
from src.validation.deterministic import *
from src.validation.semantic import SemanticValidator

class ValidationAgent:
    def __init__(self):
        self.semantic_validator = SemanticValidator()

    def invoke(self, state: dict) -> ValidationOutput:
        all_issues = []
        all_reqs = state["requirements"] + state["inferred_requirements"]

        # Run deterministic checks
        for req in all_reqs:
            all_issues.extend(check_ambiguous_verbs(req))
            all_issues.extend(check_acceptance_criteria(req))
            all_issues.extend(check_traceability(req))

        # Run semantic conflict detection
        all_issues.extend(self.semantic_validator.find_conflicts(all_reqs))

        # Calculate validation status
        error_count = len([i for i in all_issues if i.severity == "error"])
        validation_passed = error_count == 0

        # Calculate confidence (lower if many warnings)
        warning_count = len([i for i in all_issues if i.severity == "warning"])
        confidence = max(0.5, 1.0 - (warning_count * 0.05))

        return ValidationOutput(
            validation_passed=validation_passed,
            issues=all_issues,
            confidence=confidence
        )
```

**LangGraph Node Wrapper:**
```python
def validation_node(state: State) -> State:
    result = validation_agent.invoke(state)

    state["validation_issues"] = result.issues
    state["confidence"] = result.confidence
    state["last_agent"] = "validation"

    return state

def should_ask_clarification(state: State) -> str:
    """Conditional edge: validation pass → synthesis, fail → clarify"""
    error_count = len([i for i in state["validation_issues"] if i["severity"] == "error"])
    if error_count > 0 or state["confidence"] < 0.75:
        return "elicit_requirements"
    return "synthesize_rd"
```

**Test Fixtures:**
```json
{
  "test_validation_ambiguous": {
    "input": {
      "requirements": [
        {
          "id": "REQ-001",
          "action": "System should optimize performance"
        }
      ]
    },
    "expected_issues": [
      {
        "severity": "warning",
        "issue_type": "ambiguous_verb",
        "message": "uses ambiguous term 'optimize'"
      }
    ]
  },
  "test_validation_conflict": {
    "input": {
      "requirements": [
        {"id": "REQ-001", "title": "User can delete account"},
        {"id": "REQ-002", "title": "User can permanently remove account"}
      ]
    },
    "expected_issues": [
      {
        "severity": "warning",
        "issue_type": "potential_conflict",
        "message": "may conflict with REQ-002"
      }
    ]
  }
}
```

---

### 8.5 Synthesis/Template Agent

**Purpose:** Generate Requirements Document from validated requirements using industry-standard template.

**Responsibilities:**
- Map requirements to RD sections (Functional, Non-Functional, etc.)
- Generate traceability matrix
- Format output as Markdown, JSON, or PDF
- Maintain version history

**Input Schema:**
```python
class SynthesisInput(BaseModel):
    requirements: List[RequirementItem]
    inferred_requirements: List[RequirementItem]
    project_metadata: Dict
```

**Output Schema:**
```python
class SynthesisOutput(BaseModel):
    rd_content: str  # Markdown formatted
    rd_json: Dict
    version: int
    sections: List[str]
```

**Template Structure (see section 9 for full template):**
1. Project Overview
2. Stakeholders & Roles
3. Use Cases
4. Functional Requirements
5. Non-Functional Requirements
6. Data Model
7. Security & Compliance
8. Interfaces & Integration
9. Edge Cases & Error Handling
10. Acceptance Criteria & Test Cases
11. Traceability Matrix

**LangChain Implementation:**
```python
# src/agents/synthesis.py
from jinja2 import Template
from src.schemas.requirement import RequirementItem

class SynthesisAgent:
    def __init__(self):
        with open("src/templates/rd_template.md") as f:
            self.template = Template(f.read())

    def invoke(self, state: dict) -> SynthesisOutput:
        # Group requirements by type
        functional = [r for r in state["requirements"] if r.type == "functional"]
        non_functional = [r for r in state["requirements"] if r.type == "non-functional"]
        security = [r for r in state["requirements"] if r.type == "security"]

        # Render template
        rd_content = self.template.render(
            project_name=state["project_name"],
            functional_requirements=functional,
            non_functional_requirements=non_functional,
            security_requirements=security,
            inferred_requirements=state["inferred_requirements"],
            traceability_matrix=self._build_traceability_matrix(state)
        )

        # Build JSON representation
        rd_json = {
            "project": state["project_name"],
            "version": state["rd_version"] + 1,
            "requirements": [r.dict() for r in state["requirements"]],
            "inferred": [r.dict() for r in state["inferred_requirements"]]
        }

        return SynthesisOutput(
            rd_content=rd_content,
            rd_json=rd_json,
            version=state["rd_version"] + 1,
            sections=["Overview", "Functional", "Non-Functional", "Security", "Traceability"]
        )

    def _build_traceability_matrix(self, state: dict) -> List[Dict]:
        matrix = []
        for req in state["requirements"]:
            matrix.append({
                "requirement_id": req.id,
                "title": req.title,
                "source_refs": req.source_refs,
                "acceptance_criteria_count": len(req.acceptance_criteria)
            })
        return matrix
```

**LangGraph Node Wrapper:**
```python
def synthesis_node(state: State) -> State:
    result = synthesis_agent.invoke(state)

    state["rd_draft"] = result.rd_content
    state["rd_version"] = result.version
    state["last_agent"] = "synthesis"

    # Store in database
    store_rd_document(state["session_id"], result.rd_content, result.version)

    return state
```

---

### 8.6 Review & Approval Manager

**Purpose:** Manage human-in-the-loop approval cycles with inline comments and version diffs.

**Responsibilities:**
- Present RD draft to reviewer with inline commenting UI
- Capture approve/revise decisions
- Apply requested changes and regenerate RD
- Track version history and change explanations

**Input Schema:**
```python
class ReviewInput(BaseModel):
    session_id: str
    rd_version: int
    review_action: str  # "approve", "revise", "comment"
    comments: Optional[List[ReviewComment]]

class ReviewComment(BaseModel):
    requirement_id: Optional[str]
    line_number: Optional[int]
    comment_text: str
    suggested_change: Optional[str]
```

**Output Schema:**
```python
class ReviewOutput(BaseModel):
    approval_status: str  # "approved", "revision_requested"
    changes_applied: List[str]
    next_action: str  # "finalize", "regenerate_rd"
```

**Implementation:**
```python
# src/agents/review.py
class ReviewAgent:
    def invoke(self, state: dict, review_input: ReviewInput) -> ReviewOutput:
        if review_input.review_action == "approve":
            return ReviewOutput(
                approval_status="approved",
                changes_applied=[],
                next_action="finalize"
            )

        elif review_input.review_action == "revise":
            # Apply comments to requirements
            changes = []
            for comment in review_input.comments:
                if comment.suggested_change:
                    # Update requirement in state
                    req = self._find_requirement(state, comment.requirement_id)
                    if req:
                        old_value = req.title
                        req.title = comment.suggested_change
                        changes.append(f"Updated {req.id}: '{old_value}' → '{req.title}'")

            return ReviewOutput(
                approval_status="revision_requested",
                changes_applied=changes,
                next_action="regenerate_rd"
            )
```

**LangGraph Node Wrapper:**
```python
def review_node(state: State) -> State:
    # This node waits for human input via API
    # State is paused until user submits review
    state["approval_status"] = "pending"
    state["last_agent"] = "review"
    return state

def apply_review_feedback(state: State, review_input: ReviewInput) -> State:
    result = review_agent.invoke(state, review_input)
    state["approval_status"] = result.approval_status
    state["review_feedback"] = result.changes_applied
    return state
```

---

## 9. Template & RD Format

### 9.1 RD Template Structure

```markdown
# Requirements Document: {project_name}

**Version:** {version}
**Date:** {date}
**Status:** {status}

---

## 1. Project Overview

**Project Name:** {project_name}
**Objective:** {project_objective}
**Scope:** {scope_description}

---

## 2. Stakeholders & Roles

| Role | Name | Responsibilities |
|------|------|------------------|
{%- for stakeholder in stakeholders %}
| {{ stakeholder.role }} | {{ stakeholder.name }} | {{ stakeholder.responsibilities }} |
{%- endfor %}

---

## 3. Use Cases

{%- for use_case in use_cases %}
### UC-{{ loop.index }}: {{ use_case.title }}

**Actors:** {{ use_case.actors }}
**Preconditions:** {{ use_case.preconditions }}
**Flow:**
{{ use_case.flow }}
**Postconditions:** {{ use_case.postconditions }}

---
{%- endfor %}

## 4. Functional Requirements

{%- for req in functional_requirements %}
### {{ req.id }}: {{ req.title }}

**Type:** {{ req.type }}
**Actor:** {{ req.actor }}
**Action:** {{ req.action }}
**Condition:** {{ req.condition or "N/A" }}
**Priority:** {{ req.priority }}

**Acceptance Criteria:**
{%- for criteria in req.acceptance_criteria %}
- {{ criteria }}
{%- endfor %}

**Rationale:** {{ req.rationale }}
**Source:** {{ req.source_refs | join(", ") }}

---
{%- endfor %}

## 5. Non-Functional Requirements

{%- for req in non_functional_requirements %}
### {{ req.id }}: {{ req.title }}

**Metric:** {{ req.action }}
**Target:** {{ req.acceptance_criteria | join("; ") }}
**Priority:** {{ req.priority }}

**Rationale:** {{ req.rationale }}

---
{%- endfor %}

## 6. Security & Compliance Requirements

{%- for req in security_requirements %}
### {{ req.id }}: {{ req.title }}

{{ req.action }}

**Acceptance Criteria:**
{%- for criteria in req.acceptance_criteria %}
- {{ criteria }}
{%- endfor %}

---
{%- endfor %}

## 7. Inferred Requirements

{%- for req in inferred_requirements %}
### {{ req.id }}: {{ req.title }} [INFERRED]

**Confidence:** {{ req.confidence }}
**Rationale:** {{ req.rationale }}

**Acceptance Criteria:**
{%- for criteria in req.acceptance_criteria %}
- {{ criteria }}
{%- endfor %}

---
{%- endfor %}

## 8. Traceability Matrix

| Requirement ID | Title | Source References | Acceptance Criteria Count |
|----------------|-------|-------------------|---------------------------|
{%- for item in traceability_matrix %}
| {{ item.requirement_id }} | {{ item.title }} | {{ item.source_refs | join(", ") }} | {{ item.acceptance_criteria_count }} |
{%- endfor %}

---

## 9. Approval

**Reviewed by:** _____________________
**Date:** _____________________
**Status:** Approved / Revision Requested

**Comments:**

---

**End of Document**
```

### 9.2 Example Complete Requirement Item

**JSON:**
```json
{
  "id": "REQ-001",
  "title": "App cold start time",
  "type": "non-functional",
  "actor": "end-user",
  "action": "app starts and is ready to accept input",
  "condition": "on mobile devices with 4G network in India",
  "acceptance_criteria": [
    "Cold start time ≤ 2.5 seconds on median Android device with 4G in India",
    "Measured using Firebase Performance Monitoring",
    "95th percentile ≤ 4.0 seconds"
  ],
  "priority": "high",
  "confidence": 0.88,
  "inferred": false,
  "rationale": "User explicitly requested app to 'load quickly' for users in India. Converted vague term to measurable latency target based on industry standards for mobile UX.",
  "source_refs": ["chat_turn_12", "chat_turn_15"]
}
```

**Rendered Markdown:**
```markdown
### REQ-001: App cold start time

**Type:** non-functional
**Actor:** end-user
**Action:** app starts and is ready to accept input
**Condition:** on mobile devices with 4G network in India
**Priority:** high

**Acceptance Criteria:**
- Cold start time ≤ 2.5 seconds on median Android device with 4G in India
- Measured using Firebase Performance Monitoring
- 95th percentile ≤ 4.0 seconds

**Rationale:** User explicitly requested app to 'load quickly' for users in India. Converted vague term to measurable latency target based on industry standards for mobile UX.
**Source:** chat_turn_12, chat_turn_15
```

### 9.3 Conversion Rules: Chat → Measurable Criteria

| Vague Term | Measurable Conversion | Rationale |
|------------|----------------------|-----------|
| "fast" | "< 2 seconds" | Industry standard for perceived responsiveness |
| "scalable" | "handles 10,000 concurrent users" | Typical SMB scale; adjust based on context |
| "secure" | "encrypted at rest (AES-256) and in transit (TLS 1.3)" | Current security best practices |
| "reliable" | "99.9% uptime (SLA)" | Standard SaaS reliability |
| "user-friendly" | "Task completion rate > 90% in usability testing" | Measurable UX metric |

---

## 10. Validation Rules & Deterministic Checks

### 10.1 Ambiguous Verb Detection

**Regex Patterns:**
```python
AMBIGUOUS_PATTERNS = {
    r'\b(optimi[zs]e|improv|enhanc|better|good|fast|quick|secur|scalable)\b':
        "Vague quality term without measurable threshold",
    r'\b(reasonable|appropriate|sufficient|adequate)\b':
        "Subjective qualifier without objective criteria",
    r'\b(etc\.|and so on|among others)\b':
        "Incomplete enumeration"
}

def detect_ambiguous_language(text: str) -> List[str]:
    issues = []
    for pattern, message in AMBIGUOUS_PATTERNS.items():
        if re.search(pattern, text, re.IGNORECASE):
            issues.append(message)
    return issues
```

### 10.2 Structural Validation Rules

```python
VALIDATION_RULES = {
    "missing_actor": lambda req: not req.actor or len(req.actor) == 0,
    "missing_action": lambda req: not req.action or len(req.action) == 0,
    "missing_acceptance_criteria": lambda req: len(req.acceptance_criteria) == 0,
    "non_measurable_nfr": lambda req: (
        req.type == "non-functional" and
        not any(char.isdigit() for char in " ".join(req.acceptance_criteria))
    ),
    "missing_traceability": lambda req: len(req.source_refs) == 0
}

def validate_requirement(req: RequirementItem) -> List[ValidationIssue]:
    issues = []
    for rule_name, rule_func in VALIDATION_RULES.items():
        if rule_func(req):
            issues.append(ValidationIssue(
                requirement_id=req.id,
                severity="error" if "missing" in rule_name else "warning",
                issue_type=rule_name,
                message=f"Validation failed: {rule_name}",
                suggestion=SUGGESTIONS.get(rule_name, "Review requirement")
            ))
    return issues
```

### 10.3 Conflict Detection via Embeddings

**Semantic Similarity Approach:**
1. Generate embeddings for all requirement titles + actions using `text-embedding-3-small`
2. Store in ChromaDB with metadata (requirement ID, type, session)
3. For each new requirement, query ChromaDB for similar vectors (cosine similarity > 0.85)
4. Flag potential duplicates/conflicts for human review

**Implementation:**
```python
# src/utils/openai_client.py
from openai import OpenAI
import os

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def get_embedding(text: str, model: str = "text-embedding-3-small") -> List[float]:
    response = client.embeddings.create(
        input=text,
        model=model
    )
    return response.data[0].embedding

# Usage in validation
def detect_conflicts(new_req: RequirementItem, existing_reqs: List[RequirementItem]) -> List[str]:
    new_embedding = get_embedding(f"{new_req.title} {new_req.action}")
    conflicts = []

    for existing in existing_reqs:
        existing_embedding = get_embedding(f"{existing.title} {existing.action}")
        similarity = cosine_similarity(new_embedding, existing_embedding)

        if similarity > 0.85:
            conflicts.append(f"High similarity ({similarity:.2f}) with {existing.id}: {existing.title}")

    return conflicts
```

---

## 11. Human-in-the-Loop UX & Review Flow

### 11.1 UI Behaviors

**Approve/Revise Buttons:**
- Display at top and bottom of RD preview
- "Approve" → Lock RD, trigger export, send notification
- "Request Revision" → Open comment modal, allow inline comments

**Inline Comments:**
- Click any requirement to add comment
- Highlight specific text for context
- Suggest alternative phrasing in comment

**Accept Inferred Items:**
- Inferred requirements displayed with `[INFERRED]` badge and confidence score
- Toggle switches next to each: Accept / Reject
- Bulk actions: "Accept All", "Review Individually"

### 11.2 Backend APIs for Review

**Submit Review Feedback:**
```http
POST /api/v1/review/{session_id}
Authorization: Bearer {jwt_token}

{
  "review_action": "revise",
  "comments": [
    {
      "requirement_id": "REQ-001",
      "comment_text": "Please change 2.5 seconds to 2.0 seconds",
      "suggested_change": "Cold start time ≤ 2.0 seconds"
    }
  ]
}
```

**Response:**
```json
{
  "review_id": "rev_abc123",
  "status": "revision_requested",
  "changes_queued": 1,
  "next_action": "regenerate_rd"
}
```

**Approve RD:**
```http
POST /api/v1/review/{session_id}/approve
Authorization: Bearer {jwt_token}

{
  "reviewer_name": "Jane Doe",
  "comments": "Looks good, approved for implementation"
}
```

### 11.3 Version Diff Example

**Diff Generation:**
```python
import difflib

def generate_diff(old_version: str, new_version: str) -> str:
    diff = difflib.unified_diff(
        old_version.splitlines(),
        new_version.splitlines(),
        lineterm='',
        fromfile='v1',
        tofile='v2'
    )
    return '\n'.join(diff)
```

**UI Display:**
- Side-by-side view: Old version (left) | New version (right)
- Highlighting: Green for additions, Red for deletions, Yellow for modifications
- Jump to changes: "Next Change" button scrolls to next diff block

---

## 12. Testing, QA & Evaluation

### 12.1 Unit Tests

**Extraction Agent Test:**
```python
# tests/unit/test_extraction.py
import pytest
from src.agents.extraction import ExtractionAgent
from tests.fixtures.chat_sessions import SAMPLE_CHAT_1

def test_extraction_login_requirement():
    agent = ExtractionAgent()
    result = agent.invoke({"chat_history": SAMPLE_CHAT_1})

    assert len(result.requirements) >= 1
    req = result.requirements[0]
    assert req.id.startswith("REQ-")
    assert req.type == "functional"
    assert "login" in req.title.lower()
    assert len(req.acceptance_criteria) > 0
    assert len(req.source_refs) > 0
```

**Validation Agent Test:**
```python
# tests/unit/test_validation.py
def test_validation_ambiguous_verb():
    from src.validation.deterministic import check_ambiguous_verbs
    from src.schemas.requirement import RequirementItem

    req = RequirementItem(
        id="REQ-001",
        title="System should optimize performance",
        type="non-functional",
        actor="system",
        action="optimize performance",
        acceptance_criteria=["System is fast"],
        priority="high",
        confidence=0.8,
        rationale="User requested",
        source_refs=["turn_1"]
    )

    issues = check_ambiguous_verbs(req)
    assert len(issues) > 0
    assert issues[0].issue_type == "ambiguous_verb"
```

### 12.2 Integration Tests

**Full Graph Flow Test:**
```python
# tests/integration/test_graph_flow.py
from src.orchestrator.graph import build_graph

def test_happy_path_flow():
    graph = build_graph()

    initial_state = {
        "session_id": "test_session",
        "project_name": "Test Project",
        "chat_history": [
            {"role": "user", "content": "We need user login with email and password"}
        ],
        "requirements": [],
        "inferred_requirements": [],
        "current_turn": 1
    }

    # Run graph
    final_state = graph.invoke(initial_state)

    # Assertions
    assert len(final_state["requirements"]) > 0
    assert final_state["rd_draft"] is not None
    assert final_state["last_agent"] == "synthesis"
```

### 12.3 End-to-End Harness

**Simulate Full Conversation:**
```python
# tests/e2e/test_full_session.py
import pytest
from src.api.routes.sessions import create_session, send_message

@pytest.mark.e2e
async def test_full_requirements_session():
    # Create session
    session = await create_session({"project_name": "E-commerce App", "user_id": "test_user"})

    # Simulate conversation turns
    turns = [
        "We need a mobile app for e-commerce",
        "Users should log in with email and password",
        "The app should load in under 2 seconds",
        "We need to accept credit card payments"
    ]

    for turn in turns:
        response = await send_message(session.id, {"message": turn})
        assert response.status == "success"

    # Verify requirements extracted
    requirements = await get_requirements(session.id)
    assert len(requirements) >= 3  # Login, performance, payment

    # Verify RD generated
    rd = await get_rd(session.id)
    assert "REQ-" in rd.content
    assert "Traceability Matrix" in rd.content
```

### 12.4 Metrics & Measurement

**Completeness Calculation:**
```python
def calculate_completeness(ground_truth: List[str], extracted: List[RequirementItem]) -> float:
    """
    ground_truth: List of expected requirement keywords
    extracted: Actual extracted requirements
    """
    extracted_keywords = set()
    for req in extracted:
        extracted_keywords.update(req.title.lower().split())

    matches = sum(1 for keyword in ground_truth if keyword.lower() in extracted_keywords)
    return matches / len(ground_truth)

# Usage
ground_truth = ["login", "email", "password", "performance", "2 seconds", "payment"]
completeness = calculate_completeness(ground_truth, extracted_requirements)
assert completeness >= 0.95  # Target: 95%
```

**Reviewer Satisfaction Survey:**
```python
@dataclass
class ReviewerFeedback:
    session_id: str
    clarity_score: int  # 1-5
    correctness_score: int  # 1-5
    sufficiency_score: int  # 1-5
    would_recommend: bool

    @property
    def average_score(self) -> float:
        return (self.clarity_score + self.correctness_score + self.sufficiency_score) / 3
```

---

## 13. Security, Privacy & PII

### 13.1 PII Detection Pipeline

**Detection Rules:**
```python
# src/validation/pii_detector.py
import re
from typing import List, Tuple

PII_PATTERNS = {
    "email": r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
    "phone": r'\b(\+\d{1,3}[-.]?)?\(?\d{3}\)?[-.]?\d{3}[-.]?\d{4}\b',
    "ssn": r'\b\d{3}-\d{2}-\d{4}\b',
    "credit_card": r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b',
    "ip_address": r'\b(?:\d{1,3}\.){3}\d{1,3}\b'
}

def detect_pii(text: str) -> List[Tuple[str, str]]:
    """Returns list of (pii_type, matched_value)"""
    detected = []
    for pii_type, pattern in PII_PATTERNS.items():
        matches = re.findall(pattern, text)
        for match in matches:
            detected.append((pii_type, match))
    return detected

def redact_pii(text: str) -> str:
    """Replace PII with [REDACTED]"""
    for pii_type, pattern in PII_PATTERNS.items():
        text = re.sub(pattern, f"[REDACTED_{pii_type.upper()}]", text)
    return text
```

**Integration in Chat Pipeline:**
```python
# src/api/middleware/pii_filter.py
from fastapi import Request
from src.validation.pii_detector import detect_pii, redact_pii

async def pii_filter_middleware(request: Request, call_next):
    # Check request body for PII
    if request.method == "POST":
        body = await request.json()
        if "message" in body:
            pii_found = detect_pii(body["message"])
            if pii_found:
                # Log incident
                log_pii_detection(request, pii_found)
                # Redact automatically
                body["message"] = redact_pii(body["message"])

    response = await call_next(request)
    return response
```

### 13.2 Data Retention & Deletion

**Endpoints:**
```python
# src/api/routes/privacy.py
@router.delete("/sessions/{session_id}/data")
async def delete_session_data(session_id: str, user: dict = Depends(verify_token)):
    # Delete from Postgres
    delete_from_db(session_id)

    # Delete from ChromaDB
    vector_store.delete_session(session_id)

    # Delete from Redis
    redis_cache.delete(f"session:{session_id}")

    # Log deletion
    audit_log("data_deletion", session_id, user["sub"])

    return {"status": "deleted", "session_id": session_id}
```

**Automated Retention Policy:**
```python
# scripts/cleanup_old_sessions.py
from datetime import datetime, timedelta

def cleanup_old_sessions(retention_days: int = 90):
    cutoff = datetime.utcnow() - timedelta(days=retention_days)

    sessions = db.query(Session).filter(Session.created_at < cutoff, Session.status != "approved").all()

    for session in sessions:
        delete_session_data(session.id)
        logger.info(f"Deleted session {session.id} (age: {retention_days}+ days)")
```

### 13.3 Local Storage Encryption

**Postgres Encryption at Rest:**
- Enable PostgreSQL's `pgcrypto` extension
- Encrypt sensitive columns (user_id, project_name if contains PII)

```sql
CREATE EXTENSION IF NOT EXISTS pgcrypto;

-- Encrypt column example
UPDATE sessions SET project_name = pgp_sym_encrypt(project_name, 'encryption_key');

-- Decrypt on read
SELECT pgp_sym_decrypt(project_name::bytea, 'encryption_key') FROM sessions;
```

**File Encryption (Exports):**
```python
from cryptography.fernet import Fernet

def encrypt_export(file_path: str, key: bytes) -> str:
    f = Fernet(key)
    with open(file_path, 'rb') as file:
        encrypted = f.encrypt(file.read())

    encrypted_path = f"{file_path}.encrypted"
    with open(encrypted_path, 'wb') as file:
        file.write(encrypted)

    return encrypted_path
```

---

## 14. Observability & Monitoring (Local)

### 14.1 Minimal Local Dashboard

**Prometheus + Grafana (Optional):**
```yaml
# Add to docker-compose.yml
prometheus:
  image: prom/prometheus:latest
  ports:
    - "9090:9090"
  volumes:
    - ./config/prometheus.yml:/etc/prometheus/prometheus.yml
    - prometheus-data:/prometheus
  command:
    - '--config.file=/etc/prometheus/prometheus.yml'

grafana:
  image: grafana/grafana:latest
  ports:
    - "3001:3000"
  volumes:
    - grafana-data:/var/lib/grafana
  environment:
    - GF_SECURITY_ADMIN_PASSWORD=admin
```

**Prometheus Config:**
```yaml
# config/prometheus.yml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'req-eng-platform'
    static_configs:
      - targets: ['host.docker.internal:8000']
```

**Expose Metrics in FastAPI:**
```python
# src/main.py
from prometheus_fastapi_instrumentator import Instrumentator

app = FastAPI()
Instrumentator().instrument(app).expose(app)
```

### 14.2 LangSmith Tracing

**Configuration:**[61][63]
```python
# src/utils/langsmith_tracing.py
import os
os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_API_KEY"] = os.getenv("LANGSMITH_API_KEY")
os.environ["LANGSMITH_PROJECT"] = os.getenv("LANGSMITH_PROJECT", "req-eng-local")

from langsmith import traceable

@traceable(name="extract_requirements")
def extract_requirements(text: str):
    # Agent logic
    pass
```

**What to Capture:**
- Every LLM call (prompt, completion, tokens, latency)
- Agent transitions (which agent, input/output, confidence)
- Validation results (issues detected, severity)
- Human feedback (approve/revise, comments)

**View Traces:**
Navigate to https://smith.langchain.com → Select project → View traces by session ID

### 14.3 Custom Metrics

**Track in Application:**
```python
# src/utils/metrics.py
from prometheus_client import Counter, Histogram, Gauge

# Counters
requirements_extracted = Counter('requirements_extracted_total', 'Total requirements extracted')
validation_failures = Counter('validation_failures_total', 'Total validation failures', ['severity'])

# Histograms
agent_latency = Histogram('agent_latency_seconds', 'Agent execution time', ['agent_name'])
rd_generation_time = Histogram('rd_generation_seconds', 'Time to generate RD')

# Gauges
active_sessions = Gauge('active_sessions', 'Number of active sessions')
pending_reviews = Gauge('pending_reviews', 'Number of RDs pending review')

# Usage
with agent_latency.labels(agent_name="extraction").time():
    result = extraction_agent.invoke(state)
    requirements_extracted.inc(len(result.requirements))
```

---

## 15. CI / Local Dev Workflow

### 15.1 Pre-Commit Hooks

**Install Pre-Commit:**
```bash
pip install pre-commit
```

**Configuration:**
```yaml
# .pre-commit-config.yaml
repos:
  - repo: https://github.com/psf/black
    rev: 23.3.0
    hooks:
      - id: black
        language_version: python3.11

  - repo: https://github.com/pycqa/flake8
    rev: 6.0.0
    hooks:
      - id: flake8
        args: ['--max-line-length=120']

  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.3.0
    hooks:
      - id: mypy
        additional_dependencies: [pydantic]

  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.4.0
    hooks:
      - id: trailing-whitespace
      - id: end-of-file-fixer
      - id: check-yaml
      - id: check-added-large-files
```

**Install Hooks:**
```bash
pre-commit install
```

### 15.2 Linting Rules

**Flake8 Configuration:**
```ini
# .flake8
[flake8]
max-line-length = 120
exclude = .git,__pycache__,venv,build,dist
ignore = E203,W503
```

**Ruff (Modern Alternative):**
```toml
# pyproject.toml
[tool.ruff]
line-length = 120
select = ["E", "F", "W", "I"]
ignore = ["E203", "W503"]
```

### 15.3 Git Branching Strategy

**For Small Team (2-5 developers):**
- `main`: Production-ready code (protected)
- `develop`: Integration branch
- `feature/*`: Feature branches (merge to develop)
- `hotfix/*`: Critical fixes (merge to main + develop)

**Workflow:**
```bash
# Start new feature
git checkout -b feature/inference-agent develop

# Work on feature
git add .
git commit -m "feat: implement inference agent with confidence scoring"

# Push and create PR
git push origin feature/inference-agent
# Open PR on GitHub: feature/inference-agent → develop
```

### 15.4 GitHub Actions (Optional)

```yaml
# .github/workflows/test.yml
name: Test Suite

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'

      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install pytest pytest-cov

      - name: Run tests
        run: pytest tests/ --cov=src --cov-report=xml

      - name: Upload coverage
        uses: codecov/codecov-action@v3
```

---

## 16. Deliverables & Acceptance Criteria

### 16.1 Required Deliverables

1. **Runnable local environment** (Docker Compose + FastAPI)
2. **6 agents implemented** (Conversational, Extraction, Inference, Validation, Synthesis, Review)
3. **LangGraph orchestrator** with conditional branching
4. **REST API** with authentication and 8+ endpoints
5. **Database schemas** (Postgres + ChromaDB)
6. **RD template** (Markdown + Jinja2)
7. **Validation rules** (deterministic + semantic)
8. **Unit + integration tests** (pytest suite with 80%+ coverage)
9. **Documentation** (README, API docs, architecture)

### 16.2 Acceptance Checklist

- [ ] `docker-compose up` starts all services (Postgres, Redis, ChromaDB)
- [ ] `uvicorn src.main:app` starts API server without errors
- [ ] `curl http://localhost:8000/health` returns `{"status": "healthy"}`
- [ ] POST to `/sessions` creates new session and returns ID
- [ ] POST to `/sessions/{id}/messages` triggers agent orchestration
- [ ] Requirements are extracted and stored in Postgres
- [ ] Vector embeddings stored in ChromaDB
- [ ] Validation detects ambiguous verbs and missing criteria
- [ ] RD generated from template with traceability matrix
- [ ] Review endpoint accepts comments and triggers revision
- [ ] PII detection redacts email addresses in chat
- [ ] LangSmith traces visible at https://smith.langchain.com (if enabled)
- [ ] Unit tests pass: `pytest tests/unit/`
- [ ] Integration tests pass: `pytest tests/integration/`
- [ ] Exports RD to Markdown, JSON, and PDF formats

---

## 17. Implementation Plan & Prioritized Next Steps

### 17.1 MVP Phases

**MVP1: Foundation (Week 1-2)**
1. Set up local environment (Docker Compose, Postgres, Redis, ChromaDB)
2. Implement Conversational Agent (basic chat loop)
3. Implement Extraction Agent (parse chat → RequirementItem)
4. Basic API endpoints (sessions, messages)
5. Unit tests for extraction

**MVP2: Orchestration (Week 3-4)**
1. Implement LangGraph orchestrator with state management
2. Add Inference Agent (propose implicit requirements)
3. Add Validation Agent (deterministic checks)
4. Implement ChromaDB integration for semantic conflict detection
5. Integration tests for graph flow

**MVP3: Synthesis & Review (Week 5-6)**
1. Implement Synthesis Agent with Jinja2 templates
2. Add Review & Approval workflow (API endpoints)
3. Implement version history and diffs
4. Export to Markdown/PDF
5. End-to-end tests with human-in-loop simulation

**MVP4: Polish & Production-Ready (Week 7-8)**
1. PII detection and redaction
2. LangSmith tracing integration
3. Monitoring dashboards (Prometheus + Grafana)
4. Security hardening (JWT, RBAC, encryption)
5. Documentation and deployment guide

### 17.2 Dependency Graph

```
Environment Setup
├── Docker Compose Configuration
├── Database Migrations
└── API Scaffold
    ├── Authentication Middleware
    └── Logging Configuration

Agent Implementation (Parallel)
├── Conversational Agent
├── Extraction Agent
├── Inference Agent
├── Validation Agent
└── Synthesis Agent

Orchestration (Depends on Agents)
├── LangGraph State Model
├── Node Functions
└── Conditional Edges

Review Workflow (Depends on Orchestration)
├── Review API Endpoints
├── Version Diff Generator
└── Approval Flow

Testing & QA (Continuous)
├── Unit Tests (per agent)
├── Integration Tests (graph flow)
└── E2E Tests (full session)
```

### 17.3 Task Prioritization

| Priority | Task | Estimated Time | Blocker? |
|----------|------|----------------|----------|
| P0 | Docker Compose setup | 2 hours | No |
| P0 | Database schema creation | 3 hours | No |
| P0 | FastAPI scaffold + auth | 4 hours | No |
| P0 | Conversational Agent | 8 hours | No |
| P0 | Extraction Agent | 12 hours | No |
| P1 | LangGraph orchestrator | 16 hours | Needs agents |
| P1 | Inference Agent | 10 hours | No |
| P1 | Validation Agent | 8 hours | No |
| P1 | ChromaDB integration | 6 hours | No |
| P2 | Synthesis Agent + templates | 12 hours | Needs orchestrator |
| P2 | Review workflow APIs | 10 hours | Needs synthesis |
| P2 | Unit tests (all agents) | 16 hours | No |
| P3 | PII detection | 6 hours | No |
| P3 | LangSmith tracing | 4 hours | No |
| P3 | Export to PDF | 4 hours | Needs synthesis |
| P3 | Monitoring dashboards | 8 hours | No |

---

## 18. Appendix

### 18.1 Docker Compose (Complete)

```yaml
# docker-compose.yml
version: '3.8'

services:
  postgres:
    image: postgres:15-alpine
    container_name: reqeng-postgres
    environment:
      POSTGRES_USER: reqeng
      POSTGRES_PASSWORD: password
      POSTGRES_DB: reqeng_db
    ports:
      - "5432:5432"
    volumes:
      - postgres-data:/var/lib/postgresql/data
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U reqeng"]
      interval: 10s
      timeout: 5s
      retries: 5

  redis:
    image: redis:7-alpine
    container_name: reqeng-redis
    ports:
      - "6379:6379"
    volumes:
      - redis-data:/data
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 10s
      timeout: 5s
      retries: 5

  chromadb:
    image: chromadb/chroma:0.4.15
    container_name: reqeng-chromadb
    ports:
      - "8001:8000"
    volumes:
      - chromadb-data:/chroma/chroma
    environment:
      - ALLOW_RESET=true
      - ANONYMIZED_TELEMETRY=false
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/api/v1/heartbeat"]
      interval: 10s
      timeout: 5s
      retries: 5

volumes:
  postgres-data:
  redis-data:
  chromadb-data:

networks:
  default:
    name: reqeng-network
```

### 18.2 Environment Template

```bash
# .env.example
# OpenAI Configuration
OPENAI_API_KEY=sk-proj-XXXXXXXXXXXXXXXXXXXXX
OPENAI_MODEL=gpt-4-turbo-preview
OPENAI_EMBEDDING_MODEL=text-embedding-3-small
OPENAI_MAX_TOKENS=2000
OPENAI_TEMPERATURE=0.7

# LangSmith Configuration (Optional)
LANGSMITH_API_KEY=lsv2_pt_XXXXXXXXXXXXXXXXXXXXX
LANGSMITH_PROJECT=req-eng-local
LANGSMITH_TRACING=true
LANGSMITH_ENDPOINT=https://api.smith.langchain.com

# Database Configuration
DATABASE_URI=postgresql://reqeng:password@localhost:5432/reqeng_db
REDIS_URL=redis://localhost:6379/0
CHROMA_HOST=localhost
CHROMA_PORT=8001

# Application Configuration
APP_ENV=development
APP_NAME=req-eng-platform
APP_VERSION=1.0.0
LOG_LEVEL=DEBUG
CORS_ORIGINS=http://localhost:3000,http://localhost:8000

# Security
JWT_SECRET=your-jwt-secret-change-in-production-min-32-chars
JWT_ALGORITHM=HS256
JWT_EXPIRY_HOURS=24
ENCRYPTION_KEY=your-encryption-key-base64-encoded

# Agent Configuration
MAX_ITERATIONS=10
CONFIDENCE_THRESHOLD=0.60
VALIDATION_AMBIGUITY_THRESHOLD=0.75
SEMANTIC_SIMILARITY_THRESHOLD=0.85
INFERRED_CONFIDENCE_MIN=0.50
INFERRED_CONFIDENCE_MAX=0.90

# Rate Limiting
RATE_LIMIT_PER_USER=100
RATE_LIMIT_WINDOW_SECONDS=3600

# File Upload
MAX_UPLOAD_SIZE_MB=10
ALLOWED_FILE_TYPES=txt,pdf,docx,md

# Monitoring (Optional)
PROMETHEUS_ENABLED=false
GRAFANA_ENABLED=false
```

### 18.3 VSCode Launch Configuration

```json
// .vscode/launch.json
{
  "version": "0.2.0",
  "configurations": [
    {
      "name": "FastAPI Server",
      "type": "python",
      "request": "launch",
      "module": "uvicorn",
      "args": [
        "src.main:app",
        "--reload",
        "--host",
        "0.0.0.0",
        "--port",
        "8000"
      ],
      "jinja": true,
      "justMyCode": false,
      "env": {
        "PYTHONPATH": "${workspaceFolder}"
      },
      "envFile": "${workspaceFolder}/.env"
    },
    {
      "name": "Pytest: Current File",
      "type": "python",
      "request": "launch",
      "module": "pytest",
      "args": [
        "${file}",
        "-v",
        "-s"
      ],
      "console": "integratedTerminal",
      "justMyCode": false
    },
    {
      "name": "Pytest: All Tests",
      "type": "python",
      "request": "launch",
      "module": "pytest",
      "args": [
        "tests/",
        "-v",
        "--cov=src",
        "--cov-report=html"
      ],
      "console": "integratedTerminal"
    }
  ]
}
```

### 18.4 Example Agent Code (Extraction Agent)

```python
# example-agent-code.py
"""
Complete Extraction Agent implementation with LangChain and LangGraph integration
"""
from typing import List, Dict
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
import os

# Schemas
class RequirementItem(BaseModel):
    id: str = Field(..., pattern=r"^REQ-\d{3,}$")
    title: str
    type: str
    actor: str
    action: str
    condition: str = ""
    acceptance_criteria: List[str]
    priority: str
    confidence: float
    inferred: bool = False
    rationale: str
    source_refs: List[str]

class ExtractionOutput(BaseModel):
    requirements: List[RequirementItem]
    confidence: float
    ambiguous_items: List[str]

# Agent Implementation
class ExtractionAgent:
    def __init__(self):
        self.llm = ChatOpenAI(
            model=os.getenv("OPENAI_MODEL", "gpt-4-turbo-preview"),
            temperature=0.2,
            max_tokens=2000,
            openai_api_key=os.getenv("OPENAI_API_KEY")
        )

        self.parser = JsonOutputParser(pydantic_object=ExtractionOutput)

        # Load prompt template
        prompt_text = """
You are a Requirements Extraction Agent. Parse the conversation and extract structured requirements.

## Extraction Rules:
1. Generate IDs: REQ-001, REQ-002, etc.
2. Every requirement needs: id, title, type, actor, action, acceptance_criteria
3. Convert vague terms to measurable metrics
4. Link to source chat turns (chat_turn_0, chat_turn_1, etc.)

## Chat History:
{chat_history}

## Output (JSON):
{format_instructions}
"""

        self.prompt = PromptTemplate(
            template=prompt_text,
            input_variables=["chat_history"],
            partial_variables={"format_instructions": self.parser.get_format_instructions()}
        )

    def invoke(self, state: Dict) -> ExtractionOutput:
        # Create chain
        chain = self.prompt | self.llm | self.parser

        # Format chat history
        chat_history = self._format_chat_history(state["chat_history"])

        # Execute chain
        result = chain.invoke({"chat_history": chat_history})

        # Post-process: assign source refs
        for req in result["requirements"]:
            req["source_refs"] = self._find_source_turns(req, state["chat_history"])

        return ExtractionOutput(**result)

    def _format_chat_history(self, history: List[Dict]) -> str:
        lines = []
        for idx, msg in enumerate(history):
            lines.append(f"[Turn {idx}] {msg['role'].upper()}: {msg['content']}")
        return "\n".join(lines)

    def _find_source_turns(self, requirement: Dict, history: List[Dict]) -> List[str]:
        keywords = requirement["title"].lower().split()
        source_turns = []
        for idx, msg in enumerate(history):
            if any(kw in msg["content"].lower() for kw in keywords):
                source_turns.append(f"chat_turn_{idx}")
        return source_turns if source_turns else ["chat_turn_0"]

# LangGraph Node Wrapper
def extraction_node(state: Dict) -> Dict:
    """LangGraph node function for extraction agent"""
    agent = ExtractionAgent()
    result = agent.invoke(state)

    # Update state
    state["requirements"].extend(result.requirements)
    state["confidence"] = result.confidence
    state["last_agent"] = "extraction"

    # Log for traceability
    print(f"Extracted {len(result.requirements)} requirements with confidence {result.confidence:.2f}")

    return state

# Example usage
if __name__ == "__main__":
    # Mock state
    test_state = {
        "session_id": "test_123",
        "chat_history": [
            {"role": "user", "content": "Users should be able to log in with email and password."},
            {"role": "user", "content": "The login should complete in under 2 seconds."}
        ],
        "requirements": [],
        "confidence": 1.0
    }

    # Run extraction
    agent = ExtractionAgent()
    result = agent.invoke(test_state)

    print(f"\\nExtracted {len(result.requirements)} requirements:")
    for req in result.requirements:
        print(f"  - {req.id}: {req.title} (confidence: {req.confidence:.2f})")
```

### 18.5 Test Fixtures

```json
{
  "test_fixtures": {
    "chat_sessions": {
      "session_1": {
        "project_name": "E-commerce Mobile App",
        "chat_history": [
          {"role": "user", "content": "We need a mobile app for e-commerce"},
          {"role": "assistant", "content": "Great! What platforms should we support?"},
          {"role": "user", "content": "iOS and Android"},
          {"role": "user", "content": "Users should log in with email and password"},
          {"role": "user", "content": "The app should load quickly, under 2 seconds"}
        ]
      }
    },
    "expected_outputs": {
      "session_1_extraction": {
        "requirements": [
          {
            "id": "REQ-001",
            "title": "User authentication with email and password",
            "type": "functional",
            "actor": "user",
            "action": "log in using email and password credentials",
            "acceptance_criteria": [
              "User can enter email and password",
              "System authenticates credentials",
              "User is redirected to dashboard on success"
            ],
            "priority": "must",
            "confidence": 0.95,
            "source_refs": ["chat_turn_3"]
          },
          {
            "id": "REQ-002",
            "title": "App cold start time",
            "type": "non-functional",
            "actor": "system",
            "action": "loads app and displays home screen",
            "condition": "on first launch",
            "acceptance_criteria": [
              "App cold start completes in < 2 seconds on median devices"
            ],
            "priority": "high",
            "confidence": 0.88,
            "source_refs": ["chat_turn_4"]
          },
          {
            "id": "REQ-003",
            "title": "Platform support",
            "type": "constraint",
            "actor": "development team",
            "action": "builds app for iOS and Android",
            "acceptance_criteria": [
              "App runs on iOS 14+",
              "App runs on Android 8+"
            ],
            "priority": "must",
            "confidence": 0.98,
            "source_refs": ["chat_turn_2"]
          }
        ],
        "confidence": 0.93,
        "ambiguous_items": []
      }
    }
  }
}
```

### 18.6 Mermaid Diagrams

**System Architecture:**
```mermaid
graph TB
    subgraph "Client Layer"
        UI[Chat UI<br/>React/Vue]
    end

    subgraph "API Layer"
        API[FastAPI Server<br/>Port 8000]
        Auth[JWT Auth<br/>Middleware]
        Logger[Structured<br/>Logging]
    end

    subgraph "Orchestration Layer"
        Orch[LangGraph<br/>Orchestrator]
        State[Session State<br/>Manager]
    end

    subgraph "Agent Layer"
        Conv[Conversational<br/>Agent]
        Ext[Extraction<br/>Agent]
        Inf[Inference<br/>Agent]
        Val[Validation<br/>Agent]
        Syn[Synthesis<br/>Agent]
        Rev[Review<br/>Agent]
    end

    subgraph "Storage Layer"
        PG[(Postgres<br/>Artifacts)]
        Redis[(Redis<br/>Cache)]
        Chroma[(ChromaDB<br/>Vectors)]
    end

    subgraph "External Services"
        OpenAI[OpenAI API<br/>LLM + Embeddings]
        LangSmith[LangSmith<br/>Tracing]
    end

    UI -->|HTTP| API
    API --> Auth
    API --> Logger
    API --> Orch

    Orch --> State
    Orch --> Conv
    Orch --> Ext
    Orch --> Inf
    Orch --> Val
    Orch --> Syn
    Orch --> Rev

    Conv --> OpenAI
    Ext --> OpenAI
    Inf --> OpenAI
    Val --> Chroma
    Syn --> PG

    State --> Redis
    Ext --> PG
    Ext --> Chroma
    Syn --> PG

    Orch --> LangSmith

    style Orch fill:#f9f,stroke:#333,stroke-width:3px
    style OpenAI fill:#ff9,stroke:#333,stroke-width:2px
    style LangSmith fill:#9ff,stroke:#333,stroke-width:2px
```

**Happy Path Sequence:**
```mermaid
sequenceDiagram
    actor User
    participant UI
    participant API
    participant Orch as Orchestrator
    participant Conv as Conversational
    participant Ext as Extraction
    participant Val as Validation
    participant Syn as Synthesis
    participant DB as Postgres

    User->>UI: Types message
    UI->>API: POST /sessions/{id}/messages
    API->>Orch: Invoke graph

    Orch->>Conv: Process message
    Conv->>Orch: Response + metadata

    alt Requirements detected
        Orch->>Ext: Extract requirements
        Ext->>DB: Store requirements
        Ext->>Orch: Extracted items

        Orch->>Val: Validate requirements
        Val->>Orch: Validation report

        alt Validation passes
            Orch->>Syn: Generate RD
            Syn->>DB: Store RD v1
            Syn->>Orch: RD draft
        else Validation fails
            Orch->>Conv: Ask clarification
        end
    end

    Orch->>API: Return response
    API->>UI: Display message + RD preview
    UI->>User: Show result
```

### 18.7 References & Documentation

**Official Documentation:**
1. **LangChain:** https://python.langchain.com/docs/get_started/introduction
2. **LangGraph:** https://langchain-ai.github.io/langgraph/
3. **LangSmith:** https://docs.smith.langchain.com/
4. **OpenAI API:** https://platform.openai.com/docs/api-reference
5. **FastAPI:** https://fastapi.tiangolo.com/
6. **Pydantic:** https://docs.pydantic.dev/latest/
7. **ChromaDB:** https://docs.trychroma.com/
8. **Postgres:** https://www.postgresql.org/docs/
9. **Redis:** https://redis.io/documentation

**Key Tutorials:**
- Multi-Agent Systems with LangGraph: https://langchain-ai.github.io/langgraph/tutorials/multi_agent/
- Building RAG Agents: https://python.langchain.com/docs/tutorials/rag/
- LangSmith Tracing: https://docs.smith.langchain.com/tracing

**Requirements Engineering Standards:**
- IEEE 830-1998: Software Requirements Specification
- ISO/IEC/IEEE 29148: Systems and software engineering requirements processes

---

## 10-Step Implementation Checklist

Execute these steps in order on your laptop:

### Step 1: Install Prerequisites (15 min)
```bash
# Install Docker, Python 3.11, Git (see section 3.3)
docker --version  # Expected: Docker version 24.0+
python3.11 --version  # Expected: Python 3.11+
```

### Step 2: Clone & Setup Repository (10 min)
```bash
git clone https://github.com/your-org/req-eng-platform.git
cd req-eng-platform
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Step 3: Configure Environment (5 min)
```bash
cp .env.example .env
nano .env  # Add your OPENAI_API_KEY and LANGSMITH_API_KEY
```

### Step 4: Start Docker Services (5 min)
```bash
docker-compose up -d
# Expected: 3 containers running (postgres, redis, chromadb)
docker ps
```

### Step 5: Initialize Database (3 min)
```bash
python scripts/migrate.py
# Expected: "Database migrations completed successfully"
```

### Step 6: Start FastAPI Server (2 min)
```bash
uvicorn src.main:app --reload --host 0.0.0.0 --port 8000
# Expected: "Uvicorn running on http://0.0.0.0:8000"
```

### Step 7: Verify API Health (1 min)
```bash
curl http://localhost:8000/health
# Expected: {"status": "healthy", "services": {"postgres": "up", "redis": "up", "chromadb": "up"}}
```

### Step 8: Create Test Session (2 min)
```bash
# Get JWT token
TOKEN=$(curl -X POST http://localhost:8000/api/v1/auth/login \
  -H "Content-Type: application/json" \
  -d '{"username":"admin","password":"admin"}' | jq -r '.access_token')

# Create session
curl -X POST http://localhost:8000/api/v1/sessions \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"project_name":"Test Project","user_id":"test_user"}'
# Expected: {"id": "sess_xxx", "project_name": "Test Project", ...}
```

### Step 9: Run Unit Tests (5 min)
```bash
pytest tests/unit/ -v
# Expected: All tests pass (may have some skipped for incomplete agents)
```

### Step 10: View LangSmith Traces (Optional, 2 min)
```bash
# Ensure LANGSMITH_TRACING=true in .env
# Navigate to https://smith.langchain.com
# Select project: req-eng-local
# View traces from test session
```

**Total Time:** ~50 minutes to fully operational local environment

**Success Criteria:**
- ✅ All Docker containers healthy
- ✅ API returns 200 on health check
- ✅ Can create session and send message
- ✅ Unit tests pass
- ✅ Traces visible in LangSmith (if enabled)

---

**End of Design Packet 1**

**Document Version:** 1.0
**Last Updated:** November 16, 2025
**Prepared by:** AI System Architect
**Status:** Ready for Implementation
