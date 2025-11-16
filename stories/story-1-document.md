# User Story 1: Project Foundation and Core Infrastructure Setup

## Story Overview
**Story ID:** STORY-001
**Story Title:** Project Foundation and Core Infrastructure Setup
**Priority:** P0 - Critical
**Estimated Effort:** 16-24 hours
**Sprint:** Sprint 1
**Dependencies:** None (First Story)

---

## Story Description

As a **developer**, I want to **set up the complete local development environment, project structure, and core infrastructure services** so that **I can begin building the multi-agent requirements engineering platform with a solid foundation**.

This story establishes the foundational infrastructure required for all subsequent development work. It includes repository setup, Docker services configuration, database initialization, API scaffolding, and essential development tooling.

---

## Business Value

- **Enables all future development work** by providing the core infrastructure
- **Reduces setup friction** for team members joining the project
- **Establishes development standards** and patterns early
- **Provides immediate feedback** through health checks and basic API functionality
- **De-risks the project** by validating core technology stack integration

---

## Acceptance Criteria

### ✅ AC1: Repository Structure
**Given** a fresh development environment
**When** I clone the repository
**Then** the following directory structure exists:

```
req-eng-platform/
├── .env.example
├── .gitignore
├── docker-compose.yml
├── README.md
├── requirements.txt
├── pyproject.toml
├── scripts/
│   ├── migrate.py
│   └── seed_data.py
├── src/
│   ├── __init__.py
│   ├── main.py
│   ├── config.py
│   ├── api/
│   ├── agents/
│   ├── orchestrator/
│   ├── schemas/
│   ├── storage/
│   ├── validation/
│   ├── templates/
│   └── utils/
└── tests/
    ├── __init__.py
    ├── fixtures/
    ├── unit/
    └── integration/
```

**Verification:**
- Run `tree -L 2` command to verify directory structure
- All directories contain proper `__init__.py` files where required

---

### ✅ AC2: Docker Services Running
**Given** Docker and Docker Compose are installed
**When** I run `docker-compose up -d`
**Then** the following services are running and healthy:

| Service | Port | Health Check Command |
|---------|------|---------------------|
| PostgreSQL 15 | 5432 | `docker exec reqeng-postgres pg_isready -U reqeng` |
| Redis 7 | 6379 | `docker exec reqeng-redis redis-cli ping` |
| ChromaDB | 8001 | `curl -f http://localhost:8001/api/v1/heartbeat` |

**Verification:**
```bash
docker ps
# Expected: 3 containers running with status "healthy"

docker-compose logs
# Expected: No error messages, all services initialized
```

---

### ✅ AC3: Database Schema Initialized
**Given** Docker services are running
**When** I run `python scripts/migrate.py`
**Then** the following database objects are created:

**Tables:**
- `sessions` (with indexes on user_id, status, created_at)
- `chat_messages` (with indexes on session_id, timestamp)
- `requirements` (with indexes on session_id, type, priority)
- `rd_documents` (with indexes on session_id, version)
- `rd_events` (with indexes on session_id, version, timestamp)
- `audit_logs` (with indexes on session_id, user_id, timestamp)
- `langgraph_checkpoints` (with composite primary key)
- `langgraph_checkpoint_writes`

**Extensions:**
- `uuid-ossp` enabled
- `pg_crypto` enabled

**Verification:**
```sql
-- Connect to database
docker exec -it reqeng-postgres psql -U reqeng -d reqengdb

-- Verify tables
\dt
# Expected: 8 tables listed

-- Verify extensions
\dx
# Expected: uuid-ossp and pg_crypto listed

-- Verify indexes
\di
# Expected: All indexes from DDL present
```

---

### ✅ AC4: Environment Configuration
**Given** the `.env.example` template exists
**When** I copy it to `.env` and configure required variables
**Then** the application loads configuration correctly

**Required Environment Variables:**
```bash
# OpenAI API Configuration
OPENAI_API_KEY=sk-proj-XXXXXXXXXXXXXXXXXXXXX
OPENAI_MODEL=gpt-4-turbo-preview
OPENAI_EMBEDDING_MODEL=text-embedding-3-small

# Database Configuration
DATABASE_URI=postgresql://reqeng:password@localhost:5432/reqengdb
REDIS_URL=redis://localhost:6379/0
CHROMA_HOST=localhost
CHROMA_PORT=8001

# Application Configuration
APP_ENV=development
LOG_LEVEL=DEBUG
SECRET_KEY=your-secret-key-change-in-production

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

**Verification:**
```python
# Run Python in project root
from src.config import settings
assert settings.OPENAI_API_KEY is not None
assert settings.DATABASE_URI.startswith("postgresql://")
print("✅ Configuration loaded successfully")
```

---

### ✅ AC5: FastAPI Application Running
**Given** all services and configuration are set up
**When** I run `uvicorn src.main:app --reload --host 0.0.0.0 --port 8000`
**Then** the API server starts successfully

**Expected Console Output:**
```
INFO:     Started server process [12345]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
```

**Verification:**
```bash
curl http://localhost:8000/health

# Expected Response (HTTP 200):
{
  "status": "healthy",
  "services": {
    "postgres": "up",
    "redis": "up",
    "chromadb": "up"
  },
  "version": "1.0.0"
}
```

---

### ✅ AC6: Core API Endpoints Scaffold
**Given** FastAPI application is running
**When** I access the API documentation
**Then** the following endpoints are defined (stubbed):

**Endpoint Scaffold:**

| Method | Endpoint | Description | Status |
|--------|----------|-------------|--------|
| GET | `/health` | Health check | Implemented ✅ |
| POST | `/api/v1/auth/login` | User login | Stubbed 🔶 |
| POST | `/api/v1/sessions` | Create session | Stubbed 🔶 |
| GET | `/api/v1/sessions` | List sessions | Stubbed 🔶 |
| GET | `/api/v1/sessions/{id}` | Get session | Stubbed 🔶 |
| POST | `/api/v1/sessions/{id}/messages` | Send message | Stubbed 🔶 |

**Verification:**
```bash
# Access interactive API docs
open http://localhost:8000/docs

# Test health endpoint (fully implemented)
curl http://localhost:8000/health

# Test stubbed endpoint returns 501 Not Implemented
curl -X POST http://localhost:8000/api/v1/sessions \
  -H "Content-Type: application/json" \
  -d '{"project_name": "Test"}'

# Expected Response (HTTP 501):
{
  "error": "not_implemented",
  "message": "This endpoint will be implemented in subsequent stories"
}
```

---

### ✅ AC7: Development Tooling Configured
**Given** the repository is cloned
**When** I set up development tools
**Then** the following tooling is configured:

**1. Python Dependencies Installed:**
```bash
pip install -r requirements.txt
# Expected: All dependencies install without errors
```

**2. Pre-commit Hooks (Optional but Recommended):**
```bash
pip install pre-commit
pre-commit install
# Expected: Git hooks configured for black, flake8, mypy
```

**3. VSCode Configuration:**
- `.vscode/extensions.json` recommends Python, Docker, REST Client extensions
- `.vscode/launch.json` includes debug configurations

**4. Testing Framework Ready:**
```bash
pytest --version
# Expected: pytest 7.x or higher

pytest tests/ --collect-only
# Expected: Test discovery works (even if no tests yet)
```

---

### ✅ AC8: Documentation Complete
**Given** the project is set up
**When** I read the documentation
**Then** the following documentation exists:

**1. README.md includes:**
- Project overview and purpose
- Prerequisites (Python 3.11, Docker, etc.)
- Quick start guide (5-minute setup)
- Architecture diagram link
- Contributing guidelines placeholder

**2. Bootstrap Instructions:**
```bash
# Clone repository
git clone <repo-url>
cd req-eng-platform

# Set up environment
python3.11 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env with your API keys

# Start services
docker-compose up -d

# Wait for services to be healthy (30 seconds)
sleep 30

# Initialize database
python scripts/migrate.py

# Start application
uvicorn src.main:app --reload

# Verify health
curl http://localhost:8000/health
```

**Verification:**
- A new developer can follow README and complete setup in < 15 minutes
- All commands execute successfully

---

## Technical Implementation Details

### 1. Project Structure Setup

**File: `src/main.py`** (Core FastAPI Application)
```python
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import logging
from src.config import settings
from src.api.middleware.logging import logging_middleware
from src.api.middleware.error_handler import setup_exception_handlers
from src.api.routes import health, auth, sessions

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifecycle manager"""
    logger.info("🚀 Starting Requirements Engineering Platform")

    # Initialize database connections
    from src.storage.postgres import init_database
    await init_database()

    # Initialize Redis connection
    from src.storage.redis_cache import init_redis
    await init_redis()

    # Initialize ChromaDB client
    from src.storage.vectorstore import init_vector_store
    await init_vector_store()

    logger.info("✅ All services initialized")

    yield

    logger.info("🛑 Shutting down gracefully")
    # Cleanup code here

# Create FastAPI application
app = FastAPI(
    title="Requirements Engineering Platform API",
    version="1.0.0",
    description="Multi-agent conversational requirements engineering platform",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS.split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add logging middleware
app.middleware("http")(logging_middleware)

# Setup exception handlers
setup_exception_handlers(app)

# Include routers
app.include_router(health.router, tags=["System"])
app.include_router(auth.router, prefix="/api/v1/auth", tags=["Authentication"])
app.include_router(sessions.router, prefix="/api/v1/sessions", tags=["Sessions"])

@app.get("/")
async def root():
    return {
        "message": "Requirements Engineering Platform API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }
```

---

**File: `src/config.py`** (Configuration Management)
```python
from pydantic_settings import BaseSettings
from typing import Optional
import os

class Settings(BaseSettings):
    # OpenAI Configuration
    OPENAI_API_KEY: str
    OPENAI_MODEL: str = "gpt-4-turbo-preview"
    OPENAI_EMBEDDING_MODEL: str = "text-embedding-3-small"

    # Database Configuration
    DATABASE_URI: str
    REDIS_URL: str
    CHROMA_HOST: str = "localhost"
    CHROMA_PORT: int = 8001

    # Application Configuration
    APP_ENV: str = "development"
    APP_NAME: str = "req-eng-platform"
    APP_VERSION: str = "1.0.0"
    LOG_LEVEL: str = "DEBUG"

    # Security
    SECRET_KEY: str
    JWT_SECRET: str
    JWT_ALGORITHM: str = "HS256"
    JWT_EXPIRY_HOURS: int = 24

    # CORS
    CORS_ORIGINS: str = "http://localhost:3000,http://localhost:8000"

    # Agent Configuration
    MAX_ITERATIONS: int = 10
    CONFIDENCE_THRESHOLD: float = 0.60
    VALIDATION_AMBIGUITY_THRESHOLD: float = 0.75
    SEMANTIC_SIMILARITY_THRESHOLD: float = 0.85

    # LangSmith Configuration (Optional)
    LANGSMITH_API_KEY: Optional[str] = None
    LANGSMITH_PROJECT: str = "req-eng-local"
    LANGSMITH_TRACING: bool = False
    LANGSMITH_ENDPOINT: str = "https://api.smith.langchain.com"

    class Config:
        env_file = ".env"
        case_sensitive = True

settings = Settings()
```

---

**File: `src/api/routes/health.py`** (Health Check Endpoint)
```python
from fastapi import APIRouter, status
from pydantic import BaseModel
from typing import Dict, Literal
import logging
from src.storage.postgres import check_postgres_health
from src.storage.redis_cache import check_redis_health
from src.storage.vectorstore import check_chromadb_health

logger = logging.getLogger(__name__)
router = APIRouter()

class HealthResponse(BaseModel):
    status: Literal["healthy", "degraded", "unhealthy"]
    services: Dict[str, Literal["up", "down"]]
    version: str

@router.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint

    Returns the health status of the application and all dependent services.
    """
    services = {
        "postgres": "up" if await check_postgres_health() else "down",
        "redis": "up" if await check_redis_health() else "down",
        "chromadb": "up" if await check_chromadb_health() else "down"
    }

    # Determine overall health status
    if all(s == "up" for s in services.values()):
        overall_status = "healthy"
    elif any(s == "up" for s in services.values()):
        overall_status = "degraded"
    else:
        overall_status = "unhealthy"

    logger.info(f"Health check: {overall_status}, services: {services}")

    return HealthResponse(
        status=overall_status,
        services=services,
        version="1.0.0"
    )
```

---

### 2. Database Migration Script

**File: `scripts/migrate.py`**
```python
import asyncio
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from sqlalchemy import text
from src.storage.postgres import get_engine, Base
from src.models.database import (
    SessionModel,
    ChatMessageModel,
    RequirementModel,
    RDEventModel,
    RDDocumentModel,
    AuditLogModel,
    LangGraphCheckpointModel
)
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def run_migrations():
    """Run database migrations"""
    logger.info("🔧 Starting database migrations...")

    engine = get_engine()

    async with engine.begin() as conn:
        # Create extensions
        logger.info("Creating PostgreSQL extensions...")
        await conn.execute(text("CREATE EXTENSION IF NOT EXISTS \"uuid-ossp\""))
        await conn.execute(text("CREATE EXTENSION IF NOT EXISTS \"pgcrypto\""))

        # Create all tables
        logger.info("Creating database tables...")
        await conn.run_sync(Base.metadata.create_all)

        logger.info("✅ Database migrations completed successfully")

if __name__ == "__main__":
    asyncio.run(run_migrations())
```

---

### 3. Docker Compose Configuration

**File: `docker-compose.yml`**
```yaml
version: '3.8'

services:
  postgres:
    image: postgres:15-alpine
    container_name: reqeng-postgres
    environment:
      POSTGRES_USER: reqeng
      POSTGRES_PASSWORD: password
      POSTGRES_DB: reqengdb
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

---

## Testing Strategy

### Unit Tests
```python
# tests/unit/test_config.py
def test_config_loads():
    """Test configuration loads from environment"""
    from src.config import settings
    assert settings.APP_NAME == "req-eng-platform"
    assert settings.OPENAI_MODEL is not None

# tests/unit/test_health.py
@pytest.mark.asyncio
async def test_health_endpoint():
    """Test health check endpoint"""
    from src.api.routes.health import health_check
    response = await health_check()
    assert response.status in ["healthy", "degraded", "unhealthy"]
    assert "postgres" in response.services
```

### Integration Tests
```python
# tests/integration/test_database.py
@pytest.mark.asyncio
async def test_database_connection():
    """Test database connection is established"""
    from src.storage.postgres import get_session
    async with get_session() as session:
        result = await session.execute(text("SELECT 1"))
        assert result.scalar() == 1
```

---

## Definition of Done

- [ ] All Docker services start successfully with `docker-compose up -d`
- [ ] Database migrations run without errors
- [ ] Health endpoint returns status "healthy" with all services "up"
- [ ] FastAPI application starts and serves requests
- [ ] API documentation is accessible at `/docs`
- [ ] All 8 acceptance criteria are verified and passing
- [ ] README.md contains complete setup instructions
- [ ] A fresh developer can complete setup in < 15 minutes
- [ ] No errors or warnings in console output during startup
- [ ] All configuration loads correctly from `.env` file
- [ ] Repository structure matches specification
- [ ] Code follows project conventions (Black, Flake8, MyPy)
- [ ] All files have proper docstrings and type hints

---

## Dependencies for Next Stories

Once this story is complete, the following stories can begin:

- **STORY-002:** Conversational Agent Implementation
- **STORY-003:** Extraction Agent Implementation
- **STORY-004:** LangGraph Orchestrator Setup

---

## Notes for Windsurf AI Implementation

### Key Implementation Points:

1. **Start with Docker Services**: Ensure Docker Compose configuration is correct first
2. **Database Schema Priority**: The SQLAlchemy models and migration script are critical
3. **Configuration Management**: Use Pydantic Settings for type-safe configuration
4. **Health Checks**: Implement actual connectivity checks, not just stubs
5. **Error Handling**: All database operations should have proper async error handling
6. **Logging**: Set up structured logging from the start
7. **Type Safety**: Use type hints throughout for better IDE support and error catching

### Common Pitfalls to Avoid:

- ❌ Don't skip health checks on services - they prevent startup race conditions
- ❌ Don't hardcode configuration values - use environment variables
- ❌ Don't forget to close database connections properly in async context
- ❌ Don't skip the `.gitignore` for `.env` and `__pycache__` files
- ❌ Don't start other agents until this foundation is solid

### Validation Commands:

```bash
# Complete validation checklist
docker-compose up -d && sleep 30
docker ps | grep -E "(postgres|redis|chroma)" | wc -l  # Should output: 3
python scripts/migrate.py
uvicorn src.main:app --reload &
sleep 5
curl -s http://localhost:8000/health | jq '.status'  # Should output: "healthy"
pytest tests/unit/ -v
```

---

## References

- Design Document: `DesignDocument.md`
- Design Packet 1: `DesignPacket1.md`
- Design Packet 2 Part 1: `DesignPacket2-Part1.md`
- Design Packet 2 Part 2: `DesignPacket2-Part2.md`
- Database Schema: Section 5.1 in DesignPacket2-Part2.md
- API Specification: Section 4 in DesignPacket2-Part1.md

---

**End of Story 1 Document**
