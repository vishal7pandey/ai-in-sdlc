# Story 6: Comprehensive Code Audit & DFMEA - End-to-End Integration Verification

## Story Overview

**Story ID:** STORY-006
**Story Title:** Comprehensive Code Audit, Integration Testing & Design Failure Mode Effects Analysis
**Priority:** P0 - CRITICAL (Quality Gate Before Production)
**Estimated Effort:** 32-40 hours
**Sprint:** Sprint 6 (Quality Assurance)
**Dependencies:**
- STORY-001 through STORY-005 all marked "Complete"
- Claims of working orchestrator, agents, and UI

---

## Executive Summary: The Reality Check

**Current State Assessment**: Stories 1-5 claim "completion" but based on the progress log, we have:
- ✅ **Infrastructure scaffolding** exists (Docker, DB, FastAPI skeleton)
- ⚠️ **Agent implementations** with unknown integration status
- ⚠️ **Orchestrator graph** defined but untested end-to-end
- ⚠️ **Frontend UI** components built but **NOT CONNECTED** to backend
- ❌ **NO EVIDENCE** of actual Requirements Document generation working
- ❌ **NO END-TO-END TEST** proving user input → RD output

**Critical Gap**: The stated goal is to "export a requirements document as markdown" but there is **ZERO evidence** this works. We have pieces, not a system.

---

## Implementation Audit Snapshot (Post-STORY-005 Baseline)

> This section updates the pessimistic assumptions above using the **actual code as of the STORY-005 commit baseline**. The tone remains hard and critical, but findings are grounded in what is truly implemented.

### Verified Implementations

- **Backend & Infrastructure**
  - **Database schema** is richer than originally assumed:
    - `SessionModel`, `ChatMessageModel`, `RequirementModel`, `RDDocumentModel`, `RDEventModel`, `AuditLogModel`, and LangGraph checkpoint tables all exist with **CHECK constraints**, **indexes**, and **FKs**.
    - Confidence is constrained to `[0,1]`, requirement IDs follow a `REQ(-INF)?-\d{3,}` pattern, acceptance criteria arrays are enforced non-empty, and RD formats/statuses are validated at the DB level.
  - **Postgres helper** (`src/storage/postgres.py`) provides `get_session()` and `init_database()`; used from FastAPI `lifespan` to create tables at startup.
  - **Redis integration** exists via `src/storage/redis_cache.py` with a singleton `Redis` client and `init_redis()`/health check.
  - **Vector store & embeddings**:
    - `EmbeddingService` provides deterministic, local embeddings (hash-based) so tests do **not** depend on OpenAI.
    - `VectorStoreService` is an in-memory semantic search store with cosine similarity and is used by the ExtractionAgent tests.
  - **FastAPI main app** (`src/main.py`):
    - Uses `lifespan` to initialize Postgres, Redis, and vector store.
    - Configures **CORS** correctly via `CORSMiddleware` using `settings.CORS_ORIGINS`.
    - Installs logging middleware and structured exception handlers.
    - Mounts `health`, `auth`, and `sessions` routers under `/api/v1`.
  - **Auth**: `get_current_user` is a **stub dependency** that reads `X-User-Id` and falls back to `"anonymous"`; no real auth, but all APIs are wired around a user id.

- **Orchestrator & Agents**
  - **GraphState** (`src/schemas/state.py`) is a frozen Pydantic model with all orchestrator fields, including `rd_draft`, `rd_version`, `approval_status`, `ambiguity_assessment`, and checkpoint refs.
  - **Orchestrator graph** (`src/orchestrator/graph.py`):
    - Builds a `StateGraph(GraphState)` with nodes: conversational, extraction, inference, validation, synthesis, human_review, review.
    - Wires conditional routing via `decide_next_step`, `validation_router`, and `review_router`.
    - Compiles with a **DualCheckpointer** and `interrupt_before=["human_review"]`, so LangGraph checkpoints before human review.
  - **Node wrappers** (`src/orchestrator/nodes.py`):
    - `conversational_node` and `extraction_node` invoke the corresponding agents and log start/complete with correlation IDs.
    - `inference_node`, `validation_node`, `human_review_node`, and `synthesis_node`/`review_node` exist as **placeholders**; synthesis currently writes a simple `rd_draft` string into `GraphState` but does **not** persist an `RDDocumentModel`.
  - **Agents**:
    - `ConversationalAgent` and `ExtractionAgent` both extend `BaseAgent`, use shared retry-wrapped LLM clients, and log with correlation IDs.
    - Conversational agent uses `ContextManager`, `ClarificationDetector`, `ResponseFormatter`, `ConfidenceScorer`, and updates `GraphState` fields such as `pending_clarifications`, `extracted_topics`, `last_next_action`, and `last_sentiment`.
    - Extraction agent uses `AmbiguityDetector`, `TypeClassifier`, `CriteriaGenerator`, `TraceabilityLinker`, and persists embeddings via `EmbeddingService` + `VectorStoreService`.
    - `RequirementStore` bridges Pydantic requirement models to `RequirementModel` rows, including non-functional → `"non-functional"` mapping and JSONB fields.

- **API Layer**
  - `src/api/routes/sessions.py` exposes:
    - `POST /api/v1/sessions` (create), `GET /api/v1/sessions` (list), `GET /api/v1/sessions/{id}` (detail with latest `GraphState` loaded via LangGraph checkpoints).
    - `POST /api/v1/sessions/{id}/messages` executes **one orchestrator turn** by:
      - Loading or initializing `GraphState` for the session.
      - Appending the user message.
      - Calling `graph.ainvoke(state, config)` and returning an `OrchestratorTurnResponse` with `status` and updated `state`.
    - `POST /api/v1/sessions/{id}/resume-human-review` resumes the graph after a `GraphInterrupt` for human review.
  - `tests/integration/api/test_sessions_orchestrator.py` verifies the **happy path** for session creation and sending a message through the orchestrator API, including sanity checks on the returned `state`.

- **Frontend Integration (Story-005)**
  - The frontend **is wired to the backend**:
    - `frontend/src/store/sessionStore.ts` fetches sessions from `/api/v1/sessions` and creates new sessions via `POST /api/v1/sessions`, sending `X-User-Id` headers that match the backend auth stub.
    - `SessionSidebar` lists sessions, allows creation, and navigates to `/sessions/:id`.
    - `SessionPage` calls `getSessionDetail` to fetch orchestrator state and embeds both `ChatPanel` and `RequirementsSidebar`.
    - `ChatPanel` posts messages via `sendSessionMessage` to `/api/v1/sessions/{id}/messages`, updates local `GraphState`, and implements an **offline queue** with `useOnlineStatus`.
  - **E2E tests** (Playwright) exist under `frontend/tests/e2e`:
    - `smoke.spec.ts` verifies that the home page loads and the main header renders.
    - `session-flow.spec.ts` creates a new session through the real UI (handling the `window.prompt`), waits for navigation to `/sessions/:id`, and asserts the chat input is visible. This implicitly requires the backend API to be running and the `/api/v1/sessions` endpoint to be functional.

- **Testing & Type Safety**
  - There is substantial **unit and integration test coverage** for agents, orchestrator checkpointing, extraction persistence, and the sessions API, plus frontend E2E smoke and basic session flow.
  - `mypy` runs in pre-commit with a strict configuration and is currently **clean for the codebase**.
  - Ruff linting and formatting are also enforced via pre-commit, and the latest STORY-005 baseline passes all hooks.

### Still-Missing or Partial Capabilities

- **RD generation pipeline is incomplete**:
  - While `RDDocumentModel` and `RDEventModel` exist and `synthesis_node` sets a simple `rd_draft` string in `GraphState`, there is **no end-to-end flow** that:
    - Promotes `rd_draft` into a persisted `RDDocumentModel` row.
    - Exposes RD retrieval or export endpoints.
    - Surfaces RD content in the frontend.
  - The earlier “NO RD generation” assessment was directionally correct at the **feature** level (no user-visible RD), but the **DB and orchestrator scaffolding** are in place.

- **Inference, validation, and review agents remain placeholders**:
  - Nodes exist but do not perform real validation, inference, or human-review workflows.
  - Routing hooks (`validation_router`, `review_router`) are present but exercise only minimal logic in tests.

- **Resilience, performance, and security** are still largely untested:
  - No systematic error-injection tests (DB down, Redis down, Chroma health failure, LLM timeouts).
  - No load/performance baselines for orchestrator turns or extraction throughput.
  - Auth is a simple header stub; there is no real authentication, authorization, or rate limiting.

- **Frontend coverage is thin beyond the happy path**:
  - E2E tests cover basic boot and session creation → chat UI visibility, but do **not** yet assert:
    - Requirements sidebar population from real extraction output.
    - Offline queue behaviour under actual network loss.
    - Error-handling UX, accessibility, or performance metrics.

In summary, the original checklist intentionally assumed a worst-case state. The current implementation is **substantially further along** (especially for orchestrator wiring, persistence, and frontend–backend integration) but still has **hard gaps** around RD generation/export, advanced agents (inference/validation), robustness, and production-grade security.

---

## Pessimistic Audit Checklist (200+ Verification Points)

### Section 1: Foundation Audit (Story 1)

#### 1.1 Docker Infrastructure (15 checks)

| # | Check | Expected Evidence | Status | Finding |
|---|-------|-------------------|--------|---------|
| 1.1.1 | Docker Compose starts all services | `docker compose up -d` exits 0, all containers `healthy` | ❓ | **VERIFY**: Progress says "healthy" but when was this last tested? |
| 1.1.2 | Postgres accepts connections | `psql -h localhost -U reqeng -d reqeng -c "SELECT 1"` returns 1 | ❓ | **TEST**: Can we actually connect? |
| 1.1.3 | Redis accepts commands | `redis-cli -h localhost -p 6379 PING` returns PONG | ❓ | **TEST**: Redis working? |
| 1.1.4 | ChromaDB API responsive | `curl http://localhost:8001/api/v1/heartbeat` returns 200 | ❓ | **TEST**: Chroma reachable? |
| 1.1.5 | Persistent volumes created | `docker volume ls | grep reqeng` shows 3 volumes | ❓ | **TEST**: Data persists on restart? |
| 1.1.6 | Network isolation working | Containers can reach each other via service names | ❓ | **TEST**: Network DNS resolution? |
| 1.1.7 | Health checks configured | All services have `healthcheck` in docker-compose.yml | ❓ | **VERIFY**: Grep for healthcheck |
| 1.1.8 | Port mappings correct | Postgres 5432, Redis 6379, Chroma 8001, FastAPI 8000 | ❓ | **VERIFY**: No port conflicts |
| 1.1.9 | Environment variables loaded | `.env` file exists and sourced by compose | ❓ | **VERIFY**: .env template exists |
| 1.1.10 | Container restart policies | All services have `restart: unless-stopped` | ❓ | **VERIFY**: Will restart on crash |
| 1.1.11 | Resource limits set | Memory/CPU limits prevent resource exhaustion | ❌ | **LIKELY MISSING**: No mention in progress |
| 1.1.12 | Logging configured | Container logs accessible via `docker logs` | ❓ | **TEST**: Can we see logs? |
| 1.1.13 | Backup strategy exists | Database backup script or volume snapshot plan | ❌ | **MISSING**: No backup mentioned |
| 1.1.14 | Migration rollback tested | Can undo migrations without data loss | ❌ | **MISSING**: No rollback test |
| 1.1.15 | Container security scanned | Vulnerability scan results documented | ❌ | **MISSING**: No security scan |

**Foundation Risk Score: HIGH** - Basic infrastructure exists but production-readiness unverified.

#### 1.2 Database Schema Integrity (20 checks)

| # | Check | Expected Evidence | Status | Finding |
|---|-------|-------------------|--------|---------|
| 1.2.1 | `sessions` table exists | `\d sessions` shows correct schema | ✅ | **CONFIRMED (CODE)**: `SessionModel` defined with PK, timestamps, constraints; still recommended to verify via `\d sessions`. |
| 1.2.2 | `chat_messages` table exists | `\d chat_messages` shows FK to sessions | ✅ | **CONFIRMED (CODE)**: `ChatMessageModel` has FK to `sessions.id` with `ondelete="CASCADE"`; add DB-level check. |
| 1.2.3 | `requirements` table exists | `\d requirements` shows all expected columns | ✅ | **CONFIRMED (CODE)**: `RequirementModel` defines JSONB fields, confidence, priority, etc.; schema should match in DB. |
| 1.2.4 | Indexes created | `\di` shows indexes on session_id, type, confidence | ✅ | **CONFIRMED (CODE)**: Indexes declared in `__table_args__` for sessions/requirements; still verify with `\di`. |
| 1.2.5 | Foreign key constraints | Cascading deletes work (delete session → messages deleted) | ✅ | **CONFIRMED (CODE)**: FKs from chat_messages/requirements use `ondelete="CASCADE"`; cascade behaviour still needs runtime test. |
| 1.2.6 | UUID generation works | PK defaults to uuid4 in ORM; DB stores UUIDs | ✅ | **CONFIRMED (CODE)**: Python `uuid4` used as PK default; `uuidgeneratev4()` is not used. Collision risk is negligible but not formally tested. |
| 1.2.7 | Timestamps auto-populate | `created_at` defaults to NOW() | ✅ | **CONFIRMED (CODE)**: `created_at`/`updated_at` use `server_default=func.now()`; verify actual values via SELECT. |
| 1.2.8 | JSONB fields validate | Can store/retrieve arrays and objects | ❓ | **PARTIAL (CODE)**: JSONB columns defined for metadata, acceptance_criteria, source_refs; complex query behaviour not yet exercised. |
| 1.2.9 | Enum constraints enforced | Invalid requirement types rejected | ✅ | **CONFIRMED (CODE)**: `CheckConstraint`s enforce allowed values for requirement type/priority and RD formats/status; negative tests still missing. |
| 1.2.10 | Unique constraints enforced | Can't insert duplicate requirement IDs | ✅ | **CONFIRMED (CODE)**: PK on `RequirementModel.id` and `UniqueConstraint(session_id, version)` on RD documents; collision tests still TODO. |
| 1.2.11 | Check constraints work | Confidence between 0.0-1.0 enforced | ✅ | **CONFIRMED (CODE)**: `requirements_confidence` CHECK constraint enforces 0 ≤ confidence ≤ 1; previous "likely missing" note corrected. |
| 1.2.12 | Migration idempotency | Re-running migrations doesn't break | ❓ | **TEST**: Run migrate.py twice |
| 1.2.13 | Seed data idempotency | Re-running seed doesn't duplicate | ✅ | **CONFIRMED**: Progress says "idempotent" |
| 1.2.14 | Transaction isolation | Concurrent writes don't corrupt data | ❌ | **NOT TESTED**: No concurrency tests |
| 1.2.15 | Connection pooling | SQLAlchemy pool configured for load | ❌ | **UNKNOWN**: Pool size limits? |
| 1.2.16 | Query performance | All queries < 100ms on test data | ❌ | **NOT TESTED**: No performance baselines |
| 1.2.17 | Backup/restore tested | Can restore from pg_dump | ❌ | **MISSING**: No backup test |
| 1.2.18 | Schema documentation | ERD diagram exists | ❌ | **MISSING**: No schema diagram |
| 1.2.19 | Data retention policy | Old sessions pruned automatically | ❌ | **MISSING**: No retention logic |
| 1.2.20 | Audit trail exists | Who changed what when tracked | ❌ | **MISSING**: No audit logging |

**Database Risk Score: CRITICAL** - Basic schema exists but data integrity, performance, and operational aspects unverified.

#### 1.3 FastAPI Application (15 checks)

| # | Check | Expected Evidence | Status | Finding |
|---|-------|-------------------|--------|---------|
| 1.3.1 | FastAPI app starts | `uvicorn src.main:app` starts without errors | ❓ | **TEST**: App actually starts? |
| 1.3.2 | Health endpoint works | `GET /health` returns 200 with all services "up" | ✅ | **CONFIRMED**: Progress shows curl test |
| 1.3.3 | CORS configured | Frontend can call API (localhost:5173 → localhost:8000) | ✅ | **CONFIRMED (CODE)**: `CORSMiddleware` configured in `src.main` using `settings.CORS_ORIGINS`; still validate via browser/E2E tests. |
| 1.3.4 | OpenAPI docs available | `/docs` shows Swagger UI | ❓ | **TEST**: Swagger docs work? |
| 1.3.5 | Authentication middleware | JWT validation on protected routes | ❌ | **MISSING**: Auth mentioned as "stub" only |
| 1.3.6 | Error handling middleware | All exceptions return structured JSON | ❓ | **PARTIAL (CODE)**: `setup_exception_handlers(app)` is called; add tests to confirm 4xx/5xx responses are structured. |
| 1.3.7 | Logging middleware | All requests logged with correlation IDs | ✅ | **CONFIRMED (CODE)**: `logging_middleware` attached in `src.main`; still inspect logs under load. |
| 1.3.8 | Request validation | Pydantic models reject invalid input | ❓ | **TEST**: Send bad JSON, verify 422 |
| 1.3.9 | Response models | All endpoints return typed responses | ❓ | **TEST**: Response schemas correct? |
| 1.3.10 | Rate limiting | Prevents DOS attacks | ❌ | **MISSING**: No rate limiting mentioned |
| 1.3.11 | Async consistency | All DB calls use `async with` correctly | ❓ | **CODE REVIEW**: Check all `AsyncSession` usage |
| 1.3.12 | Connection cleanup | DB connections closed properly | ❓ | **TEST**: No connection leaks under load? |
| 1.3.13 | Graceful shutdown | SIGTERM handled correctly | ❌ | **NOT TESTED**: Shutdown signal handling |
| 1.3.14 | Environment config | All settings from .env, no hardcoding | ❓ | **CODE REVIEW**: Grep for hardcoded values |
| 1.3.15 | Security headers | HTTPS, CSP, X-Frame-Options set | ❌ | **MISSING**: No security headers mentioned |

**FastAPI Risk Score: HIGH** - Basic app works but production concerns (auth, CORS, security) unaddressed.

---

### Section 2: Agent Implementation Audit (Stories 2-3)

#### 2.1 Conversational Agent Deep Dive (25 checks)

| # | Check | Expected Evidence | Status | Finding |
|---|-------|-------------------|--------|---------|
| 2.1.1 | Agent class exists | `src/agents/conversational/agent.py` with ConversationalAgent | ✅ | **CONFIRMED (CODE)**: ConversationalAgent implemented and used by orchestrator nodes. |
| 2.1.2 | BaseAgent inheritance | ConversationalAgent extends BaseAgent correctly | ✅ | **CONFIRMED (CODE)**: Inherits BaseAgent, reusing shared LLM wrapper, logging, and error-handling helpers. |
| 2.1.3 | LLM integration | OpenAI API actually called with correct prompts | ✅ | **PARTIAL (CODE)**: Uses `ChatOpenAI` with retry wrapper; tests stub `agent.llm` so real OpenAI calls are not exercised in CI. |
| 2.1.4 | Prompt template loaded | `conversational.txt` exists and used | ❓ | **PARTIAL (CODE)**: PromptBuilder loads `templates/prompts/conversational.txt`; template path wired, but file content/length not yet audited. |
| 2.1.5 | Context extraction | ContextManager identifies topics/actors/domain | ✅ | **CONFIRMED (CODE)**: ContextManager infers domain, actors, features, implicit needs, gaps, sentiment, and momentum from recent messages; behaviour still needs empirical evaluation. |
| 2.1.6 | Clarification detection | AmbiguityDetector flags vague terms | ✅ | **CONFIRMED (CODE)**: ClarificationDetector implements heuristic detection (vague adjectives, missing metrics) and generates clarifying questions. |
| 2.1.7 | Token budget enforced | Prompts stay under 8000 tokens | ❓ | **PARTIAL (CODE)**: TokenBudgetManager exists and constrains `max_response_tokens`, but prompt-truncation via token counts is not yet used in ConversationalAgent. |
| 2.1.8 | Response formatting | ResponseFormatter validates output schema | ✅ | **CONFIRMED (CODE+TEST)**: ResponseFormatter builds `ConversationalResponse` with Pydantic validation; tests cover normal and parse-error fallback paths. |
| 2.1.9 | Confidence scoring | Formula: 0.5×LLM + 0.3×parse + 0.2×clarity | ✅ | **CONFIRMED (CODE)**: ConfidenceScorer implements the STORY-002 formula; targeted tests would still be useful. |
| 2.1.10 | Error handling | LLM timeout/rate limit gracefully handled | ✅ | **PARTIAL (CODE+TEST)**: Generic `Exception` from LLM or parser triggers a structured fallback state; specific rate-limit/timeout cases are not yet simulated in tests. |
| 2.1.11 | Retry logic | Exponential backoff on failures | ✅ | **CONFIRMED (CODE)**: BaseAgent wraps `llm._generate` with Tenacity-based exponential backoff for LLM exceptions; no soak tests yet. |
| 2.1.12 | Circuit breaker | Stops calling LLM after 5 consecutive failures | ❌ | **MISSING**: No dedicated circuit breaker; repeated failures will still attempt LLM calls until the caller intervenes. |
| 2.1.13 | Logging instrumentation | All steps logged with correlation IDs | ✅ | **CONFIRMED (CODE)**: Extensive log_with_context calls in agent and nodes use `correlation_id`; logs inspected in integration tests. |
| 2.1.14 | State updates | Returns valid GraphState with updated fields | ✅ | **CONFIRMED (CODE+TEST)**: AgentResult updates chat_history and state fields; tests assert pending_clarifications, extracted_topics, last_next_action, last_sentiment, and confidence. |
| 2.1.15 | Next action determination | Sets `next_action` correctly (continue/extract) | ✅ | **CONFIRMED (CODE+TEST)**: Next action is parsed from LLM output; tests cover clarify vs non-clarify flows feeding into routing. |
| 2.1.16 | Sentiment analysis | Detects positive/neutral/negative user tone | ✅ | **CONFIRMED (CODE)**: ContextManager uses simple keyword heuristics for sentiment; accuracy not yet benchmarked. |
| 2.1.17 | Conversation momentum | Calculates engagement (0.0-1.0) | ✅ | **CONFIRMED (CODE)**: Momentum derived from `current_turn/10` clamped into [0.1, 1.0]; no UX-level tuning yet. |
| 2.1.18 | Few-shot examples | Examples selected by conversation stage | ❌ | **MISSING**: Prompt template is static; there is no dynamic few-shot example selection logic today. |
| 2.1.19 | Fallback responses | Template response when LLM fails | ✅ | **CONFIRMED (CODE+TEST)**: `_fallback_result` returns a deterministic assistant message; error tests assert fallback behaviour. |
| 2.1.20 | Multi-turn consistency | State persists across multiple invocations | ✅ | **CONFIRMED (TEST)**: Integration tests cover multi-turn conversational flows via the orchestrator graph. |
| 2.1.21 | Token counting accurate | tiktoken counts match actual usage | ❌ | **MISSING**: TokenBudgetManager can count tokens but ConversationalAgent does not currently feed real counts into budget decisions. |
| 2.1.22 | Prompt optimization | System prompt < 800 tokens as designed | ❓ | **UNKNOWN**: No automated checks enforce prompt length; template should be reviewed and potentially constrained. |
| 2.1.23 | User input sanitization | Harmful prompts rejected | ❌ | **MISSING**: No prompt injection defense |
| 2.1.24 | PII detection | Personally identifiable info flagged | ❌ | **MISSING**: No PII scanning |
| 2.1.25 | Rate limit awareness | Backs off when approaching OpenAI limits | ❌ | **MISSING**: No rate limit tracking |

**Conversational Agent Risk Score: CRITICAL** - Core logic exists but integration, error handling, and security unverified.

#### 2.2 Extraction Agent Deep Dive (30 checks)

| # | Check | Expected Evidence | Status | Finding |
|---|-------|-------------------|--------|---------|
| 2.2.1 | ExtractionAgent exists | `src/agents/extraction/agent.py` with full implementation | ✅ | **CONFIRMED (CODE)**: ExtractionAgent orchestrates type classification, criteria generation, ambiguity detection, traceability, and embeddings. |
| 2.2.2 | RequirementItem schema | Pydantic model with 13 fields + validators | ✅ | **CONFIRMED (CODE)**: Requirement schemas defined in `src/schemas.requirement`; DB mapping handled by RequirementStore. |
| 2.2.3 | Entity extraction | Actors, actions, conditions extracted via regex | ✅ | **CONFIRMED (CODE)**: EntityExtractor uses regex/heuristics to find actors, action spans, and conditions; quality not yet measured on real data. |
| 2.2.4 | Type classification | 7 requirement types classified correctly | ✅ | **CONFIRMED (CODE)**: TypeClassifier implements rule-based mapping over RequirementType; no accuracy metrics or golden set yet. |
| 2.2.5 | Acceptance criteria | Generated from templates with context | ✅ | **CONFIRMED (CODE)**: CriteriaGenerator returns templated, testable acceptance criteria with sensible defaults for performance/auth flows. |
| 2.2.6 | Traceability linking | Requirements link to source chat turns | ✅ | **CONFIRMED (CODE)**: TraceabilityLinker produces `chat:turn:{idx}` refs based on keyword overlap, with fallback to most recent user message. |
| 2.2.7 | Ambiguity detection | Flags vague terms and suggests clarifications | ✅ | **CONFIRMED (CODE)**: AmbiguityDetector computes ambiguity_score, ambiguous_terms, and suggestions; exposed via ExtractionAgent and tested indirectly. |
| 2.2.8 | Requirement ID generation | REQ-001, REQ-002, etc. sequential and unique | ❓ | **PARTIAL (CODE)**: DB enforces ID pattern and uniqueness; ID allocation policy (sequential vs UUID-like) is not centrally defined or tested. |
| 2.2.9 | Title generation | First 50 chars of action, truncated properly | ❓ | **UNKNOWN**: Title derivation logic is present in requirement schemas/agent, but no explicit tests assert length and readability constraints. |
| 2.2.10 | Confidence calculation | Weighted by ambiguity + parse quality | ✅ | **PARTIAL (CODE)**: ExtractionAgent adjusts requirement confidence based on ambiguity flags; no holistic quality metric (precision/recall) yet. |
| 2.2.11 | Multiple requirements | Extracts all requirements from conversation | ❓ | **PARTIAL (TEST)**: Integration tests cover single-requirement flows; behaviour on long, multi-requirement conversations not stress-tested. |
| 2.2.12 | Requirement deduplication | Doesn't extract same requirement twice | ❌ | **MISSING**: No explicit deduplication layer; repeated statements may yield duplicated RequirementModel rows. |
| 2.2.13 | Database persistence | `RequirementStore` saves to Postgres | ✅ | **CONFIRMED (CODE+TEST)**: RequirementStore maps Pydantic models to RequirementModel and back; integration tests cover save + load round-trips. |
| 2.2.14 | JSONB fields | `acceptance_criteria` and `source_refs` stored correctly | ❓ | **PARTIAL (CODE)**: JSONB columns are defined and populated from lists; query patterns and index usage for JSONB fields remain untested. |
| 2.2.15 | Enum mapping | Pydantic enums → DB strings correctly | ✅ | **CONFIRMED (CODE)**: RequirementType/Priority map to DB strings with non-functional → "non-functional" normalization; reverse mapping handled in RequirementStore. |
| 2.2.16 | Vector embeddings | Each requirement embedded and stored | ✅ | **CONFIRMED (CODE+TEST)**: ExtractionAgent calls EmbeddingService and VectorStoreService; integration tests verify vectors are persisted and queryable. |
| 2.2.17 | Semantic search | Can find similar requirements by embedding | ✅ | **CONFIRMED (TEST)**: Extraction embedding tests assert semantic_search returns the expected requirement with score above threshold. |
| 2.2.18 | Embedding consistency | Same text → same embedding | ✅ | **CONFIRMED (CODE)**: EmbeddingService uses deterministic hashing; identical text yields identical embeddings by design. |
| 2.2.19 | Extraction performance | < 2 seconds per requirement | ❌ | **NOT TESTED**: No timing assertions or perf baselines for ExtractionAgent or downstream persistence. |
| 2.2.20 | Error recovery | Partial extraction saved even if some fail | ❓ | **PARTIAL (CODE)**: ExtractionAgent has a coarse-grained fallback path; mid-stream errors within multi-requirement outputs are not explicitly handled or tested. |
| 2.2.21 | Logging completeness | All extraction steps logged | ✅ | **CONFIRMED (CODE)**: ExtractionAgent logs start, end, and embedding failures with correlation IDs; log coverage under edge cases still to be reviewed. |
| 2.2.22 | State updates | `requirements[]` and `ambiguous_items[]` updated | ✅ | **CONFIRMED (CODE+TEST)**: _convert_output populates requirements_update and ambiguous_items; tests assert resulting GraphState fields. |
| 2.2.23 | Metadata enrichment | Extraction metadata (tokens, duration) tracked | ✅ | **PARTIAL (CODE+TEST)**: Extraction metadata (tokens_used, duration_ms, model) is stored in state; tests assert presence, not yet used for monitoring. |
| 2.2.24 | Concurrent extraction | Multiple extractions don't interfere | ❌ | **NOT TESTED**: No concurrency or multi-session stress tests; in-memory vector store and DB writes assumed safe but unproven under load. |
| 2.2.25 | Requirement validation | Schema validators reject invalid data | ❓ | **PARTIAL (CODE)**: Pydantic schemas enforce types and constraints; negative-path tests for invalid requirements are missing. |
| 2.2.26 | Type fallback | Defaults to "functional" if classification unsure | ✅ | **CONFIRMED (CODE)**: TypeClassifier returns RequirementType.FUNCTIONAL when no indicators match. |
| 2.2.27 | Actor fallback | Defaults to "user" or "system" if no actor found | ✅ | **CONFIRMED (CODE)**: EntityExtractor falls back to ["system"] when no actor pattern matches. |
| 2.2.28 | Criteria minimum | Always generates ≥1 acceptance criterion | ✅ | **CONFIRMED (CODE)**: CriteriaGenerator appends a simple `{actor} can {action}` criterion if templated generation fails. |
| 2.2.29 | Source refs minimum | Always has ≥1 source reference | ✅ | **CONFIRMED (CODE)**: TraceabilityLinker guarantees at least one `chat:turn:{idx}` reference via fallbacks. |
| 2.2.30 | Extraction metrics | Precision, recall, F1 score measured | ❌ | **MISSING**: No dataset or metrics pipeline to quantify extraction quality; only functional correctness is asserted. |

**Extraction Agent Risk Score: HIGH** - Core extraction works but quality metrics, performance, and edge cases unverified.

---

### Section 3: Orchestrator Audit (Story 4)

#### 3.1 LangGraph Graph Definition (20 checks)

| # | Check | Expected Evidence | Status | Finding |
|---|-------|-------------------|--------|---------|
| 3.1.1 | Graph file exists | `src/orchestrator/graph.py` with `build_graph()` | ✅ | **CONFIRMED (CODE)**: Graph constructed and compiled at import time; used throughout API and tests. |
| 3.1.2 | GraphState schema | TypedDict with all 20+ fields defined | ✅ | **CONFIRMED (CODE)**: `GraphState` is a frozen Pydantic model in `src/schemas/state.py` with orchestrator fields including rd_draft and approval_status. |
| 3.1.3 | Nodes added | conversational, extraction, inference, validation, synthesis, human_review, review | ✅ | **CONFIRMED (CODE)**: All nodes registered in `build_graph()`; inference/validation/review remain placeholders. |
| 3.1.4 | Entry point set | `set_entry_point('conversational')` | ✅ | **CONFIRMED (CODE)**: Entry edge from START → "conversational" configured via `workflow.add_edge(START, "conversational")`. |
| 3.1.5 | Normal edges | extraction → validation, synthesis → review | ✅ | **CONFIRMED (CODE)**: Graph wires extraction→validation, inference→validation, synthesis→human_review→review. |
| 3.1.6 | Conditional edges | 3 routing functions implemented | ✅ | **CONFIRMED (CODE)**: `decide_next_step`, `validation_router`, and `review_router` are defined in `routing.py` and used in `build_graph()`. |
| 3.1.7 | Routing: conversational | `decide_next_step()` returns extract/continue/validate | ❓ | **PARTIAL (CODE)**: Logic covers error escalation, next_action, and document-intent keywords; behaviour under varied real inputs not yet exhaustively tested. |
| 3.1.8 | Routing: validation | `validation_router()` returns pass/fail/needs_inference | ❓ | **PARTIAL (CODE)**: Routes based on confidence, critical validation_issues, and inferred requirements; minimal tests. |
| 3.1.9 | Routing: review | `review_router()` returns approved/revision/pending | ❓ | **PARTIAL (CODE)**: Maps approval_status to approved/revision/pending; end-to-end review flows not yet exercised. |
| 3.1.10 | Graph compiles | `workflow.compile()` succeeds without errors | ✅ | **CONFIRMED (CODE+TEST)**: Graph compiles on import; orchestrator tests invoke `graph.ainvoke` successfully. |
| 3.1.11 | Graph visualization | `draw_ascii()` shows expected structure | ❌ | **NOT TESTED**: No visualization run |
| 3.1.12 | Checkpointing configured | PostgresSaver or RedisSaver attached | ✅ | **CONFIRMED (CODE+TEST)**: DualCheckpointer writes to Redis + Postgres; checkpoint integration tested in `test_checkpointing`. |
| 3.1.13 | Thread ID usage | Each session has unique thread_id | ✅ | **CONFIRMED (CODE+TEST)**: Thread IDs use `session-{session_id}`; checkpoint tests and sessions API rely on this convention. |
| 3.1.14 | State persistence | State saved after each node | ✅ | **PARTIAL (TEST)**: Checkpoint tests verify rows in `langgraph_checkpoints`; per-node persistence frequency is governed by LangGraph internals and not fully characterised. |
| 3.1.15 | State retrieval | `get_state()` returns latest checkpoint | ✅ | **PARTIAL (CODE+TEST)**: `graph.aget_state` used in sessions API and checkpoint tests; more edge-case coverage (missing/corrupt state) needed. |
| 3.1.16 | Interrupt mechanism | Human review node pauses execution | ✅ | **PARTIAL (CODE)**: Graph compiled with `interrupt_before=["human_review"]` and GraphInterrupt is handled in sessions routes; no dedicated integration tests for full human-review loop. |
| 3.1.17 | Error node exists | Error handler node for failures | ❓ | **CODE REVIEW**: Error handler added? |
| 3.1.18 | Loop prevention | Max 10 iterations enforced | ❌ | **MISSING**: `should_continue_iteration` helper exists but is not wired into the graph; no guard against infinite routing loops beyond business logic. |
| 3.1.19 | Timeout handling | Long-running graphs timeout gracefully | ❌ | **NOT TESTED**: No timeout tests |
| 3.1.20 | Graph metrics | Execution time per node tracked | ❌ | **MISSING**: No metrics mentioned |

**LangGraph Risk Score: HIGH** - Graph structure exists but execution, checkpointing, and error handling unverified.

#### 3.2 Node Implementation (25 checks)

| # | Check | Expected Evidence | Status | Finding |
|---|-------|-------------------|--------|---------|
| 3.2.1 | Node wrappers exist | `src/orchestrator/nodes.py` with all 6 nodes | ✅ | **CONFIRMED (CODE)**: conversational, extraction, inference, validation, synthesis, human_review, and review nodes implemented. |
| 3.2.2 | conversational_node | Calls ConversationalAgent.invoke() | ✅ | **CONFIRMED (CODE+TEST)**: Node delegates to `_conversational_agent.invoke(state)` and is exercised in conversational flow and API integration tests. |
| 3.2.3 | extraction_node | Calls ExtractionAgent.invoke() | ✅ | **CONFIRMED (CODE+TEST)**: Node delegates to `_extraction_agent.invoke(state)`; extraction flow and E2E extraction tests cover this path. |
| 3.2.4 | inference_node | Inference agent integrated | ❌ | **PLACEHOLDER**: Node logs a no-op and returns state unchanged; no inference agent connected. |
| 3.2.5 | validation_node | Validation logic implemented | ❌ | **PLACEHOLDER**: Node logs and returns state unchanged; no validation agent yet, though routing logic exists. |
| 3.2.6 | synthesis_node | RD generation implemented | ❌ | **PARTIAL**: Node logs and sets a simple `rd_draft` string and increments `rd_version`; no RDDocument persistence or user-facing RD yet. |
| 3.2.7 | review_node | Human review flow implemented | ❌ | **PLACEHOLDER**: Node logs approval_status and returns state unchanged; human-in-the-loop workflow not implemented. |
| 3.2.8 | Node error handling | Try/except in each node | ❓ | **PARTIAL (CODE)**: Nodes rely on BaseAgent.invoke error handling and retries; they do not wrap agent calls in additional try/except. |
| 3.2.9 | State immutability | Nodes return new state dict, don't mutate | ✅ | **CONFIRMED (CODE)**: GraphState is frozen; agents and synthesis_node use `with_updates` to return new instances. |
| 3.2.10 | State merging | Updated state merged correctly | ✅ | **PARTIAL (CODE+TEST)**: BaseAgent._merge_result merges chat_history, requirements, and state_updates; agent tests cover typical merges, but not all edge combinations. |
| 3.2.11 | last_agent tracking | Every node sets `last_agent` | ✅ | **PARTIAL (CODE)**: BaseAgent defaults last_agent to agent name; conversational/extraction nodes respect this, placeholders rely on prior state. |
| 3.2.12 | iterations increment | Every node increments `iterations` | ✅ | **PARTIAL (CODE)**: BaseAgent increments iterations on each successful invoke; placeholder nodes do not increment by design. |
| 3.2.13 | error_count tracking | Failures increment `error_count` | ✅ | **PARTIAL (CODE)**: BaseAgent._handle_failure increments error_count and degrades confidence; no dedicated tests for multi-failure scenarios. |
| 3.2.14 | confidence updates | Confidence adjusted by each agent | ✅ | **PARTIAL (CODE+TEST)**: Conversational and extraction agents update confidence; validation/inference/review do not yet contribute. |
| 3.2.15 | chat_history updates | Messages appended correctly | ✅ | **CONFIRMED (CODE+TEST)**: BaseAgent merges chat_history_update into state; conversational agent tests assert resulting chat history length. |
| 3.2.16 | requirements updates | New requirements added to list | ✅ | **CONFIRMED (CODE+TEST)**: ExtractionAgent produces requirements_update; integration tests assert requirements length and contents. |
| 3.2.17 | Async consistency | All nodes are async functions | ✅ | **CONFIRMED (CODE)**: All node wrappers are `async def` and used by LangGraph. |
| 3.2.18 | Agent initialization | Agents initialized once, reused | ✅ | **CONFIRMED (CODE)**: `_conversational_agent` and `_extraction_agent` created at module import time and reused as singletons. |
| 3.2.19 | Node logging | Each node logs start/complete | ✅ | **PARTIAL (CODE+TEST)**: Nodes log start/completion with correlation IDs; logs observed in integration tests but not formally asserted. |
| 3.2.20 | Node performance | Nodes complete in < 5 seconds | ❌ | **NOT TESTED**: No timing budgets or performance assertions for node execution. |
| 3.2.21 | Node isolation | One node failure doesn't crash graph | ❓ | **PARTIAL (CODE)**: BaseAgent degrades state on failures, but orchestrator-level behaviour under repeated node failures is not systematically tested. |
| 3.2.22 | Node retries | Failed nodes can be retried | ❌ | **MISSING**: LLM calls are retried, but there is no graph-level node retry mechanism or replay control. |
| 3.2.23 | Node caching | Expensive ops cached | ❌ | **NOT IMPLEMENTED**: No caching |
| 3.2.24 | Node observability | Metrics exported per node | ❌ | **MISSING**: No Prometheus metrics |
| 3.2.25 | Node testing | Each node has unit tests | ❓ | **TEST**: Unit tests exist? |

**Node Risk Score: CRITICAL** - Only 2 of 6 nodes implemented (conversational, extraction). **Synthesis node missing = NO RD GENERATION**.

#### 3.3 API Integration (20 checks)

| # | Check | Expected Evidence | Status | Finding |
|---|-------|-------------------|--------|---------|
| 3.3.1 | Sessions endpoint | POST /api/v1/sessions creates session | ✅ | **CONFIRMED (CODE+TEST)**: Implemented in `src/api/routes/sessions.py`; integration test `test_sessions_orchestrator` verifies creation and response shape. |
| 3.3.2 | Send message endpoint | POST /api/v1/sessions/{id}/messages | ✅ | **CONFIRMED (CODE+TEST)**: Implemented and exercised in `test_sessions_orchestrator`, which asserts returned orchestrator state. |
| 3.3.3 | Get requirements | GET /api/v1/sessions/{id}/requirements | ❌ | **MISSING**: No dedicated requirements endpoint; requirements are only exposed via GraphState in session detail/turn responses. |
| 3.3.4 | Generate RD endpoint | POST /api/v1/rd/{session_id}/generate | ❌ | **CRITICAL**: No RD generation endpoint |
| 3.3.5 | Get RD endpoint | GET /api/v1/rd/{session_id} | ❌ | **CRITICAL**: Can't retrieve RD |
| 3.3.6 | Export RD endpoint | GET /api/v1/rd/{session_id}/export | ❌ | **CRITICAL**: No export endpoint |
| 3.3.7 | Graph invocation | API calls `graph.ainvoke(state, config)` | ✅ | **CONFIRMED (CODE+TEST)**: Sessions routes call `graph.ainvoke` with thread-specific config; integration tests confirm basic end-to-end execution. |
| 3.3.8 | Thread ID usage | Uses `session-{id}` as thread_id | ✅ | **CONFIRMED (CODE)**: All orchestrator calls use `thread_id = f"session-{session_id}"`; checkpointing tests depend on this convention. |
| 3.3.9 | State loading | Loads session state before invocation | ✅ | **PARTIAL (CODE+TEST)**: `GET /sessions/{id}` and message/resume endpoints use `graph.aget_state`; behaviour under missing or corrupt checkpoints only partially covered. |
| 3.3.10 | State saving | Saves state after invocation | ✅ | **PARTIAL (TEST)**: Checkpoint tests show rows written after graph runs; explicit tests for every transition are missing. |
| 3.3.11 | Response extraction | Extracts assistant message from state | ❓ | **PARTIAL (CODE)**: API returns full GraphState; assistant messages are present in `chat_history`, but there is no thin response DTO just for the latest assistant utterance. |
| 3.3.12 | Error responses | Returns structured errors (500, 404, 422) | ❓ | **TEST**: Error handling works? |
| 3.3.13 | Async endpoints | All endpoints use `async def` | ✅ | **CONFIRMED (CODE)**: Sessions routes are async functions using async DB + LangGraph calls. |
| 3.3.14 | Request validation | Pydantic models validate input | ✅ | **PARTIAL (CODE)**: CreateSessionRequest, SendMessageRequest, and HumanReviewDecision used as request models; negative-path tests are limited. |
| 3.3.15 | Response models | Return types specified | ✅ | **CONFIRMED (CODE)**: Response models declared on routes (SessionResponse, SessionDetailResponse, OrchestratorTurnResponse). |
| 3.3.16 | CORS headers | Frontend can call API | ✅ | **CONFIRMED (CODE)**: CORS configured globally in `src.main`; browser/E2E verification still recommended. |
| 3.3.17 | Authentication | JWT tokens validated | ❌ | **NOT IMPLEMENTED**: Auth stub only |
| 3.3.18 | Rate limiting | API protected from abuse | ❌ | **MISSING**: No rate limits |
| 3.3.19 | Request logging | All API calls logged | ❓ | **TEST**: Logs show API requests? |
| 3.3.20 | OpenAPI docs | /docs endpoint works | ❓ | **TEST**: Swagger UI loads? |

**API Risk Score: CRITICAL** - Basic endpoints may exist but **NO RD GENERATION OR EXPORT ENDPOINTS = CORE FEATURE MISSING**.

---

### Section 4: Frontend Audit (Story 5)

#### 4.1 Frontend Build & Deployment (15 checks)

| # | Check | Expected Evidence | Status | Finding |
|---|-------|-------------------|--------|---------|
| 4.1.1 | Frontend directory exists | `frontend/` with package.json | ✅ | **CONFIRMED (CODE)**: `frontend/package.json` defines a Vite + React + TS app with Playwright E2E support; runtime build still needs verification. |
| 4.1.2 | Dependencies installed | `node_modules/` exists | ❓ | **TEST**: npm install worked? |
| 4.1.3 | TypeScript compiles | `tsc` exits 0 with no errors | ❓ | **TEST**: No TS errors? |
| 4.1.4 | Vite dev server | `npm run dev` starts on port 5173 | ❓ | **TEST**: Dev server works? |
| 4.1.5 | Production build | `npm run build` succeeds | ❓ | **TEST**: Build passes? |
| 4.1.6 | Build artifacts | `dist/` contains index.html, assets/ | ❓ | **VERIFY**: Dist folder created? |
| 4.1.7 | Bundle size | Main bundle < 200KB gzipped | ❌ | **NOT MEASURED**: No bundle analysis |
| 4.1.8 | Preview server | `npm run preview` serves build | ❓ | **TEST**: Preview works? |
| 4.1.9 | Environment variables | `.env` with VITE_API_URL, VITE_WS_URL | ❓ | **VERIFY**: .env exists? |
| 4.1.10 | ESLint passes | `npm run lint` exits 0 | ❓ | **TEST**: No lint errors? |
| 4.1.11 | Prettier formatted | All files formatted | ❓ | **TEST**: npm run format works? |
| 4.1.12 | No console errors | Browser console clean on load | ❓ | **TEST**: Check console? |
| 4.1.13 | No TypeScript errors | No red squiggles in VS Code | ❓ | **VERIFY**: IDE clean? |
| 4.1.14 | Hot module replacement | Changes reflect instantly | ❓ | **TEST**: HMR works? |
| 4.1.15 | Source maps | Can debug TypeScript in browser | ❌ | **NOT TESTED**: Source maps work? |

**Frontend Build Risk Score: MEDIUM** - Build tooling probably works but not verified.

#### 4.2 Frontend-Backend Integration (30 checks)

| # | Check | Expected Evidence | Status | Finding |
|---|-------|-------------------|--------|---------|
| 4.2.1 | API client configured | `src/lib/api/client.ts` with base URL | ✅ | **CONFIRMED (CODE)**: `apiFetch` uses `/api/v1` base URL and attaches `X-User-Id`; helpers exist for `getSessionDetail` and `sendSessionMessage`. |
| 4.2.2 | CORS working | No CORS errors in browser console | ❓ | **PARTIAL (CODE)**: Backend CORS is configured and frontend uses same-origin `/api/v1` paths; browser verification under real deploy still recommended. |
| 4.2.3 | Create session call | Frontend → POST /api/v1/sessions | ✅ | **CONFIRMED (CODE+E2E)**: `SessionSidebar` + `sessionStore` POST to `/api/v1/sessions`; `session-flow.spec.ts` exercises this via the real backend. |
| 4.2.4 | Send message call | Frontend → POST /api/v1/sessions/{id}/messages | ✅ | **CONFIRMED (CODE)**: ChatPanel calls `sendSessionMessage` which POSTs to `/api/v1/sessions/{id}/messages`; no E2E test yet asserts the full message round-trip. |
| 4.2.5 | Get requirements call | Frontend → GET /api/v1/requirements | ❌ | **MISSING**: No dedicated requirements API call; requirements are consumed via `GraphState` returned from the sessions API. |
| 4.2.6 | WebSocket connection | Frontend connects to ws://localhost:8000/ws | ❌ | **NOT TESTED**: WS connection works? |
| 4.2.7 | WS authentication | Token sent in query param or header | ❌ | **NOT IMPLEMENTED**: No WS auth |
| 4.2.8 | WS message handling | Frontend processes `message.chunk` events | ❌ | **NOT TESTED**: Event handling works? |
| 4.2.9 | WS reconnection | Auto-reconnects on disconnect | ❌ | **NOT TESTED**: Reconnection logic? |
| 4.2.10 | WS heartbeat | Ping/pong every 30s | ❌ | **NOT TESTED**: Heartbeat works? |
| 4.2.11 | Optimistic updates | Messages appear instantly in UI | ❌ | **NOT TESTED**: Optimistic UI works? |
| 4.2.12 | Streaming response | Characters appear one-by-one | ❌ | **NOT TESTED**: Streaming works? |
| 4.2.13 | Requirement cards | Requirements appear in right panel | ❌ | **NOT TESTED**: Cards render? |
| 4.2.14 | Confidence scores | Progress bars show correct values | ❌ | **NOT TESTED**: Scores accurate? |
| 4.2.15 | RD generation trigger | "Generate RD" button calls API | ❌ | **BROKEN**: No API endpoint exists |
| 4.2.16 | RD preview | RD content displays in viewer | ❌ | **BROKEN**: No RD to display |
| 4.2.17 | Markdown export | Download button triggers file download | ❌ | **BROKEN**: No RD to export |
| 4.2.18 | Offline mode | Queues messages when offline | ❓ | **PARTIAL (CODE)**: ChatPanel uses `useOnlineStatus` and a local queue to buffer messages; no automated tests simulate offline/online transitions. |
| 4.2.19 | Error toasts | API errors show as notifications | ❌ | **MISSING**: ChatPanel shows inline error banners; there is no central toast system wired to API failures. |
| 4.2.20 | Loading states | Spinners show during API calls | ❓ | **PARTIAL (CODE)**: SessionSidebar and SessionPage expose basic loading text; no spinners or E2E assertions yet. |
| 4.2.21 | Empty states | Friendly messages when no data | ✅ | **CONFIRMED (CODE)**: ChatPanel and SessionSidebar render helpful empty states when there are no messages or sessions. |
| 4.2.22 | Session persistence | LocalStorage saves current session | ❌ | **MISSING**: Only `userId` is persisted in localStorage; selected session is not stored. |
| 4.2.23 | Session resume | Can resume after page refresh | ❌ | **MISSING**: No dedicated resume logic beyond server-side session detail fetch. |
| 4.2.24 | Multi-session support | Can switch between sessions | ✅ | **PARTIAL (CODE)**: SessionSidebar allows selecting and navigating between multiple sessions; no E2E coverage yet. |
| 4.2.25 | Message history | Scrolling loads older messages | ❌ | **NOT IMPLEMENTED**: Infinite scroll? |
| 4.2.26 | Markdown rendering | Chat messages render markdown | ❌ | **NOT TESTED**: Markdown works? |
| 4.2.27 | Code syntax highlight | Code blocks highlighted | ❌ | **NOT IMPLEMENTED**: Syntax highlighting? |
| 4.2.28 | Responsive design | Works on mobile/tablet | ❓ | **PARTIAL (CODE)**: Layout uses responsive Tailwind classes with a mobile toggle for chat/requirements; no responsive or cross-device tests. |
| 4.2.29 | Accessibility | Keyboard navigation works | ❓ | **PARTIAL (CODE)**: Some ARIA roles/labels are present (e.g. offline banner, send button), but there is no systematic accessibility testing. |
| 4.2.30 | Performance | Lighthouse score ≥85 | ❌ | **NOT TESTED**: No Lighthouse run |

**Frontend Integration Risk Score: HIGH** - Frontend is wired to the backend for sessions and chat, with basic E2E coverage, but RD features, WebSockets, requirements UI assertions, and non-functional aspects remain unimplemented or untested.

---

### Section 5: End-to-End Integration Tests (40 checks)

#### 5.1 Happy Path Flow (20 checks)

| # | Check | Expected Evidence | Status | Finding |
|---|-------|-------------------|--------|---------|
| 5.1.1 | User opens UI | http://localhost:5173 loads | ✅ | **CONFIRMED (E2E)**: `smoke.spec.ts` loads `/` and asserts the main header is visible. |
| 5.1.2 | Create session | Clicks "New Session", enters name | ✅ | **CONFIRMED (E2E)**: `session-flow.spec.ts` clicks "New", handles the prompt, and creates a session via the real backend. |
| 5.1.3 | Session appears | Sidebar shows new session | ✅ | **PARTIAL (E2E)**: `session-flow.spec.ts` navigates to `/sessions/:id` after creation and verifies the chat UI; the sidebar entry is implied but not explicitly asserted. |
| 5.1.4 | Send message | Types "Users need login", presses Enter | ❌ | **NOT TESTED**: No E2E test |
| 5.1.5 | Message echoes | User message appears in chat | ❌ | **NOT TESTED**: No E2E test |
| 5.1.6 | API called | POST /api/v1/sessions/{id}/messages | ❌ | **NOT TESTED**: No E2E test |
| 5.1.7 | Orchestrator invoked | Graph executes conversational node | ❌ | **NOT TESTED**: No E2E test |
| 5.1.8 | LLM called | OpenAI API receives prompt | ❌ | **NOT TESTED**: No E2E test |
| 5.1.9 | Response streamed | WS sends message.chunk events | ❌ | **NOT TESTED**: No E2E test |
| 5.1.10 | Response displayed | AI message appears character-by-character | ❌ | **NOT TESTED**: No E2E test |
| 5.1.11 | Confidence shown | Confidence badge appears | ❌ | **NOT TESTED**: No E2E test |
| 5.1.12 | Second message | User provides more details | ❌ | **NOT TESTED**: No E2E test |
| 5.1.13 | Extraction triggered | Orchestrator routes to extraction node | ❌ | **NOT TESTED**: No E2E test |
| 5.1.14 | Requirements extracted | ExtractionAgent runs | ❌ | **NOT TESTED**: No E2E test |
| 5.1.15 | Requirements saved | Postgres receives INSERT | ❌ | **NOT TESTED**: No E2E test |
| 5.1.16 | Embeddings stored | ChromaDB receives embeddings | ❌ | **NOT TESTED**: No E2E test |
| 5.1.17 | Requirements displayed | Cards appear in right panel | ❌ | **NOT TESTED**: No E2E test |
| 5.1.18 | Generate RD clicked | User clicks "Generate RD" | ❌ | **NOT TESTED**: No E2E test |
| 5.1.19 | RD generated | Synthesis agent creates markdown | ❌ | **BROKEN**: Synthesis agent doesn't exist |
| 5.1.20 | RD displayed | Markdown preview shows | ❌ | **BROKEN**: No RD to display |

**Happy Path Risk Score: HIGH** - Only minimal Playwright tests exist (UI boot + session creation). There is **no end-to-end coverage** for the full conversation → extraction → RD generation → export flow.

#### 5.2 Error Scenarios (20 checks)

| # | Check | Expected Evidence | Status | Finding |
|---|-------|-------------------|--------|---------|
| 5.2.1 | OpenAI rate limit | Handles 429 gracefully | ❌ | **NOT TESTED**: No error test |
| 5.2.2 | OpenAI timeout | Handles request timeout | ❌ | **NOT TESTED**: No error test |
| 5.2.3 | Database connection lost | Handles DB disconnect | ❌ | **NOT TESTED**: No error test |
| 5.2.4 | Redis unavailable | Handles Redis down | ❌ | **NOT TESTED**: No error test |
| 5.2.5 | ChromaDB unavailable | Handles Chroma down | ❌ | **NOT TESTED**: No error test |
| 5.2.6 | Invalid user input | Rejects malformed JSON | ❌ | **NOT TESTED**: No error test |
| 5.2.7 | Invalid requirement | Rejects bad requirement schema | ❌ | **NOT TESTED**: No error test |
| 5.2.8 | State corruption | Handles invalid GraphState | ❌ | **NOT TESTED**: No error test |
| 5.2.9 | Checkpoint load failure | Handles missing checkpoint | ❌ | **NOT TESTED**: No error test |
| 5.2.10 | WebSocket disconnect | Auto-reconnects | ❌ | **NOT TESTED**: No error test |
| 5.2.11 | Network offline | Shows offline banner | ❌ | **NOT TESTED**: No error test |
| 5.2.12 | Message queue overflow | Limits queued messages | ❌ | **NOT TESTED**: No error test |
| 5.2.13 | Session not found | Returns 404 | ❌ | **NOT TESTED**: No error test |
| 5.2.14 | Unauthorized request | Returns 401 | ❌ | **NOT TESTED**: No error test (no auth) |
| 5.2.15 | Concurrent requests | Handles race conditions | ❌ | **NOT TESTED**: No concurrency test |
| 5.2.16 | Large message | Handles 10KB+ messages | ❌ | **NOT TESTED**: No size test |
| 5.2.17 | Long conversation | Handles 100+ turns | ❌ | **NOT TESTED**: No long test |
| 5.2.18 | Graph timeout | Handles stuck graph | ❌ | **NOT TESTED**: No timeout test |
| 5.2.19 | Agent crash | Handles agent exception | ❌ | **NOT TESTED**: No crash test |
| 5.2.20 | Memory leak | No memory growth over time | ❌ | **NOT TESTED**: No memory test |

**Error Scenario Risk Score: CRITICAL** - **NO ERROR TESTING AT ALL**. System will fail in production.

---

## Critical Missing Components Identified

### 🔴 SHOWSTOPPER ISSUES (Must Fix Before ANY Demo)

1. **NO SYNTHESIS AGENT** ❌
   - **Impact**: Cannot generate Requirements Document
   - **Evidence**: Progress log mentions "placeholder" only
   - **Required**: Full synthesis agent implementation
   - **Effort**: 16-24 hours

2. **NO RD GENERATION ENDPOINT** ❌
   - **Impact**: Frontend has no way to trigger RD generation
   - **Evidence**: No POST /api/v1/rd/{id}/generate endpoint mentioned
   - **Required**: API endpoint + synthesis node integration
   - **Effort**: 4-8 hours

3. **NO RD EXPORT FUNCTIONALITY** ❌
   - **Impact**: Can't download the markdown file
   - **Evidence**: No export endpoint or file download logic
   - **Required**: Export endpoint + file serving
   - **Effort**: 2-4 hours

4. **FRONTEND NOT CONNECTED TO BACKEND** ❌
   - **Impact**: UI doesn't work, all features broken
   - **Evidence**: No CORS, no integration tests, no E2E tests
   - **Required**: CORS configuration + integration testing
   - **Effort**: 8-12 hours

5. **NO END-TO-END TESTS** ❌
   - **Impact**: No proof the system works at all
   - **Evidence**: Progress shows unit tests only, no E2E
   - **Required**: Full E2E test suite (Playwright/Cypress)
   - **Effort**: 16-24 hours

**Total Showstopper Fix Effort: 46-72 hours**

### 🟠 HIGH-PRIORITY GAPS (Breaks User Experience)

6. **NO INFERENCE AGENT** ❌
   - Mentioned as "placeholder" in progress
   - Required for proposing implicit requirements
   - Effort: 12-16 hours

7. **NO VALIDATION AGENT** ❌
   - No validation logic mentioned
   - Required for checking requirement quality
   - Effort: 12-16 hours

8. **NO HUMAN REVIEW FLOW** ❌
   - No review node or approval workflow
   - Required for human-in-the-loop
   - Effort: 8-12 hours

9. **NO AUTHENTICATION** ❌
   - Auth mentioned as "stub" only
   - Required for multi-user support
   - Effort: 8-12 hours

10. **NO WEBSOCKET IMPLEMENTATION** ❌
    - WebSocket protocol designed but not implemented
    - Required for real-time streaming
    - Effort: 12-16 hours

**Total High-Priority Fix Effort: 52-72 hours**

### 🟡 MEDIUM-PRIORITY GAPS (Quality/Reliability)

11. No error handling tests
12. No performance benchmarks
13. No security testing
14. No load testing
15. No backup/restore procedures
16. No monitoring/alerting
17. No rate limiting
18. No caching layer
19. No graceful degradation
20. No rollback procedures

**Total Medium-Priority Fix Effort: 40-60 hours**

---

## Design Failure Mode & Effects Analysis (DFMEA)

### DFMEA Template for Each Component

| Component | Failure Mode | Effect | Severity (1-10) | Occurrence (1-10) | Detection (1-10) | RPN | Recommended Action |
|-----------|--------------|--------|-----------------|-------------------|------------------|-----|-------------------|
| **Synthesis Agent** | Not implemented | No RD generation | 10 | 10 | 1 | **1000** | IMPLEMENT IMMEDIATELY |
| **RD Export API** | Endpoint missing | Cannot download RD | 10 | 10 | 1 | **1000** | IMPLEMENT IMMEDIATELY |
| **Frontend-Backend** | Not connected | UI doesn't work | 10 | 10 | 1 | **1000** | FIX CORS + TEST |
| **E2E Tests** | Don't exist | No proof system works | 9 | 10 | 1 | **900** | WRITE E2E TESTS |
| **WebSocket** | Not implemented | No real-time updates | 8 | 10 | 1 | **800** | IMPLEMENT WS SERVER |
| **Validation Agent** | Not implemented | Poor requirement quality | 8 | 9 | 2 | **720** | IMPLEMENT VALIDATION |
| **Inference Agent** | Not implemented | Incomplete requirements | 7 | 9 | 2 | **630** | IMPLEMENT INFERENCE |
| **Authentication** | Stub only | Security vulnerability | 9 | 8 | 1 | **720** | IMPLEMENT AUTH |
| **Error Handling** | Not tested | Production failures | 9 | 7 | 3 | **567** | ADD ERROR TESTS |
| **Performance** | Not measured | Slow/unusable | 7 | 7 | 4 | **392** | RUN BENCHMARKS |

**RPN = Risk Priority Number (Severity × Occurrence × Detection)**

**High-Risk Items (RPN > 500)**: 8 of 10 failures are HIGH RISK

---

## Recommended Test Plan (160+ Tests)

### Phase 1: Unit Tests (60 tests, 12 hours)
- [ ] 20 tests for conversational agent components
- [ ] 20 tests for extraction agent components
- [ ] 10 tests for orchestrator routing logic
- [ ] 10 tests for API endpoints

### Phase 2: Integration Tests (40 tests, 16 hours)
- [ ] 10 tests for agent → database flow
- [ ] 10 tests for agent → vector store flow
- [ ] 10 tests for orchestrator → agents flow
- [ ] 10 tests for API → orchestrator flow

### Phase 3: End-to-End Tests (20 tests, 24 hours)
- [ ] 10 tests for happy path scenarios
- [ ] 10 tests for error scenarios

### Phase 4: Non-Functional Tests (40 tests, 16 hours)
- [ ] 10 performance tests (load, stress, spike)
- [ ] 10 security tests (auth, XSS, injection)
- [ ] 10 resilience tests (chaos, failure injection)
- [ ] 10 usability tests (accessibility, UX)

**Total Test Effort: 68 hours**

---

## Audit Findings Summary

### Coverage Analysis

| Category | Implemented | Tested | Working | Coverage |
|----------|-------------|--------|---------|----------|
| Infrastructure | 80% | 30% | 70% | **MEDIUM** |
| Database | 90% | 40% | 80% | **MEDIUM** |
| FastAPI | 70% | 40% | 60% | **LOW** |
| Conversational Agent | 90% | 70% | 75% | **MEDIUM** |
| Extraction Agent | 95% | 70% | 80% | **MEDIUM** |
| Orchestrator Graph | 80% | 40% | 60% | **LOW** |
| Orchestrator Nodes | 60% | 30% | 40% | **HIGH** |
| API Endpoints | 60% | 30% | 40% | **HIGH** |
| Frontend Build | 80% | 20% | 60% | **MEDIUM** |
| Frontend-Backend | 70% | 30% | 60% | **HIGH** |
| End-to-End Flow | 30% | 10% | 20% | **SHOWSTOPPER** |

### Risk Assessment

| Risk Category | Risk Level | Justification |
|---------------|------------|---------------|
| **Core Feature Delivery** | 🔴 **CRITICAL** | RD generation/export still missing end-to-end despite RD models and a simple synthesis placeholder. |
| **System Integration** | 🔴 **HIGH** | Backend, agents, and frontend are wired for sessions + chat with minimal E2E, but there is no integrated RD flow and only partial orchestrator coverage. |
| **Production Readiness** | 🔴 **CRITICAL** | Central logging and exception handlers exist, but there is no monitoring, alerting, load/perf/chaos testing, or backup/restore validation. |
| **Code Quality** | 🟠 **HIGH** | Strong typing and many unit/integration tests are in place, but non-functional tests, negative paths, and concurrency/perf coverage are largely absent. |
| **User Experience** | � **HIGH** | Core UI for sessions and chat works with basic E2E checks, but RD UX, streaming, rich error/offline handling, and accessibility are missing or untested. |
| **Data Integrity** | 🟠 **HIGH** | Schema constraints and requirement persistence are solid, yet there is no tested backup, retention policy, or fully used audit trail. |
| **Security** | 🔴 **CRITICAL** | Auth is a header stub with no real authentication/authorization, no rate limiting, and no systematic security testing beyond basic Pydantic validation. |
| **Scalability** | 🟡 **MEDIUM** | Architecture can likely handle small loads, but there is no load testing, caching strategy, or capacity planning. |

---

## Recommended Story 6 Acceptance Criteria

### ✅ AC1: Synthesis Agent Implementation
**Given** extraction has requirements
**When** synthesis node is invoked
**Then** a complete Requirements Document is generated in markdown format

**Verification:**
- [ ] `src/agents/synthesis/agent.py` exists with SynthesisAgent class
- [ ] Agent generates valid markdown with all sections
- [ ] All extracted requirements included in RD
- [ ] Traceability preserved in RD
- [ ] Unit tests pass for synthesis agent
- [ ] Integration test: extraction → synthesis → RD

### ✅ AC2: RD Generation API Endpoints
**Given** a session with requirements
**When** RD generation is requested
**Then** API returns generated RD and enables download

**Verification:**
- [ ] POST /api/v1/rd/{session_id}/generate endpoint exists
- [ ] GET /api/v1/rd/{session_id} returns RD content
- [ ] GET /api/v1/rd/{session_id}/export triggers file download
- [ ] Integration test: API → synthesis → response
- [ ] OpenAPI docs show all RD endpoints

### ✅ AC3: Frontend-Backend Integration
**Given** frontend and backend running
**When** frontend makes API calls
**Then** CORS allows requests and data flows correctly

**Verification:**
- [ ] CORS configured for localhost:5173
- [ ] Frontend can create session (POST /sessions)
- [ ] Frontend can send messages (POST /sessions/{id}/messages)
- [ ] Frontend can get requirements (GET /requirements)
- [ ] Frontend can generate RD (POST /rd/{id}/generate)
- [ ] Frontend can download RD (GET /rd/{id}/export)
- [ ] No CORS errors in browser console
- [ ] Integration test: UI → API → DB → UI

### ✅ AC4: WebSocket Implementation
**Given** frontend and backend running
**When** user sends message
**Then** response streams in real-time via WebSocket

**Verification:**
- [ ] WebSocket server endpoint /ws exists
- [ ] Frontend connects successfully
- [ ] Authentication on connection (if auth implemented)
- [ ] Heartbeat every 30 seconds
- [ ] message.chunk events stream correctly
- [ ] requirements.extracted events fire
- [ ] agent.update events fire
- [ ] Auto-reconnection on disconnect
- [ ] Integration test: WS connection → messages → disconnect → reconnect

### ✅ AC5: End-to-End Happy Path Test
**Given** clean database and running system
**When** E2E test executes
**Then** full flow completes successfully

**Test Flow:**
1. User opens UI (http://localhost:5173)
2. User creates session "Test Project"
3. User sends message "Users need login with email/password"
4. AI responds with clarifying questions
5. User provides details "Login under 2 seconds on 4G"
6. Requirements extracted (2 requirements appear)
7. User clicks "Generate RD"
8. RD preview shows with both requirements
9. User clicks "Export Markdown"
10. File downloads with correct content

**Verification:**
- [ ] Playwright/Cypress E2E test exists
- [ ] Test passes consistently (5 consecutive runs)
- [ ] Test completes in < 2 minutes
- [ ] Screenshots captured at each step
- [ ] Downloaded RD file contains both requirements
- [ ] RD markdown is valid and parseable

### ✅ AC6: Error Scenario Testing
**Given** various failure conditions
**When** system encounters errors
**Then** graceful degradation occurs

**Test Scenarios:**
- [ ] OpenAI rate limit: Shows error, offers retry
- [ ] Database disconnect: Shows error, caches locally
- [ ] WebSocket disconnect: Auto-reconnects, queues messages
- [ ] Invalid input: Shows validation errors
- [ ] Network offline: Shows offline banner, queues operations
- [ ] Agent timeout: Shows timeout, offers manual entry
- [ ] Concurrent requests: No race conditions
- [ ] Large input (10KB): Handles gracefully
- [ ] Long conversation (100 turns): No memory leaks
- [ ] Browser refresh: Session resumes correctly

### ✅ AC7: Performance Baseline
**Given** test data and load generator
**When** performance tests run
**Then** system meets performance targets

**Targets:**
- [ ] API response time: < 200ms (p95)
- [ ] LLM call time: < 3s (p95)
- [ ] RD generation: < 5s for 10 requirements
- [ ] WebSocket latency: < 100ms
- [ ] Frontend load time: < 2s on 3G
- [ ] Concurrent users: 10 users without degradation
- [ ] Memory usage: Stable over 1 hour
- [ ] Database queries: All < 100ms

### ✅ AC8: Code Quality & Documentation
**Given** completed codebase
**When** quality checks run
**Then** all standards met

**Verification:**
- [ ] ESLint: 0 errors, 0 warnings (frontend)
- [ ] Ruff: 0 errors (backend)
- [ ] MyPy: 100% type coverage (backend)
- [ ] Test coverage: ≥80% (backend), ≥70% (frontend)
- [ ] All functions have docstrings
- [ ] README has complete setup instructions
- [ ] API documentation complete (OpenAPI)
- [ ] Architecture diagram exists
- [ ] Sequence diagrams for key flows

### ✅ AC9: Security Baseline
**Given** security testing tools
**When** security scan runs
**Then** no critical vulnerabilities

**Verification:**
- [ ] Dependency scan (npm audit, pip-audit): 0 critical
- [ ] SQL injection test: Protected by Pydantic + SQLAlchemy
- [ ] XSS test: All inputs sanitized
- [ ] CSRF protection: CSRF tokens on state-changing endpoints
- [ ] Authentication: JWT validation working (if implemented)
- [ ] Authorization: Role-based access control (if implemented)
- [ ] Secrets: No hardcoded API keys, use .env
- [ ] HTTPS: SSL/TLS in production config

### ✅ AC10: Deployment Readiness
**Given** production-like environment
**When** deployment checklist verified
**Then** system ready for production

**Verification:**
- [ ] Docker images build successfully
- [ ] Docker Compose production config exists
- [ ] Environment variables documented
- [ ] Database migrations tested (up + down)
- [ ] Backup script exists and tested
- [ ] Monitoring configured (health checks, metrics)
- [ ] Logging configured (structured logs, rotation)
- [ ] Error tracking configured (Sentry or equivalent)
- [ ] Load balancer config (if applicable)
- [ ] Deployment runbook exists

---

## Definition of Done for Story 6

**Only mark complete when ALL of the following are TRUE:**

- [ ] Synthesis agent implemented and tested
- [ ] RD generation API endpoints working
- [ ] Frontend connected to backend (CORS working)
- [ ] WebSocket server implemented and tested
- [ ] End-to-end happy path test passes
- [ ] 10 error scenario tests pass
- [ ] Performance benchmarks meet targets
- [ ] Code quality checks pass (linting, types, coverage)
- [ ] Security scan passes (no critical vulnerabilities)
- [ ] Deployment checklist complete
- [ ] **MOST IMPORTANT**: Demo to product owner showing full flow from conversation → download RD file

**Until ALL these are done, system is NOT READY FOR DEMO.**

---

## Estimated Effort to Complete System

### Remaining Work Breakdown

| Task Category | Estimated Hours | Priority |
|---------------|----------------|----------|
| **Synthesis Agent** | 16-24 | P0 - Showstopper |
| **RD API Endpoints** | 4-8 | P0 - Showstopper |
| **WebSocket Server** | 12-16 | P0 - Showstopper |
| **Frontend Integration** | 8-12 | P0 - Showstopper |
| **End-to-End Tests** | 16-24 | P0 - Showstopper |
| **Inference Agent** | 12-16 | P1 - High |
| **Validation Agent** | 12-16 | P1 - High |
| **Error Handling** | 8-12 | P1 - High |
| **Performance Testing** | 8-12 | P2 - Medium |
| **Security Hardening** | 8-12 | P2 - Medium |
| **Documentation** | 8-12 | P2 - Medium |
| **Deployment Config** | 4-8 | P2 - Medium |

**Total Remaining Effort: 116-172 hours** (approximately **3-4 weeks** of full-time work)

---

## Immediate Action Items (Next 24 Hours)

1. **RUN CURRENT SYSTEM** ⏱️ 2 hours
   - [ ] Docker compose up
   - [ ] Verify all services healthy
   - [ ] Run existing tests
   - [ ] Document what actually works

2. **IMPLEMENT SYNTHESIS AGENT** ⏱️ 8 hours
   - [ ] Create synthesis agent class
   - [ ] Implement markdown generation
   - [ ] Write unit tests
   - [ ] Integrate with orchestrator

3. **CREATE RD ENDPOINTS** ⏱️ 4 hours
   - [ ] POST /api/v1/rd/{id}/generate
   - [ ] GET /api/v1/rd/{id}
   - [ ] GET /api/v1/rd/{id}/export
   - [ ] Test with curl

4. **VERIFY CORS & API PROXY** ⏱️ 1 hour
   - [ ] Confirm existing `CORSMiddleware` settings allow frontend calls without errors
   - [ ] Manually test from frontend (create session + send message)
   - [ ] Adjust allowed origins only if issues are observed

5. **STABILIZE & EXTEND E2E TESTS** ⏱️ 8 hours
   - [ ] Run existing Playwright tests (smoke + session-flow) against local stack
   - [ ] Fix any flakiness and document current coverage limits
   - [ ] Define next E2E scenario for conversation → extraction → RD once RD endpoints exist

**Total Immediate Work: 23 hours (3 days)**

---

## Conclusion

**Current State: ~60-70% of core architecture implemented**

**Reality Check:**
- ✅ Infrastructure and database schema are in place with constraints, indexes, and checkpointing.
- ✅ Conversational and extraction agents are wired through the orchestrator and covered by unit/integration tests.
- ⚠️ Orchestrator graph and nodes are implemented, but inference/validation/review remain placeholders and are only lightly tested via routing.
- ⚠️ Frontend is connected to the backend for sessions and chat, but has no RD UI, streaming, or rich error/offline coverage.
- ❌ No end-to-end RD generation/export flow exists yet (core demo feature still missing).
- ❌ No single E2E test exercises the full conversation → extraction → RD generation → export pipeline.

**Recommendation**: **DO NOT DEMO** until Story 6 acceptance criteria are met. Current system is 60-70% of the way there but the missing 30-40% are the most critical parts.

**Path Forward:**
1. Complete Story 6 (this document) - 3-4 weeks
2. Only then can you have a working demo
3. Future stories can add enhancements (collaboration, PDF export, etc.)

**This document provides the roadmap to get from "code exists" to "system works".**
