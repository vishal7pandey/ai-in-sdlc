# Project Progress Log

This document captures the end-of-story status for the project. Each story entry summarizes what was implemented, how it was verified, and any follow-up considerations. Future stories should append their own sections using the same template.

## Story Index

| Story ID | Title                                      | Status     | Completion Date | Notes                                     |
|----------|--------------------------------------------|------------|-----------------|-------------------------------------------|
| STORY-001| Project Foundation & Core Infrastructure     | Completed  | 2025-11-16      | Baseline repo + tooling ready             |
| STORY-002| Conversational Agent Implementation          | Completed  | 2025-11-16      | Conversational agent + tests              |
| STORY-003| Requirements Extraction Agent Implementation | Completed  | 2025-11-16      | Extraction agent + persistence + search   |
| STORY-004| LangGraph Orchestrator - Multi-Agent Workflow | Completed | 2025-11-17      | Orchestrator graph + checkpointing + API  |
| STORY-005| Frontend MVP - Interactive Chat UI with Requirements Visualization | Completed | 2025-11-18      | Chat UI + requirements sidebar + offline + E2E |

---

## STORY-001 – Project Foundation & Core Infrastructure Setup
**Status:** ✅ Completed — 2025-11-16
**Scope:** Establish repository structure, Docker services, database schema, FastAPI skeleton, tooling, and documentation needed to begin feature development.

### Implementation Overview
- Created full repository scaffold (src modules, scripts, tests, stories) with consistent `__init__.py` placement and documentation of requirements (@stories/story-1-document.md).
- Provisioned Docker Compose stack (Postgres 15, Redis 7, ChromaDB 0.4.24) with persistent volumes, health checks, and tuned Chroma defaults plus Python-based heartbeat probe (@docker-compose.yml#1-58).
- Implemented SQLAlchemy Async models, migration script, and idempotent seed script to bootstrap session + requirement data (@scripts/migrate.py, @scripts/seed_data.py).
- Built FastAPI application entrypoint wiring config, middleware (logging + error handling), routers (`/health`, auth, sessions stubs), and async storage helpers (Postgres, Redis, Chroma) (@src/main.py and related modules).
- Configured development tooling: `uv` environment & lockfile, `.python-version`, `.pre-commit-config.yaml`, setup scripts, and README quick-start instructions for onboarding (@README.md, scripts/setup-dev.*).

### Verification Evidence
- **Docker:** `docker compose up -d` → `reqeng-postgres`, `reqeng-redis`, `reqeng-chromadb` all report `(healthy)` via `docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"`.
- **Migrations:** `uv run python scripts/migrate.py` → logs "✅ Database migrations completed successfully".
- **Seed data:** `uv run python scripts/seed_data.py` → logs "Demo requirement already present; skipping insert" followed by "✅ Seed data ensured" (idempotent reruns).
- **Tests:** `uv run pytest tests/unit tests/integration -q` → 3 tests passed (`test_config`, `test_health`, `test_database`).
- **Health endpoint:** `uv run uvicorn src.main:app --host 127.0.0.1 --port 8000` (background) + `curl http://localhost:8000/health` → `{"status":"healthy","services":{"postgres":"up","redis":"up","chromadb":"up"},"version":"1.0.0"}` confirming all dependencies reachable.

### Follow-ups / Next Stories
1. STORY-002 and beyond can build conversational agents, requirements extraction workflows, and LangGraph orchestration on top of this foundation.
2. Keep this progress log updated after each story by appending a new section with the same headings (Implementation Overview, Verification Evidence, Follow-ups).

---

## STORY-002 – Conversational Agent Implementation
**Status:** ✅ Completed — 2025-11-16
**Scope:** Build the initial conversational agent (AC1–AC12): module scaffolding, BaseAgent, prompt system, context/clarification components, response formatting, token budgeting, confidence scoring, error handling, logging, and documentation.

### Implementation Overview
- Created conversational package (`agent.py`, `prompt_builder.py`, `context_manager.py`, `clarification_detector.py`, `response_formatter.py`, `token_budget.py`, `confidence_scorer.py`) plus schemas/state fields to persist topics, clarifications, next action, and sentiment.
- Added fallback + retry logic, structured logging, and confidence scoring to `ConversationalAgent`; instrumented logs for start/context/LLM/parse/completion (see README logging section).
- Expanded unit coverage (helpers, token budget, confidence scorer, error handling) and multi-turn integration test to satisfy AC5–AC12.
- Documented STORY-002 in README with module overview and `uv run python scripts/log_convo_sample.py` verification command; script emits log sample for AC12 evidence.

### Verification Evidence
- `uv run pytest tests/unit tests/integration -q` → 16 tests passed (unit + integration suites for conversational components, agent flow, DB connectivity).
- `uv run python scripts/log_convo_sample.py` → emits INFO/DEBUG logs showing correlation IDs, context extraction, LLM call, and completion with confidence (AC12 proof).

### Follow-ups / Next Stories
1. Future stories can build extraction/validation agents on top of the conversational outputs now persisted in `GraphState`.
2. Consider migrating Pydantic configs to ConfigDict to silence deprecation warnings when convenient.

---

## STORY-003 – Requirements Extraction Agent Implementation
**Status:** ✅ Completed — 2025-11-16
**Scope:** Implement the ExtractionAgent (AC1–AC12): module scaffolding, requirement schema, extraction helpers (entities, types, criteria, traceability, ambiguity), agent orchestration, confidence scoring, logging, database persistence, semantic embeddings, and end-to-end tests.

### Implementation Overview
- Created extraction package (`agent.py`, `prompt_builder.py`, `entity_extractor.py`, `type_classifier.py`, `criteria_generator.py`, `traceability_linker.py`, `ambiguity_detector.py`, `schemas.py`) plus state fields to persist `ambiguous_items` and `extraction_metadata` (@src/agents/extraction/*, @src/schemas/state.py).
- Implemented `RequirementItem` schema with validation for IDs, acceptance criteria, and `source_refs` (@src/schemas/requirement.py), and wired it into `GraphState.requirements`.
- Built `ExtractionAgent` orchestration using `BaseAgent`, LangChain `PydanticOutputParser`, and the extraction helpers to enrich requirements with actors, conditions, types, generated criteria, traceability links, and ambiguity assessment (@src/agents/extraction/agent.py).
- Added extraction-specific confidence scoring that combines LLM-reported extraction confidence, per-requirement confidence, and penalties for ambiguous items; fallback paths reduce session confidence while preserving state.
- Implemented `RequirementStore` for persisting `RequirementItem` instances to Postgres, including type/priority mapping between enums and the `RequirementModel` ORM (@src/storage/requirement_store.py, @src/models/database.py).
- Implemented `EmbeddingService` for deterministic, test-friendly text embeddings and an in-memory `VectorStoreService` that exposes add/search APIs suitable for semantic requirement search (@src/services/embedding_service.py, @src/storage/vectorstore.py).
- Wired `ExtractionAgent` to call `_store_embeddings(...)` after successful extraction so that each requirement’s embedding and metadata are available for semantic similarity queries (@src/agents/extraction/agent.py).

### Verification Evidence
- **Unit & agent tests:**
  - `uv run pytest tests/unit/agents/extraction -q` — covers entity extraction, type classification, criteria generation, traceability linking, ambiguity detection, prompt building, and the ExtractionAgent happy path.
- **Integration tests (agent flow):**
  - `uv run pytest tests/integration/agents/test_extraction_flow.py -q` — validates multi-turn extraction flow, enrichment of `GraphState.requirements`, `ambiguous_items`, and `extraction_metadata`.
- **Integration tests (persistence & embeddings):**
  - `uv run pytest tests/integration/test_requirement_store.py -q` — verifies `RequirementStore` can save and retrieve requirements, including JSONB fields and non-functional type mapping.
  - `uv run pytest tests/integration/agents/test_extraction_embedding.py -q` — verifies `ExtractionAgent` populates the in-memory vector store and supports semantic search over extracted requirements.
  - `uv run pytest tests/integration/agents/test_extraction_e2e.py -q` — exercises the end-to-end flow: conversation → extraction → DB persistence → embedding storage → semantic search.

### Follow-ups / Next Stories
1. Replace the in-memory `VectorStoreService` with a concrete Chroma-backed implementation once semantic infrastructure is ready for production use.
2. Add API endpoints and orchestrator wiring so that persisted requirements and their embeddings can be queried from the web API.
3. Promote STORY-003 from "In Progress" to "Completed" once all integration commands above have been run and validated in CI.

---

## STORY-004 – LangGraph Orchestrator Implementation – Agent Workflow & Checkpointing
**Status:** ✅ Completed — 2025-11-17
**Scope:** Implement the LangGraph-based orchestrator that coordinates conversational and extraction agents in a stateful workflow, with Redis/Postgres checkpointing, conditional routing, human-in-the-loop review, and API integration.

### Implementation Overview
- Defined a LangGraph `StateGraph` using the frozen Pydantic `GraphState` model as the state schema, and wired nodes for `conversational`, `extraction`, `inference`, `validation`, `synthesis`, `human_review`, and `review` (@src/orchestrator/graph.py, @src/orchestrator/nodes.py).
- Implemented routing helpers (`decide_next_step`, `validation_router`, `review_router`, `should_continue_iteration`) that operate on `GraphState` and drive conditional transitions (@src/orchestrator/routing.py).
- Implemented a `DualCheckpointer` that conforms to LangGraph's `BaseCheckpointSaver` interface and persists checkpoints to Redis (fast) and Postgres (durable), including JSON-safe serialization of Pydantic models (@src/orchestrator/checkpointer.py, @src/models/database.py#LangGraphCheckpointModel).
- Compiled the orchestrator graph with `interrupt_before=["human_review"]`, so the workflow can pause for human review before the `review` node executes, and added logging in `human_review_node` for observability (@src/orchestrator/graph.py, @src/orchestrator/nodes.py).
- Wired FastAPI session endpoints:
  - `POST /api/v1/sessions` to create `SessionModel` rows.
  - `GET /api/v1/sessions` and `GET /api/v1/sessions/{session_id}` to list and inspect sessions plus latest `GraphState`.
  - `POST /api/v1/sessions/{session_id}/messages` to run one orchestrator turn, returning an `OrchestratorTurnResponse` that captures normal completion or an interrupt.
  - `POST /api/v1/sessions/{session_id}/resume-human-review` to apply a human decision and resume the graph after a human-review interrupt.
  (@src/api/routes/sessions.py, @src/schemas/api.py, @src/api/dependencies/auth.py).
- Integrated a lightweight auth dependency `get_current_user` that sources `user_id` from the `X-User-Id` header and threads it into `SessionModel.user_id` and `GraphState.user_id` (@src/api/dependencies/auth.py, @src/api/routes/sessions.py).

### Verification Evidence
- **Checkpointing integration:** `uv run pytest tests/integration/orchestrator/test_checkpointing.py -q` — verifies that running the compiled graph stores checkpoints in Postgres via `DualCheckpointer` and that checkpoints can be restored via the checkpointer API.
- **API integration:** `uv run pytest tests/integration/api/test_sessions_orchestrator.py -q` — exercises the happy path for `POST /api/v1/sessions` plus `POST /api/v1/sessions/{session_id}/messages`, asserting that:
  - Sessions are created with correct metadata and user_id.
  - Orchestrator turns return an `OrchestratorTurnResponse` with `status` in `{ "ok", "interrupt" }` and a well-formed `state` whose `session_id`, `project_name`, and `user_id` match the created session.
- **Agent-level tests:** existing STORY-002 and STORY-003 tests continue to pass, confirming that the orchestrator uses the already-verified conversational and extraction agents.

### Design Deviations & Notes
- **State schema vs story TypedDict:** The story described `GraphState` as a `TypedDict` in `src/orchestrator/state.py` with keys such as `next_action`, `validated`, `last_updated`, and `checkpoint_id`. The implementation instead uses a **Pydantic model** in `src/schemas/state.py`, re-exported via `src/orchestrator/state.py`. Some field names differ:
  - `last_next_action` instead of `next_action`.
  - No explicit `validated` or `last_updated` fields; the same concepts are captured via confidence, validation metadata, and logging.
  - `checkpoint_ref` / `parent_checkpoint_ref` instead of `checkpoint_id` / `parent_checkpoint_id` to avoid clashing with LangGraph's internal checkpoint channels.
  This is functionally equivalent and better aligned with Pydantic v2 and LangGraph, but future design docs should reference the **Pydantic `GraphState` model** and its actual field names.
- **StateManager abstraction:** The story's AC7 showed a `StateManager` (e.g., `src/storage/state_manager`) responsible for loading and saving session state. The implementation instead delegates directly to LangGraph's checkpointer (`graph.aget_state`, `graph.ainvoke`) plus the existing SQLAlchemy `SessionModel`, with no separate `StateManager` class. LangGraph's checkpointing effectively *is* the state manager for orchestrator state. Future design can either:
  - Embrace this pattern and treat the checkpointer as the canonical state store, or
  - Introduce a thin `StateManager` wrapper around `graph.aget_state` / `graph.ainvoke` if additional indirection or cross-cutting concerns (masking, analytics) are needed.
- **API contract shape:** The story's example response model focused on returning a `ChatMessageResponse` (assistant message + basic context). The final implementation returns an `OrchestratorTurnResponse` that includes a `status` flag (`"ok"` vs `"interrupt"`), an optional `interrupt_type`, and the full `GraphState`. This richer contract better supports human-in-the-loop flows and debugging. Future client designs should depend on `OrchestratorTurnResponse` as the canonical orchestrator response type.
- **Human-in-the-loop implementation:** The story described an interrupt-style `human_review` node. The implementation uses LangGraph's `interrupt_before=["human_review"]` mechanism plus dedicated FastAPI endpoints to surface and resume interrupts. This is slightly more sophisticated than the illustrative design, but consistent with LangGraph's current interrupt APIs.

### Follow-ups / Next Stories
1. If future stories introduce additional agents (e.g., inference, validation, synthesis) with real implementations, update the graph wiring and routing rules to reflect their richer logic while preserving the existing checkpointing and interrupt patterns.
2. Consider adding more API tests around the human-review interrupt and resume flow (e.g., simulate an interrupt, call `resume-human-review`, assert that `approval_status` and downstream behavior are correct).
3. Update design documentation for `GraphState` and API contracts to match the implemented Pydantic models and `OrchestratorTurnResponse` shape, so future design work starts from the current reality.

---

## STORY-005 – Frontend MVP: Interactive Chat UI with Requirements Visualization

**Status:** ✅ Completed — 2025-11-18
**Scope:** Deliver a React/Vite frontend MVP that lets users create sessions, chat with the AI, see extracted requirements, preview/export an RD draft, and exercise the end-to-end orchestrator via the browser. This story focuses on the core workflow and defers full WebSocket streaming and deeper perf/a11y work to follow-up stories.

### Implementation Overview

- **Frontend scaffold and tooling (AC1):**
  - Initialized a Vite 5 + React 18 + TypeScript SPA in `frontend/` with TailwindCSS 3, Zustand, and basic ESLint configuration.
  - Implemented core structure under `src/`: layout components, session store, chat panel, requirements sidebar, pages, and API types/client.
  - Added tooling scripts: `npm run dev`, `build`, `lint`, `format` (ESLint `--fix`), and Playwright `test:e2e`.

- **Layout and navigation (AC2):**
  - Implemented `AppLayout` with header, left `SessionSidebar`, and main content area hosting chat + requirements (`AppLayout.tsx`, `Header.tsx`).
  - Defined routes in `App.tsx` using React Router (`/` → `HomePage`, `/sessions/:sessionId` → `SessionPage`), with lazy loading via `React.lazy` + `Suspense` for better initial load behavior.
  - `HomePage` provides a simple landing view instructing users to create/select a session.

- **Session management UI (AC3):**
  - Built `useSessionStore` (Zustand) to call backend REST endpoints for `GET /api/v1/sessions` and `POST /api/v1/sessions`, mapping `SessionResponse` into a lightweight `Session` model.
  - Implemented `SessionSidebar` to:
    - Load sessions on mount (`loadSessions`).
    - Prompt for a project name and create a new session via `createSession`, then navigate to `/sessions/:id`.
    - Show a list of sessions with active highlighting and navigation on click.

- **Chat interface and GraphState wiring (AC4):**
  - Implemented `ChatPanel` to:
    - Display conversation history from `GraphState.chat_history` with basic styling for user vs assistant messages.
    - Handle user input and send messages via `sendSessionMessage` (`POST /api/v1/sessions/{id}/messages`), updating the current `GraphState` via an `onStateUpdate` callback.
  - `SessionPage` owns the `GraphState | null` state, loads initial state via `getSessionDetail`, and passes it into `ChatPanel` and `RequirementsSidebar`.

- **Requirements sidebar and RD draft (AC6 & AC7):**
  - Replaced the earlier `RequirementsPanel` with a unified `RequirementsSidebar` that:
    - Renders requirement cards from `state.requirements` with title, type, actor/action, and a numeric confidence indicator.
    - Adds a simple color-coded confidence bar under each requirement (green/yellow/orange bands by confidence bucket).
    - Shows a plain-text RD draft preview from `state.rd_draft`.
    - Provides an "Export MD" button that downloads the RD draft as a `.md` file for external editing.

- **Offline handling and message queue (AC8 partial):**
  - Added `useOnlineStatus` hook that tracks `navigator.onLine` and `online`/`offline` events.
  - Wired `ChatPanel` to:
    - Show an offline banner (`role="status"`, `aria-live="polite"`) when disconnected.
    - Disable the chat input/send button after a small queued-message threshold.
    - Queue messages locally while offline and automatically flush them sequentially to the backend when connectivity returns, updating `GraphState` on each successful send.

- **Responsive design for mobile/desktop (AC9):**
  - Updated `AppLayout` so the session sidebar is collapsible on small screens via a header "menu" button (`Header` → `onToggleSidebar`).
  - In `SessionPage`, added a mobile-only toggle bar to switch between Chat and Requirements views (`mobileView` state), while keeping side-by-side Chat + Requirements on `md+` breakpoints.

- **Performance and build configuration (AC10 partial):**
  - Configured `vite.config.ts` with a modest `build.chunkSizeWarningLimit` and ensured the dev server proxies `/api` to `VITE_API_URL` (default `http://localhost:8000`).
  - Implemented route-level code-splitting by lazy-loading `HomePage` and `SessionPage` with `React.lazy` + a full‑screen loading fallback.

- **Accessibility improvements (AC11 partial):**
  - Marked the offline banner as a live region (`role="status"`, `aria-live="polite"`).
  - Added `aria-label="Send message"` on the chat send button and maintained good color contrast in the dark theme.
  - Kept semantics reasonably clean (headings, buttons, inputs), though a full WCAG/axe pass is deferred.

- **E2E tests and scripts (AC12 partial):**
  - Added Playwright config (`playwright.config.ts`) with a Chromium project, dev-server integration (`npm run dev`), and base URL `http://localhost:5173`.
  - Implemented two E2E tests under `tests/e2e/`:
    - `smoke.spec.ts` — verifies the app boots and renders the main header text.
    - `session-flow.spec.ts` — creates a new session from the sidebar (handling `window.prompt`), navigates to `/sessions/:id`, and asserts the visible chat input is present, exercising the REST API and UI wiring.
  - Added `npm run test:e2e` and `npm run format` scripts and resolved initial ESLint issues (`SessionPage` effects, `sessionStore` typing).

### Verification Evidence

- **Local development and manual flows:**
  - `npm install` (in `frontend/`) followed by `npm run dev` → Vite dev server on `http://localhost:5173` with no TypeScript or ESLint errors.
  - `uv run uvicorn src.main:app --port 8000` (in repo root) → backend REST API available, orchestrator endpoints working.
  - In the browser:
    - Use the Sessions sidebar to create a new session; verify navigation to `/sessions/{id}` and chat panel visibility.
    - Conduct a short conversation that triggers the extraction node and observe requirements appearing in the sidebar and the RD draft text updating.
    - Click "Export MD" and confirm a Markdown file downloads with the current RD draft.
    - Toggle the offline banner by simulating offline mode in DevTools; send messages while offline and observe queued state + auto-flush on reconnection.

- **Automated checks:**
  - `npm run lint` → passes after addressing initial effect and `any` issues.
  - `npm run format` → ESLint `--fix` runs cleanly.
  - With backend running: `npm run test:e2e` (in `frontend/`) → Playwright runs both `smoke.spec.ts` and `session-flow.spec.ts`, asserting end‑to‑end session creation and chat UI visibility.

### Design Deviations & Notes

- **WebSocket + streaming (AC5):**
  - STORY-005 originally described a full WebSocket integration (`useWebSocket` hook, streaming message chunks, real‑time requirements updates). The current frontend uses **REST-only** calls (`sendSessionMessage`) with conventional request/response turns and no streaming UI. WebSocket support (including the protocol defined in `designing/websocket-protocol.md`) is deferred to a dedicated future story.

- **Stores and component breakdown:**
  - The story’s illustrative design referenced separate `chatStore`, `requirementsStore`, shadcn UI primitives, and smaller components (e.g. `MessageBubble`, `RequirementCard`, `RDViewer`). The implementation takes a slightly more compact approach:
    - A single `ChatPanel` component renders messages from `GraphState.chat_history` instead of a separate chat store.
    - Requirements are rendered directly from `GraphState.requirements` inside `RequirementsSidebar` rather than a dedicated `requirementsStore`.
    - RD preview + export live inside `RequirementsSidebar` instead of a standalone `RDViewer`/`ExportMenu` component.
  - This keeps the MVP simpler while preserving the core UX and can be refactored into a more granular component hierarchy later.

- **Offline and error handling scope:**
  - AC8 envisioned toasts, retry buttons, and broader error surfaces. The current implementation focuses on the critical path:
    - Offline banner + message queueing + auto-flush.
    - Simple inline error banners for message send and session load failures.
  - More sophisticated toast/notification handling is left for a future UX polish story.

- **Performance, accessibility, and metrics:**
  - AC10/AC11 mention Lighthouse ≥ 85, WCAG 2.1 AA, and automated axe checks in CI. This story implements reasonable defaults (lazy routes, dark theme contrast, ARIA tweaks) but does **not** include formal Lighthouse/axe audits or CI enforcement. Those remain follow-up work.

### Follow-ups / Next Stories

1. **WebSocket streaming & real-time updates:** Implement backend WebSocket endpoint and a `useWebSocket` hook on the frontend, including streaming message chunks, live requirement updates, agent status indicators, reconnection, and heartbeats as described in `designing/websocket-protocol.md` (completing the originally described AC5 behavior).
2. **Deeper UX polish & componentization:** Extract `MessageBubble`, richer requirement cards, RD viewer, and shadcn-style UI primitives; add inline metadata (confidence badges, source refs, inferred tags) and traceability links in the UI.
3. **Accessibility & performance audit:** Run Lighthouse and axe-core checks, address any contrast/focus issues, and tune bundle splitting, caching, and code-splitting strategy for production.
4. **Expanded E2E coverage:** Add E2E flows for sending chat messages, seeing requirements appear, offline queueing behavior, and RD export; wire these into CI alongside existing backend tests.
5. **Session and RD lifecycle UX:** Introduce richer session management (renaming, archiving), RD version history, and better affordances around approval status and review feedback once backend capabilities mature.
