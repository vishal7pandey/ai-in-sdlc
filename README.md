# ai-in-sdlc – Requirements Engineering Platform

Toolkit of backend agents and a React frontend that turn conversational chat into validated software requirements and a living requirements document.

This repo contains:

- A **FastAPI** backend with LangGraph-based agents (conversational + extraction) and persistence in Postgres/Redis/Chroma.
- A **React/Vite** frontend that lets users create sessions, chat with the AI, see extracted requirements, and export an RD draft.

This README is intentionally short and focused on what a new developer needs to run and work on the project.

---

## 1. Prerequisites

- **Python** 3.11+ (managed via `uv`)
- **Node.js** 20+ and **npm**
- **Docker** (for Postgres, Redis, Chroma via `docker compose`)

---

## 2. Backend: FastAPI + LangGraph

Backend code lives under `src/`.

### 2.1 Start infrastructure

From the repo root:

```bash
docker compose up -d
```

This brings up Postgres, Redis, and Chroma with the expected ports and health checks.

### 2.2 Configure environment

Create a local `.env` based on the example:

```bash
cp .env.example .env
# Edit .env to set OPENAI_API_KEY and any other secrets
```

### 2.3 Run the backend API

Still from the repo root:

```bash
uv run uvicorn src.main:app --host 127.0.0.1 --port 8000
```

Useful URLs:

- API docs: http://localhost:8000/docs
- Health check: http://localhost:8000/health

---

## 3. Frontend: React + Vite

Frontend lives under `frontend/` and talks to the backend via `/api` (proxied to `http://localhost:8000`).

### 3.1 Install and run

```bash
cd frontend
npm install
npm run dev
```

Then open: http://localhost:5173

### 3.2 Basic workflow

- Use the **Sessions** sidebar to create a new session ("New" button).
- The app navigates to `/sessions/:id` and opens the **Chat** panel.
- Chat about requirements; extracted requirements appear in the **Requirements** sidebar.
- Use the RD section to preview and export a Markdown requirements document.

---

## 4. Tests and Tooling

### 4.1 Python (backend)

From the repo root:

```bash
uv run pytest tests/unit tests/integration -q
```

Static checks are wired via pre-commit (`ruff`, `mypy`, etc.). To run them manually:

```bash
uv run ruff check .
uv run mypy .
```

### 4.2 Frontend

From `frontend/`:

```bash
npm run lint       # ESLint
npm run format     # ESLint with --fix
npm run test:e2e   # Playwright E2E tests (requires backend running)
```

The E2E suite currently includes:

- A smoke test that asserts the app loads and the header renders.
- A flow test that creates a session, navigates to the chat view, and verifies the chat input is visible.

---

## 5. Project Structure & Further Docs

High-level structure:

- `src/` – FastAPI app, agents, orchestrator, schemas, storage.
- `frontend/` – React/Vite frontend.
- `stories/` – Detailed story specs (STORY-001–STORY-00x).
- `docs/PROGRESS.md` – Per-story implementation and verification notes.

For deeper architecture and story-level details, start with:

- `stories/story-5-document.md` – Frontend MVP, chat UI + requirements visualization.
- `docs/PROGRESS.md` – What each story implemented and how it was verified.
