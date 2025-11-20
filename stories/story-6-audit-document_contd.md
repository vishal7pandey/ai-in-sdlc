<img src="https://r2cdn.perplexity.ai/pplx-full-logo-primary-dark%402x.png" style="height:64px;margin-right:32px"/>

# How to Proceed: A Pragmatic Action Plan

Based on the **brutal audit** I just provided, here's your **clear path forward**:

## 🎯 **Two Parallel Tracks: Fix Critical Gaps + Verify What Exists**


***

## **TRACK 1: Immediate Verification (Next 24-48 Hours)**

**Before writing ANY new code, verify what actually works:**

### **Step 1: System Health Check** (2-4 hours)

```bash
# 1. Start all services
cd /path/to/project
docker compose up -d

# 2. Verify all containers healthy
docker ps
# Expected: 4 containers running (postgres, redis, chroma, fastapi)

# 3. Check logs
docker logs reqeng-api
# Look for: "Application startup complete"

# 4. Test health endpoint
curl http://localhost:8000/health
# Expected: {"status":"healthy","services":{"postgres":"up","redis":"up","chroma":"up"}}

# 5. Check database
docker exec -it reqeng-postgres psql -U reqeng -d reqeng -c "\dt"
# Expected: sessions, chat_messages, requirements tables listed

# 6. Run existing tests
uv run pytest tests/ -v
# Document which tests pass, which fail
```

**OUTPUT**: Create `VERIFICATION_REPORT.md` with:

- ✅ What works
- ❌ What's broken
- ❓ What's unknown

***

### **Step 2: Code Inventory** (2-4 hours)

```bash
# 1. List all Python files
find src/ -name "*.py" -type f | sort > FILES_BACKEND.txt

# 2. Check for missing agents
ls -la src/agents/
# Should see: conversational/, extraction/
# Missing: synthesis/, inference/, validation/

# 3. Check API endpoints
grep -r "@router" src/api/routes/
# Document which endpoints exist

# 4. Check orchestrator nodes
grep -r "def.*_node" src/orchestrator/
# Document which nodes are implemented vs stubs

# 5. Frontend check (if exists)
ls -la frontend/src/
# Document directory structure
```

**OUTPUT**: Update `VERIFICATION_REPORT.md` with inventory.

***

## **TRACK 2: Critical Implementation (Weeks 1-3)**

**Work on these in strict priority order:**

### **🔴 Week 1: Make ONE Feature Work End-to-End (40 hours)**

**Goal**: User can type a message and get a response (no RD yet, just prove integration works)

#### **Monday-Tuesday: Backend Connectivity** (16h)

**Priority 1: CORS Fix** (1h)

```python
# src/main.py
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

**Priority 2: Basic Session Endpoint** (4h)

```python
# src/api/routes/sessions.py
@router.post("/")
async def create_session(
    project_name: str,
    db: AsyncSession = Depends(get_db)
) -> SessionResponse:
    session = Session(id=uuid4(), project_name=project_name)
    db.add(session)
    await db.commit()
    return SessionResponse.from_orm(session)
```

**Priority 3: Message Endpoint** (8h)

```python
# src/api/routes/messages.py
@router.post("/{session_id}/messages")
async def send_message(
    session_id: str,
    message: MessageRequest,
    db: AsyncSession = Depends(get_db)
) -> MessageResponse:
    # 1. Save user message
    # 2. Invoke conversational agent
    # 3. Return AI response
    # (Don't worry about extraction yet)
```

**Priority 4: Basic E2E Test** (3h)

```python
# tests/e2e/test_basic_flow.py
async def test_create_session_and_send_message():
    # POST /sessions
    # POST /sessions/{id}/messages
    # Assert response returned
```


***

#### **Wednesday-Thursday: Frontend Connection** (16h)

**Priority 1: Frontend Environment** (2h)

```bash
cd frontend
npm install
# Create .env
echo "VITE_API_URL=http://localhost:8000/api/v1" > .env
```

**Priority 2: API Client** (4h)

```typescript
// frontend/src/lib/api/client.ts
const API_BASE = import.meta.env.VITE_API_URL;

export async function createSession(projectName: string) {
  const res = await fetch(`${API_BASE}/sessions`, {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({project_name: projectName})
  });
  return res.json();
}
```

**Priority 3: Basic UI** (8h)

```typescript
// frontend/src/App.tsx
function App() {
  const [sessionId, setSessionId] = useState('');
  const [messages, setMessages] = useState([]);

  const handleCreateSession = async () => {
    const session = await createSession('Test');
    setSessionId(session.id);
  };

  const handleSendMessage = async (text) => {
    const response = await sendMessage(sessionId, text);
    setMessages([...messages, {role: 'user', text}, {role: 'ai', text: response.content}]);
  };

  return (
    <div>
      <button onClick={handleCreateSession}>Create Session</button>
      <ChatPanel messages={messages} onSend={handleSendMessage} />
    </div>
  );
}
```

**Priority 4: Integration Test** (2h)

```bash
# Run both servers
docker compose up -d
cd frontend && npm run dev

# Manual test:
# 1. Open http://localhost:5173
# 2. Click "Create Session"
# 3. Type message
# 4. Verify AI responds
```


***

#### **Friday: E2E Test** (8h)

```typescript
// tests/e2e/happy-path.spec.ts
import { test, expect } from '@playwright/test';

test('user can chat with AI', async ({ page }) => {
  await page.goto('http://localhost:5173');

  // Create session
  await page.click('text=Create Session');
  await expect(page.locator('text=Session created')).toBeVisible();

  // Send message
  await page.fill('textarea', 'Hello AI');
  await page.press('textarea', 'Enter');

  // Verify response
  await expect(page.locator('text=Hello')).toBeVisible();
});
```

**✅ MILESTONE 1: You can now demo conversation (no RD yet, but integration proven)**

***

### **🟠 Week 2: Add Extraction + Synthesis (40 hours)**

**Goal**: User can extract requirements and download RD file

#### **Monday-Wednesday: Synthesis Agent** (24h)

**Priority 1: Agent Implementation** (16h)

```python
# src/agents/synthesis/agent.py
class SynthesisAgent(BaseAgent):
    async def generate_rd(self, requirements: List[RequirementItem]) -> str:
        """Generate markdown Requirements Document"""

        # 1. Group by type (functional, non-functional, etc.)
        # 2. Create markdown sections
        # 3. Add traceability links
        # 4. Format acceptance criteria

        md = f"""# Requirements Document

## 1. Functional Requirements

"""
        for req in functional_reqs:
            md += f"""### {req.id}: {req.title}

**Actor**: {req.actor}
**Action**: {req.action}

**Acceptance Criteria**:
{self._format_criteria(req.acceptance_criteria)}

**Source**: {', '.join(req.source_refs)}
**Confidence**: {req.confidence * 100:.0f}%

---

"""
        return md
```

**Priority 2: Tests** (4h)

```python
# tests/unit/test_synthesis_agent.py
def test_generate_rd_with_requirements():
    agent = SynthesisAgent()
    requirements = [create_test_requirement()]
    rd = agent.generate_rd(requirements)

    assert "# Requirements Document" in rd
    assert "REQ-001" in rd
    assert "Acceptance Criteria" in rd
```

**Priority 3: Integration** (4h)

```python
# src/orchestrator/nodes.py
async def synthesis_node(state: GraphState) -> GraphState:
    agent = SynthesisAgent()
    requirements = state["requirements"]
    rd_content = await agent.generate_rd(requirements)

    return {
        **state,
        "rd_content": rd_content,
        "rd_generated": True,
        "last_agent": "synthesis"
    }
```


***

#### **Thursday-Friday: RD API Endpoints** (16h)

**Priority 1: Generation Endpoint** (8h)

```python
# src/api/routes/rd.py
@router.post("/{session_id}/generate")
async def generate_rd(
    session_id: str,
    db: AsyncSession = Depends(get_db)
) -> RDResponse:
    # 1. Get session requirements
    requirements = await get_requirements(db, session_id)

    # 2. Invoke synthesis agent
    agent = SynthesisAgent()
    rd_content = await agent.generate_rd(requirements)

    # 3. Save to session
    await save_rd(db, session_id, rd_content)

    return RDResponse(content=rd_content)
```

**Priority 2: Export Endpoint** (4h)

```python
@router.get("/{session_id}/export")
async def export_rd(
    session_id: str,
    db: AsyncSession = Depends(get_db)
) -> FileResponse:
    rd_content = await get_rd_content(db, session_id)

    # Return as downloadable file
    return FileResponse(
        content=rd_content.encode('utf-8'),
        media_type='text/markdown',
        filename=f'requirements-{session_id}.md'
    )
```

**Priority 3: Frontend Integration** (4h)

```typescript
// frontend/src/components/RDViewer.tsx
async function generateRD() {
  setLoading(true);
  const response = await fetch(`${API_BASE}/rd/${sessionId}/generate`, {method: 'POST'});
  const data = await response.json();
  setRdContent(data.content);
  setLoading(false);
}

async function downloadRD() {
  const response = await fetch(`${API_BASE}/rd/${sessionId}/export`);
  const blob = await response.blob();
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = `requirements-${sessionId}.md`;
  a.click();
}
```

**✅ MILESTONE 2: Full flow works - conversation → extraction → RD → download**

***

### **🟡 Week 3: Stabilization + Testing (40 hours)**

#### **Monday-Tuesday: WebSocket** (16h)

- Implement /ws endpoint
- Add streaming response
- Test reconnection


#### **Wednesday: Error Handling** (8h)

- Add try/catch to all endpoints
- Return structured errors
- Test failure scenarios


#### **Thursday-Friday: E2E Tests** (16h)

- Write 10 happy path tests
- Write 5 error scenario tests
- Achieve 80%+ coverage

**✅ MILESTONE 3: System is demo-ready and tested**

***

## **Decision Points**

### **After Week 1:**

**Question**: Does conversation work end-to-end?

- ✅ YES → Continue to Week 2
- ❌ NO → Debug Week 1 until working (don't proceed)


### **After Week 2:**

**Question**: Can you generate and download RD?

- ✅ YES → Continue to Week 3
- ❌ NO → Fix Week 2 before stabilization


### **After Week 3:**

**Question**: Do E2E tests pass?

- ✅ YES → **System ready for demo**
- ❌ NO → Fix tests until passing

***

## **What to Defer (Not in Critical Path)**

These can wait until AFTER you have working demo:

- ❌ Inference agent (nice to have)
- ❌ Validation agent (nice to have)
- ❌ Human review workflow (nice to have)
- ❌ Authentication (add later)
- ❌ Multi-user collaboration (add later)
- ❌ PDF export (markdown is enough for demo)
- ❌ Advanced filtering (add later)

***

## **Success Criteria (Demo-Ready Definition)**

You are ready to demo when you can do this **without errors**:

1. Start system: `docker compose up -d && cd frontend && npm run dev`
2. Open browser: `http://localhost:5173`
3. Create session: Click button, enter "E-commerce App"
4. Send message: "Users need login with email/password"
5. See AI response within 3 seconds
6. Send message: "Login under 2 seconds on 4G"
7. See requirements extracted (2 cards in right panel)
8. Click "Generate RD"
9. See RD preview appear
10. Click "Export Markdown"
11. File downloads
12. Open file - see both requirements with acceptance criteria

**If all 12 steps work: YOU'RE DEMO-READY** 🎉

***

## **My Recommendation**

**Start with TRACK 1 (verification) tomorrow**. Spend 4-8 hours documenting what actually exists. Then you'll know exactly which parts of TRACK 2 you can skip vs must implement.

**Focus ruthlessly on the critical path**: Conversation → Extraction → RD → Download. Everything else is optional until this works.

**The audit document is your quality gate**. Don't claim "complete" until you can check off all 10 acceptance criteria in Story 6.

Ready to start? I recommend:

1. Run the verification scripts
2. Report back with VERIFICATION_REPORT.md
3. Then we'll prioritize the exact tasks based on what's actually missing

**Want me to generate specific implementation files for Week 1 (CORS fix, session endpoint, message endpoint, basic E2E test)?**
