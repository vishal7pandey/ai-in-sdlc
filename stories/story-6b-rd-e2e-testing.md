# User Story 6B: Requirements Document End-to-End Testing

## Story Overview

**Story ID:** STORY-006B
**Story Title:** Requirements Document E2E Testing - Browser-Level Validation
**Priority:** P0 - CRITICAL (Quality Gate for Demo Readiness)
**Estimated Effort:** 16-24 hours
**Sprint:** Sprint 6 (Quality Assurance - Phase 2)
**Dependencies:**
- STORY-006A: Synthesis Agent Implementation (Complete)
- STORY-004: Orchestrator Implementation (Complete)
- STORY-005: Frontend MVP (Complete)
- Backend RD API endpoints implemented
- Backend integration test `test_rd_endpoints.py` passing

---

## Executive Summary: Closing the Critical Gap

### **Current State Assessment**

✅ **Backend RD Flow Complete**:
- Integration test `tests/integration/api/test_rd_endpoints.py` exists
- Exercises full backend flow: Session → Message → Extraction → Synthesis → RD Export
- All API endpoints working at HTTP level
- **Proof**: Backend can generate and export RD

❌ **Frontend-Backend Integration Unverified**:
- No browser-level test proving UI → API → RD flow works
- No proof that actual users can generate and download RD via UI
- **Gap**: Backend works, but can users access it through the frontend?

### **What This Story Delivers**

This story **closes the final gap** by implementing **Playwright E2E tests** that prove:
1. Real users can create sessions via UI
2. Users can send messages that trigger extraction
3. Requirements appear in the right panel
4. "Generate RD" button creates RD preview
5. "Export MD" button downloads valid markdown file
6. Downloaded file contains actual requirements with proper formatting

**After this story**: System is **fully demo-ready** with end-to-end proof of core workflow.

---

## Socratic Design Questions

### Q1: What's the difference between backend integration test and browser E2E test?

**Answer:**

| Aspect | Backend Integration Test | Browser E2E Test |
|--------|-------------------------|------------------|
| **Scope** | API endpoints only | Full UI → API flow |
| **Client** | Python requests library | Real browser (Chromium/Firefox) |
| **What it proves** | API works correctly | Users can accomplish task via UI |
| **User perspective** | No | Yes - exactly what user experiences |
| **UI bugs caught** | No | Yes - broken buttons, missing handlers, CORS, etc. |
| **Example failure** | API returns 500 | Button exists but click handler not wired |

**Insight**: Backend test proves *server works*. E2E test proves *users can use it*.

---

### Q2: Why can't we just trust that frontend components work if backend works?

**Answer:**

**Integration failures happen at boundaries**:

1. **CORS not configured** → Backend works, frontend gets CORS error
2. **Event handler missing** → Button renders, click does nothing
3. **State update bug** → API returns data, UI doesn't update
4. **WebSocket vs HTTP mismatch** → Backend expects WS, frontend sends HTTP
5. **File download broken** → API returns data, browser doesn't trigger download
6. **Selector mismatch** → Test can't find button due to class name change

**Real-world example**:
```typescript
// Backend returns 200 with RD content
// But frontend does this:
const response = await fetch('/api/v1/rd/${sessionId}');
const text = await response.text(); // ✅ Works
// But forgot to update state:
// setRdContent(text); // ❌ Missing - UI never updates
```

Backend test passes. User sees nothing. **E2E test catches this.**

---

### Q3: What makes an E2E test "good"?

**Answer:**

**Good E2E Test Characteristics**:

| Characteristic | Why It Matters | Example |
|----------------|----------------|---------|
| **User-centric** | Tests what users do, not implementation | "Click 'Generate RD'" not "Call generateRD()" |
| **Deterministic** | Same input → same output, no flakiness | Stub LLM responses for consistency |
| **Fast enough** | < 60 seconds, can run frequently | Mock slow operations (LLM calls) |
| **Comprehensive** | Covers happy path + key error cases | Success + network error + empty state |
| **Isolated** | Each test independent, no shared state | Fresh session per test |
| **Debuggable** | Screenshots, videos, clear assertions | Capture on failure, descriptive expects |

**Anti-patterns to avoid**:
- ❌ Testing implementation details (checking internal state)
- ❌ Hardcoded waits (`sleep(5000)`) instead of proper waiting
- ❌ Brittle selectors (CSS classes that change frequently)
- ❌ One giant test that does everything (hard to debug failures)

---

### Q4: How do we handle LLM non-determinism in tests?

**Answer:**

**Problem**: Real LLM calls are:
- **Slow** (2-5 seconds per call)
- **Non-deterministic** (same prompt → different responses)
- **Expensive** (costs money)
- **Rate-limited** (can hit OpenAI limits)
- **Flaky** (network issues, timeouts)

**Solutions (in priority order)**:

**Option 1: Mock LLM at API level** (Best for E2E)
```python
# Backend in test mode
if os.getenv("TEST_MODE") == "true":
    class MockLLM:
        def ainvoke(self, prompt):
            # Deterministic response based on prompt
            if "login" in prompt.lower():
                return "REQ-001: User authentication via email/password"
            return "REQ-XXX: Generic requirement"
```

**Option 2: Stub entire agent** (Faster)
```python
# tests/conftest.py
@pytest.fixture
def stub_agents():
    # Return pre-recorded agent outputs
    return {
        "conversational": "Great! Can you clarify the authentication method?",
        "extraction": [RequirementItem(...)]
    }
```

**Option 3: Use deterministic prompts** (Slower but more realistic)
```typescript
// In test, use specific prompts known to produce consistent responses
await chatInput.fill("User must log in with email and password");
// This specific phrasing reliably produces REQ-001
```

**Option 4: Assert on patterns, not exact text** (Most flexible)
```typescript
// Don't assert exact requirement text
await expect(page.getByText("REQ-001: User authentication")).toBeVisible(); // ❌ Brittle

// Assert on structure/patterns
await expect(page.getByText(/REQ-\d{3}/)).toBeVisible(); // ✅ Flexible
const reqText = await page.locator('.requirement-card').first().textContent();
expect(reqText).toContain('login'); // ✅ Semantic check
```

**Recommendation**: Start with **Option 3** (deterministic prompts) + **Option 4** (pattern assertions) for quick wins. Add **Option 1** (mock LLM) later for speed.

---

### Q5: What exactly should the E2E test verify about the downloaded file?

**Answer:**

**Downloaded file must prove**:

1. **File exists and is downloadable** ✅
   - Playwright captures download event
   - File has non-zero size

2. **Correct filename format** ✅
   - Pattern: `requirements-{sessionId}.md`
   - Extension is `.md`

3. **Valid markdown structure** ✅
   - Starts with `# Requirements Document`
   - Has proper heading hierarchy (##, ###)

4. **Contains actual requirements** ✅
   - At least one `REQ-XXX` identifier
   - Requirement text matches what user typed

5. **Has expected sections** ✅
   - Functional Requirements
   - Non-Functional Requirements (if any)
   - Acceptance Criteria
   - Source references

6. **Traceability present** ✅
   - `**Source**: Turn X` or similar
   - Links back to chat conversation

**Verification code**:
```typescript
const download = await page.waitForEvent('download');
const path = await download.path();
const content = await fs.readFile(path, 'utf-8');

// Structure checks
expect(content).toMatch(/^# Requirements Document/);
expect(content).toContain('## 1. Functional Requirements');

// Content checks
expect(content).toMatch(/REQ-\d{3}/); // At least one requirement
expect(content).toContain('login'); // User's requirement text
expect(content).toContain('Acceptance Criteria'); // Has criteria
expect(content).toMatch(/\*\*Source\*\*:/); // Has traceability

// Format checks
expect(download.suggestedFilename()).toMatch(/requirements-.*\.md$/);
const stats = await fs.stat(path);
expect(stats.size).toBeGreaterThan(100); // Non-empty file
```

---

### Q6: How does this test relate to the existing `test_rd_endpoints.py`?

**Answer:**

**Complementary tests at different levels**:

```
┌─────────────────────────────────────────────────┐
│  Browser E2E Test (rd-flow.spec.ts)            │
│  ✓ User clicks buttons                          │
│  ✓ UI updates correctly                         │
│  ✓ File downloads work                          │
│  ✓ Full user journey                            │
└──────────────────┬──────────────────────────────┘
                   │ Uses
                   ↓
┌─────────────────────────────────────────────────┐
│  Backend Integration Test (test_rd_endpoints.py)│
│  ✓ API endpoints correct                        │
│  ✓ Data flow through services                   │
│  ✓ Database persistence                         │
│  ✓ Business logic correct                       │
└─────────────────────────────────────────────────┘
```

**What each test catches**:

| Failure Type | Backend Test | E2E Test |
|-------------|--------------|----------|
| API returns wrong data | ✅ Catches | ✅ Catches |
| Database save fails | ✅ Catches | ✅ Catches |
| Button not wired to API | ❌ Misses | ✅ Catches |
| CORS not configured | ❌ Misses | ✅ Catches |
| Download not triggered | ❌ Misses | ✅ Catches |
| UI doesn't update | ❌ Misses | ✅ Catches |

**Both tests are necessary**. Backend test is **faster** (30s) and **precise** (exact error location). E2E test is **slower** (2min) but **comprehensive** (proves user experience works).

---

### Q7: What's the "test pyramid" and where does this fit?

**Answer:**

**Standard Test Pyramid**:

```
           ┌──────────┐
           │  E2E     │ 10% - Slow, expensive, comprehensive
           │  (10)    │
           └──────────┘
        ┌──────────────┐
        │ Integration  │ 20% - Medium speed, component boundaries
        │   (30)       │
        └──────────────┘
     ┌──────────────────┐
     │   Unit Tests     │ 70% - Fast, cheap, many scenarios
     │     (100+)       │
     └──────────────────┘
```

**Your current test distribution**:

| Test Type | Count | Coverage | Status |
|-----------|-------|----------|--------|
| **Unit Tests** | 40+ | Agents, utilities, models | ✅ Good |
| **Integration Tests** | 10+ | API, DB, orchestrator | ✅ Good |
| **E2E Tests** | 2 | Smoke, session flow | ⚠️ Need RD flow |

**After this story**:
- Add 1-2 E2E tests for RD flow
- Total E2E tests: 3-4 (sufficient for MVP)
- Pyramid shape: ✅ Healthy

**Guideline**:
- Unit tests: Test everything
- Integration tests: Test boundaries
- E2E tests: Test critical user journeys only

**Critical user journeys** (need E2E):
1. ✅ User can access app (smoke test)
2. ✅ User can create session (session-flow test)
3. ❌ User can generate RD (**THIS STORY**)
4. 🔮 Future: User can edit requirements (Story 7)

---

### Q8: What error scenarios should the E2E test cover?

**Answer:**

**Happy path** (must work):
- User creates session → sends message → generates RD → downloads file ✅

**Error scenarios** (should gracefully fail):

1. **Backend unavailable** (Network error)
   - Expected: "Connection error. Please try again." toast
   - Test: Stop backend, click "Generate RD", verify error message

2. **Session not found** (Data error)
   - Expected: "Session not found" error, redirect to home
   - Test: Use invalid session ID in URL, verify redirect

3. **No requirements to generate RD from** (Business logic)
   - Expected: "No requirements extracted yet" message, button disabled
   - Test: Create session, immediately click "Generate RD", verify message

4. **RD generation timeout** (LLM timeout)
   - Expected: "Generation taking longer than expected..." message
   - Test: Mock slow synthesis, verify timeout handling

5. **Download blocked by browser** (Browser security)
   - Expected: "Download blocked. Please check permissions." message
   - Test: Block downloads in browser settings, verify message

**Recommendation**: Start with **happy path only**. Add error scenarios in later refinement (Story 6C).

**Why**:
- Happy path proves core functionality works (80% value)
- Error scenarios add robustness (20% value)
- MVP needs happy path working, errors can be handled later

---

### Q9: How long should this E2E test take to run?

**Answer:**

**Target timing breakdown**:

| Phase | Duration | Cumulative |
|-------|----------|------------|
| Browser launch | 2-3s | 3s |
| Navigate to app | 1-2s | 5s |
| Create session | 2-3s | 8s |
| Send message | 1-2s | 10s |
| **Wait for extraction** | **10-15s** | **25s** |
| Generate RD | 3-5s | 30s |
| **Synthesis + RD generation** | **10-15s** | **45s** |
| Export + download | 2-3s | 48s |
| File validation | 1-2s | 50s |

**Total: ~50 seconds** (without mocking)

**With LLM mocking**:
- Extraction: 15s → 2s (saved 13s)
- Synthesis: 15s → 3s (saved 12s)
- **Total: ~25 seconds**

**Performance targets**:

| Scenario | Target | Acceptable | Too Slow |
|----------|--------|------------|----------|
| With real LLM | 60s | 90s | >120s |
| With mocked LLM | 30s | 45s | >60s |

**Why this matters**:
- **< 30s**: Developers run frequently during development ✅
- **30-60s**: Run before commits ✅
- **> 60s**: Only run in CI, developers skip ❌

**Recommendation**: Start without mocking (60s test). If too slow for development workflow, add mocking later.

---

### Q10: What makes this test "demo-ready proof"?

**Answer:**

**Demo Scenario**:
```
Stakeholder: "Show me how users generate requirements documents."
You: "Let me run the E2E test..."
[Test runs in browser, visible on screen]
✅ Session created
✅ Message sent
✅ Requirements extracted (cards appear)
✅ RD generated (preview shows)
✅ File downloaded
✅ File contains requirements
Stakeholder: "Great! This works end-to-end."
```

**What the passing E2E test proves to stakeholders**:

1. **System works** ✅
   - Not just "backend works" or "frontend looks nice"
   - **Entire workflow** from user input to downloaded file

2. **Ready for users** ✅
   - Real browser, real clicks, real download
   - Not mocked components or fake data

3. **Repeatable** ✅
   - Test runs consistently
   - Not "works on my machine" demo

4. **Maintainable** ✅
   - Automated test, not manual demo script
   - Can run before every release

5. **Comprehensive** ✅
   - Covers all 12 steps from audit success criteria
   - No gaps in user journey

**"Demo-ready" = You can show the E2E test running live as proof the system works.**

**Alternative**: Manual demo script is NOT demo-ready because:
- ❌ Requires human to remember steps
- ❌ Might break day before demo (no automated check)
- ❌ Can't prove it runs consistently
- ❌ Doesn't scale (can't test 10 scenarios manually)

---

## Acceptance Criteria

### ✅ AC1: Playwright Test Suite Setup

**Given** Playwright is configured in `frontend/`
**When** E2E test directory structure is created
**Then** test infrastructure is ready

**Verification Checklist**:
- [ ] `frontend/tests/e2e/rd-flow.spec.ts` exists
- [ ] `frontend/playwright.config.ts` includes RD test
- [ ] Test can be run with `npm run test:e2e:rd`
- [ ] Test uses same browser config as other E2E tests
- [ ] Test has proper timeout settings (60s for LLM calls)
- [ ] Screenshots captured on failure
- [ ] Video recording enabled for debugging

**File Structure**:
```
frontend/
├── tests/
│   ├── e2e/
│   │   ├── smoke.spec.ts           # ✅ Exists
│   │   ├── session-flow.spec.ts    # ✅ Exists
│   │   └── rd-flow.spec.ts         # ⬅️ NEW (this story)
│   └── setup/
│       └── test-helpers.ts         # Shared utilities
├── playwright.config.ts            # Updated with RD test config
└── package.json                    # Scripts for running tests
```

---

### ✅ AC2: Session Creation via UI

**Given** user opens app in browser
**When** user creates new session via UI
**Then** session is created and user lands on chat page

**Test Steps**:
```typescript
test('User can create session', async ({ page }) => {
  // 1. Navigate to app
  await page.goto('/');

  // 2. Trigger session creation (modal or inline form)
  await page.click('[data-testid="new-session-button"]');

  // 3. Enter project name
  await page.fill('[data-testid="project-name-input"]', 'E2E Test Project');

  // 4. Submit
  await page.click('[data-testid="create-session-submit"]');

  // 5. Wait for navigation
  await page.waitForURL('**/sessions/*');

  // 6. Verify session page loaded
  await expect(page.getByText('E2E Test Project')).toBeVisible();
  await expect(page.getByPlaceholder(/Describe a feature/)).toBeVisible();
});
```

**Verification**:
- [ ] Browser navigates to `/sessions/{uuid}`
- [ ] Session ID in URL is valid UUID format
- [ ] Project name displays in UI header
- [ ] Chat input is focused and ready
- [ ] Requirements sidebar shows "No requirements yet"

---

### ✅ AC3: Message Sending and Requirements Extraction

**Given** user is on session chat page
**When** user sends message about a requirement
**Then** message is sent, AI responds, and requirements are extracted

**Test Steps**:
```typescript
test('User sends message and sees requirements', async ({ page }) => {
  // ... (after session creation from AC2)

  // 1. Locate chat input
  const chatInput = page.getByPlaceholder(/Describe a feature/);

  // 2. Type requirement message
  await chatInput.fill('Users must be able to log in with email and password');

  // 3. Send message
  await chatInput.press('Enter');

  // 4. Verify user message appears
  await expect(page.getByText('Users must be able to log in')).toBeVisible();

  // 5. Wait for AI response
  await expect(page.getByText(/Great|Thank you|Can you clarify/)).toBeVisible({ timeout: 15000 });

  // 6. Wait for requirements extraction
  await expect(page.getByText(/REQ-\d{3}/)).toBeVisible({ timeout: 20000 });

  // 7. Verify requirement card appears in sidebar
  const requirementCard = page.locator('[data-testid="requirement-card"]').first();
  await expect(requirementCard).toBeVisible();

  // 8. Verify requirement contains user's text
  const cardText = await requirementCard.textContent();
  expect(cardText).toContain('log in');
});
```

**Verification**:
- [ ] User message appears in chat immediately
- [ ] AI response streams in (visible progress)
- [ ] At least one requirement card appears in sidebar
- [ ] Requirement card has REQ-XXX identifier
- [ ] Requirement card has confidence score badge
- [ ] Requirement text relates to user's message

---

### ✅ AC4: RD Preview Generation

**Given** requirements have been extracted
**When** user clicks "Generate RD" button
**Then** RD preview appears with formatted content

**Test Steps**:
```typescript
test('User generates RD preview', async ({ page }) => {
  // ... (after requirements extraction from AC3)

  // 1. Verify "Generate RD" button is enabled
  const generateButton = page.getByRole('button', { name: /Generate RD/i });
  await expect(generateButton).toBeEnabled();

  // 2. Click button
  await generateButton.click();

  // 3. Wait for generation (synthesis agent runs)
  await expect(page.getByText(/Generating document/)).toBeVisible();

  // 4. Wait for preview to appear
  const rdPreview = page.locator('[data-testid="rd-preview"]');
  await expect(rdPreview).toBeVisible({ timeout: 20000 });

  // 5. Verify preview contains expected sections
  await expect(rdPreview.getByText('# Requirements Document')).toBeVisible();
  await expect(rdPreview.getByText(/## 1\. Functional Requirements/)).toBeVisible();

  // 6. Verify requirements from extraction appear in preview
  await expect(rdPreview.getByText(/REQ-\d{3}/)).toBeVisible();

  // 7. Verify export button now enabled
  const exportButton = page.getByRole('button', { name: /Export MD/i });
  await expect(exportButton).toBeEnabled();
});
```

**Verification**:
- [ ] Loading indicator shown during generation
- [ ] Preview appears within 20 seconds
- [ ] Preview shows markdown-rendered content
- [ ] All extracted requirements appear in preview
- [ ] Preview has proper heading hierarchy
- [ ] Acceptance criteria visible for each requirement
- [ ] Source traceability links present
- [ ] Export button becomes enabled

---

### ✅ AC5: Markdown File Export

**Given** RD preview is visible
**When** user clicks "Export MD" button
**Then** markdown file downloads with correct content

**Test Steps**:
```typescript
test('User exports RD as markdown', async ({ page }) => {
  // ... (after RD preview generation from AC4)

  // 1. Set up download listener
  const downloadPromise = page.waitForEvent('download');

  // 2. Click export button
  const exportButton = page.getByRole('button', { name: /Export MD/i });
  await exportButton.click();

  // 3. Wait for download to start
  const download = await downloadPromise;

  // 4. Verify filename format
  const filename = download.suggestedFilename();
  expect(filename).toMatch(/requirements-[a-f0-9-]+\.md$/);

  // 5. Save file and read content
  const path = await download.path();
  const content = await fs.readFile(path, 'utf-8');

  // 6. Verify file structure
  expect(content).toMatch(/^# Requirements Document/m);
  expect(content).toContain('## 1. Functional Requirements');

  // 7. Verify requirements present
  expect(content).toMatch(/REQ-\d{3}/);
  expect(content).toContain('log in'); // User's original text

  // 8. Verify acceptance criteria
  expect(content).toContain('**Acceptance Criteria**');

  // 9. Verify traceability
  expect(content).toMatch(/\*\*Source\*\*:/);

  // 10. Verify file is non-empty
  expect(content.length).toBeGreaterThan(200);
});
```

**Verification**:
- [ ] Download event fires within 5 seconds
- [ ] Filename matches pattern `requirements-{uuid}.md`
- [ ] File extension is `.md`
- [ ] File size > 200 bytes (non-empty)
- [ ] File contains markdown heading structure
- [ ] All requirements from UI appear in file
- [ ] Acceptance criteria formatted correctly
- [ ] Traceability section present
- [ ] File is valid markdown (can be parsed)

---

### ✅ AC6: End-to-End Happy Path Test

**Given** clean browser and running system
**When** complete E2E test runs
**Then** all 12 steps from audit success criteria pass

**Comprehensive Test**:
```typescript
test('Complete RD generation workflow', async ({ page }) => {
  // Step 1: Navigate to app
  await page.goto('/');
  await expect(page).toHaveTitle(/Requirements Engineering/);

  // Step 2: Create session
  await page.click('[data-testid="new-session-button"]');
  await page.fill('[data-testid="project-name-input"]', 'E2E Complete Test');
  await page.click('[data-testid="create-session-submit"]');
  await page.waitForURL('**/sessions/*');

  // Step 3: Extract session ID from URL
  const sessionId = page.url().split('/sessions/')[1];
  expect(sessionId).toMatch(/^[a-f0-9-]+$/);

  // Step 4: Send first requirement message
  const chatInput = page.getByPlaceholder(/Describe a feature/);
  await chatInput.fill('Users must be able to log in with email and password');
  await chatInput.press('Enter');

  // Step 5: Wait for extraction
  await expect(page.getByText(/REQ-001/)).toBeVisible({ timeout: 20000 });

  // Step 6: Send second requirement message
  await chatInput.fill('Login must complete within 2 seconds on 4G network');
  await chatInput.press('Enter');

  // Step 7: Wait for second extraction
  await expect(page.getByText(/REQ-002/)).toBeVisible({ timeout: 20000 });

  // Step 8: Verify requirement count
  const requirementCards = page.locator('[data-testid="requirement-card"]');
  await expect(requirementCards).toHaveCount(2);

  // Step 9: Generate RD
  const generateButton = page.getByRole('button', { name: /Generate RD/i });
  await generateButton.click();

  // Step 10: Wait for preview
  const rdPreview = page.locator('[data-testid="rd-preview"]');
  await expect(rdPreview).toBeVisible({ timeout: 20000 });

  // Step 11: Verify both requirements in preview
  await expect(rdPreview.getByText(/REQ-001/)).toBeVisible();
  await expect(rdPreview.getByText(/REQ-002/)).toBeVisible();

  // Step 12: Export and validate file
  const downloadPromise = page.waitForEvent('download');
  const exportButton = page.getByRole('button', { name: /Export MD/i });
  await exportButton.click();

  const download = await downloadPromise;
  const path = await download.path();
  const content = await fs.readFile(path, 'utf-8');

  // Final validations
  expect(content).toContain('REQ-001');
  expect(content).toContain('REQ-002');
  expect(content).toContain('log in');
  expect(content).toContain('2 seconds');
  expect(content).toContain('Acceptance Criteria');

  // Success metrics
  const endTime = Date.now();
  console.log(`Complete E2E test passed in ${(endTime - startTime) / 1000}s`);
});
```

**Verification**:
- [ ] Test completes in < 90 seconds
- [ ] All 12 steps pass without errors
- [ ] No console errors in browser
- [ ] No network failures (CORS, 404, 500)
- [ ] Downloaded file is valid and complete
- [ ] Test is repeatable (passes 5 consecutive times)

---

### ✅ AC7: Error Handling - No Requirements Scenario

**Given** user creates session
**When** user clicks "Generate RD" without any requirements
**Then** appropriate error message shown

**Test Steps**:
```typescript
test('Generate RD with no requirements shows error', async ({ page }) => {
  // 1. Create session
  await page.goto('/');
  await page.click('[data-testid="new-session-button"]');
  await page.fill('[data-testid="project-name-input"]', 'Empty Session Test');
  await page.click('[data-testid="create-session-submit"]');

  // 2. Verify "Generate RD" button is disabled
  const generateButton = page.getByRole('button', { name: /Generate RD/i });
  await expect(generateButton).toBeDisabled();

  // 3. Verify helper text shown
  await expect(page.getByText(/No requirements extracted yet/)).toBeVisible();

  // OR if button is enabled but shows error on click:

  // 2. Click "Generate RD"
  await generateButton.click();

  // 3. Verify error message
  await expect(page.getByText(/Cannot generate document without requirements/)).toBeVisible();

  // 4. Verify no preview shown
  const rdPreview = page.locator('[data-testid="rd-preview"]');
  await expect(rdPreview).not.toBeVisible();
});
```

**Verification**:
- [ ] Button disabled when no requirements exist, OR
- [ ] Error toast shown when clicked with no requirements
- [ ] Error message is clear and actionable
- [ ] No API call made when no requirements
- [ ] UI remains responsive after error

---

### ✅ AC8: Test Configuration and Documentation

**Given** test suite implemented
**When** developers need to run tests
**Then** documentation and scripts are clear

**Documentation Requirements**:

**1. README section**:
```markdown
# E2E Tests

## Running RD Flow Tests

```bash
# Run all E2E tests
npm run test:e2e

# Run only RD flow test
npm run test:e2e:rd

# Run with UI (headed mode)
npm run test:e2e:rd -- --headed

# Debug mode with browser DevTools
npm run test:e2e:rd -- --debug
```

## Prerequisites

- Backend running on http://localhost:8000
- Frontend running on http://localhost:5173
- Database seeded with test data (optional)

## Test Data

Tests use deterministic prompts for consistent results:
- "Users must be able to log in with email and password" → REQ-001
- "Login must complete within 2 seconds" → REQ-002

## Troubleshooting

**Test fails with "Download timeout"**:
- Increase timeout in playwright.config.ts
- Check that Export button is wired to download handler

**Test fails with "REQ-001 not found"**:
- Verify extraction agent is running
- Check that backend orchestrator routes to extraction
- Enable LLM mocking if using real OpenAI API
```

**2. Package.json scripts**:
```json
{
  "scripts": {
    "test:e2e": "playwright test",
    "test:e2e:rd": "playwright test rd-flow",
    "test:e2e:headed": "playwright test --headed",
    "test:e2e:debug": "playwright test --debug"
  }
}
```

**3. CI/CD integration**:
```yaml
# .github/workflows/e2e-tests.yml
name: E2E Tests

on: [push, pull_request]

jobs:
  e2e:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-node@v3
      - name: Install dependencies
        run: npm ci
      - name: Install Playwright browsers
        run: npx playwright install --with-deps
      - name: Start backend
        run: docker compose up -d
      - name: Wait for backend
        run: npx wait-on http://localhost:8000/health
      - name: Start frontend
        run: npm run dev &
      - name: Wait for frontend
        run: npx wait-on http://localhost:5173
      - name: Run E2E tests
        run: npm run test:e2e
      - name: Upload test results
        if: failure()
        uses: actions/upload-artifact@v3
        with:
          name: playwright-report
          path: playwright-report/
```

**Verification**:
- [ ] README has E2E testing section
- [ ] All commands documented with examples
- [ ] Troubleshooting guide included
- [ ] CI/CD workflow exists
- [ ] Test can be run by any developer without manual setup

---

### ✅ AC9: Test Maintainability and Robustness

**Given** E2E test is implemented
**When** codebase changes over time
**Then** test remains stable and maintainable

**Maintainability Checklist**:

**1. Selector Strategy**:
```typescript
// ❌ Brittle - breaks when CSS changes
await page.click('.btn.btn-primary.generate-rd');

// ✅ Semantic - stable across refactors
await page.getByRole('button', { name: /Generate RD/i });

// ✅ Test IDs - explicit test hooks
await page.click('[data-testid="generate-rd-button"]');
```

**2. Wait Strategy**:
```typescript
// ❌ Brittle - arbitrary timeout
await page.waitForTimeout(5000);

// ✅ Robust - wait for specific condition
await expect(page.getByText(/REQ-001/)).toBeVisible({ timeout: 20000 });
```

**3. Test Isolation**:
```typescript
// Each test gets fresh state
test.beforeEach(async ({ page, context }) => {
  // Clear cookies and local storage
  await context.clearCookies();
  await page.goto('/');
  await page.evaluate(() => localStorage.clear());
});
```

**4. Debugging Support**:
```typescript
test('RD flow', async ({ page }) => {
  // Capture screenshot before each assertion
  await page.screenshot({ path: 'before-generate.png', fullPage: true });

  // Log important state
  const reqCount = await page.locator('[data-testid="requirement-card"]').count();
  console.log(`Requirements extracted: ${reqCount}`);

  // ... rest of test
});
```

**Verification**:
- [ ] All selectors use semantic roles or test IDs
- [ ] No hardcoded `waitForTimeout()` calls
- [ ] Each test is independent (can run in isolation)
- [ ] Screenshots captured at key steps
- [ ] Failures are debuggable (clear error messages)
- [ ] Test passes consistently (10 consecutive runs)

---

### ✅ AC10: Performance and Reliability Benchmarks

**Given** E2E test suite complete
**When** tests run in CI/CD pipeline
**Then** performance meets targets

**Performance Targets**:

| Metric | Target | Current | Status |
|--------|--------|---------|--------|
| Test duration (with real LLM) | < 90s | TBD | ⏳ |
| Test duration (with mocked LLM) | < 45s | TBD | ⏳ |
| Test reliability | 95% pass rate | TBD | ⏳ |
| Time to first failure detection | < 5min | TBD | ⏳ |
| Browser memory usage | < 500MB | TBD | ⏳ |

**Reliability Metrics**:
- [ ] Test passes 95% of the time (19 out of 20 runs)
- [ ] No false positives (failures that pass on retry)
- [ ] No flaky assertions (timing-dependent failures)
- [ ] Clear failure messages (can debug without re-running)

**Performance Optimization**:
```typescript
// Parallel execution where possible
test.describe.configure({ mode: 'parallel' });

// Share browser context across tests
test.describe.serial('RD flow', () => {
  // Tests run in sequence in same browser
});

// Skip expensive operations in setup
test.beforeAll(async () => {
  // One-time setup (seed database, etc.)
});
```

**Verification**:
- [ ] Test duration measured and logged
- [ ] Reliability tracked over 20 runs
- [ ] Performance optimizations applied
- [ ] CI/CD pipeline includes E2E tests
- [ ] Failures trigger notifications

---

## Implementation Guide

### Phase 1: Test Setup (4 hours)

**Tasks**:
1. Create `frontend/tests/e2e/rd-flow.spec.ts`
2. Add test configuration to `playwright.config.ts`
3. Create test helper utilities in `tests/setup/test-helpers.ts`
4. Add npm scripts for running tests
5. Write basic smoke test to verify setup

**Deliverables**:
- [ ] Test file structure created
- [ ] Can run `npm run test:e2e:rd` successfully
- [ ] Basic test passes (navigates to app)

---

### Phase 2: Happy Path Implementation (8 hours)

**Tasks**:
1. Implement session creation test (AC2)
2. Implement message sending and extraction test (AC3)
3. Implement RD preview generation test (AC4)
4. Implement markdown export test (AC5)
5. Combine into single comprehensive test (AC6)

**Deliverables**:
- [ ] Complete E2E test covering all 12 steps
- [ ] Test passes consistently (5 consecutive runs)
- [ ] Downloaded file validated

---

### Phase 3: Error Handling and Edge Cases (4 hours)

**Tasks**:
1. Implement "no requirements" error test (AC7)
2. Add test for multiple requirements
3. Add test for long requirement text
4. Add test for special characters in requirements

**Deliverables**:
- [ ] Error scenarios covered
- [ ] Edge cases handled gracefully
- [ ] All tests passing

---

### Phase 4: Documentation and CI/CD (4 hours)

**Tasks**:
1. Write README documentation (AC8)
2. Add troubleshooting guide
3. Create CI/CD workflow
4. Run reliability test (20 consecutive runs)
5. Measure and document performance

**Deliverables**:
- [ ] README complete with examples
- [ ] CI/CD pipeline running E2E tests
- [ ] Performance benchmarks documented
- [ ] Test suite ready for team use

---

## Testing Strategy

### Test Determinism

**Problem**: LLM responses are non-deterministic

**Solutions**:

**Option 1: Use deterministic prompts** (Quick win)
```typescript
// Specific prompts known to produce consistent requirements
const DETERMINISTIC_PROMPTS = {
  auth: "Users must be able to log in with email and password",
  performance: "Login must complete within 2 seconds on 4G network"
};
```

**Option 2: Mock LLM at backend** (Best long-term)
```python
# Backend test mode
if os.getenv("TEST_MODE") == "true":
    CONVERSATIONAL_RESPONSES = {
        "login": "Great! Can you clarify the authentication method?",
        "performance": "Understood. What's the acceptable response time?"
    }
```

**Option 3: Assert on patterns** (Most flexible)
```typescript
// Don't assert exact text
expect(content).toContain("REQ-001: User authentication"); // ❌ Brittle

// Assert on structure
expect(content).toMatch(/REQ-\d{3}:/); // ✅ Flexible
expect(content).toContain("login"); // ✅ Semantic
```

---

### Debugging Failed Tests

**When test fails**:

1. **Check screenshots** in `playwright-report/`
2. **Check video** recording (if enabled)
3. **Check browser console** logs
4. **Check network tab** for failed requests
5. **Run in headed mode**: `npm run test:e2e:rd -- --headed`
6. **Run with debugger**: `npm run test:e2e:rd -- --debug`

**Common failure modes**:

| Failure | Likely Cause | Fix |
|---------|--------------|-----|
| "Button not found" | Selector changed | Update selector or add test ID |
| "Timeout waiting for REQ-001" | Extraction not triggered | Check backend logs, verify orchestrator |
| "Download not started" | Handler not wired | Verify Export button onClick handler |
| "CORS error" | CORS not configured | Add CORS middleware to FastAPI |
| "File content empty" | Synthesis failed | Check synthesis agent logs |

---

## Success Metrics

### Quantitative Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| **Test pass rate** | ≥ 95% | Passes 19/20 runs |
| **Test duration** | < 90s | Measured in CI |
| **Time to detect failure** | < 5min | CI pipeline time |
| **False positive rate** | < 5% | Flaky test detection |
| **Code coverage** | ≥ 80% | Integration points |

### Qualitative Metrics

**✅ Demo Readiness**:
- [ ] Can show test running live to stakeholders
- [ ] Test proves end-to-end workflow works
- [ ] Downloaded file is presentable quality

**✅ Developer Experience**:
- [ ] Test is easy to run (`npm run test:e2e:rd`)
- [ ] Failures are easy to debug (clear messages)
- [ ] Test completes quickly enough for dev workflow

**✅ Maintainability**:
- [ ] Test survives UI refactors (semantic selectors)
- [ ] Test is self-documenting (clear test names)
- [ ] Test has low maintenance burden (< 1 hour/month)

---

## Definition of Done

**Mark this story complete when ALL of the following are TRUE**:

1. ✅ **Test exists and passes**:
   - [ ] `frontend/tests/e2e/rd-flow.spec.ts` implemented
   - [ ] Test covers all 10 acceptance criteria
   - [ ] Test passes 19 out of 20 consecutive runs

2. ✅ **Full workflow validated**:
   - [ ] Session creation via UI ✓
   - [ ] Message sending and extraction ✓
   - [ ] RD preview generation ✓
   - [ ] Markdown file download ✓
   - [ ] File content validation ✓

3. ✅ **Documentation complete**:
   - [ ] README has E2E testing section
   - [ ] All commands documented
   - [ ] Troubleshooting guide included
   - [ ] Example outputs shown

4. ✅ **CI/CD integration**:
   - [ ] E2E test runs in CI pipeline
   - [ ] Test results visible in pull requests
   - [ ] Failures block merges

5. ✅ **Performance targets met**:
   - [ ] Test completes in < 90 seconds
   - [ ] Pass rate ≥ 95%
   - [ ] No flaky assertions

6. ✅ **Team enablement**:
   - [ ] Any developer can run test
   - [ ] No manual setup required
   - [ ] Clear failure messages

7. ✅ **Demo ready**:
   - [ ] Can show test running live
   - [ ] Downloaded file is presentable
   - [ ] Proves system works end-to-end

**Until ALL checkboxes are checked, the system is NOT demo-ready.**

---

## Next Steps After Story 6B

**With E2E RD test passing, you can**:

1. **Demo to stakeholders** ✅
   - Show test running in browser
   - Download file and open in editor
   - Prove end-to-end workflow works

2. **Story 7: WebSocket Integration** (Optional)
   - Add real-time streaming to E2E test
   - Verify character-by-character response

3. **Story 8: Error Handling** (Hardening)
   - Add more error scenario tests
   - Test network failures, timeouts, etc.

4. **Story 9: Performance Optimization** (Polish)
   - Reduce test duration with LLM mocking
   - Optimize synthesis agent speed

5. **Story 10: UI Enhancements** (Features)
   - Add requirement editing
   - Add comments and reviews
   - Test these flows in E2E

---

## Appendix A: Complete Test File Template

```typescript
// frontend/tests/e2e/rd-flow.spec.ts
import { test, expect } from '@playwright/test';
import fs from 'fs/promises';

test.describe('Requirements Document E2E Flow', () => {
  test.beforeEach(async ({ page, context }) => {
    // Clear state
    await context.clearCookies();
    await page.goto('/');
    await page.evaluate(() => localStorage.clear());
  });

  test('Complete RD generation workflow', async ({ page }) => {
    const startTime = Date.now();

    // Step 1-2: Navigate and create session
    await page.goto('/');
    await page.click('[data-testid="new-session-button"]');
    await page.fill('[data-testid="project-name-input"]', 'E2E Test Project');
    await page.click('[data-testid="create-session-submit"]');
    await page.waitForURL('**/sessions/*');

    // Step 3-4: Send requirement messages
    const chatInput = page.getByPlaceholder(/Describe a feature/);

    await chatInput.fill('Users must be able to log in with email and password');
    await chatInput.press('Enter');
    await expect(page.getByText(/REQ-001/)).toBeVisible({ timeout: 20000 });

    await chatInput.fill('Login must complete within 2 seconds on 4G network');
    await chatInput.press('Enter');
    await expect(page.getByText(/REQ-002/)).toBeVisible({ timeout: 20000 });

    // Step 5: Generate RD
    const generateButton = page.getByRole('button', { name: /Generate RD/i });
    await generateButton.click();

    const rdPreview = page.locator('[data-testid="rd-preview"]');
    await expect(rdPreview).toBeVisible({ timeout: 20000 });
    await expect(rdPreview.getByText(/REQ-001/)).toBeVisible();
    await expect(rdPreview.getByText(/REQ-002/)).toBeVisible();

    // Step 6: Export and validate
    const downloadPromise = page.waitForEvent('download');
    const exportButton = page.getByRole('button', { name: /Export MD/i });
    await exportButton.click();

    const download = await downloadPromise;
    const path = await download.path();
    const content = await fs.readFile(path, 'utf-8');

    // Assertions
    expect(content).toContain('# Requirements Document');
    expect(content).toContain('REQ-001');
    expect(content).toContain('REQ-002');
    expect(content).toContain('log in');
    expect(content).toContain('2 seconds');
    expect(content).toContain('Acceptance Criteria');

    const duration = (Date.now() - startTime) / 1000;
    console.log(`✅ E2E test passed in ${duration}s`);
  });

  test('Error: Generate RD with no requirements', async ({ page }) => {
    // Create session without sending messages
    await page.goto('/');
    await page.click('[data-testid="new-session-button"]');
    await page.fill('[data-testid="project-name-input"]', 'Empty Session');
    await page.click('[data-testid="create-session-submit"]');

    // Verify button disabled or error on click
    const generateButton = page.getByRole('button', { name: /Generate RD/i });
    await expect(generateButton).toBeDisabled();
  });
});
```

---

**End of Story 6B Document**
