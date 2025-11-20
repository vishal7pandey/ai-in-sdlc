# Story 7: Inference, Validation & Human Review Implementation

## Story Overview

**Story ID:** STORY-007
**Story Title:** Complete Agent Orchestration - Inference, Validation & Human-in-the-Loop
**Priority:** P1 - HIGH (Production Readiness)
**Estimated Effort:** 32-40 hours
**Sprint:** Sprint 7 (Advanced Agent Features)
**Dependencies:**
- STORY-006B: RD E2E Testing (Complete)
- Orchestrator routing logic exists
- Human review API endpoints implemented
- Backend infrastructure ready

---

## Executive Summary

### **Current State**

✅ **What's Working**:
- Conversational agent fully implemented
- Extraction agent fully implemented
- Synthesis agent fully implemented
- Human review API endpoints exist (`/resume-human-review`)
- Orchestrator routing logic defined
- Graph interrupts before `human_review` node

❌ **What's Missing**:
- **InferenceAgent**: Node is no-op placeholder
- **ValidationAgent**: Node is no-op placeholder
- **Review logic**: Node only logs, doesn't update state
- **Human review UI**: No frontend for approval/rejection
- **End-to-end tests**: No tests covering full review cycle

### **What This Story Delivers**

**Complete agent orchestration** with:
1. InferenceAgent that proposes implicit requirements
2. ValidationAgent that checks requirement quality
3. Review logic that manages RD lifecycle
4. Human review UI for approval/revision workflow
5. E2E tests proving human-in-the-loop works

**After this story**: System has **production-grade** requirement engineering with quality gates.

---

## Design Decisions (Answers to Your Questions)

### **Section A: InferenceAgent Design**

#### **A1. Goal of Inference**

**Inference discovers implicit requirements across 4 categories**:

| Category | Examples | When to Infer |
|----------|----------|---------------|
| **Security** | Authentication, authorization, data encryption, input validation | Any feature with user data or actions |
| **Performance** | Response time targets, throughput, scalability | Any user-facing feature |
| **Reliability** | Uptime, error handling, data backup, recovery | Any critical business feature |
| **Compliance** | GDPR, accessibility (WCAG), audit logging | Any feature with PII or legal requirements |

**Concrete example**:

```
Explicit requirement: "Users can log in with email/password"

Inferred requirements:
1. REQ-INF-001: Password must be hashed with bcrypt (Security)
2. REQ-INF-002: Failed login attempts must be rate-limited (Security)
3. REQ-INF-003: Login must complete within 2 seconds on 4G (Performance)
4. REQ-INF-004: Session expires after 24 hours of inactivity (Security)
5. REQ-INF-005: Login attempts must be logged for audit (Compliance)
```

**Inference triggers**:
- After extraction completes
- When `validation_router` returns `"needs_inference"`
- Only runs once per extraction batch (avoid infinite inference loops)

---

#### **A2. Constraints - Marking Inferred Requirements**

**All inferred requirements MUST be clearly distinguished**:

```python
@dataclass
class RequirementItem:
    id: str                    # REQ-INF-001 (vs REQ-001 for explicit)
    title: str
    type: RequirementType
    inferred: bool = False     # ⬅️ ALWAYS True for inference
    inference_category: str | None = None  # "security" | "performance" | ...
    confidence: float          # Lower confidence (0.6-0.8) for inferred
    parent_req_id: str | None  # Links to explicit requirement that triggered inference
    # ... other fields
```

**ID Naming Convention**:
- Explicit: `REQ-001`, `REQ-002`, ...
- Inferred: `REQ-INF-001`, `REQ-INF-002`, ...

**UI Treatment**:
- Inferred requirements have distinct badge: `[Inferred]` or `[Suggested]`
- Different card styling (e.g., dashed border, lighter background)
- Grouped separately in sidebar: "Explicit Requirements" vs "Inferred Requirements"

---

#### **A3. How Aggressive - Confidence Thresholds**

**Inference is conservative - only high-confidence inferences**:

| Confidence | Action | Example |
|------------|--------|---------|
| **≥ 0.75** | Auto-add to `inferred_requirements` | "Login needs password hashing" (security best practice) |
| **0.60-0.74** | Add but flag for review | "Login might need 2FA" (depends on security requirements) |
| **< 0.60** | Don't infer | Too speculative |

**Inference rules prioritized by confidence**:

```python
HIGH_CONFIDENCE_RULES = {
    "authentication": [
        ("password_hashing", 0.95),      # Industry standard
        ("rate_limiting", 0.90),         # Security best practice
        ("session_timeout", 0.85),       # Common requirement
    ],
    "data_storage": [
        ("backup_strategy", 0.90),
        ("encryption_at_rest", 0.85),
    ],
    "user_facing": [
        ("response_time_sla", 0.80),
        ("error_handling", 0.85),
    ]
}

MEDIUM_CONFIDENCE_RULES = {
    "authentication": [
        ("two_factor_auth", 0.70),       # Depends on security posture
        ("password_policy", 0.65),
    ]
}
```

**Strategy**: Start with **HIGH_CONFIDENCE_RULES only**. Expand to medium-confidence rules in later iterations based on user feedback.

---

#### **A4. User Interaction - Auto-add vs Confirmation**

**Two-phase approach**:

**Phase 1 (Initial - Story 7)**: **Auto-add with clear marking**
- Inferred requirements automatically added to `state.inferred_requirements`
- Clearly marked in UI as "Inferred" or "Suggested"
- User can delete unwanted inferred requirements
- **Rationale**: Simplest implementation, gets feedback quickly

**Phase 2 (Future - Story 8)**: **Confirmation workflow**
- Show inferred requirements in separate "Suggestions" panel
- User can:
  - ✅ Accept (moves to main requirements list)
  - ❌ Reject (removes from suggestions)
  - ✏️ Edit (modify before accepting)
- **Rationale**: More control, but requires additional UI complexity

**For Story 7, implement Phase 1**: Auto-add with clear distinction.

**Why**:
- Faster to implement and test
- Users can easily ignore/delete unwanted suggestions
- Gather feedback on inference quality before adding confirmation UI

---

### **Section B: ValidationAgent Design**

#### **B1. Validation Rules**

**Validation checks requirements against 3 rule categories**:

##### **1. Structural Validation (MUST PASS)**

Every requirement MUST have:

| Field | Rule | Severity if Missing |
|-------|------|---------------------|
| `actor` | Non-empty string | Critical |
| `action` | Non-empty string | Critical |
| `acceptance_criteria` | List with ≥1 item | Critical |
| `source_refs` | List with ≥1 item (except inferred) | Critical |
| `type` | Valid RequirementType enum | Critical |
| `id` | Unique within session | Critical |

**Example failure**:
```json
{
  "issue_id": "VAL-001",
  "requirement_id": "REQ-003",
  "severity": "critical",
  "category": "structure",
  "message": "Requirement is missing acceptance criteria",
  "suggested_fix": "Add at least one testable acceptance criterion"
}
```

##### **2. Content Quality Validation (SHOULD PASS)**

| Check | Rule | Severity if Failed |
|-------|------|-------------------|
| **Title clarity** | Length 10-80 chars, no vague terms | Warning |
| **Testability** | Acceptance criteria are measurable | Warning |
| **Specificity** | No ambiguous terms ("fast", "secure", "user-friendly") | Warning |
| **Completeness** | Action has object ("login **with what**") | Warning |
| **Traceability** | Source refs point to existing chat turns | Warning |

**Example warnings**:
```json
[
  {
    "issue_id": "VAL-002",
    "requirement_id": "REQ-004",
    "severity": "warning",
    "category": "clarity",
    "message": "Title contains vague term: 'fast'",
    "suggested_fix": "Replace 'fast' with specific time target (e.g., '< 2 seconds')"
  },
  {
    "issue_id": "VAL-003",
    "requirement_id": "REQ-004",
    "severity": "warning",
    "category": "testability",
    "message": "Acceptance criterion is not measurable: 'System should be responsive'",
    "suggested_fix": "Add specific metric: 'Page load time < 2s on 4G network'"
  }
]
```

##### **3. Business Logic Validation (SHOULD PASS)**

| Check | Rule | Severity |
|-------|------|----------|
| **Conflict detection** | Requirements don't contradict each other | Warning |
| **Dependency check** | If REQ-A depends on REQ-B, REQ-B exists | Warning |
| **Coverage** | All major actors have at least one requirement | Info |

**Example**:
```json
{
  "issue_id": "VAL-004",
  "requirement_id": "REQ-005",
  "severity": "warning",
  "category": "conflict",
  "message": "REQ-005 conflicts with REQ-002: different authentication methods",
  "suggested_fix": "Clarify which authentication method is required or support both"
}
```

---

#### **B2. Confidence Model - How Validation Affects Confidence**

**Confidence calculation after validation**:

```python
def calculate_confidence_after_validation(
    requirement: RequirementItem,
    validation_issues: list[ValidationIssue]
) -> float:
    """
    Adjust requirement confidence based on validation results.

    Initial confidence from extraction: 0.0-1.0
    Validation adjustments:
    - Critical issue: -0.3
    - Warning: -0.1
    - Info: -0.05

    Minimum confidence: 0.1 (never zero to allow fixing)
    """
    base_confidence = requirement.confidence

    for issue in validation_issues:
        if issue.requirement_id == requirement.id:
            if issue.severity == "critical":
                base_confidence -= 0.3
            elif issue.severity == "warning":
                base_confidence -= 0.1
            elif issue.severity == "info":
                base_confidence -= 0.05

    # Clamp to [0.1, 1.0]
    return max(0.1, min(1.0, base_confidence))
```

**Overall session confidence**:

```python
def calculate_session_confidence(state: GraphState) -> float:
    """
    Session confidence is weighted average of all requirements.

    Weights:
    - Explicit requirements: 1.0
    - Inferred requirements: 0.5 (less certain)
    """
    if not state.requirements:
        return 0.0

    total_weight = 0.0
    weighted_confidence = 0.0

    for req in state.requirements:
        weight = 0.5 if req.inferred else 1.0
        weighted_confidence += req.confidence * weight
        total_weight += weight

    for req in state.inferred_requirements:
        weight = 0.5
        weighted_confidence += req.confidence * weight
        total_weight += weight

    return weighted_confidence / total_weight if total_weight > 0 else 0.0
```

---

#### **B3. Validation Issue Output Schema**

**Complete ValidationIssue schema**:

```python
@dataclass
class ValidationIssue:
    """
    Represents a single validation problem found in a requirement.
    """
    issue_id: str              # VAL-001, VAL-002, ...
    requirement_id: str        # REQ-001, REQ-INF-003, ...
    severity: str              # "critical" | "warning" | "info"
    category: str              # "structure" | "clarity" | "testability" | "conflict" | ...
    message: str               # Human-readable issue description
    suggested_fix: str | None  # Actionable suggestion for resolving
    field: str | None          # Which field has issue (e.g., "acceptance_criteria")
    timestamp: str             # ISO 8601 when issue was detected

# Example usage
validation_issue = ValidationIssue(
    issue_id="VAL-001",
    requirement_id="REQ-003",
    severity="critical",
    category="structure",
    message="Requirement is missing acceptance criteria",
    suggested_fix="Add at least one testable acceptance criterion describing expected behavior",
    field="acceptance_criteria",
    timestamp="2025-11-19T14:30:00.000Z"
)
```

**Storage in GraphState**:

```python
@dataclass
class GraphState:
    # ... existing fields

    validation_issues: list[dict[str, Any]] = field(default_factory=list)
    validation_completed: bool = False
    validation_timestamp: str | None = None
```

**Mapping to database (optional persistence)**:

```python
# src/database/models/validation.py
class ValidationIssueModel(Base):
    __tablename__ = "validation_issues"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    session_id = Column(UUID(as_uuid=True), ForeignKey("sessions.id"))
    requirement_id = Column(String, nullable=False)  # REQ-001
    issue_id = Column(String, nullable=False)         # VAL-001
    severity = Column(Enum("critical", "warning", "info"), nullable=False)
    category = Column(String, nullable=False)
    message = Column(Text, nullable=False)
    suggested_fix = Column(Text, nullable=True)
    field = Column(String, nullable=True)
    timestamp = Column(DateTime(timezone=True), server_default=func.now())
    resolved = Column(Boolean, default=False)

    # Relationships
    session = relationship("Session", back_populates="validation_issues")
```

---

### **Section C: Review / RD Lifecycle Design**

#### **C1. RD Status State Machine**

**RD lifecycle states**:

```
[draft] → [under_review] → [approved]
           ↓                    ↓
    [revision_requested]    [archived]
           ↓
      [draft] (cycle repeats)
```

**State definitions**:

| Status | Meaning | Next States | Who Can Transition |
|--------|---------|-------------|-------------------|
| `draft` | RD being generated/edited | `under_review` | System (synthesis agent) |
| `under_review` | Awaiting human approval | `approved`, `revision_requested` | Human reviewer |
| `revision_requested` | Changes needed | `draft` | System (on new generation) |
| `approved` | Final version locked | `archived` | Human reviewer |
| `archived` | Historical record | - | Admin/system |

**Database schema**:

```python
# src/database/models/rd_document.py
class RDDocumentModel(Base):
    __tablename__ = "rd_documents"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    session_id = Column(UUID(as_uuid=True), ForeignKey("sessions.id"))
    version = Column(Integer, nullable=False)  # 1, 2, 3, ...
    status = Column(
        Enum("draft", "under_review", "revision_requested", "approved", "archived"),
        nullable=False,
        default="draft"
    )
    content = Column(Text, nullable=False)  # Markdown content
    metadata = Column(JSONB, nullable=False)  # Requirements count, confidence, etc.
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    # Who approved/requested revision
    reviewer_user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=True)
    review_timestamp = Column(DateTime(timezone=True), nullable=True)
    review_feedback = Column(Text, nullable=True)

    # Relationships
    session = relationship("Session", back_populates="rd_documents")
    events = relationship("RDEventModel", back_populates="document")
```

---

#### **C2. RD Events - Audit Trail**

**Every status change creates an RDEvent**:

```python
# src/database/models/rd_event.py
class RDEventModel(Base):
    __tablename__ = "rd_events"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    document_id = Column(UUID(as_uuid=True), ForeignKey("rd_documents.id"))
    event_type = Column(
        Enum(
            "draft_generated",
            "submitted_for_review",
            "approved",
            "revision_requested",
            "regenerated",
            "archived"
        ),
        nullable=False
    )
    previous_status = Column(String, nullable=True)
    new_status = Column(String, nullable=False)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=True)
    timestamp = Column(DateTime(timezone=True), server_default=func.now())
    metadata = Column(JSONB, nullable=True)  # Event-specific data

    # Relationships
    document = relationship("RDDocumentModel", back_populates="events")
```

**Event creation triggers**:

| Trigger | Event Type | Previous Status | New Status | Metadata |
|---------|-----------|----------------|------------|----------|
| Synthesis completes | `draft_generated` | - | `draft` | `{version, req_count, confidence}` |
| Review node runs | `submitted_for_review` | `draft` | `under_review` | `{submitted_at, version}` |
| Approval granted | `approved` | `under_review` | `approved` | `{reviewer, feedback}` |
| Revision requested | `revision_requested` | `under_review` | `revision_requested` | `{reviewer, feedback, issues}` |
| New synthesis after revision | `regenerated` | `revision_requested` | `draft` | `{version, changes}` |

---

#### **C3. UI Semantics - Post-Approval Behavior**

**After approval (`status = "approved"`):**

| Action | Behavior | Rationale |
|--------|----------|-----------|
| **Further edits** | ❌ Blocked (read-only) | Approved RD is immutable |
| **Export** | ✅ Allowed (shows "Approved" badge) | User can download final version |
| **New conversation** | ✅ Allowed (creates new draft, version++) | User can iterate on approved version |
| **View history** | ✅ Allowed (see all versions) | Audit trail |

**Approved RD UI treatment**:
- Green "Approved ✓" badge in header
- Lock icon on requirements
- Export button shows "Export Final Version"
- Timestamp of approval visible
- Reviewer name/feedback visible

**Starting new iteration**:
- User clicks "Request Changes" or "New Version"
- System creates new draft (version = N+1)
- Previous approved version remains read-only
- New draft status = `draft`

**Example user flow**:
```
Version 1: draft → under_review → approved ✓ (immutable)
Version 2: draft → under_review → revision_requested
Version 3: draft → under_review → approved ✓ (immutable)
```

---

### **Section D: Human Review UX Design**

#### **D1. Trigger Points - When to Interrupt**

**Human review interrupts the graph when ANY of these conditions are true**:

| Condition | Threshold | Why |
|-----------|-----------|-----|
| **Low confidence** | Session confidence < 0.70 | Quality concerns |
| **Critical validation issues** | ≥1 issue with `severity="critical"` | Structural problems |
| **Explicit request** | User clicked "Submit for Review" | User initiated |
| **High-value session** | ≥10 requirements extracted | Significant work |

**Implementation in routing**:

```python
def validation_router(state: GraphState) -> str:
    """
    After validation_node completes.

    Returns:
    - "fail" → back to conversation (critical issues, don't proceed)
    - "needs_inference" → to inference_node
    - "pass" → to synthesis_node

    After synthesis → check if human review needed.
    """
    # Critical failures
    if state.confidence < 0.60:
        return "fail"

    critical_issues = [
        issue for issue in state.validation_issues
        if issue.get("severity") == "critical"
    ]
    if critical_issues:
        return "fail"

    # Check if inference needed
    if state.requirements and not state.inferred_requirements:
        return "needs_inference"

    # Otherwise proceed to synthesis
    return "pass"

def synthesis_router(state: GraphState) -> str:
    """
    After synthesis_node completes.

    Returns:
    - "human_review" → interrupt before human_review_node
    - "auto_approve" → skip to review_node with approval_status="approved"
    """
    # Conditions requiring human review
    requires_review = (
        state.confidence < 0.70 or                     # Low confidence
        any(i.get("severity") == "critical"
            for i in state.validation_issues) or       # Critical issues
        len(state.requirements) >= 10 or               # High-value session
        state.get("user_requested_review", False)      # Explicit request
    )

    if requires_review:
        return "human_review"
    else:
        # Auto-approve if high quality
        return "auto_approve"
```

**Graph edges**:
```python
# In graph.py
workflow.add_conditional_edges(
    "synthesis",
    synthesis_router,
    {
        "human_review": "human_review",  # Interrupt point
        "auto_approve": "review"         # Skip human review
    }
)
```

---

#### **D2. Review Surface - UI Component**

**Review UI is a full-screen modal dialog** (not a tab):

**Why modal**:
- Focuses attention on review task
- Blocks other actions until decision made
- Clear call-to-action
- Prevents accidental edits during review

**Modal structure**:

```
┌─────────────────────────────────────────────────────────────┐
│  Review Requirements Document                          [×]   │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  Session: E-commerce Platform                                │
│  Version: 2                                                   │
│  Requirements: 8 explicit, 5 inferred                         │
│  Confidence: 78%                                              │
│  Validation: 2 warnings                                       │
│                                                               │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ [Requirements]  [RD Preview]  [Validation Issues]   │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                               │
│  [Tab content: Preview of RD with all requirements]         │
│                                                               │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ Review Feedback (optional)                           │   │
│  │ ┌─────────────────────────────────────────────────┐ │   │
│  │ │ Looks good, but please clarify REQ-003's perfor-│ │   │
│  │ │ mance criteria.                                  │ │   │
│  │ └─────────────────────────────────────────────────┘ │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                               │
├─────────────────────────────────────────────────────────────┤
│                                      [Request Revision]      │
│                                      [Approve] ✓             │
└─────────────────────────────────────────────────────────────┘
```

**Component breakdown**:

```typescript
// frontend/src/components/HumanReviewModal.tsx

interface HumanReviewModalProps {
  session: Session;
  rdContent: string;
  requirements: Requirement[];
  inferredRequirements: Requirement[];
  validationIssues: ValidationIssue[];
  confidence: number;
  onApprove: (feedback?: string) => Promise<void>;
  onRequestRevision: (feedback: string) => Promise<void>;
  onClose: () => void;
}

export function HumanReviewModal({ ... }: HumanReviewModalProps) {
  const [activeTab, setActiveTab] = useState<'requirements' | 'preview' | 'issues'>('preview');
  const [feedback, setFeedback] = useState('');
  const [isSubmitting, setIsSubmitting] = useState(false);

  const handleApprove = async () => {
    setIsSubmitting(true);
    try {
      await onApprove(feedback || undefined);
      toast.success('Requirements approved successfully');
    } catch (error) {
      toast.error('Failed to approve requirements');
    } finally {
      setIsSubmitting(false);
    }
  };

  const handleRequestRevision = async () => {
    if (!feedback.trim()) {
      toast.error('Please provide feedback for revision');
      return;
    }

    setIsSubmitting(true);
    try {
      await onRequestRevision(feedback);
      toast.success('Revision requested');
    } catch (error) {
      toast.error('Failed to request revision');
    } finally {
      setIsSubmitting(false);
    }
  };

  return (
    <Dialog open onOpenChange={onClose}>
      <DialogContent className="max-w-4xl max-h-[90vh]">
        <DialogHeader>
          <DialogTitle>Review Requirements Document</DialogTitle>
          <DialogDescription>
            Review and approve the generated requirements document
          </DialogDescription>
        </DialogHeader>

        {/* Metadata summary */}
        <div className="grid grid-cols-4 gap-4 p-4 bg-gray-50 rounded">
          <MetricCard label="Requirements" value={requirements.length} />
          <MetricCard label="Inferred" value={inferredRequirements.length} />
          <MetricCard label="Confidence" value={`${Math.round(confidence * 100)}%`} />
          <MetricCard label="Issues" value={validationIssues.length} />
        </div>

        {/* Tabs */}
        <Tabs value={activeTab} onValueChange={setActiveTab}>
          <TabsList>
            <TabsTrigger value="preview">RD Preview</TabsTrigger>
            <TabsTrigger value="requirements">
              Requirements ({requirements.length + inferredRequirements.length})
            </TabsTrigger>
            <TabsTrigger value="issues">
              Validation Issues ({validationIssues.length})
            </TabsTrigger>
          </TabsList>

          <TabsContent value="preview" className="max-h-96 overflow-y-auto">
            <MarkdownPreview content={rdContent} />
          </TabsContent>

          <TabsContent value="requirements" className="max-h-96 overflow-y-auto">
            <RequirementsList
              requirements={requirements}
              inferredRequirements={inferredRequirements}
            />
          </TabsContent>

          <TabsContent value="issues" className="max-h-96 overflow-y-auto">
            <ValidationIssuesList issues={validationIssues} />
          </TabsContent>
        </Tabs>

        {/* Feedback textarea */}
        <div>
          <Label htmlFor="feedback">Review Feedback (optional)</Label>
          <Textarea
            id="feedback"
            placeholder="Add comments or suggestions..."
            value={feedback}
            onChange={(e) => setFeedback(e.target.value)}
            rows={4}
          />
        </div>

        {/* Action buttons */}
        <DialogFooter>
          <Button
            variant="outline"
            onClick={handleRequestRevision}
            disabled={isSubmitting}
          >
            Request Revision
          </Button>
          <Button
            onClick={handleApprove}
            disabled={isSubmitting}
          >
            {isSubmitting ? 'Submitting...' : 'Approve ✓'}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
```

---

#### **D3. Decision Options**

**Three decision states** (sufficient for MVP):

| Decision | `approval_status` Value | Next Action | Use Case |
|----------|------------------------|-------------|----------|
| **Approve** | `"approved"` | RD status → approved, end | Requirements are good |
| **Request Revision** | `"revision_requested"` | Back to conversation | Changes needed |
| **Close/Cancel** | `"pending"` (no change) | Resume later | Need more time to review |

**Why no "Defer" button**:
- Closing modal without decision keeps status as `"pending"`
- User can resume review later via "Resume Review" button
- Simpler UX (2 actions instead of 3)

**Button behavior**:

```typescript
// Approve
{
  approval_status: "approved",
  review_feedback: feedback || null,  // Optional
  // Backend sets:
  // - rd_status → "approved"
  // - Creates RDEvent(event_type="approved")
  // - Locks RD for editing
}

// Request Revision
{
  approval_status: "revision_requested",
  review_feedback: feedback,          // Required
  // Backend sets:
  // - rd_status → "revision_requested"
  // - Creates RDEvent(event_type="revision_requested")
  // - Unlocks conversation for edits
}

// Close modal (no API call)
// - approval_status remains "pending"
// - Can resume later
```

---

## Implementation Specifications

### **Part 1: InferenceAgent Implementation**

#### **File: `src/agents/inference/agent.py`**

```python
"""
Inference Agent: Discovers implicit requirements based on explicit requirements.

Infers cross-cutting concerns like security, performance, reliability, compliance.
"""

import logging
from typing import Any
from dataclasses import dataclass

from src.agents.base import BaseAgent, AgentResult
from src.models.state import GraphState
from src.models.requirement import RequirementItem, RequirementType
from src.utils.logging import log_with_context

logger = logging.getLogger(__name__)


@dataclass
class InferenceRule:
    """Defines a rule for inferring requirements."""
    trigger_keywords: list[str]     # Keywords that trigger this rule
    category: str                   # "security" | "performance" | "reliability" | "compliance"
    confidence: float               # Base confidence for inferences from this rule
    requirement_template: dict[str, Any]  # Template for generating RequirementItem


class InferenceAgent(BaseAgent):
    """
    Agent that infers implicit requirements.

    Analyzes explicit requirements and conversation context to propose
    implicit requirements in 4 categories:
    - Security (auth, encryption, rate limiting)
    - Performance (response times, throughput)
    - Reliability (error handling, backup, uptime)
    - Compliance (GDPR, accessibility, audit logging)
    """

    def __init__(self):
        super().__init__()
        self.rules = self._load_inference_rules()

    def _load_inference_rules(self) -> list[InferenceRule]:
        """
        Load high-confidence inference rules.

        These are hardcoded rules based on industry best practices.
        Future: Load from config or database.
        """
        return [
            # Authentication security
            InferenceRule(
                trigger_keywords=["login", "signin", "authenticate", "password"],
                category="security",
                confidence=0.95,
                requirement_template={
                    "title": "Passwords must be securely hashed",
                    "actor": "system",
                    "action": "hash user passwords using bcrypt with salt",
                    "condition": "when storing user credentials",
                    "acceptance_criteria": [
                        "Passwords never stored in plaintext",
                        "BCrypt hashing algorithm used (cost factor ≥12)",
                        "Unique salt per password"
                    ],
                    "type": "security",
                }
            ),
            InferenceRule(
                trigger_keywords=["login", "signin", "authenticate"],
                category="security",
                confidence=0.90,
                requirement_template={
                    "title": "Failed login attempts must be rate-limited",
                    "actor": "system",
                    "action": "limit failed login attempts to prevent brute force",
                    "condition": "when user attempts authentication",
                    "acceptance_criteria": [
                        "Maximum 5 failed attempts per IP address per 15 minutes",
                        "Account locked for 15 minutes after 5 failures",
                        "User notified of account lockout"
                    ],
                    "type": "security",
                }
            ),
            InferenceRule(
                trigger_keywords=["login", "authenticate", "session"],
                category="security",
                confidence=0.85,
                requirement_template={
                    "title": "User sessions must expire after inactivity",
                    "actor": "system",
                    "action": "automatically expire idle sessions",
                    "condition": "after 24 hours of inactivity",
                    "acceptance_criteria": [
                        "Session expires after 24 hours of no activity",
                        "User redirected to login page on expired session",
                        "Session expiry time configurable per environment"
                    ],
                    "type": "security",
                }
            ),

            # Performance
            InferenceRule(
                trigger_keywords=["user", "interface", "page", "screen", "view"],
                category="performance",
                confidence=0.80,
                requirement_template={
                    "title": "User-facing operations must complete within 2 seconds",
                    "actor": "system",
                    "action": "respond to user requests within acceptable timeframe",
                    "condition": "for 95% of requests under normal load",
                    "acceptance_criteria": [
                        "Page load time < 2 seconds (p95) on 4G network",
                        "API response time < 500ms (p95)",
                        "Loading indicators shown for operations > 200ms"
                    ],
                    "type": "performance",
                }
            ),

            # Reliability
            InferenceRule(
                trigger_keywords=["data", "store", "save", "database"],
                category="reliability",
                confidence=0.90,
                requirement_template={
                    "title": "User data must be backed up regularly",
                    "actor": "system",
                    "action": "create automated backups of all user data",
                    "condition": "at least daily",
                    "acceptance_criteria": [
                        "Full database backup every 24 hours",
                        "Backup retention for 30 days",
                        "Backup restore tested monthly",
                        "Backup stored in geographically separate location"
                    ],
                    "type": "data",
                }
            ),

            # Compliance (GDPR)
            InferenceRule(
                trigger_keywords=["user", "account", "profile", "personal", "email"],
                category="compliance",
                confidence=0.85,
                requirement_template={
                    "title": "User data handling must comply with GDPR",
                    "actor": "system",
                    "action": "handle personal data according to GDPR requirements",
                    "condition": "for all users in EU/EEA",
                    "acceptance_criteria": [
                        "Users can request data export (right to portability)",
                        "Users can request account deletion (right to erasure)",
                        "Consent recorded for data processing",
                        "Privacy policy clearly displayed"
                    ],
                    "type": "compliance",
                }
            ),

            # Audit logging
            InferenceRule(
                trigger_keywords=["login", "access", "admin", "modify", "delete"],
                category="compliance",
                confidence=0.85,
                requirement_template={
                    "title": "Security-sensitive actions must be audit logged",
                    "actor": "system",
                    "action": "log all authentication attempts and data modifications",
                    "condition": "for security and compliance purposes",
                    "acceptance_criteria": [
                        "All login attempts logged (success and failure)",
                        "All data modifications logged with user, timestamp, changes",
                        "Logs stored for minimum 90 days",
                        "Logs tamper-proof (append-only)"
                    ],
                    "type": "compliance",
                }
            ),
        ]

    async def invoke(self, state: GraphState) -> AgentResult:
        """
        Infer implicit requirements based on explicit requirements.

        Args:
            state: Current graph state with requirements

        Returns:
            AgentResult with inferred_requirements populated
        """
        session_id = state.get("session_id", "unknown")
        correlation_id = state.get("correlation_id", "unknown")

        log_with_context(
            logger.info,
            "Starting inference",
            agent="inference",
            session_id=session_id,
            correlation_id=correlation_id,
            explicit_req_count=len(state.get("requirements", []))
        )

        # Get explicit requirements
        explicit_requirements = state.get("requirements", [])
        if not explicit_requirements:
            log_with_context(
                logger.info,
                "No explicit requirements to infer from",
                agent="inference",
                session_id=session_id
            )
            return AgentResult(
                updates={},
                metadata={"inferred_count": 0}
            )

        # Collect all text from requirements + chat history
        context_text = self._build_context(state)

        # Apply inference rules
        inferred_requirements = []
        next_inf_id = 1

        for rule in self.rules:
            # Check if any trigger keywords match context
            if any(keyword in context_text.lower() for keyword in rule.trigger_keywords):
                # Generate inferred requirement
                req_id = f"REQ-INF-{next_inf_id:03d}"
                next_inf_id += 1

                # Find parent requirement (first explicit req with matching keywords)
                parent_id = None
                for explicit_req in explicit_requirements:
                    req_text = f"{explicit_req.get('title', '')} {explicit_req.get('action', '')}".lower()
                    if any(kw in req_text for kw in rule.trigger_keywords):
                        parent_id = explicit_req.get("id")
                        break

                inferred_req = RequirementItem(
                    id=req_id,
                    title=rule.requirement_template["title"],
                    actor=rule.requirement_template["actor"],
                    action=rule.requirement_template["action"],
                    condition=rule.requirement_template.get("condition", ""),
                    acceptance_criteria=rule.requirement_template["acceptance_criteria"],
                    type=rule.requirement_template.get("type", "functional"),
                    confidence=rule.confidence,
                    inferred=True,
                    inference_category=rule.category,
                    parent_req_id=parent_id,
                    source_refs=[f"Inferred from {parent_id}" if parent_id else "Inferred from context"],
                    ambiguity_score=0.1,  # Low ambiguity for rule-based inference
                    timestamp=state.get("timestamp", "")
                )

                inferred_requirements.append(inferred_req.to_dict())

        log_with_context(
            logger.info,
            "Inference completed",
            agent="inference",
            session_id=session_id,
            correlation_id=correlation_id,
            inferred_count=len(inferred_requirements)
        )

        return AgentResult(
            updates={
                "inferred_requirements": inferred_requirements,
                "last_agent": "inference"
            },
            metadata={
                "inferred_count": len(inferred_requirements),
                "categories": list(set(rule.category for rule in self.rules if any(kw in context_text.lower() for kw in rule.trigger_keywords)))
            }
        )

    def _build_context(self, state: GraphState) -> str:
        """
        Build context string from requirements and chat history.

        Args:
            state: Current graph state

        Returns:
            Combined text of all requirements and recent chat messages
        """
        context_parts = []

        # Add explicit requirements
        for req in state.get("requirements", []):
            context_parts.append(f"{req.get('title', '')} {req.get('action', '')}")

        # Add recent chat history (last 5 turns)
        for msg in state.get("chat_history", [])[-10:]:
            if msg.get("role") == "user":
                context_parts.append(msg.get("content", ""))

        return " ".join(context_parts)
```

---

#### **File: `src/orchestrator/nodes.py` - Update inference_node**

```python
async def inference_node(state: GraphState) -> GraphState:
    """
    Inference node: Discover implicit requirements.

    Wraps InferenceAgent to populate inferred_requirements.
    """
    session_id = state.get("session_id", "unknown")
    correlation_id = state.get("correlation_id", "unknown")

    log_with_context(
        logger.info,
        "Inference node started",
        agent="inference",
        session_id=session_id,
        correlation_id=correlation_id
    )

    # Initialize agent (or get from singleton)
    if not hasattr(inference_node, "_agent"):
        from src.agents.inference.agent import InferenceAgent
        inference_node._agent = InferenceAgent()

    # Invoke agent
    result = await inference_node._agent.invoke(state)

    # Merge updates into state
    new_state = state.with_updates(**result.updates)

    log_with_context(
        logger.info,
        "Inference node completed",
        agent="inference",
        session_id=session_id,
        correlation_id=correlation_id,
        inferred_count=len(result.updates.get("inferred_requirements", []))
    )

    return new_state
```

---

### **Part 2: ValidationAgent Implementation**

#### **File: `src/agents/validation/agent.py`**

```python
"""
Validation Agent: Checks requirement quality and completeness.

Validates requirements against structural, content, and business logic rules.
"""

import logging
import re
from typing import Any
from dataclasses import dataclass
from datetime import datetime

from src.agents.base import BaseAgent, AgentResult
from src.models.state import GraphState
from src.models.requirement import RequirementItem
from src.utils.logging import log_with_context

logger = logging.getLogger(__name__)


@dataclass
class ValidationIssue:
    """Represents a validation problem."""
    issue_id: str
    requirement_id: str
    severity: str  # "critical" | "warning" | "info"
    category: str  # "structure" | "clarity" | "testability" | "conflict"
    message: str
    suggested_fix: str | None
    field: str | None
    timestamp: str


class ValidationAgent(BaseAgent):
    """
    Agent that validates requirement quality.

    Checks requirements against three rule categories:
    1. Structural: Must have actor, action, acceptance criteria, etc.
    2. Content Quality: Clarity, testability, specificity
    3. Business Logic: Conflicts, dependencies, coverage
    """

    VAGUE_TERMS = {
        "fast", "slow", "quick", "quickly", "responsive", "user-friendly",
        "easy", "simple", "complex", "secure", "safe", "reliable",
        "scalable", "performant", "efficient", "robust"
    }

    async def invoke(self, state: GraphState) -> AgentResult:
        """
        Validate all requirements in state.

        Args:
            state: Current graph state with requirements

        Returns:
            AgentResult with validation_issues populated
        """
        session_id = state.get("session_id", "unknown")
        correlation_id = state.get("correlation_id", "unknown")

        log_with_context(
            logger.info,
            "Starting validation",
            agent="validation",
            session_id=session_id,
            correlation_id=correlation_id
        )

        # Combine explicit and inferred requirements
        all_requirements = (
            state.get("requirements", []) +
            state.get("inferred_requirements", [])
        )

        if not all_requirements:
            return AgentResult(
                updates={"validation_completed": True},
                metadata={"validated_count": 0}
            )

        # Run validation checks
        validation_issues = []
        issue_counter = 1

        for req in all_requirements:
            # Structural validation
            issues = self._validate_structure(req, issue_counter)
            validation_issues.extend(issues)
            issue_counter += len(issues)

            # Content quality validation
            issues = self._validate_content_quality(req, issue_counter)
            validation_issues.extend(issues)
            issue_counter += len(issues)

        # Business logic validation (across all requirements)
        issues = self._validate_business_logic(all_requirements, issue_counter)
        validation_issues.extend(issues)

        # Calculate adjusted confidence
        adjusted_confidence = self._calculate_confidence_after_validation(
            state.get("confidence", 1.0),
            validation_issues
        )

        log_with_context(
            logger.info,
            "Validation completed",
            agent="validation",
            session_id=session_id,
            correlation_id=correlation_id,
            total_issues=len(validation_issues),
            critical_issues=sum(1 for i in validation_issues if i["severity"] == "critical"),
            warnings=sum(1 for i in validation_issues if i["severity"] == "warning")
        )

        return AgentResult(
            updates={
                "validation_issues": [issue.__dict__ for issue in validation_issues],
                "validation_completed": True,
                "validation_timestamp": datetime.utcnow().isoformat(),
                "confidence": adjusted_confidence,
                "last_agent": "validation"
            },
            metadata={
                "validated_count": len(all_requirements),
                "issue_count": len(validation_issues)
            }
        )

    def _validate_structure(self, req: dict[str, Any], start_id: int) -> list[ValidationIssue]:
        """
        Validate requirement has all required structural fields.

        Args:
            req: Requirement dict
            start_id: Starting issue ID number

        Returns:
            List of ValidationIssue objects
        """
        issues = []
        issue_num = start_id
        req_id = req.get("id", "UNKNOWN")
        timestamp = datetime.utcnow().isoformat()

        # Check required fields
        required_fields = {
            "actor": "Actor (who performs the action)",
            "action": "Action (what is done)",
            "acceptance_criteria": "Acceptance criteria (how to test)",
        }

        # Source refs required for explicit requirements only
        if not req.get("inferred", False):
            required_fields["source_refs"] = "Source references (traceability)"

        for field, description in required_fields.items():
            value = req.get(field)

            if field == "acceptance_criteria":
                # Must be non-empty list
                if not value or not isinstance(value, list) or len(value) == 0:
                    issues.append(ValidationIssue(
                        issue_id=f"VAL-{issue_num:03d}",
                        requirement_id=req_id,
                        severity="critical",
                        category="structure",
                        message=f"Missing {description}",
                        suggested_fix="Add at least one testable acceptance criterion describing expected behavior",
                        field=field,
                        timestamp=timestamp
                    ))
                    issue_num += 1
            elif field == "source_refs":
                # Must be non-empty list
                if not value or not isinstance(value, list) or len(value) == 0:
                    issues.append(ValidationIssue(
                        issue_id=f"VAL-{issue_num:03d}",
                        requirement_id=req_id,
                        severity="critical",
                        category="structure",
                        message=f"Missing {description}",
                        suggested_fix="Add reference to chat turn or document that triggered this requirement",
                        field=field,
                        timestamp=timestamp
                    ))
                    issue_num += 1
            else:
                # Must be non-empty string
                if not value or not isinstance(value, str) or not value.strip():
                    issues.append(ValidationIssue(
                        issue_id=f"VAL-{issue_num:03d}",
                        requirement_id=req_id,
                        severity="critical",
                        category="structure",
                        message=f"Missing {description}",
                        suggested_fix=f"Provide clear {field} for the requirement",
                        field=field,
                        timestamp=timestamp
                    ))
                    issue_num += 1

        return issues

    def _validate_content_quality(self, req: dict[str, Any], start_id: int) -> list[ValidationIssue]:
        """
        Validate content quality: clarity, testability, specificity.

        Args:
            req: Requirement dict
            start_id: Starting issue ID number

        Returns:
            List of ValidationIssue objects
        """
        issues = []
        issue_num = start_id
        req_id = req.get("id", "UNKNOWN")
        timestamp = datetime.utcnow().isoformat()

        title = req.get("title", "")
        action = req.get("action", "")
        acceptance_criteria = req.get("acceptance_criteria", [])

        # Check title length
        if len(title) < 10:
            issues.append(ValidationIssue(
                issue_id=f"VAL-{issue_num:03d}",
                requirement_id=req_id,
                severity="warning",
                category="clarity",
                message="Title is too short (< 10 characters)",
                suggested_fix="Expand title to clearly describe the requirement",
                field="title",
                timestamp=timestamp
            ))
            issue_num += 1
        elif len(title) > 80:
            issues.append(ValidationIssue(
                issue_id=f"VAL-{issue_num:03d}",
                requirement_id=req_id,
                severity="warning",
                category="clarity",
                message="Title is too long (> 80 characters)",
                suggested_fix="Shorten title to be more concise",
                field="title",
                timestamp=timestamp
            ))
            issue_num += 1

        # Check for vague terms
        combined_text = f"{title} {action}".lower()
        vague_found = [term for term in self.VAGUE_TERMS if term in combined_text]

        for term in vague_found:
            issues.append(ValidationIssue(
                issue_id=f"VAL-{issue_num:03d}",
                requirement_id=req_id,
                severity="warning",
                category="clarity",
                message=f"Contains vague term: '{term}'",
                suggested_fix=f"Replace '{term}' with specific, measurable criteria",
                field="title" if term in title.lower() else "action",
                timestamp=timestamp
            ))
            issue_num += 1

        # Check testability of acceptance criteria
        for i, criterion in enumerate(acceptance_criteria):
            if not self._is_testable(criterion):
                issues.append(ValidationIssue(
                    issue_id=f"VAL-{issue_num:03d}",
                    requirement_id=req_id,
                    severity="warning",
                    category="testability",
                    message=f"Acceptance criterion is not measurable: '{criterion[:50]}...'",
                    suggested_fix="Add specific metric or observable behavior (e.g., '< 2 seconds', 'returns 200 status')",
                    field="acceptance_criteria",
                    timestamp=timestamp
                ))
                issue_num += 1

        return issues

    def _validate_business_logic(self, all_requirements: list[dict[str, Any]], start_id: int) -> list[ValidationIssue]:
        """
        Validate business logic: conflicts, dependencies, coverage.

        Args:
            all_requirements: List of all requirement dicts
            start_id: Starting issue ID number

        Returns:
            List of ValidationIssue objects
        """
        issues = []
        issue_num = start_id
        timestamp = datetime.utcnow().isoformat()

        # Check for potential conflicts (simplified heuristic)
        # Look for requirements with conflicting keywords
        conflict_pairs = [
            (["synchronous", "blocking"], ["asynchronous", "non-blocking"]),
            (["encrypted", "encryption"], ["plaintext", "unencrypted"]),
            (["public", "open"], ["private", "restricted"]),
        ]

        for i, req1 in enumerate(all_requirements):
            req1_text = f"{req1.get('title', '')} {req1.get('action', '')}".lower()

            for j, req2 in enumerate(all_requirements[i+1:], start=i+1):
                req2_text = f"{req2.get('title', '')} {req2.get('action', '')}".lower()

                for set1, set2 in conflict_pairs:
                    has_set1 = any(kw in req1_text for kw in set1)
                    has_set2 = any(kw in req2_text for kw in set2)

                    if has_set1 and has_set2:
                        issues.append(ValidationIssue(
                            issue_id=f"VAL-{issue_num:03d}",
                            requirement_id=req1.get("id", "UNKNOWN"),
                            severity="warning",
                            category="conflict",
                            message=f"Potential conflict with {req2.get('id', 'UNKNOWN')}: contradictory approaches",
                            suggested_fix=f"Clarify relationship between {req1.get('id')} and {req2.get('id')}",
                            field=None,
                            timestamp=timestamp
                        ))
                        issue_num += 1

        return issues

    def _is_testable(self, criterion: str) -> bool:
        """
        Check if acceptance criterion is testable (has measurable element).

        Args:
            criterion: Acceptance criterion string

        Returns:
            True if appears testable, False otherwise
        """
        # Heuristics for testability
        testable_patterns = [
            r'\d+',                          # Contains numbers
            r'<|>|=|≤|≥',                    # Comparison operators
            r'must|should|shall',             # Requirement verbs
            r'returns?|displays?|shows?',     # Observable outcomes
            r'within|after|before',           # Time constraints
            r'error|success|failure',         # Status outcomes
        ]

        return any(re.search(pattern, criterion, re.IGNORECASE) for pattern in testable_patterns)

    def _calculate_confidence_after_validation(
        self,
        base_confidence: float,
        issues: list[ValidationIssue]
    ) -> float:
        """
        Adjust confidence based on validation results.

        Args:
            base_confidence: Starting confidence (0.0-1.0)
            issues: List of validation issues

        Returns:
            Adjusted confidence (0.1-1.0)
        """
        adjusted = base_confidence

        for issue in issues:
            if issue.severity == "critical":
                adjusted -= 0.3
            elif issue.severity == "warning":
                adjusted -= 0.1
            elif issue.severity == "info":
                adjusted -= 0.05

        # Clamp to [0.1, 1.0]
        return max(0.1, min(1.0, adjusted))
```

---

#### **File: `src/orchestrator/nodes.py` - Update validation_node**

```python
async def validation_node(state: GraphState) -> GraphState:
    """
    Validation node: Check requirement quality.

    Wraps ValidationAgent to populate validation_issues.
    """
    session_id = state.get("session_id", "unknown")
    correlation_id = state.get("correlation_id", "unknown")

    log_with_context(
        logger.info,
        "Validation node started",
        agent="validation",
        session_id=session_id,
        correlation_id=correlation_id
    )

    # Initialize agent
    if not hasattr(validation_node, "_agent"):
        from src.agents.validation.agent import ValidationAgent
        validation_node._agent = ValidationAgent()

    # Invoke agent
    result = await validation_node._agent.invoke(state)

    # Merge updates into state
    new_state = state.with_updates(**result.updates)

    log_with_context(
        logger.info,
        "Validation node completed",
        agent="validation",
        session_id=session_id,
        correlation_id=correlation_id,
        issue_count=len(result.updates.get("validation_issues", [])),
        confidence=result.updates.get("confidence", state.get("confidence"))
    )

    return new_state
```

---

### **Part 3: Review Node & RD Lifecycle**

#### **File: `src/orchestrator/nodes.py` - Update review_node**

```python
async def review_node(state: GraphState) -> GraphState:
    """
    Review node: Handle post-human-review actions.

    Updates RD status based on approval_status and creates audit events.
    """
    session_id = state.get("session_id", "unknown")
    correlation_id = state.get("correlation_id", "unknown")
    approval_status = state.get("approval_status", "pending")

    log_with_context(
        logger.info,
        "Review node started",
        agent="review",
        session_id=session_id,
        correlation_id=correlation_id,
        approval_status=approval_status
    )

    # Import here to avoid circular dependency
    from src.database.repositories.rd_document import RDDocumentRepository
    from src.database.repositories.rd_event import RDEventRepository

    # Get latest RD document
    async with get_async_session() as db:
        rd_repo = RDDocumentRepository(db)
        event_repo = RDEventRepository(db)

        latest_rd = await rd_repo.get_latest_by_session(session_id)

        if not latest_rd:
            log_with_context(
                logger.warning,
                "No RD document found for review",
                agent="review",
                session_id=session_id
            )
            return state

        # Update RD status based on approval_status
        if approval_status == "approved":
            # Approve the RD
            await rd_repo.update_status(
                document_id=latest_rd.id,
                new_status="approved",
                reviewer_user_id=state.get("reviewer_user_id"),  # From API
                review_feedback=state.get("review_feedback")
            )

            # Create approved event
            await event_repo.create_event(
                document_id=latest_rd.id,
                event_type="approved",
                previous_status="under_review",
                new_status="approved",
                user_id=state.get("reviewer_user_id"),
                metadata={
                    "feedback": state.get("review_feedback"),
                    "confidence": state.get("confidence"),
                    "requirements_count": len(state.get("requirements", []))
                }
            )

            log_with_context(
                logger.info,
                "RD approved",
                agent="review",
                session_id=session_id,
                document_id=str(latest_rd.id),
                version=latest_rd.version
            )

        elif approval_status == "revision_requested":
            # Request revision
            await rd_repo.update_status(
                document_id=latest_rd.id,
                new_status="revision_requested",
                reviewer_user_id=state.get("reviewer_user_id"),
                review_feedback=state.get("review_feedback")
            )

            # Create revision_requested event
            await event_repo.create_event(
                document_id=latest_rd.id,
                event_type="revision_requested",
                previous_status="under_review",
                new_status="revision_requested",
                user_id=state.get("reviewer_user_id"),
                metadata={
                    "feedback": state.get("review_feedback"),
                    "issues": [issue.get("message") for issue in state.get("validation_issues", [])]
                }
            )

            log_with_context(
                logger.info,
                "RD revision requested",
                agent="review",
                session_id=session_id,
                document_id=str(latest_rd.id),
                feedback=state.get("review_feedback")
            )

        else:
            # approval_status == "pending" - no action
            log_with_context(
                logger.info,
                "Review pending, no status change",
                agent="review",
                session_id=session_id
            )

    return state
```

---

### **Part 4: Frontend Human Review UI**

#### **File: `frontend/src/hooks/useHumanReview.ts`**

```typescript
/**
 * Hook for handling human review workflow.
 *
 * Detects when orchestrator interrupts for human review,
 * manages review modal state, and calls resume-human-review API.
 */

import { useState, useCallback } from 'react';
import { toast } from 'sonner';

interface HumanReviewDecision {
  approval_status: 'approved' | 'revision_requested';
  review_feedback?: string;
}

interface UseHumanReviewProps {
  sessionId: string;
  onReviewComplete?: () => void;
}

export function useHumanReview({ sessionId, onReviewComplete }: UseHumanReviewProps) {
  const [isReviewModalOpen, setIsReviewModalOpen] = useState(false);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [rdContent, setRdContent] = useState<string>('');
  const [requirements, setRequirements] = useState<any[]>([]);
  const [inferredRequirements, setInferredRequirements] = useState<any[]>([]);
  const [validationIssues, setValidationIssues] = useState<any[]>([]);
  const [confidence, setConfidence] = useState<number>(1.0);

  /**
   * Call this when orchestrator response has status="interrupt".
   *
   * Opens review modal and populates it with current state.
   */
  const triggerReview = useCallback((state: any) => {
    setRdContent(state.rd_draft || '');
    setRequirements(state.requirements || []);
    setInferredRequirements(state.inferred_requirements || []);
    setValidationIssues(state.validation_issues || []);
    setConfidence(state.confidence || 1.0);
    setIsReviewModalOpen(true);
  }, []);

  /**
   * User approves the RD.
   */
  const approve = useCallback(async (feedback?: string) => {
    setIsSubmitting(true);

    try {
      const decision: HumanReviewDecision = {
        approval_status: 'approved',
        review_feedback: feedback
      };

      const response = await fetch(
        `/api/v1/sessions/${sessionId}/resume-human-review`,
        {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(decision)
        }
      );

      if (!response.ok) {
        throw new Error('Failed to approve requirements');
      }

      const result = await response.json();

      toast.success('Requirements approved successfully');
      setIsReviewModalOpen(false);

      onReviewComplete?.();

      return result;
    } catch (error) {
      console.error('Error approving requirements:', error);
      toast.error('Failed to approve requirements');
      throw error;
    } finally {
      setIsSubmitting(false);
    }
  }, [sessionId, onReviewComplete]);

  /**
   * User requests revision.
   */
  const requestRevision = useCallback(async (feedback: string) => {
    if (!feedback.trim()) {
      toast.error('Please provide feedback for revision');
      return;
    }

    setIsSubmitting(true);

    try {
      const decision: HumanReviewDecision = {
        approval_status: 'revision_requested',
        review_feedback: feedback
      };

      const response = await fetch(
        `/api/v1/sessions/${sessionId}/resume-human-review`,
        {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(decision)
        }
      );

      if (!response.ok) {
        throw new Error('Failed to request revision');
      }

      const result = await response.json();

      toast.success('Revision requested');
      setIsReviewModalOpen(false);

      onReviewComplete?.();

      return result;
    } catch (error) {
      console.error('Error requesting revision:', error);
      toast.error('Failed to request revision');
      throw error;
    } finally {
      setIsSubmitting(false);
    }
  }, [sessionId, onReviewComplete]);

  /**
   * User closes modal without decision (status remains "pending").
   */
  const closeReview = useCallback(() => {
    setIsReviewModalOpen(false);
  }, []);

  return {
    isReviewModalOpen,
    isSubmitting,
    rdContent,
    requirements,
    inferredRequirements,
    validationIssues,
    confidence,
    triggerReview,
    approve,
    requestRevision,
    closeReview
  };
}
```

---

#### **File: `frontend/src/components/HumanReviewModal.tsx`**

**Complete modal component (200+ lines)**:

```typescript
/**
 * Modal for human review of generated requirements document.
 *
 * Displays:
 * - RD preview (markdown)
 * - Requirements list (explicit + inferred)
 * - Validation issues
 * - Metadata (confidence, counts)
 *
 * Actions:
 * - Approve (with optional feedback)
 * - Request Revision (requires feedback)
 * - Close (no decision, status remains "pending")
 */

import React, { useState } from 'react';
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle
} from '@/components/ui/dialog';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Button } from '@/components/ui/button';
import { Label } from '@/components/ui/label';
import { Textarea } from '@/components/ui/textarea';
import { Badge } from '@/components/ui/badge';
import { AlertCircle, CheckCircle2, Clock, FileText } from 'lucide-react';
import { MarkdownPreview } from './MarkdownPreview';

interface HumanReviewModalProps {
  isOpen: boolean;
  isSubmitting: boolean;
  rdContent: string;
  requirements: any[];
  inferredRequirements: any[];
  validationIssues: any[];
  confidence: number;
  onApprove: (feedback?: string) => Promise<void>;
  onRequestRevision: (feedback: string) => Promise<void>;
  onClose: () => void;
}

export function HumanReviewModal({
  isOpen,
  isSubmitting,
  rdContent,
  requirements,
  inferredRequirements,
  validationIssues,
  confidence,
  onApprove,
  onRequestRevision,
  onClose
}: HumanReviewModalProps) {
  const [activeTab, setActiveTab] = useState<'preview' | 'requirements' | 'issues'>('preview');
  const [feedback, setFeedback] = useState('');

  const handleApprove = async () => {
    await onApprove(feedback || undefined);
    setFeedback(''); // Clear feedback after submit
  };

  const handleRequestRevision = async () => {
    if (!feedback.trim()) {
      return; // Hook will show error toast
    }
    await onRequestRevision(feedback);
    setFeedback('');
  };

  // Count issues by severity
  const criticalCount = validationIssues.filter(i => i.severity === 'critical').length;
  const warningCount = validationIssues.filter(i => i.severity === 'warning').length;
  const infoCount = validationIssues.filter(i => i.severity === 'info').length;

  return (
    <Dialog open={isOpen} onOpenChange={onClose}>
      <DialogContent className="max-w-5xl max-h-[90vh] overflow-hidden flex flex-col">
        <DialogHeader>
          <DialogTitle className="flex items-center gap-2">
            <FileText className="h-5 w-5" />
            Review Requirements Document
          </DialogTitle>
          <DialogDescription>
            Review the generated requirements and provide your decision
          </DialogDescription>
        </DialogHeader>

        {/* Metadata Summary */}
        <div className="grid grid-cols-4 gap-4 p-4 bg-gray-50 dark:bg-gray-800 rounded-lg">
          <MetricCard
            label="Explicit"
            value={requirements.length}
            icon={<CheckCircle2 className="h-4 w-4 text-green-600" />}
          />
          <MetricCard
            label="Inferred"
            value={inferredRequirements.length}
            icon={<Clock className="h-4 w-4 text-blue-600" />}
          />
          <MetricCard
            label="Confidence"
            value={`${Math.round(confidence * 100)}%`}
            icon={getConfidenceIcon(confidence)}
          />
          <MetricCard
            label="Issues"
            value={validationIssues.length}
            icon={<AlertCircle className={`h-4 w-4 ${criticalCount > 0 ? 'text-red-600' : 'text-yellow-600'}`} />}
            detail={criticalCount > 0 ? `${criticalCount} critical` : warningCount > 0 ? `${warningCount} warnings` : undefined}
          />
        </div>

        {/* Tabs */}
        <Tabs value={activeTab} onValueChange={(v: any) => setActiveTab(v)} className="flex-1 flex flex-col overflow-hidden">
          <TabsList className="grid w-full grid-cols-3">
            <TabsTrigger value="preview">
              RD Preview
            </TabsTrigger>
            <TabsTrigger value="requirements">
              Requirements ({requirements.length + inferredRequirements.length})
            </TabsTrigger>
            <TabsTrigger value="issues">
              Validation ({validationIssues.length})
            </TabsTrigger>
          </TabsList>

          <TabsContent value="preview" className="flex-1 overflow-y-auto mt-4 p-4 border rounded-lg">
            <MarkdownPreview content={rdContent} />
          </TabsContent>

          <TabsContent value="requirements" className="flex-1 overflow-y-auto mt-4 space-y-4">
            {/* Explicit Requirements */}
            {requirements.length > 0 && (
              <div>
                <h3 className="font-semibold mb-2">Explicit Requirements</h3>
                <div className="space-y-2">
                  {requirements.map((req) => (
                    <RequirementCard key={req.id} requirement={req} />
                  ))}
                </div>
              </div>
            )}

            {/* Inferred Requirements */}
            {inferredRequirements.length > 0 && (
              <div>
                <h3 className="font-semibold mb-2 flex items-center gap-2">
                  Inferred Requirements
                  <Badge variant="outline">Suggested</Badge>
                </h3>
                <div className="space-y-2">
                  {inferredRequirements.map((req) => (
                    <RequirementCard key={req.id} requirement={req} isInferred />
                  ))}
                </div>
              </div>
            )}
          </TabsContent>

          <TabsContent value="issues" className="flex-1 overflow-y-auto mt-4 space-y-2">
            {validationIssues.length === 0 ? (
              <div className="flex flex-col items-center justify-center h-full text-gray-500">
                <CheckCircle2 className="h-12 w-12 mb-2" />
                <p>No validation issues found</p>
              </div>
            ) : (
              validationIssues.map((issue) => (
                <ValidationIssueCard key={issue.issue_id} issue={issue} />
              ))
            )}
          </TabsContent>
        </Tabs>

        {/* Feedback Textarea */}
        <div className="space-y-2">
          <Label htmlFor="feedback">Review Feedback (optional for approval, required for revision)</Label>
          <Textarea
            id="feedback"
            placeholder="Add your comments or suggestions..."
            value={feedback}
            onChange={(e) => setFeedback(e.target.value)}
            rows={3}
            className="resize-none"
          />
          <p className="text-xs text-gray-500">
            {feedback.trim().length === 0 && 'Provide feedback if requesting revision'}
          </p>
        </div>

        {/* Action Buttons */}
        <DialogFooter className="flex gap-2">
          <Button
            variant="outline"
            onClick={handleRequestRevision}
            disabled={isSubmitting || !feedback.trim()}
          >
            Request Revision
          </Button>
          <Button
            onClick={handleApprove}
            disabled={isSubmitting}
            className="bg-green-600 hover:bg-green-700"
          >
            {isSubmitting ? 'Submitting...' : 'Approve ✓'}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}

// Helper Components

function MetricCard({ label, value, icon, detail }: { label: string; value: string | number; icon: React.ReactNode; detail?: string }) {
  return (
    <div className="flex flex-col">
      <div className="flex items-center gap-2 text-sm text-gray-500 mb-1">
        {icon}
        <span>{label}</span>
      </div>
      <div className="text-2xl font-semibold">{value}</div>
      {detail && <div className="text-xs text-gray-500 mt-1">{detail}</div>}
    </div>
  );
}

function RequirementCard({ requirement, isInferred = false }: { requirement: any; isInferred?: boolean }) {
  return (
    <div className={`p-4 border rounded-lg ${isInferred ? 'border-dashed border-blue-300 dark:border-blue-700' : 'border-gray-200 dark:border-gray-700'}`}>
      <div className="flex items-start justify-between mb-2">
        <div className="flex items-center gap-2">
          <span className="font-mono text-sm font-semibold">{requirement.id}</span>
          {isInferred && <Badge variant="outline">Inferred</Badge>}
          <Badge variant="outline">{requirement.type}</Badge>
        </div>
        <ConfidenceBadge confidence={requirement.confidence} />
      </div>
      <h4 className="font-medium mb-1">{requirement.title}</h4>
      <p className="text-sm text-gray-600 dark:text-gray-400 mb-2">
        <span className="font-semibold">{requirement.actor}</span> {requirement.action}
        {requirement.condition && <> {requirement.condition}</>}
      </p>
      {requirement.acceptance_criteria && requirement.acceptance_criteria.length > 0 && (
        <div className="text-sm">
          <span className="font-semibold">Acceptance Criteria:</span>
          <ul className="list-disc list-inside ml-2 mt-1">
            {requirement.acceptance_criteria.slice(0, 2).map((ac: string, i: number) => (
              <li key={i} className="text-gray-600 dark:text-gray-400">{ac}</li>
            ))}
            {requirement.acceptance_criteria.length > 2 && (
              <li className="text-gray-500 italic">+{requirement.acceptance_criteria.length - 2} more</li>
            )}
          </ul>
        </div>
      )}
    </div>
  );
}

function ValidationIssueCard({ issue }: { issue: any }) {
  const severityConfig = {
    critical: { color: 'red', icon: <AlertCircle className="h-4 w-4" /> },
    warning: { color: 'yellow', icon: <AlertCircle className="h-4 w-4" /> },
    info: { color: 'blue', icon: <CheckCircle2 className="h-4 w-4" /> }
  };

  const config = severityConfig[issue.severity as keyof typeof severityConfig] || severityConfig.info;

  return (
    <div className={`p-4 border-l-4 border-${config.color}-500 bg-${config.color}-50 dark:bg-${config.color}-900/10 rounded`}>
      <div className="flex items-start gap-2">
        <div className={`text-${config.color}-600 dark:text-${config.color}-400 mt-0.5`}>
          {config.icon}
        </div>
        <div className="flex-1">
          <div className="flex items-center gap-2 mb-1">
            <span className="font-mono text-xs">{issue.issue_id}</span>
            <Badge variant="outline" className={`text-${config.color}-600`}>
              {issue.severity}
            </Badge>
            <span className="text-xs text-gray-500">{issue.category}</span>
          </div>
          <p className="text-sm font-medium mb-1">
            {issue.requirement_id}: {issue.message}
          </p>
          {issue.suggested_fix && (
            <p className="text-sm text-gray-600 dark:text-gray-400 italic">
              💡 {issue.suggested_fix}
            </p>
          )}
        </div>
      </div>
    </div>
  );
}

function ConfidenceBadge({ confidence }: { confidence: number }) {
  const percentage = Math.round(confidence * 100);
  const color = percentage >= 80 ? 'green' : percentage >= 60 ? 'yellow' : 'red';

  return (
    <Badge variant="outline" className={`text-${color}-600`}>
      {percentage}%
    </Badge>
  );
}

function getConfidenceIcon(confidence: number) {
  if (confidence >= 0.8) return <CheckCircle2 className="h-4 w-4 text-green-600" />;
  if (confidence >= 0.6) return <Clock className="h-4 w-4 text-yellow-600" />;
  return <AlertCircle className="h-4 w-4 text-red-600" />;
}
```

---

#### **File: `frontend/src/pages/SessionPage.tsx` - Integration**

**Add human review detection to message handling**:

```typescript
// In SessionPage.tsx

import { useHumanReview } from '@/hooks/useHumanReview';
import { HumanReviewModal } from '@/components/HumanReviewModal';

export function SessionPage() {
  const { sessionId } = useParams();

  // Add human review hook
  const {
    isReviewModalOpen,
    isSubmitting,
    rdContent,
    requirements,
    inferredRequirements,
    validationIssues,
    confidence,
    triggerReview,
    approve,
    requestRevision,
    closeReview
  } = useHumanReview({
    sessionId: sessionId!,
    onReviewComplete: () => {
      // Refresh session data after review
      refetchSession();
    }
  });

  const handleSendMessage = async (content: string) => {
    try {
      const response = await fetch(`/api/v1/sessions/${sessionId}/messages`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ content })
      });

      const result = await response.json();

      // Check if orchestrator interrupted for human review
      if (result.status === 'interrupt' && result.interrupt_type === 'human_review') {
        // Trigger human review modal
        triggerReview(result.state);
      } else {
        // Normal message flow
        // ... update chat history, etc.
      }
    } catch (error) {
      toast.error('Failed to send message');
    }
  };

  return (
    <div>
      {/* ... existing session UI */}

      {/* Human Review Modal */}
      <HumanReviewModal
        isOpen={isReviewModalOpen}
        isSubmitting={isSubmitting}
        rdContent={rdContent}
        requirements={requirements}
        inferredRequirements={inferredRequirements}
        validationIssues={validationIssues}
        confidence={confidence}
        onApprove={approve}
        onRequestRevision={requestRevision}
        onClose={closeReview}
      />
    </div>
  );
}
```

---

### **Part 5: End-to-End Testing**

#### **File: `frontend/tests/e2e/human-review-flow.spec.ts`**

```typescript
/**
 * E2E test for human review workflow.
 *
 * Tests:
 * 1. Orchestrator interrupts for review
 * 2. Review modal appears
 * 3. User approves RD
 * 4. RD status changes to "approved"
 * 5. User requests revision
 * 6. System returns to conversation
 */

import { test, expect } from '@playwright/test';

test.describe('Human Review Workflow', () => {
  test.beforeEach(async ({ page }) => {
    // Start from clean state
    await page.goto('/');
  });

  test('Complete review approval flow', async ({ page }) => {
    // 1. Create session
    await page.click('[data-testid="new-session-button"]');
    await page.fill('[data-testid="project-name-input"]', 'Review Test Project');
    await page.click('[data-testid="create-session-submit"]');
    await page.waitForURL('**/sessions/*');

    // 2. Send messages to trigger extraction
    const chatInput = page.getByPlaceholder(/Describe a feature/);

    await chatInput.fill('Users must be able to log in with email and password');
    await chatInput.press('Enter');
    await expect(page.getByText(/REQ-001/)).toBeVisible({ timeout: 20000 });

    await chatInput.fill('Login must be secure with rate limiting');
    await chatInput.press('Enter');
    await expect(page.getByText(/REQ-002/)).toBeVisible({ timeout: 20000 });

    // 3. Trigger RD generation (which should trigger human review)
    //    Note: This assumes confidence/validation triggers review
    const generateButton = page.getByRole('button', { name: /Generate RD/i });
    await generateButton.click();

    // 4. Wait for human review modal to appear
    //    (This happens when orchestrator interrupts)
    const reviewModal = page.getByRole('dialog', { name: /Review Requirements/i });
    await expect(reviewModal).toBeVisible({ timeout: 30000 });

    // 5. Verify modal shows expected content
    await expect(reviewModal.getByText(/Explicit Requirements/i)).toBeVisible();
    await expect(reviewModal.getByText(/REQ-001/)).toBeVisible();
    await expect(reviewModal.getByText(/REQ-002/)).toBeVisible();

    // 6. Approve the RD
    const approveButton = reviewModal.getByRole('button', { name: /Approve/i });
    await approveButton.click();

    // 7. Wait for modal to close
    await expect(reviewModal).not.toBeVisible({ timeout: 5000 });

    // 8. Verify RD status changed (check UI indicator)
    await expect(page.getByText(/Approved/i)).toBeVisible();

    // 9. Verify RD is now read-only (no edit buttons)
    const rdEditButton = page.getByRole('button', { name: /Edit RD/i });
    await expect(rdEditButton).not.toBeVisible();
  });

  test('Request revision flow', async ({ page }) => {
    // 1-4. Same setup as above (create session, extract requirements, trigger review)
    // ... (omitted for brevity)

    // 5. Wait for review modal
    const reviewModal = page.getByRole('dialog', { name: /Review Requirements/i });
    await expect(reviewModal).toBeVisible({ timeout: 30000 });

    // 6. Add feedback and request revision
    const feedbackTextarea = reviewModal.getByLabelText(/Review Feedback/i);
    await feedbackTextarea.fill('Please clarify the rate limiting requirements for login');

    const revisionButton = reviewModal.getByRole('button', { name: /Request Revision/i });
    await revisionButton.click();

    // 7. Wait for modal to close
    await expect(reviewModal).not.toBeVisible({ timeout: 5000 });

    // 8. Verify conversation is re-enabled
    const chatInput = page.getByPlaceholder(/Describe a feature/);
    await expect(chatInput).toBeEnabled();

    // 9. Verify feedback appears in conversation (AI acknowledges revision request)
    await expect(page.getByText(/clarify/i)).toBeVisible({ timeout: 10000 });
  });

  test('Close review without decision (pending state)', async ({ page }) => {
    // 1-4. Setup (omitted for brevity)

    // 5. Wait for review modal
    const reviewModal = page.getByRole('dialog', { name: /Review Requirements/i });
    await expect(reviewModal).toBeVisible({ timeout: 30000 });

    // 6. Close modal without decision (click X or outside)
    const closeButton = reviewModal.getByRole('button', { name: /close/i });
    await closeButton.click();

    // 7. Modal closes
    await expect(reviewModal).not.toBeVisible();

    // 8. Verify "Resume Review" button appears
    const resumeButton = page.getByRole('button', { name: /Resume Review/i });
    await expect(resumeButton).toBeVisible();

    // 9. Click resume to reopen modal
    await resumeButton.click();
    await expect(reviewModal).toBeVisible();
  });

  test('Validation issues displayed in review modal', async ({ page }) => {
    // 1. Create session with intentionally low-quality requirement
    // ... (setup omitted)

    const chatInput = page.getByPlaceholder(/Describe a feature/);
    await chatInput.fill('The system should be fast'); // Vague term "fast"
    await chatInput.press('Enter');

    // 2. Trigger review
    const generateButton = page.getByRole('button', { name: /Generate RD/i });
    await generateButton.click();

    // 3. Open review modal
    const reviewModal = page.getByRole('dialog', { name: /Review Requirements/i });
    await expect(reviewModal).toBeVisible({ timeout: 30000 });

    // 4. Switch to "Validation" tab
    const validationTab = reviewModal.getByRole('tab', { name: /Validation/i });
    await validationTab.click();

    // 5. Verify validation issue appears
    await expect(reviewModal.getByText(/vague term/i)).toBeVisible();
    await expect(reviewModal.getByText(/fast/i)).toBeVisible();
    await expect(reviewModal.getByText(/suggested_fix/i)).toBeVisible();
  });
});
```

---

## Acceptance Criteria

### **AC1: InferenceAgent Implementation**

**Given** extraction has completed
**When** inference_node executes
**Then** implicit requirements are inferred and added to state

**Verification**:
- [ ] `src/agents/inference/agent.py` exists with InferenceAgent class
- [ ] Agent loads ≥5 high-confidence inference rules
- [ ] Agent infers security, performance, reliability, compliance requirements
- [ ] Inferred requirements have `id` starting with `REQ-INF-`
- [ ] Inferred requirements have `inferred=True` flag
- [ ] Inferred requirements link to parent explicit requirement via `parent_req_id`
- [ ] Unit test: Given explicit "login" requirement, infers password hashing
- [ ] Integration test: inference_node populates `inferred_requirements` in state

---

### **AC2: ValidationAgent Implementation**

**Given** requirements (explicit + inferred) exist
**When** validation_node executes
**Then** validation issues are identified and confidence adjusted

**Verification**:
- [ ] `src/agents/validation/agent.py` exists with ValidationAgent class
- [ ] Agent checks structural validation (actor, action, criteria present)
- [ ] Agent checks content quality (title length, vague terms, testability)
- [ ] Agent checks business logic (conflicts, dependencies)
- [ ] Validation issues have schema: `issue_id`, `requirement_id`, `severity`, `category`, `message`, `suggested_fix`
- [ ] Confidence adjusted: critical issue = -0.3, warning = -0.1
- [ ] Unit test: Missing acceptance criteria → critical issue
- [ ] Unit test: Vague term "fast" → warning issue
- [ ] Integration test: validation_node populates `validation_issues` in state

---

### **AC3: Review Node & RD Lifecycle**

**Given** RD generated and human review completed
**When** review_node executes
**Then** RD status updates and events created

**Verification**:
- [ ] `RDDocumentModel.status` enum includes: draft, under_review, revision_requested, approved, archived
- [ ] `RDEventModel` table exists with event types: draft_generated, submitted_for_review, approved, revision_requested
- [ ] review_node updates RD status to "approved" when `approval_status="approved"`
- [ ] review_node updates RD status to "revision_requested" when `approval_status="revision_requested"`
- [ ] review_node creates RDEvent with metadata (feedback, confidence, requirements_count)
- [ ] Approved RD is immutable (no further edits allowed)
- [ ] Integration test: Approval flow creates correct DB records
- [ ] Integration test: Revision flow creates correct DB records

---

### **AC4: Human Review Modal UI**

**Given** orchestrator interrupts for human review
**When** frontend detects `status="interrupt", interrupt_type="human_review"`
**Then** review modal appears with RD preview and actions

**Verification**:
- [ ] `HumanReviewModal` component exists
- [ ] Modal shows 3 tabs: RD Preview, Requirements, Validation Issues
- [ ] Modal shows metadata: requirements count, confidence, issues count
- [ ] Modal has feedback textarea
- [ ] Modal has "Approve" and "Request Revision" buttons
- [ ] "Request Revision" disabled if feedback empty
- [ ] "Approve" calls `/api/v1/sessions/{id}/resume-human-review` with `approval_status="approved"`
- [ ] "Request Revision" calls resume endpoint with `approval_status="revision_requested"` and feedback
- [ ] Modal closes after successful decision
- [ ] Toast notification shown on success/error

---

### **AC5: Human Review Integration**

**Given** user sends message
**When** orchestrator returns `status="interrupt"`
**Then** frontend triggers review modal

**Verification**:
- [ ] `useHumanReview` hook exists
- [ ] Hook provides `triggerReview(state)` method
- [ ] `SessionPage` checks `result.status === "interrupt" && result.interrupt_type === "human_review"`
- [ ] On interrupt, calls `triggerReview(result.state)`
- [ ] Modal opens automatically
- [ ] Modal populated with state data (rd_content, requirements, validation_issues, confidence)
- [ ] Integration test: Mock interrupt response → modal appears

---

### **AC6: Orchestrator Routing Updates**

**Given** validation completes
**When** validation_router runs
**Then** correct next node determined

**Verification**:
- [ ] `validation_router` returns "fail" if confidence < 0.60
- [ ] `validation_router` returns "fail" if ≥1 critical validation issue
- [ ] `validation_router` returns "needs_inference" if no inferred_requirements
- [ ] `validation_router` returns "pass" otherwise (proceed to synthesis)
- [ ] `synthesis_router` added: returns "human_review" if low confidence OR critical issues OR ≥10 requirements
- [ ] `synthesis_router` returns "auto_approve" if high quality
- [ ] Graph edges updated: synthesis → {human_review, auto_approve}
- [ ] Unit test: Low confidence triggers human_review route

---

### **AC7: End-to-End Human Review Test**

**Given** clean system and browser
**When** E2E test runs
**Then** complete review workflow passes

**Verification**:
- [ ] `frontend/tests/e2e/human-review-flow.spec.ts` exists
- [ ] Test creates session, sends messages, triggers extraction
- [ ] Test clicks "Generate RD", waits for review modal
- [ ] Test verifies modal shows requirements and validation issues
- [ ] Test approves RD with feedback
- [ ] Test verifies RD status changes to "approved"
- [ ] Test verifies RD is now read-only
- [ ] Alternative test: Request revision with feedback
- [ ] Alternative test: Close without decision (pending state)
- [ ] All tests pass consistently (5 consecutive runs)

---

### **AC8: Inferred Requirements UI**

**Given** inferred requirements exist
**When** requirements sidebar renders
**Then** inferred requirements visually distinguished

**Verification**:
- [ ] Inferred requirements have distinct badge: "[Inferred]" or "[Suggested]"
- [ ] Inferred requirements have different card styling (dashed border)
- [ ] Inferred requirements grouped separately in sidebar
- [ ] Sidebar sections: "Explicit Requirements" and "Inferred Requirements"
- [ ] Inferred requirement cards show `inference_category` (Security, Performance, etc.)
- [ ] Inferred requirement cards link to parent requirement
- [ ] User can dismiss/delete inferred requirements
- [ ] Visual test: Screenshot comparison shows distinct styling

---

### **AC9: Validation Issues Display**

**Given** validation issues exist
**When** review modal "Validation" tab selected
**Then** issues displayed with severity, category, suggestions

**Verification**:
- [ ] Validation tab shows count badge: "Validation (N)"
- [ ] Critical issues have red border/icon
- [ ] Warning issues have yellow border/icon
- [ ] Info issues have blue border/icon
- [ ] Each issue shows: issue_id, requirement_id, message, suggested_fix
- [ ] Issues grouped by severity (critical first)
- [ ] Empty state shown if no issues: "No validation issues found ✓"
- [ ] Visual test: Screenshot shows correct severity colors

---

### **AC10: Documentation & Testing Complete**

**Given** all features implemented
**When** documentation reviewed
**Then** complete guides and examples exist

**Verification**:
- [ ] README has "Inference & Validation" section
- [ ] README has "Human Review Workflow" section
- [ ] API documentation updated (resume-human-review endpoint)
- [ ] Sequence diagram for human review flow exists
- [ ] Unit test coverage ≥80% for new agents
- [ ] Integration test coverage for all nodes
- [ ] E2E test for human review workflow
- [ ] All tests passing in CI/CD

---

## Definition of Done

**Mark Story 7 complete when ALL of the following are TRUE**:

1. ✅ **InferenceAgent implemented**:
   - [ ] Agent class exists with ≥5 inference rules
   - [ ] inference_node populates inferred_requirements
   - [ ] Unit + integration tests pass

2. ✅ **ValidationAgent implemented**:
   - [ ] Agent class exists with structural, content, business logic checks
   - [ ] validation_node populates validation_issues and adjusts confidence
   - [ ] Unit + integration tests pass

3. ✅ **Review lifecycle complete**:
   - [ ] RD status state machine implemented
   - [ ] RDEvent audit trail working
   - [ ] review_node updates status and creates events
   - [ ] Integration tests pass

4. ✅ **Human review UI working**:
   - [ ] HumanReviewModal component implemented
   - [ ] useHumanReview hook implemented
   - [ ] Modal triggers on orchestrator interrupt
   - [ ] Approve/revision actions call correct APIs
   - [ ] UI tests pass

5. ✅ **Routing updated**:
   - [ ] validation_router checks confidence and issues
   - [ ] synthesis_router determines human_review vs auto_approve
   - [ ] Graph edges correct
   - [ ] Routing tests pass

6. ✅ **E2E test complete**:
   - [ ] human-review-flow.spec.ts covers approval, revision, pending
   - [ ] Test passes consistently (5 runs)
   - [ ] Test completes in < 2 minutes

7. ✅ **UI polish**:
   - [ ] Inferred requirements visually distinct
   - [ ] Validation issues displayed correctly
   - [ ] Approved RD shows locked state
   - [ ] Visual regression tests pass

8. ✅ **Documentation complete**:
   - [ ] README sections added
   - [ ] API docs updated
   - [ ] Sequence diagrams exist
   - [ ] Troubleshooting guide included

**Until ALL checkboxes are checked, the story is incomplete.**

---

## Implementation Timeline

### **Week 1: Backend Agents** (16-20 hours)
- Day 1-2: InferenceAgent implementation + tests (8h)
- Day 3-4: ValidationAgent implementation + tests (8h)
- Day 5: Review node updates + RD lifecycle (4h)

### **Week 2: Frontend & Integration** (16-20 hours)
- Day 1-2: HumanReviewModal component (8h)
- Day 3: useHumanReview hook + SessionPage integration (4h)
- Day 4: Inferred requirements UI + validation display (4h)
- Day 5: E2E tests (4h)

**Total: 32-40 hours**

---

## Next Steps After Story 7

**With complete agent orchestration, you can**:

1. **Story 8: WebSocket Real-Time Streaming** (Optional)
   - Stream inference/validation progress
   - Live updates during review

2. **Story 9: Requirement Editing & Collaboration** (Features)
   - Edit requirements after approval
   - Multi-user collaboration
   - Comment threads

3. **Story 10: Advanced Inference** (Enhancements)
   - ML-based inference (vs rule-based)
   - Learn from user feedback
   - Contextual inference

4. **Story 11: Export Formats** (Polish)
   - PDF export with styling
   - DOCX export with formatting
   - Jira/Linear integration

---

**End of Story 7 Document**
