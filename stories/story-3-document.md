# User Story 3: Requirements Extraction Agent Implementation

## Story Overview

**Story ID:** STORY-003
**Story Title:** Requirements Extraction Agent - Structured Requirement Parsing from Conversation
**Priority:** P0 - Critical
**Estimated Effort:** 24-32 hours
**Sprint:** Sprint 3
**Dependencies:**
- STORY-001 (Project Foundation) - Complete ✅
- STORY-002 (Conversational Agent) - Complete ✅

---

## Socratic Design Questions & Answers

### Q1: What is the fundamental purpose of the Extraction Agent?
**A:** The Extraction Agent transforms unstructured conversational text into **structured, machine-readable requirement objects** that conform to a standardized schema. It bridges natural language (what users say) and formal specifications (what engineers need).

### Q2: What makes a "good" extraction versus a "bad" one?
**A:** Good extraction:
- **Completeness**: Captures all explicit requirements mentioned
- **Accuracy**: Maps natural language to correct requirement types
- **Traceability**: Links each requirement back to source conversation turns
- **Measurability**: Converts vague terms ("fast", "secure") into testable criteria
- **Non-hallucination**: Only extracts what was actually stated or clearly implied

Bad extraction:
- Invents requirements not mentioned
- Misclassifies types (functional vs non-functional)
- Loses context (who, what, when, why)
- Creates ambiguous acceptance criteria

### Q3: What are the key challenges in extraction?
**A:**
1. **Ambiguity**: Users say "fast" - what does that mean? (2s? 5s? 10s?)
2. **Implicit context**: "Users should log in" - which users? What auth method?
3. **Scattered information**: Requirements mentioned across multiple turns
4. **Contradictions**: User says "must support 1000 users" then later "maybe 500 is enough"
5. **Type classification**: Is "app should be secure" functional or non-functional?

### Q4: How does extraction differ from conversation?
**A:** Conversational Agent focuses on **elicitation** (asking questions, clarifying). Extraction Agent focuses on **formalization** (structuring what was said). They work in tandem:
- Conversation: "Tell me more about your login requirements"
- Extraction: Converts "users log in with email/password" → `RequirementItem(id="REQ-001", actor="user", action="authenticate using email and password", ...)`

### Q5: What schema should extracted requirements follow?
**A:** Based on industry standards (IEEE 830, IREB), each requirement needs:
```python
RequirementItem:
  - id: Unique identifier (REQ-001, REQ-002)
  - title: Short summary (50 chars)
  - type: functional | non-functional | business | security | data | interface | constraint
  - actor: Who performs the action (user, system, admin)
  - action: What happens (specific verb + object)
  - condition: Under what circumstances (optional)
  - acceptance_criteria: List of testable criteria (minimum 1)
  - priority: low | medium | high | must
  - confidence: 0.0-1.0 (LLM confidence in extraction)
  - inferred: boolean (explicitly stated vs inferred)
  - rationale: Why this is a requirement
  - source_refs: Chat turn IDs where mentioned
```

### Q6: How do we handle vague language?
**A:** Three strategies:
1. **Flag ambiguity**: Mark requirements with low confidence if terms are vague
2. **Propose measurable alternatives**: "fast" → "page loads in < 2 seconds"
3. **Request clarification**: Add to `ambiguous_items` for Conversational Agent follow-up

### Q7: What's the relationship between Extraction and Inference agents?
**A:**
- **Extraction**: Captures **explicit** requirements (what user said)
- **Inference**: Proposes **implicit** requirements (what user didn't say but likely needs)

Example:
- User says: "Users should upload profile photos"
- Extraction creates: `REQ-001: User can upload profile photo`
- Inference proposes: `REQ-INF-001: System validates photo format (JPEG/PNG)`, `REQ-INF-002: System enforces 5MB file size limit`

### Q8: How do we ensure traceability?
**A:** Every requirement must have `source_refs` pointing to chat turn indices:
```python
source_refs: ["chat:turn:3", "chat:turn:7"]
```
This enables:
- Auditability (where did this come from?)
- Conflict resolution (if contradictory statements exist)
- Human review (reviewer can see original context)

### Q9: What output format maximizes downstream usability?
**A:** Pydantic models with validation:
```python
class ExtractionOutput(BaseModel):
    requirements: List[RequirementItem]  # Extracted items
    confidence: float  # Overall extraction confidence
    ambiguous_items: List[str]  # Items needing clarification
    extraction_metadata: Dict  # Tokens used, duration, model version
```

### Q10: How do we measure extraction quality?
**A:** Three metrics:
1. **Precision**: Of extracted requirements, how many are correct? (Target: >90%)
2. **Recall**: Of all mentioned requirements, how many did we extract? (Target: >95%)
3. **F1 Score**: Harmonic mean of precision and recall (Target: >92%)

---

## Story Description

As a **requirements engineer**, I want to **implement the Extraction Agent that parses conversational text and produces structured requirement objects** so that **explicit user needs are formally captured with full traceability, type classification, and measurable acceptance criteria**.

This story implements the **second critical agent** in the pipeline. After the Conversational Agent gathers information, the Extraction Agent:

- Parses chat messages for requirement statements
- Classifies requirements by type (functional, non-functional, etc.)
- Generates unique IDs following the REQ-XXX pattern
- Converts vague terms into measurable criteria where possible
- Links each requirement to source chat turns
- Flags ambiguous statements for follow-up
- Outputs validated `RequirementItem` objects ready for inference and validation

---

## Business Value

- **Reduces manual documentation** by 80%+ (automated extraction vs manual transcription)
- **Ensures completeness** through systematic parsing of all conversation turns
- **Enables traceability** linking every requirement to its conversational origin
- **Accelerates review cycles** by providing structured, not prose-based requirements
- **Improves quality** through standardized classification and acceptance criteria generation
- **Facilitates change management** via clear versioning and source tracking

---

## Acceptance Criteria

### ✅ AC1: Agent Module Structure

**Given** the project source code
**When** I examine the `src/agents/extraction/` directory
**Then** the Extraction Agent is properly structured:

**Directory Structure:**
```
src/agents/extraction/
├── __init__.py
├── agent.py                    # Main ExtractionAgent class
├── prompt_builder.py           # Extraction prompt construction
├── entity_extractor.py         # Extract actors, actions, conditions
├── type_classifier.py          # Classify requirement types
├── criteria_generator.py       # Generate acceptance criteria
├── traceability_linker.py      # Link requirements to source turns
├── ambiguity_detector.py       # Flag vague language
└── schemas.py                  # Extraction-specific schemas
```

**Verification:**
```bash
find src/agents/extraction -name "*.py" | wc -l
# Expected: 9 files total

ls -la src/agents/extraction/
# Expected: All files above present

python -c "from src.agents.extraction.agent import ExtractionAgent; print('✅ Import successful')"
# Expected: ✅ Import successful
```

---

### ✅ AC2: RequirementItem Schema Implementation

**Given** the schemas module
**When** I examine `src/schemas/requirement.py`
**Then** it defines complete requirement data models:

**RequirementItem Schema:**
```python
from pydantic import BaseModel, Field, validator
from typing import List, Optional, Literal
from datetime import datetime
from enum import Enum

class RequirementType(str, Enum):
    FUNCTIONAL = "functional"
    NON_FUNCTIONAL = "non_functional"
    BUSINESS = "business"
    SECURITY = "security"
    DATA = "data"
    INTERFACE = "interface"
    CONSTRAINT = "constraint"

class Priority(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    MUST = "must"

class RequirementItem(BaseModel):
    id: str = Field(..., pattern=r"^REQ(-INF)?-\d{3}$")
    title: str = Field(..., min_length=10, max_length=500)
    type: RequirementType
    actor: str = Field(..., min_length=1, max_length=200)
    action: str = Field(..., min_length=5)
    condition: Optional[str] = None
    acceptance_criteria: List[str] = Field(..., min_items=1)
    priority: Priority = Priority.MEDIUM
    confidence: float = Field(..., ge=0.0, le=1.0)
    inferred: bool = False
    rationale: str = Field(..., min_length=20)
    source_refs: List[str] = Field(..., min_items=1)
    created_at: datetime = Field(default_factory=datetime.utcnow)

    @validator('acceptance_criteria')
    def criteria_must_be_testable(cls, v):
        """Ensure acceptance criteria are testable"""
        for criterion in v:
            if len(criterion) < 10:
                raise ValueError(f"Criterion too short: {criterion}")
        return v

    @validator('source_refs')
    def source_refs_format(cls, v):
        """Validate source reference format"""
        for ref in v:
            if not ref.startswith('chat:turn:'):
                raise ValueError(f"Invalid source ref format: {ref}")
        return v

    class Config:
        json_schema_extra = {
            "example": {
                "id": "REQ-001",
                "title": "User authentication with email and password",
                "type": "functional",
                "actor": "user",
                "action": "authenticate using email and password credentials",
                "condition": "when accessing protected resources",
                "acceptance_criteria": [
                    "User can enter email address in valid format",
                    "User can enter password with minimum 8 characters",
                    "System validates credentials against database",
                    "User is redirected to dashboard on success",
                    "User sees error message on failure"
                ],
                "priority": "must",
                "confidence": 0.95,
                "inferred": False,
                "rationale": "User explicitly requested email/password login as primary authentication method",
                "source_refs": ["chat:turn:3", "chat:turn:5"],
                "created_at": "2025-11-16T10:30:00Z"
            }
        }
```

**Verification:**
```python
from src.schemas.requirement import RequirementItem, RequirementType, Priority

# Test valid requirement
req = RequirementItem(
    id="REQ-001",
    title="User login with email and password",
    type=RequirementType.FUNCTIONAL,
    actor="user",
    action="authenticate using email and password",
    acceptance_criteria=["User can enter credentials"],
    confidence=0.9,
    rationale="User explicitly requested this feature",
    source_refs=["chat:turn:1"]
)

assert req.id == "REQ-001"
assert req.type == RequirementType.FUNCTIONAL
print("✅ RequirementItem validation successful")
```

---

### ✅ AC3: Extraction Agent Core Implementation

**Given** the base agent class from Story 2
**When** I instantiate and invoke the ExtractionAgent
**Then** it correctly extracts structured requirements from conversation:

**Agent Initialization:**
```python
from src.agents.extraction.agent import ExtractionAgent
from src.config import settings

agent = ExtractionAgent(
    openai_api_key=settings.OPENAI_API_KEY,
    model=settings.OPENAI_MODEL,
    temperature=0.2,  # Low temperature for structured extraction
    max_tokens=2000
)

# Verify initialization
assert agent is not None
assert agent.name == "extraction"
print("✅ ExtractionAgent initialized successfully")
```

**Agent Invocation (Happy Path):**
```python
from src.schemas.state import GraphState
from src.schemas.chat import Message
from datetime import datetime

state = GraphState(
    session_id="test-session-002",
    project_name="E-commerce Platform",
    user_id="user-123",
    chat_history=[
        Message(
            id="msg-1",
            role="user",
            content="Users should be able to log in with email and password",
            timestamp=datetime.utcnow()
        ),
        Message(
            id="msg-2",
            role="user",
            content="The login should complete in under 2 seconds",
            timestamp=datetime.utcnow()
        ),
        Message(
            id="msg-3",
            role="user",
            content="We need to support both iOS and Android platforms",
            timestamp=datetime.utcnow()
        )
    ],
    current_turn=3,
    requirements=[],
    confidence=1.0,
    last_agent="conversational"
)

result = await agent.invoke(state)

# Verify extraction
assert result is not None
assert isinstance(result, GraphState)
assert len(result.requirements) >= 3  # Should extract at least 3 requirements
assert result.last_agent == "extraction"
assert all(req.id.startswith("REQ-") for req in result.requirements)
print(f"✅ Extracted {len(result.requirements)} requirements successfully")
```

---

### ✅ AC4: Entity Extraction Components

**Given** conversation text
**When** EntityExtractor processes it
**Then** it identifies actors, actions, and conditions:

**EntityExtractor Implementation:**
```python
# src/agents/extraction/entity_extractor.py
from typing import Dict, List
import re

class EntityExtractor:
    """Extracts structured entities from natural language"""

    ACTOR_PATTERNS = [
        r"\b(user|admin|customer|system|manager|developer|stakeholder)s?\b",
        r"\b(the |a )?(application|platform|service|API)\b"
    ]

    ACTION_VERBS = [
        "authenticate", "log in", "sign in", "access", "view", "create",
        "update", "delete", "edit", "submit", "download", "upload",
        "send", "receive", "process", "validate", "verify"
    ]

    def extract_actors(self, text: str) -> List[str]:
        """Extract all mentioned actors"""
        actors = []
        text_lower = text.lower()

        for pattern in self.ACTOR_PATTERNS:
            matches = re.finditer(pattern, text_lower, re.IGNORECASE)
            for match in matches:
                actor = match.group(0).strip()
                if actor and actor not in actors:
                    actors.append(actor)

        return actors or ["system"]  # Default to "system" if none found

    def extract_action(self, text: str) -> str:
        """Extract primary action from text"""
        text_lower = text.lower()

        # Look for explicit action verbs
        for verb in self.ACTION_VERBS:
            if verb in text_lower:
                # Extract context around verb
                idx = text_lower.find(verb)
                context = text_lower[max(0, idx):min(len(text_lower), idx+100)]
                return context.strip()

        # Fallback: use first verb phrase
        words = text.split()
        for i, word in enumerate(words):
            if self._is_verb(word):
                return " ".join(words[i:i+5])  # Take verb + 4 words

        return text[:100]  # Fallback to first 100 chars

    def extract_condition(self, text: str) -> Optional[str]:
        """Extract conditional clauses (when, if, while, etc.)"""
        condition_markers = ["when", "if", "while", "during", "after", "before"]
        text_lower = text.lower()

        for marker in condition_markers:
            if marker in text_lower:
                idx = text_lower.find(marker)
                condition = text[idx:min(len(text), idx+100)]
                return condition.strip()

        return None

    def _is_verb(self, word: str) -> bool:
        """Simple verb detection (enhance with NLP if needed)"""
        return word.lower() in self.ACTION_VERBS

# Test
extractor = EntityExtractor()
text = "Users should be able to log in with email when accessing protected pages"
actors = extractor.extract_actors(text)
action = extractor.extract_action(text)
condition = extractor.extract_condition(text)

print(f"Actors: {actors}")       # ['users']
print(f"Action: {action}")       # 'log in with email'
print(f"Condition: {condition}") # 'when accessing protected pages'
```

**Verification:**
```bash
uv run pytest tests/unit/agents/extraction/test_entity_extractor.py -v

# Expected test output:
# ✅ test_extract_actors_from_text PASSED
# ✅ test_extract_action_verbs PASSED
# ✅ test_extract_conditional_clauses PASSED
# ✅ test_multiple_actors PASSED
# ✅ test_default_actor_when_none_found PASSED
```

---

### ✅ AC5: Requirement Type Classification

**Given** extracted requirement text
**When** TypeClassifier analyzes it
**Then** it correctly categorizes the requirement type:

**TypeClassifier Implementation:**
```python
# src/agents/extraction/type_classifier.py
from src.schemas.requirement import RequirementType
from typing import Dict
import re

class TypeClassifier:
    """Classifies requirements by type using rules + LLM"""

    # Keyword-based classification rules
    TYPE_INDICATORS = {
        RequirementType.FUNCTIONAL: [
            "user can", "system shall", "application must",
            "feature", "functionality", "action", "process",
            "log in", "upload", "download", "create", "update", "delete"
        ],
        RequirementType.NON_FUNCTIONAL: [
            "performance", "response time", "latency", "throughput",
            "scalability", "availability", "uptime", "reliability",
            "load", "concurrent", "speed", "fast", "slow"
        ],
        RequirementType.SECURITY: [
            "authentication", "authorization", "encrypt", "secure",
            "permission", "access control", "password", "token",
            "vulnerability", "attack", "protect", "privacy"
        ],
        RequirementType.DATA: [
            "database", "data model", "schema", "table", "field",
            "store", "persist", "backup", "retention", "migration"
        ],
        RequirementType.INTERFACE: [
            "API", "endpoint", "integration", "webhook", "protocol",
            "interface", "UI", "screen", "page", "form", "button"
        ],
        RequirementType.BUSINESS: [
            "business rule", "policy", "regulation", "compliance",
            "legal", "contract", "SLA", "KPI", "ROI", "revenue"
        ],
        RequirementType.CONSTRAINT: [
            "budget", "timeline", "deadline", "technology stack",
            "platform", "device", "browser", "version", "limit"
        ]
    }

    def classify(self, text: str, action: str = "") -> RequirementType:
        """Classify requirement type using keyword matching"""
        text_lower = (text + " " + action).lower()

        # Score each type
        scores: Dict[RequirementType, int] = {t: 0 for t in RequirementType}

        for req_type, keywords in self.TYPE_INDICATORS.items():
            for keyword in keywords:
                if keyword in text_lower:
                    scores[req_type] += 1

        # Return type with highest score
        if max(scores.values()) > 0:
            return max(scores, key=scores.get)

        # Default to functional if no matches
        return RequirementType.FUNCTIONAL

    async def classify_with_llm(self, text: str, action: str) -> Dict:
        """Use LLM for ambiguous cases"""
        prompt = f"""
        Classify this requirement into one of these types:
        - functional (user actions, features)
        - non_functional (performance, scalability)
        - security (auth, encryption, access control)
        - data (storage, schemas, persistence)
        - interface (APIs, UI/UX, integrations)
        - business (rules, compliance, legal)
        - constraint (budget, timeline, technical limits)

        Requirement: {text}
        Action: {action}

        Return JSON: {{"type": "...", "confidence": 0.0-1.0, "reasoning": "..."}}
        """

        # LLM call here (implementation in AC6)
        # For now, fall back to rule-based
        return {
            "type": self.classify(text, action).value,
            "confidence": 0.8,
            "reasoning": "Rule-based classification"
        }

# Test
classifier = TypeClassifier()

# Test cases
tests = [
    ("Users should log in with email and password", RequirementType.FUNCTIONAL),
    ("The app should load in under 2 seconds", RequirementType.NON_FUNCTIONAL),
    ("User passwords must be encrypted at rest", RequirementType.SECURITY),
    ("Support iOS 14+ and Android 10+", RequirementType.CONSTRAINT),
]

for text, expected_type in tests:
    result = classifier.classify(text)
    assert result == expected_type, f"Failed: {text} -> {result} (expected {expected_type})"

print("✅ Type classification tests passed")
```

**Verification:**
```bash
uv run pytest tests/unit/agents/extraction/test_type_classifier.py -v

# Expected:
# ✅ test_functional_classification PASSED
# ✅ test_non_functional_classification PASSED
# ✅ test_security_classification PASSED
# ✅ test_data_classification PASSED
# ✅ test_interface_classification PASSED
# ✅ test_business_classification PASSED
# ✅ test_constraint_classification PASSED
```

---

### ✅ AC6: Acceptance Criteria Generation

**Given** a requirement statement
**When** CriteriaGenerator processes it
**Then** it produces testable acceptance criteria:

**CriteriaGenerator Implementation:**
```python
# src/agents/extraction/criteria_generator.py
from typing import List
import re

class CriteriaGenerator:
    """Generates testable acceptance criteria from requirements"""

    CRITERIA_TEMPLATES = {
        "authentication": [
            "User can enter {auth_method} in valid format",
            "System validates {auth_method} against database",
            "User is granted access on successful validation",
            "User sees error message on validation failure",
            "Failed attempts are logged for security audit"
        ],
        "performance": [
            "{action} completes in {threshold} for {percentile} of requests",
            "System maintains performance under {load} concurrent users",
            "Response time is measured and logged",
            "Performance metrics are monitored in production"
        ],
        "data_upload": [
            "User can select file from {device}",
            "System validates file format is {formats}",
            "System enforces file size limit of {size_limit}",
            "Upload progress is displayed to user",
            "User receives confirmation on successful upload"
        ],
        "generic": [
            "{actor} can {action}",
            "System validates {action} meets requirements",
            "{actor} receives confirmation of {action}",
            "Errors during {action} are handled gracefully"
        ]
    }

    def generate(
        self,
        actor: str,
        action: str,
        req_type: str,
        context: dict = None
    ) -> List[str]:
        """Generate acceptance criteria based on requirement details"""

        context = context or {}
        criteria = []

        # Determine template category
        if "log in" in action.lower() or "authenticate" in action.lower():
            template_key = "authentication"
            context.setdefault("auth_method", "email and password")
        elif "load" in action.lower() or "response" in action.lower():
            template_key = "performance"
            context.setdefault("action", action)
            context.setdefault("threshold", "< 2 seconds")
            context.setdefault("percentile", "95%")
        elif "upload" in action.lower():
            template_key = "data_upload"
            context.setdefault("device", "local device")
            context.setdefault("formats", "JPEG, PNG")
            context.setdefault("size_limit", "5MB")
        else:
            template_key = "generic"

        # Apply templates
        templates = self.CRITERIA_TEMPLATES.get(template_key, self.CRITERIA_TEMPLATES["generic"])

        for template in templates:
            try:
                criterion = template.format(
                    actor=actor,
                    action=action,
                    **context
                )
                criteria.append(criterion)
            except KeyError:
                # Skip template if context missing
                continue

        # Ensure minimum criteria
        if not criteria:
            criteria.append(f"{actor.capitalize()} can {action}")
            criteria.append(f"System confirms {action} completion")

        return criteria

# Test
generator = CriteriaGenerator()

# Test case 1: Authentication
criteria = generator.generate(
    actor="user",
    action="authenticate using email and password",
    req_type="functional"
)
print("Authentication Criteria:")
for c in criteria:
    print(f"  - {c}")

# Test case 2: Performance
criteria = generator.generate(
    actor="system",
    action="load application homepage",
    req_type="non_functional",
    context={"threshold": "< 2.5 seconds", "percentile": "99%"}
)
print("\nPerformance Criteria:")
for c in criteria:
    print(f"  - {c}")
```

**Verification:**
```bash
uv run pytest tests/unit/agents/extraction/test_criteria_generator.py -v

# Expected:
# ✅ test_authentication_criteria PASSED
# ✅ test_performance_criteria PASSED
# ✅ test_upload_criteria PASSED
# ✅ test_generic_criteria PASSED
# ✅ test_minimum_criteria_generated PASSED
```

---

### ✅ AC7: Traceability Linking

**Given** extracted requirements and chat history
**When** TraceabilityLinker processes them
**Then** each requirement is linked to source chat turns:

**TraceabilityLinker Implementation:**
```python
# src/agents/extraction/traceability_linker.py
from typing import List, Dict
from src.schemas.chat import Message

class TraceabilityLinker:
    """Links requirements to source conversation turns"""

    def link_to_sources(
        self,
        requirement_text: str,
        chat_history: List[Message]
    ) -> List[str]:
        """Find chat turns that contributed to this requirement"""

        # Extract keywords from requirement
        keywords = self._extract_keywords(requirement_text)

        # Search chat history for matching turns
        source_refs = []

        for turn_idx, message in enumerate(chat_history):
            if message.role != "user":
                continue

            # Calculate keyword overlap
            overlap = self._calculate_overlap(keywords, message.content.lower())

            # If significant overlap, add as source
            if overlap >= 0.3:  # 30% keyword match threshold
                source_refs.append(f"chat:turn:{turn_idx}")

        # Ensure at least one source
        if not source_refs and chat_history:
            # Default to last user message
            for i in range(len(chat_history) - 1, -1, -1):
                if chat_history[i].role == "user":
                    source_refs.append(f"chat:turn:{i}")
                    break

        return source_refs

    def _extract_keywords(self, text: str) -> set:
        """Extract meaningful keywords from text"""
        # Remove common words
        stop_words = {
            "the", "a", "an", "and", "or", "but", "in", "on", "at",
            "to", "for", "of", "with", "by", "from", "is", "are",
            "was", "were", "be", "been", "being", "have", "has", "had"
        }

        words = text.lower().split()
        keywords = {w.strip(".,!?") for w in words if len(w) > 3 and w not in stop_words}
        return keywords

    def _calculate_overlap(self, keywords: set, text: str) -> float:
        """Calculate keyword overlap percentage"""
        if not keywords:
            return 0.0

        text_words = set(text.split())
        matches = keywords.intersection(text_words)
        return len(matches) / len(keywords)

# Test
linker = TraceabilityLinker()

chat_history = [
    Message(id="1", role="user", content="Users should log in with email and password"),
    Message(id="2", role="assistant", content="Got it. Any specific requirements?"),
    Message(id="3", role="user", content="Yes, the login should be fast, under 2 seconds"),
]

req_text = "User authentication using email and password credentials"
sources = linker.link_to_sources(req_text, chat_history)

print(f"Requirement: {req_text}")
print(f"Linked to: {sources}")
# Expected: ['chat:turn:0'] (first user message)
```

**Verification:**
```bash
uv run pytest tests/unit/agents/extraction/test_traceability_linker.py -v

# Expected:
# ✅ test_links_to_matching_turns PASSED
# ✅ test_multiple_source_turns PASSED
# ✅ test_keyword_overlap_calculation PASSED
# ✅ test_fallback_to_last_turn PASSED
```

---

### ✅ AC8: Ambiguity Detection Integration

**Given** extracted requirement text
**When** AmbiguityDetector analyzes it
**Then** it flags vague or unmeasurable language:

**AmbiguityDetector Implementation:**
```python
# src/agents/extraction/ambiguity_detector.py
from typing import List, Dict
import re

class AmbiguityDetector:
    """Detects ambiguous and non-measurable language"""

    AMBIGUOUS_TERMS = {
        "adjectives": ["fast", "slow", "quick", "easy", "simple", "good", "bad",
                      "better", "worse", "nice", "intuitive", "user-friendly"],
        "verbs": ["optimize", "improve", "enhance", "maximize", "minimize"],
        "quantifiers": ["many", "few", "some", "several", "various", "multiple"],
        "qualifiers": ["reasonable", "appropriate", "sufficient", "adequate", "acceptable"]
    }

    def detect(self, text: str) -> Dict:
        """Detect ambiguous language in requirement"""

        text_lower = text.lower()
        found_terms = []

        # Check for ambiguous terms
        for category, terms in self.AMBIGUOUS_TERMS.items():
            for term in terms:
                if re.search(rf"\b{term}\b", text_lower):
                    found_terms.append({
                        "term": term,
                        "category": category,
                        "position": text_lower.find(term)
                    })

        # Calculate ambiguity score
        ambiguity_score = min(len(found_terms) * 0.2, 1.0)

        # Generate clarifying questions
        suggestions = self._generate_suggestions(found_terms)

        return {
            "is_ambiguous": len(found_terms) > 0,
            "ambiguity_score": ambiguity_score,
            "ambiguous_terms": found_terms,
            "suggestions": suggestions
        }

    def _generate_suggestions(self, terms: List[Dict]) -> List[str]:
        """Generate clarifying questions for ambiguous terms"""
        suggestions = []

        for term_info in terms:
            term = term_info["term"]

            if term in ["fast", "quick", "slow"]:
                suggestions.append(f"What is the specific time threshold for '{term}'? (e.g., < 2 seconds)")
            elif term in ["scalable", "many", "few"]:
                suggestions.append(f"What is the specific quantity for '{term}'? (e.g., 1000 concurrent users)")
            elif term in ["easy", "simple", "intuitive"]:
                suggestions.append(f"What measurable criteria defines '{term}'? (e.g., 90% task completion rate)")
            elif term in ["optimize", "improve", "enhance"]:
                suggestions.append(f"What metric should be optimized? (e.g., reduce latency by 30%)")
            else:
                suggestions.append(f"Please clarify what '{term}' means in measurable terms")

        return suggestions[:3]  # Limit to top 3 suggestions

# Test
detector = AmbiguityDetector()

test_cases = [
    "The app should load quickly and be user-friendly",
    "System must handle many concurrent users efficiently",
    "Login should complete in under 2 seconds with 99% success rate"
]

for text in test_cases:
    result = detector.detect(text)
    print(f"\nText: {text}")
    print(f"Ambiguous: {result['is_ambiguous']}")
    print(f"Score: {result['ambiguity_score']:.2f}")
    if result['ambiguous_terms']:
        print(f"Terms: {[t['term'] for t in result['ambiguous_terms']]}")
        print(f"Suggestions: {result['suggestions']}")
```

**Verification:**
```bash
uv run pytest tests/unit/agents/extraction/test_ambiguity_detector.py -v

# Expected:
# ✅ test_detects_ambiguous_adjectives PASSED
# ✅ test_detects_ambiguous_verbs PASSED
# ✅ test_calculates_ambiguity_score PASSED
# ✅ test_generates_clarifying_suggestions PASSED
# ✅ test_no_ambiguity_in_measurable_text PASSED
```

---

### ✅ AC9: Complete Extraction Pipeline

**Given** conversational state with chat history
**When** ExtractionAgent runs full pipeline
**Then** it produces validated RequirementItem objects:

**Full Agent Implementation:**
```python
# src/agents/extraction/agent.py
from src.agents.base import BaseAgent
from src.schemas.state import GraphState
from src.schemas.requirement import RequirementItem, RequirementType, Priority
from src.agents.extraction.entity_extractor import EntityExtractor
from src.agents.extraction.type_classifier import TypeClassifier
from src.agents.extraction.criteria_generator import CriteriaGenerator
from src.agents.extraction.traceability_linker import TraceabilityLinker
from src.agents.extraction.ambiguity_detector import AmbiguityDetector
from typing import Dict, Any, List
from datetime import datetime
import uuid

class ExtractionAgent(BaseAgent):
    """Extract structured requirements from conversation"""

    def __init__(self, **kwargs):
        super().__init__(name="extraction", temperature=0.2, **kwargs)

        # Initialize sub-components
        self.entity_extractor = EntityExtractor()
        self.type_classifier = TypeClassifier()
        self.criteria_generator = CriteriaGenerator()
        self.traceability_linker = TraceabilityLinker()
        self.ambiguity_detector = AmbiguityDetector()

        self.requirement_counter = 0

    async def execute(self, state: GraphState) -> Dict[str, Any]:
        """Execute extraction pipeline"""

        # Extract user messages only
        user_messages = [
            msg for msg in state.chat_history
            if msg.role == "user"
        ]

        if not user_messages:
            return {
                "requirements": [],
                "confidence": 1.0,
                "ambiguous_items": []
            }

        # Combine all user messages for context
        full_context = " ".join([msg.content for msg in user_messages])

        # Extract requirements from each message
        requirements = []
        ambiguous_items = []

        for msg in user_messages:
            extracted = await self._extract_from_message(
                msg.content,
                state.chat_history,
                state.session_id
            )

            requirements.extend(extracted["requirements"])
            ambiguous_items.extend(extracted["ambiguous_items"])

        # Calculate overall confidence
        if requirements:
            avg_confidence = sum(r.confidence for r in requirements) / len(requirements)
        else:
            avg_confidence = 1.0

        return {
            "requirements": requirements,
            "confidence": avg_confidence,
            "ambiguous_items": ambiguous_items,
            "extraction_metadata": {
                "total_requirements": len(requirements),
                "ambiguous_count": len(ambiguous_items),
                "processed_messages": len(user_messages)
            }
        }

    async def _extract_from_message(
        self,
        message_text: str,
        chat_history: List,
        session_id: str
    ) -> Dict:
        """Extract requirements from a single message"""

        # Check if message contains requirement-like statements
        if not self._contains_requirement(message_text):
            return {"requirements": [], "ambiguous_items": []}

        # Extract entities
        actors = self.entity_extractor.extract_actors(message_text)
        action = self.entity_extractor.extract_action(message_text)
        condition = self.entity_extractor.extract_condition(message_text)

        # Classify type
        req_type = self.type_classifier.classify(message_text, action)

        # Generate acceptance criteria
        criteria = self.criteria_generator.generate(
            actor=actors[0] if actors else "user",
            action=action,
            req_type=req_type.value
        )

        # Link to source turns
        source_refs = self.traceability_linker.link_to_sources(
            message_text,
            chat_history
        )

        # Detect ambiguity
        ambiguity_result = self.ambiguity_detector.detect(message_text)

        # Generate requirement ID
        self.requirement_counter += 1
        req_id = f"REQ-{self.requirement_counter:03d}"

        # Create title (first 50 chars of action)
        title = action[:50] if len(action) <= 50 else action[:47] + "..."

        # Calculate confidence
        confidence = self._calculate_confidence(ambiguity_result, criteria)

        # Build requirement
        requirement = RequirementItem(
            id=req_id,
            title=title,
            type=req_type,
            actor=actors[0] if actors else "user",
            action=action,
            condition=condition,
            acceptance_criteria=criteria,
            priority=Priority.MEDIUM,
            confidence=confidence,
            inferred=False,
            rationale=f"Extracted from user message: '{message_text[:100]}...'",
            source_refs=source_refs,
            created_at=datetime.utcnow()
        )

        # Collect ambiguous items
        ambiguous = []
        if ambiguity_result["is_ambiguous"]:
            ambiguous.append({
                "requirement_id": req_id,
                "terms": [t["term"] for t in ambiguity_result["ambiguous_terms"]],
                "suggestions": ambiguity_result["suggestions"]
            })

        return {
            "requirements": [requirement],
            "ambiguous_items": ambiguous
        }

    def _contains_requirement(self, text: str) -> bool:
        """Check if text contains a requirement statement"""
        requirement_indicators = [
            "should", "must", "shall", "need", "require", "want",
            "can", "able to", "support", "allow", "enable"
        ]
        text_lower = text.lower()
        return any(indicator in text_lower for indicator in requirement_indicators)

    def _calculate_confidence(
        self,
        ambiguity_result: Dict,
        criteria: List[str]
    ) -> float:
        """Calculate extraction confidence"""

        # Start with base confidence
        confidence = 0.9

        # Reduce for ambiguity
        confidence -= ambiguity_result["ambiguity_score"] * 0.3

        # Boost for quality criteria
        if len(criteria) >= 3:
            confidence += 0.05

        # Ensure bounds
        return max(0.5, min(1.0, confidence))
```

**Verification:**
```bash
uv run pytest tests/integration/agents/test_extraction_agent.py -v

# Expected:
# ✅ test_full_extraction_pipeline PASSED
# ✅ test_multiple_requirements_extraction PASSED
# ✅ test_ambiguity_flagging PASSED
# ✅ test_traceability_links PASSED
# ✅ test_type_classification_accuracy PASSED
```

---

### ✅ AC10: Database Persistence Integration

**Given** extracted requirements
**When** they are persisted to database
**Then** all fields are correctly stored and retrievable:

**Database Model:**
```python
# src/models/database.py
from sqlalchemy import Column, String, Float, Boolean, DateTime, ARRAY, Text, Integer, ForeignKey
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import relationship
from datetime import datetime
import uuid

class RequirementModel(Base):
    __tablename__ = "requirements"

    id = Column(String(50), primary_key=True)  # REQ-001
    session_id = Column(UUID(as_uuid=True), ForeignKey("sessions.id"), nullable=False)

    # Core fields
    title = Column(String(500), nullable=False)
    type = Column(String(50), nullable=False)
    actor = Column(String(200), nullable=False)
    action = Column(Text, nullable=False)
    condition = Column(Text, nullable=True)

    # Criteria and metadata
    acceptance_criteria = Column(JSONB, nullable=False)  # List of strings
    priority = Column(String(20), nullable=False)
    confidence = Column(Float, nullable=False)
    inferred = Column(Boolean, default=False)
    rationale = Column(Text, nullable=False)
    source_refs = Column(JSONB, nullable=False)  # List of strings

    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, onupdate=datetime.utcnow)

    # Relationships
    session = relationship("SessionModel", back_populates="requirements")

    # Indexes for performance
    __table_args__ = (
        Index('idx_requirements_session', 'session_id'),
        Index('idx_requirements_type', 'type'),
        Index('idx_requirements_confidence', 'confidence'),
    )
```

**Storage Service:**
```python
# src/storage/requirement_store.py
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from src.models.database import RequirementModel
from src.schemas.requirement import RequirementItem
from typing import List
from uuid import UUID

class RequirementStore:
    """Persist and retrieve requirements"""

    async def save_requirement(
        self,
        db: AsyncSession,
        session_id: UUID,
        requirement: RequirementItem
    ) -> RequirementModel:
        """Save requirement to database"""

        db_req = RequirementModel(
            id=requirement.id,
            session_id=session_id,
            title=requirement.title,
            type=requirement.type.value,
            actor=requirement.actor,
            action=requirement.action,
            condition=requirement.condition,
            acceptance_criteria=requirement.acceptance_criteria,
            priority=requirement.priority.value,
            confidence=requirement.confidence,
            inferred=requirement.inferred,
            rationale=requirement.rationale,
            source_refs=requirement.source_refs,
            created_at=requirement.created_at
        )

        db.add(db_req)
        await db.commit()
        await db.refresh(db_req)

        return db_req

    async def get_requirements(
        self,
        db: AsyncSession,
        session_id: UUID
    ) -> List[RequirementItem]:
        """Retrieve all requirements for a session"""

        query = select(RequirementModel).where(
            RequirementModel.session_id == session_id
        ).order_by(RequirementModel.created_at)

        result = await db.execute(query)
        db_reqs = result.scalars().all()

        # Convert to Pydantic models
        return [self._to_pydantic(req) for req in db_reqs]

    def _to_pydantic(self, db_req: RequirementModel) -> RequirementItem:
        """Convert DB model to Pydantic"""
        return RequirementItem(
            id=db_req.id,
            title=db_req.title,
            type=db_req.type,
            actor=db_req.actor,
            action=db_req.action,
            condition=db_req.condition,
            acceptance_criteria=db_req.acceptance_criteria,
            priority=db_req.priority,
            confidence=db_req.confidence,
            inferred=db_req.inferred,
            rationale=db_req.rationale,
            source_refs=db_req.source_refs,
            created_at=db_req.created_at
        )
```

**Verification:**
```bash
uv run pytest tests/integration/storage/test_requirement_store.py -v

# Expected:
# ✅ test_save_requirement PASSED
# ✅ test_retrieve_requirements PASSED
# ✅ test_pydantic_conversion PASSED
# ✅ test_jsonb_fields_preserved PASSED
```

---

### ✅ AC11: Vector Embedding for Semantic Search

**Given** extracted requirements
**When** they are embedded and stored in ChromaDB
**Then** semantic similarity search works:

**Vector Store Integration:**
```python
# src/agents/extraction/agent.py (addition)
async def _store_embeddings(
    self,
    requirements: List[RequirementItem],
    session_id: str
):
    """Store requirement embeddings for semantic search"""

    from src.services.embedding_service import EmbeddingService
    from src.storage.vectorstore import VectorStoreService

    embedding_svc = EmbeddingService()
    vector_store = VectorStoreService()

    for req in requirements:
        # Create searchable text
        text = f"{req.title} {req.action}"

        # Generate embedding
        embedding = await embedding_svc.get_embedding(text)

        # Store in ChromaDB
        await vector_store.add_requirement(
            requirement_id=req.id,
            title=req.title,
            action=req.action,
            embedding=embedding,
            metadata={
                "session_id": session_id,
                "type": req.type.value,
                "priority": req.priority.value,
                "confidence": req.confidence,
                "inferred": req.inferred
            }
        )
```

**Verification:**
```bash
uv run pytest tests/integration/agents/test_extraction_embedding.py -v

# Expected:
# ✅ test_embedding_generation PASSED
# ✅ test_vector_storage PASSED
# ✅ test_semantic_similarity_search PASSED
# ✅ test_duplicate_detection PASSED
```

---

### ✅ AC12: End-to-End Extraction Flow

**Given** a complete conversation session
**When** ExtractionAgent processes it
**Then** all requirements are extracted, stored, and retrievable:

**Integration Test:**
```python
# tests/integration/agents/test_extraction_e2e.py
import pytest
from src.agents.extraction.agent import ExtractionAgent
from src.schemas.state import GraphState
from src.schemas.chat import Message
from datetime import datetime
from uuid import uuid4

@pytest.mark.asyncio
async def test_end_to_end_extraction_flow(db_session):
    """Test complete extraction pipeline"""

    # Setup
    agent = ExtractionAgent()
    session_id = str(uuid4())

    # Create conversation
    state = GraphState(
        session_id=session_id,
        project_name="E-commerce Platform",
        user_id="user-123",
        chat_history=[
            Message(
                id="msg-1",
                role="user",
                content="Users should be able to log in with email and password",
                timestamp=datetime.utcnow()
            ),
            Message(
                id="msg-2",
                role="assistant",
                content="Got it. Any performance requirements?",
                timestamp=datetime.utcnow()
            ),
            Message(
                id="msg-3",
                role="user",
                content="Yes, login should complete in under 2 seconds",
                timestamp=datetime.utcnow()
            ),
            Message(
                id="msg-4",
                role="user",
                content="Also, we need to support iOS and Android platforms",
                timestamp=datetime.utcnow()
            )
        ],
        current_turn=4,
        requirements=[],
        confidence=1.0
    )

    # Execute extraction
    result = await agent.invoke(state)

    # Assertions
    assert len(result.requirements) >= 3

    # Check requirement 1: Authentication
    auth_req = result.requirements[0]
    assert auth_req.id == "REQ-001"
    assert auth_req.type.value == "functional"
    assert "log in" in auth_req.action or "authenticate" in auth_req.action
    assert len(auth_req.acceptance_criteria) >= 1
    assert len(auth_req.source_refs) >= 1
    assert auth_req.source_refs[0].startswith("chat:turn:")

    # Check requirement 2: Performance
    perf_req = result.requirements[1]
    assert perf_req.id == "REQ-002"
    assert perf_req.type.value == "non_functional"
    assert "2 seconds" in " ".join(perf_req.acceptance_criteria)

    # Check requirement 3: Platform support
    platform_req = result.requirements[2]
    assert platform_req.id == "REQ-003"
    assert platform_req.type.value == "constraint"
    assert "iOS" in platform_req.action or "Android" in platform_req.action

    # Check overall quality
    assert result.confidence >= 0.7
    assert result.last_agent == "extraction"

    print(f"✅ Extracted {len(result.requirements)} requirements successfully")
    for req in result.requirements:
        print(f"  - {req.id}: {req.title} (confidence: {req.confidence:.2f})")
```

**Verification:**
```bash
uv run pytest tests/integration/agents/test_extraction_e2e.py -v --tb=short

# Expected output:
# ✅ Extracted 3 requirements successfully
#   - REQ-001: User authentication using email and password (confidence: 0.88)
#   - REQ-002: Login completion time under 2 seconds (confidence: 0.90)
#   - REQ-003: Platform support for iOS and Android (confidence: 0.85)
# PASSED
```

---

## Technical Implementation Summary

### Key Files Created

1. **`src/agents/extraction/agent.py`** (350+ lines)
   - Main ExtractionAgent class
   - Pipeline orchestration
   - Confidence calculation

2. **`src/agents/extraction/entity_extractor.py`** (150+ lines)
   - Actor, action, condition extraction
   - Regex patterns for entities

3. **`src/agents/extraction/type_classifier.py`** (200+ lines)
   - Rule-based + LLM classification
   - Type indicators dictionary

4. **`src/agents/extraction/criteria_generator.py`** (180+ lines)
   - Template-based criteria generation
   - Context-aware formatting

5. **`src/agents/extraction/traceability_linker.py`** (100+ lines)
   - Keyword overlap calculation
   - Source turn linking

6. **`src/agents/extraction/ambiguity_detector.py`** (120+ lines)
   - Vague term detection
   - Clarification suggestions

7. **`src/schemas/requirement.py`** (150+ lines)
   - RequirementItem model
   - Pydantic validation
   - JSON schema

8. **`src/models/database.py`** (80+ lines - addition)
   - RequirementModel SQLAlchemy model
   - Database indexes

9. **`src/storage/requirement_store.py`** (120+ lines)
   - CRUD operations
   - Pydantic ↔ SQLAlchemy conversion

---

## Testing Strategy

### Unit Tests (30+ tests)
- Entity extraction components
- Type classification accuracy
- Criteria generation templates
- Traceability linking logic
- Ambiguity detection rules
- Schema validation

### Integration Tests (15+ tests)
- Full extraction pipeline
- Database persistence
- Vector embedding storage
- End-to-end flow
- Multi-requirement extraction

### Quality Metrics
- **Precision Target**: >90% (correct extractions / total extractions)
- **Recall Target**: >95% (extracted requirements / total mentioned)
- **F1 Score Target**: >92%
- **Traceability Coverage**: 100% (all requirements linked)

---

## Definition of Done

- [ ] All 12 acceptance criteria passed and verified
- [ ] ExtractionAgent fully functional with sub-components
- [ ] RequirementItem schema with validation implemented
- [ ] Entity extraction working for actors, actions, conditions
- [ ] Type classification accurate (>85% on test cases)
- [ ] Acceptance criteria generation producing testable criteria
- [ ] Traceability linking to source chat turns
- [ ] Ambiguity detection flagging vague terms
- [ ] Database persistence complete with indexes
- [ ] Vector embeddings stored in ChromaDB
- [ ] End-to-end integration test passing
- [ ] Unit tests: 100% pass rate (30+ tests)
- [ ] Integration tests: 100% pass rate (15+ tests)
- [ ] Code follows project conventions (Black, Ruff, MyPy)
- [ ] All docstrings present and clear
- [ ] Type hints complete throughout

---

## Dependencies for Next Stories

Once Story 3 is complete, the following stories can proceed:

- **STORY-004:** Inference Agent Implementation (proposes implicit requirements)
- **STORY-005:** Validation Agent Implementation (checks extracted requirements)
- **STORY-006:** LangGraph Orchestrator Integration (connects all agents)

---

## Notes for Windsurf AI Implementation

### Key Implementation Priorities

1. **Start with RequirementItem schema** - Defines the output contract
2. **Build EntityExtractor first** - Foundation for all extraction
3. **Implement TypeClassifier** - Critical for correct categorization
4. **Add CriteriaGenerator** - Ensures testability
5. **Integrate TraceabilityLinker** - Auditability requirement
6. **Add AmbiguityDetector** - Quality control
7. **Complete full agent pipeline** - Orchestrate all components

### Critical Design Decisions

**Q: Why separate entity extraction, type classification, and criteria generation?**
**A:** Single Responsibility Principle. Each component is independently testable and replaceable. For example, we could enhance EntityExtractor with NLP (spaCy) without affecting other components.

**Q: Why use both rule-based and LLM-based approaches?**
**A:** Rule-based is fast, deterministic, and cheap. LLM is flexible but slower and costs tokens. Hybrid approach: rules first, LLM for edge cases.

**Q: How do we prevent over-extraction?**
**A:** `_contains_requirement()` method filters messages. Only messages with requirement indicators ("should", "must", "need") are processed.

**Q: What if users contradict themselves?**
**A:** Traceability enables conflict detection. Validation Agent (Story 5) will compare embeddings and flag conflicts.

### Testing Execution

```bash
# Run all extraction tests
uv run pytest tests/unit/agents/extraction/ -v --tb=short

# Run integration tests
uv run pytest tests/integration/agents/test_extraction_agent.py -v --tb=short

# Test specific component
uv run pytest tests/unit/agents/extraction/test_entity_extractor.py::test_extract_actors -v

# With coverage
uv run pytest tests/unit/agents/extraction/ --cov=src/agents/extraction --cov-report=html
```

### Common Pitfalls to Avoid

- ❌ Don't hallucinate requirements - only extract what was stated
- ❌ Don't skip traceability - every requirement needs source_refs
- ❌ Don't generate generic criteria - use context for specificity
- ❌ Don't ignore ambiguity - flag it for clarification
- ❌ Don't forget type hints - critical for schema validation
- ❌ Don't skip database persistence - requirements must be stored
- ❌ Don't ignore embeddings - needed for duplicate/conflict detection

### Performance Considerations

- **Token optimization**: Extract from user messages only (skip assistant messages)
- **Batch processing**: If extracting from many messages, batch LLM calls
- **Caching**: Cache entity extraction results for common patterns
- **Parallelization**: Extract from multiple messages in parallel

---

## References

- **Design Packet 1:** Section 8.2 - Extraction Agent
- **Design Packet 2:** Section 6 - Agent Implementation Blueprints
- **Story 1:** Project Foundation (prerequisite)
- **Story 2:** Conversational Agent (prerequisite)
- **IEEE 830:** Requirements specification standard
- **IREB Foundation Level:** Requirements engineering body of knowledge

---

**End of Story 3 Document**

**Next Steps:**
1. Implement RequirementItem schema with full validation
2. Build EntityExtractor with regex patterns
3. Create TypeClassifier with keyword mapping
4. Develop CriteriaGenerator with templates
5. Integrate all components into ExtractionAgent
6. Test extraction quality with test conversations
7. Prepare for Story 4 (Inference Agent) which builds on extracted requirements
