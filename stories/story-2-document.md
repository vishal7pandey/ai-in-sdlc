# User Story 2: Conversational Agent Implementation

## Story Overview

**Story ID:** STORY-002
**Story Title:** Conversational Agent Implementation
**Priority:** P0 - Critical
**Estimated Effort:** 20-28 hours
**Sprint:** Sprint 2
**Dependencies:** STORY-001 (Project Foundation must be complete)

---

## Story Description

As a **requirements engineer**, I want to **implement the foundational Conversational Agent that handles natural language interaction with stakeholders** so that **I can begin gathering requirements through guided, context-aware dialogue that maintains conversation flow and detects when clarification is needed**.

This story implements the **first critical agent** in the orchestration pipeline—the **Conversational Agent**. This agent drives the entire workflow by:

- Greeting users and setting conversation context
- Asking targeted clarifying questions without overwhelming the user
- Detecting when sufficient information has been gathered for requirement extraction
- Maintaining conversation flow and tone
- Escalating to human review when confidence drops below threshold

---

## Business Value

- **Enables stakeholder engagement** through natural conversation rather than forms
- **Reduces analysis burden** on requirements engineers through AI-guided dialogue
- **Improves requirement quality** by asking clarifying questions early
- **Accelerates first-pass discovery** via intelligent conversation flow
- **Establishes baseline** for measuring LLM output quality and confidence

---

## Acceptance Criteria

### ✅ AC1: Agent Module Structure

**Given** the project source code
**When** I examine the `src/agents/` directory
**Then** the Conversational Agent is properly structured:

**Directory Structure:**
```
src/agents/
├── __init__.py
├── base.py                      # Base agent class
├── conversational/
│   ├── __init__.py
│   ├── agent.py                 # Main ConversationalAgent class
│   ├── prompt_builder.py         # Dynamic prompt construction
│   ├── context_manager.py        # Conversation context tracking
│   ├── clarification_detector.py # Ambiguity detection logic
│   └── response_formatter.py     # Response parsing and validation
├── extraction/                   # (Placeholder for Story 3)
├── inference/                    # (Placeholder for Story 3)
├── validation/                   # (Placeholder for Story 3)
└── synthesis/                    # (Placeholder for Story 4)
```

**Verification:**
```bash
find src/agents/conversational -name "*.py" | wc -l
# Expected: 6 files total

ls -la src/agents/conversational/
# Expected: All files above present

python -c "from src.agents.conversational.agent import ConversationalAgent; print('✅ Import successful')"
# Expected: ✅ Import successful
```

---

### ✅ AC2: Base Agent Class

**Given** the agents module structure
**When** I review `src/agents/base.py`
**Then** a generic base class exists with:

**Base Agent Class Properties:**
```python
# src/agents/base.py should define:
class BaseAgent(ABC):
    - self.llm: ChatOpenAI
    - self.name: str
    - self.logger: Logger
    - self.retry_policy: RetryPolicy
    - async def invoke(state: GraphState) -> GraphState  # Abstract method
    - async def execute(state: GraphState) -> Dict        # For subclasses
    - Error handling with circuit breaker pattern
    - Instrumentation for metrics and logging
    - Token budget management
```

**Verification:**
```python
from src.agents.base import BaseAgent
from src.schemas.state import GraphState
import inspect

# Verify abstract base class
assert inspect.isabstract(BaseAgent), "BaseAgent should be abstract"

# Verify required methods
methods = [m for m in dir(BaseAgent) if not m.startswith('_')]
print(f"✅ BaseAgent methods: {methods}")
```

---

### ✅ AC3: Conversational Agent Core Implementation

**Given** the base agent class
**When** I instantiate and invoke the ConversationalAgent
**Then** it correctly processes user messages and generates responses:

**Agent Initialization:**
```python
from src.agents.conversational.agent import ConversationalAgent
from src.config import settings

agent = ConversationalAgent(
    openai_api_key=settings.OPENAI_API_KEY,
    model=settings.OPENAI_MODEL,
    temperature=0.7,  # Slightly creative for natural conversation
    max_tokens=500
)

# Verify initialization
assert agent is not None
assert agent.name == "conversational"
print("✅ ConversationalAgent initialized successfully")
```

**Agent Invocation (Happy Path):**
```python
from src.schemas.state import GraphState
from src.schemas.chat import Message
from datetime import datetime

state = GraphState(
    session_id="test-session-001",
    project_name="E-commerce Platform v2",
    user_id="user-123",
    chat_history=[
        Message(
            id="msg-1",
            role="user",
            content="We need a mobile app for e-commerce",
            timestamp=datetime.utcnow()
        )
    ],
    current_turn=1,
    requirements=[],
    inferred_requirements=[],
    validation_issues=[],
    confidence=1.0,
    rd_draft=None,
    rd_version=0,
    approval_status="pending",
    review_feedback=None,
    last_agent="system",
    iterations=0,
    error_count=0
)

result = await agent.invoke(state)

# Verify result
assert result is not None
assert isinstance(result, GraphState)
assert len(result.chat_history) > len(state.chat_history)  # New message added
assert result.last_agent == "conversational"
assert 0.0 <= result.confidence <= 1.0
print("✅ ConversationalAgent produced valid output")
```

---

### ✅ AC4: Dynamic Prompt Template System

**Given** the Conversational Agent
**When** I examine prompt construction
**Then** prompts adapt to conversation context:

**Prompt Template File:**
```
File: src/templates/prompts/conversational.txt

Template sections:
- System role and responsibilities
- Conversation guidelines
- Clarifying question strategy
- Output format specifications
- Few-shot examples

Key characteristics:
- Respects token budget (max 800 tokens for system prompt)
- Includes context variables: {project_name}, {current_turn}, {requirement_count}, {chat_history}
- Adapts tone based on turn number
- Escalation logic for low confidence
```

**Verification:**
```bash
cat src/templates/prompts/conversational.txt | wc -l
# Expected: 50+ lines

grep -c "{" src/templates/prompts/conversational.txt
# Expected: 4+ template variables

grep -c "question\|clarif\|ask" src/templates/prompts/conversational.txt
# Expected: 3+ mentions of clarifying questions
```

---

### ✅ AC5: Context Manager

**Given** conversation history in state
**When** the ContextManager processes it
**Then** it extracts actionable context:

**ContextManager Responsibilities:**
```python
# src/agents/conversational/context_manager.py
class ContextManager:
    async def extract_context(state: GraphState) -> Dict:
        # Return:
        # - conversation_summary: str
        # - identified_domain: str  (e.g., "e-commerce", "saas", "mobile")
        # - mentioned_actors: List[str]
        # - mentioned_features: List[str]
        # - implicit_needs: List[str]
        # - clarification_gaps: List[str]
        # - turn_sentiment: str  (positive, neutral, negative)
        # - conversation_momentum: float (0.0-1.0, is user engaged?)
```

**Verification:**
```python
from src.agents.conversational.context_manager import ContextManager

context_mgr = ContextManager()
state = GraphState(...)  # Sample state with conversation history
context = await context_mgr.extract_context(state)

assert isinstance(context, dict)
assert "conversation_summary" in context
assert "identified_domain" in context
assert "clarification_gaps" in context
print(f"✅ Context extracted: {list(context.keys())}")
```

---

### ✅ AC6: Clarification Detection Logic

**Given** user input and conversation history
**When** the ClarificationDetector runs
**Then** it flags ambiguous statements:

**Clarification Detector Output:**
```python
class ClarificationResult:
    is_ambiguous: bool
    ambiguity_score: float  # 0.0 = clear, 1.0 = very ambiguous
    ambiguous_terms: List[str]  # e.g., ["fast", "easy", "scalable"]
    clarifying_questions: List[str]
    confidence: float  # Confidence in the ambiguity assessment

# Example output:
{
    "is_ambiguous": True,
    "ambiguity_score": 0.82,
    "ambiguous_terms": ["load quickly", "performant"],
    "clarifying_questions": [
        "What's your target load time? (e.g., 2 seconds on 4G)",
        "Which devices are priority? (iOS, Android, web)"
    ],
    "confidence": 0.88
}
```

**Verification:**
```bash
uv run pytest tests/unit/agents/test_clarification_detector.py -v

# Expected test output:
# ✅ test_detects_ambiguous_verbs PASSED
# ✅ test_flags_missing_metrics PASSED
# ✅ test_recognizes_vague_adjectives PASSED
# ✅ test_confidence_calculation PASSED
```

---

### ✅ AC7: Response Formatting and Validation

**Given** raw LLM output
**When** ResponseFormatter processes it
**Then** it returns structured, validated response:

**ResponseFormatter Output Schema:**
```python
from pydantic import BaseModel
from typing import List, Optional, Literal

class ConversationalResponse(BaseModel):
    message: str  # Response to user (100-500 chars, conversational tone)
    next_action: Literal[
        "continue_eliciting",  # Keep asking questions
        "extract_requirements",  # Enough info, move to extraction
        "clarify",              # Need clarification on previous statement
        "wait_for_input"        # Waiting for user response
    ]
    clarifying_questions: Optional[List[str]] = None  # If next_action == "clarify"
    confidence: float  # 0.0-1.0, confidence in this response quality
    extracted_topics: List[str] = []  # Topics mentioned by user
    sentiment: Literal["positive", "neutral", "negative"] = "neutral"

# Validation rules:
# - message length: 50-500 characters
# - confidence: must be between 0.0 and 1.0
# - clarifying_questions: only present if next_action == "clarify"
# - All fields must be present
```

**Verification:**
```python
from src.agents.conversational.response_formatter import ResponseFormatter
from src.agents.conversational.schemas import ConversationalResponse

formatter = ResponseFormatter()

# Test with valid raw LLM output
raw_output = """
thought: User is asking about login functionality, which is clearly a functional requirement.
They mentioned email, but didn't specify password reset needs.

response: Got it! So users need to log in with email and password. Just to clarify -
should the app support password recovery if someone forgets their password?

nextAction: clarify
clarifyingQuestions:
  - Should the app support password reset?
  - How should users verify their email?
confidence: 0.85
"""

response = formatter.parse_and_validate(raw_output)
assert isinstance(response, ConversationalResponse)
assert response.next_action == "clarify"
assert len(response.clarifying_questions) > 0
print(f"✅ Response validated: {response}")
```

---

### ✅ AC8: Turn-by-Turn Message Processing

**Given** a sequence of user inputs
**When** the agent processes messages sequentially
**Then** conversation state accumulates correctly:

**Test Scenario - 3 Turn Conversation:**

**Turn 1: Initial Request**
```
User: "We need a mobile app for e-commerce"

Agent Response:
  message: "Great! I'll help you define requirements for your e-commerce mobile app.
            To start, could you tell me what platforms you're targeting - iOS, Android, or both?"
  next_action: continue_eliciting
  confidence: 0.90
  extracted_topics: ["mobile app", "e-commerce", "platforms"]
```

**Turn 2: Follow-up Answer**
```
User: "Both iOS and Android. Also needs to be fast for users in India."

Agent Response:
  message: "Excellent! Cross-platform coverage and performance are important.
            When you say 'fast', what would be an acceptable app load time?
            For example, 2-3 seconds on 4G networks?"
  next_action: continue_eliciting
  confidence: 0.82
  extracted_topics: ["iOS", "Android", "performance", "India", "load time"]
```

**Turn 3: Measurement Provided**
```
User: "Yeah, under 2.5 seconds cold start time is the goal."

Agent Response:
  message: "Perfect! I've captured your core requirements. Let me extract and organize
            what we've discussed so far."
  next_action: extract_requirements
  confidence: 0.88
  extracted_topics: ["cold start", "2.5 seconds", "performance target"]
```

**Verification:**
```bash
uv run pytest tests/integration/agents/test_conversational_flow.py -v

# Expected:
# ✅ test_three_turn_conversation PASSED
# ✅ test_accumulates_topics_correctly PASSED
# ✅ test_confidence_progression PASSED
# ✅ test_next_action_transitions PASSED
```

---

### ✅ AC9: Token Budget Management

**Given** a Conversational Agent with token limits
**When** processing long conversations
**Then** tokens stay within budget (max 8000 total):

**Token Budget Allocation:**
```
Total budget: 8000 tokens

System prompt:          800 tokens (10%)
Chat history:         3000 tokens (37.5%) - truncate oldest messages
Few-shot examples:    1200 tokens (15%)
Response buffer:      3000 tokens (37.5%)

Test case:
- Start with 20 messages (2000 tokens total)
- Agent should include: last 15 messages (1500 tokens)
- Total usage: 800 + 1500 + 1200 + response ≤ 8000
```

**Verification:**
```python
from src.agents.conversational.token_budget import TokenBudgetManager
import tiktoken

manager = TokenBudgetManager(model="gpt-4-turbo-preview", max_tokens=8000)

# Generate sample messages
messages = [
    {"role": "user", "content": f"Message number {i}" * 50}
    for i in range(20)
]

truncated = manager.truncate_chat_history(messages)
tokens = manager.count_tokens(str(truncated))

assert tokens <= 3000, f"Chat history exceeded budget: {tokens} > 3000"
print(f"✅ Token budget respected: {tokens}/3000 tokens used")
```

---

### ✅ AC10: Confidence Scoring

**Given** agent output
**When** confidence is calculated
**Then** it reflects output quality accurately:

**Confidence Scoring Formula:**
```
confidence = (
    0.5 × llm_confidence +
    0.3 × parse_quality +
    0.2 × source_clarity
)

Where:
- llm_confidence: LLM's self-reported confidence in response (0.0-1.0)
- parse_quality: Percentage of required fields successfully extracted
  - Required: message, next_action, confidence
  - Calculated as: fields_present / 3
- source_clarity: Inverse of ambiguity in user's original message
  - Full clarity (no ambiguous terms): 1.0
  - Some ambiguity (1-2 vague terms): 0.7
  - High ambiguity (3+ vague terms): 0.4

Test case:
- LLM confidence: 0.90
- Parse quality: 1.0 (all 3 fields present)
- Source clarity: 0.85 (slightly ambiguous user input)
- Final: (0.5 × 0.90) + (0.3 × 1.0) + (0.2 × 0.85) = 0.62 + 0.30 + 0.17 = 0.89
```

**Verification:**
```python
from src.agents.conversational.confidence_scorer import ConfidenceScorer

scorer = ConfidenceScorer()

result = scorer.calculate(
    llm_confidence=0.90,
    parsed_fields={"message": True, "next_action": True, "confidence": True},
    source_text="Users should be able to log in fast and easily"
)

# Parsed fields: 3/3 = 1.0
# Ambiguous terms in source: "fast", "easily" = clarity ~0.7
# Expected: (0.5 × 0.90) + (0.3 × 1.0) + (0.2 × 0.7) ≈ 0.80

assert 0.75 <= result <= 0.85, f"Confidence out of expected range: {result}"
print(f"✅ Confidence score calculated correctly: {result:.2f}")
```

---

### ✅ AC11: Error Handling and Fallbacks

**Given** failure conditions (LLM timeout, rate limit, parsing error)
**When** error occurs during agent invocation
**Then** graceful degradation kicks in:

**Error Scenarios:**

| Scenario | Detection | Fallback | Outcome |
|----------|-----------|----------|---------|
| **LLM Rate Limit** | `RateLimitError` from OpenAI | Retry with exponential backoff (max 3 attempts) | User doesn't perceive interruption if successful |
| **LLM Timeout** | Request exceeds 30s | Use cached response from similar input | Last-known response for this context |
| **Parse Failure** | Invalid JSON from LLM | Use template-based response ("I'd like to know more about...") | Graceful degradation, low confidence |
| **No LLM Response** | Empty content | Offer manual data entry option | User can continue via form |
| **Circuit Breaker Open** | Consecutive failures (5+) | Disable LLM temporarily, show banner | User notified, can continue manually |

**Verification:**
```bash
uv run pytest tests/unit/agents/test_conversational_errors.py -v

# Expected test results:
# ✅ test_rate_limit_retry PASSED
# ✅ test_timeout_fallback PASSED
# ✅ test_parse_failure_template PASSED
# ✅ test_circuit_breaker_activation PASSED
# ✅ test_graceful_degradation PASSED
```

---

### ✅ AC12: Instrumentation and Logging

**Given** agent execution
**When** agent processes a message
**Then** comprehensive logs are emitted:

**Log Output Requirements:**
```
Every agent invocation should emit logs at these key points:

1. Start of execution:
   {
     "level": "INFO",
     "message": "Conversational agent started",
     "correlation_id": "corr-xyz",
     "agent": "conversational",
     "session_id": "sess-123",
     "turn": 1
   }

2. Context extraction:
   {
     "level": "DEBUG",
     "message": "Context extracted",
     "correlation_id": "corr-xyz",
     "topics_identified": ["mobile app", "e-commerce"],
     "actors_mentioned": ["user", "admin"],
     "confidence": 0.88
   }

3. LLM call:
   {
     "level": "DEBUG",
     "message": "LLM call started",
     "correlation_id": "corr-xyz",
     "model": "gpt-4-turbo-preview",
     "tokens_input": 1245
   }

4. Response parsing:
   {
     "level": "DEBUG",
     "message": "Response parsed successfully",
     "correlation_id": "corr-xyz",
     "next_action": "continue_eliciting",
     "parse_quality": 1.0
   }

5. Completion:
   {
     "level": "INFO",
     "message": "Conversational agent completed",
     "correlation_id": "corr-xyz",
     "duration_ms": 2345,
     "confidence": 0.85,
     "tokens_total": 2890
   }
```

**Verification:**
```bash
# Run agent and capture logs
uv run python -m src.main > test_logs.json 2>&1 &
sleep 2
curl -X POST http://localhost:8000/api/v1/sessions/test-sess/messages \
  -H "Content-Type: application/json" \
  -d '{"message":"We need a mobile app"}'

# Parse logs
grep "conversational agent" test_logs.json
# Expected: INFO and DEBUG logs present

grep "correlation_id" test_logs.json | head -1
# Expected: Correlation ID consistent across related logs
```

---

## Technical Implementation Details

### 1. Base Agent Class

**File: `src/agents/base.py`**

```python
import logging
import asyncio
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Optional, Dict, Any
from tenacity import retry, stop_after_attempt, wait_exponential
from openai import RateLimitError, APIError, Timeout
from src.schemas.state import GraphState
from src.utils.logging import get_logger, log_with_context

logger = get_logger(__name__)

class BaseAgent(ABC):
    """Base class for all agent implementations"""

    def __init__(
        self,
        name: str,
        model: str = "gpt-4-turbo-preview",
        temperature: float = 0.2,
        max_tokens: int = 2000,
        openai_api_key: Optional[str] = None
    ):
        """Initialize agent with LLM configuration"""
        from langchain_openai import ChatOpenAI
        import os

        self.name = name
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.logger = get_logger(self.__class__.__name__)

        self.llm = ChatOpenAI(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            openai_api_key=openai_api_key or os.getenv("OPENAI_API_KEY")
        )

        self.start_time: Optional[datetime] = None

    @abstractmethod
    async def execute(self, state: GraphState) -> Dict[str, Any]:
        """Execute agent logic. Override in subclasses."""
        pass

    async def invoke(self, state: GraphState) -> GraphState:
        """Main entry point. Handles error management and instrumentation."""
        self.start_time = datetime.utcnow()
        correlation_id = state.get("correlation_id", "unknown")

        try:
            log_with_context(
                self.logger, "info",
                f"{self.name} agent started",
                agent=self.name,
                session_id=state.session_id,
                turn=state.current_turn,
                correlation_id=correlation_id
            )

            result = await self.execute(state)

            duration_ms = (datetime.utcnow() - self.start_time).total_seconds() * 1000
            log_with_context(
                self.logger, "info",
                f"{self.name} agent completed",
                agent=self.name,
                duration_ms=duration_ms,
                confidence=result.get("confidence", 0.0),
                correlation_id=correlation_id
            )

            # Update state with result
            updated_state = {
                **state,
                "last_agent": self.name,
                "confidence": result.get("confidence", state.get("confidence", 1.0)),
                "iterations": state.get("iterations", 0) + 1
            }

            # Merge agent-specific updates
            if "chat_history_update" in result:
                updated_state["chat_history"].extend(result["chat_history_update"])
            if "requirements_update" in result:
                updated_state["requirements"].extend(result["requirements_update"])

            return GraphState(**updated_state)

        except Exception as e:
            error_count = state.get("error_count", 0) + 1
            log_with_context(
                self.logger, "error",
                f"{self.name} agent failed",
                agent=self.name,
                error=str(e),
                error_type=type(e).__name__,
                correlation_id=correlation_id
            )

            # Escalate if too many errors
            if error_count >= 3:
                raise

            # Return degraded state
            return GraphState(
                **state,
                error_count=error_count,
                last_agent=self.name,
                confidence=max(0.0, state.get("confidence", 1.0) - 0.2)
            )
```

---

### 2. Conversational Agent

**File: `src/agents/conversational/agent.py`**

```python
import os
from datetime import datetime
from typing import List, Dict, Any, Optional
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from src.agents.base import BaseAgent
from src.schemas.state import GraphState, Message
from src.schemas.chat import ConversationalResponse
from src.agents.conversational.prompt_builder import PromptBuilder
from src.agents.conversational.context_manager import ContextManager
from src.agents.conversational.clarification_detector import ClarificationDetector
from src.agents.conversational.response_formatter import ResponseFormatter
from src.agents.conversational.token_budget import TokenBudgetManager
from src.utils.logging import get_logger, log_with_context

logger = get_logger(__name__)

class ConversationalAgent(BaseAgent):
    """Conversational agent for requirements elicitation"""

    def __init__(self, **kwargs):
        super().__init__(name="conversational", **kwargs)
        self.prompt_builder = PromptBuilder()
        self.context_manager = ContextManager()
        self.clarification_detector = ClarificationDetector()
        self.response_formatter = ResponseFormatter()
        self.token_manager = TokenBudgetManager(model=self.model)
        self.parser = PydanticOutputParser(pydantic_object=ConversationalResponse)

    async def execute(self, state: GraphState) -> Dict[str, Any]:
        """Execute conversational agent logic"""

        # Extract context from conversation history
        context = await self.context_manager.extract_context(state)
        log_with_context(
            logger, "debug",
            "Context extracted",
            topics=context.get("identified_topics", []),
            gaps=context.get("clarification_gaps", [])
        )

        # Detect if user input is ambiguous
        user_message = state.chat_history[-1].content if state.chat_history else ""
        ambiguity_result = self.clarification_detector.detect(user_message)
        log_with_context(
            logger, "debug",
            "Ambiguity detection complete",
            is_ambiguous=ambiguity_result["is_ambiguous"],
            score=ambiguity_result["ambiguity_score"]
        )

        # Build optimized prompt
        prompt_text = self.prompt_builder.build(
            project_name=state.project_name,
            current_turn=state.current_turn,
            requirements_count=len(state.requirements),
            context=context,
            ambiguity_result=ambiguity_result,
            chat_history=state.chat_history
        )

        # Manage token budget
        truncated_history = self.token_manager.truncate_chat_history(state.chat_history)

        # Build and invoke LLM chain
        prompt = PromptTemplate(
            template=prompt_text,
            input_variables=[],
            partial_variables={
                "format_instructions": self.parser.get_format_instructions()
            }
        )

        chain = prompt | self.llm | self.parser

        # Invoke with retry logic
        log_with_context(logger, "debug", "LLM invocation starting")

        try:
            response = await chain.ainvoke({})
        except Exception as e:
            log_with_context(
                logger, "error",
                "LLM invocation failed",
                error=str(e),
                error_type=type(e).__name__
            )
            # Fallback to template response
            response = self.response_formatter.generate_fallback_response(state)

        # Parse and validate response
        log_with_context(
            logger, "debug",
            "Response parsed",
            next_action=response.next_action,
            confidence=response.confidence
        )

        # Build assistant message
        assistant_message = Message(
            id=f"msg-{datetime.utcnow().timestamp()}",
            role="assistant",
            content=response.message,
            timestamp=datetime.utcnow(),
            metadata={
                "agent": "conversational",
                "next_action": response.next_action,
                "confidence": response.confidence,
                "clarifying_questions": response.clarifying_questions or [],
                "extracted_topics": response.extracted_topics,
                "sentiment": response.sentiment
            }
        )

        return {
            "confidence": response.confidence,
            "chat_history_update": [assistant_message],
            "next_action": response.next_action,
            "clarifying_questions": response.clarifying_questions
        }
```

---

### 3. Prompt Builder

**File: `src/agents/conversational/prompt_builder.py`**

```python
from typing import Dict, List, Any
from src.agents.conversational.token_budget import TokenBudgetManager

class PromptBuilder:
    """Constructs optimized prompts for conversational agent"""

    def __init__(self):
        self.token_manager = TokenBudgetManager()
        self.system_prompt_template = self._load_system_prompt()

    def _load_system_prompt(self) -> str:
        """Load system prompt from template file"""
        with open("src/templates/prompts/conversational.txt", "r") as f:
            return f.read()

    def build(
        self,
        project_name: str,
        current_turn: int,
        requirements_count: int,
        context: Dict[str, Any],
        ambiguity_result: Dict[str, Any],
        chat_history: List
    ) -> str:
        """Build prompt with appropriate context and examples"""

        # Truncate chat history to fit token budget
        truncated_history = self.token_manager.truncate_chat_history(chat_history)

        # Format chat history
        chat_text = "\n".join([
            f"Turn {i+1} [{msg.role.upper()}]: {msg.content}"
            for i, msg in enumerate(truncated_history[-10:])  # Last 10 messages
        ])

        # Select examples based on requirements_count
        examples = self._select_examples(requirements_count)

        # Build final prompt
        prompt = self.system_prompt_template.format(
            project_name=project_name,
            current_turn=current_turn,
            requirements_count=requirements_count,
            identified_domain=context.get("identified_domain", "unknown"),
            mentioned_actors=", ".join(context.get("mentioned_actors", [])),
            chat_history=chat_text,
            examples=examples,
            format_instructions="{format_instructions}"  # Filled by LangChain
        )

        return prompt

    def _select_examples(self, requirements_count: int) -> str:
        """Select few-shot examples based on conversation stage"""
        if requirements_count == 0:
            return self._load_examples("initial")
        elif requirements_count < 5:
            return self._load_examples("gathering")
        else:
            return self._load_examples("synthesizing")

    def _load_examples(self, stage: str) -> str:
        """Load examples for conversation stage"""
        with open(f"src/templates/examples/conversational_{stage}.txt", "r") as f:
            return f.read()
```

---

### 4. Context Manager

**File: `src/agents/conversational/context_manager.py`**

```python
from typing import Dict, List, Any, Set
from src.schemas.state import GraphState
from src.utils.logging import get_logger, log_with_context

logger = get_logger(__name__)

class ContextManager:
    """Extracts actionable context from conversation history"""

    # Lists of common terms to identify
    ACTOR_KEYWORDS = {"user", "admin", "customer", "stakeholder", "system", "manager"}
    DOMAIN_INDICATORS = {
        "ecommerce": ["product", "checkout", "cart", "payment", "shipping"],
        "saas": ["subscription", "billing", "license", "api", "integration"],
        "mobile": ["ios", "android", "mobile", "app", "smartphone"],
        "social": ["post", "follow", "comment", "share", "like", "profile"]
    }

    async def extract_context(self, state: GraphState) -> Dict[str, Any]:
        """Extract comprehensive context from chat history"""

        if not state.chat_history:
            return self._empty_context()

        # Combine all user messages
        user_messages = [
            msg.content for msg in state.chat_history
            if msg.role == "user"
        ]
        full_text = " ".join(user_messages).lower()

        return {
            "conversation_summary": self._summarize(user_messages),
            "identified_domain": self._identify_domain(full_text),
            "mentioned_actors": self._extract_actors(full_text),
            "mentioned_features": self._extract_features(user_messages),
            "implicit_needs": self._infer_implicit_needs(full_text),
            "clarification_gaps": self._identify_gaps(user_messages),
            "turn_sentiment": self._assess_sentiment(user_messages[-1]),
            "conversation_momentum": self._measure_momentum(state)
        }

    def _summarize(self, messages: List[str], max_length: int = 200) -> str:
        """Create brief summary of conversation"""
        if not messages:
            return ""
        # Use first and last user message for summary
        first = messages[0][:100]
        last = messages[-1][:100]
        return f"{first}... [continuing to] ...{last}"

    def _identify_domain(self, text: str) -> str:
        """Identify problem domain from conversation"""
        for domain, keywords in self.DOMAIN_INDICATORS.items():
            if any(kw in text for kw in keywords):
                return domain
        return "generic"

    def _extract_actors(self, text: str) -> List[str]:
        """Extract mentioned actors/roles"""
        actors = []
        for actor in self.ACTOR_KEYWORDS:
            if actor in text:
                actors.append(actor)
        return actors

    def _extract_features(self, messages: List[str]) -> List[str]:
        """Extract mentioned features"""
        # Simple noun extraction - could enhance with NLP
        features = []
        feature_keywords = ["login", "payment", "notification", "search", "upload", "export"]
        for msg in messages:
            for feature in feature_keywords:
                if feature in msg.lower():
                    features.append(feature)
        return list(set(features))

    def _infer_implicit_needs(self, text: str) -> List[str]:
        """Infer implicit requirements"""
        implicit = []
        if "mobile" in text or "app" in text:
            implicit.extend(["offline support", "battery optimization", "responsive design"])
        if "payment" in text:
            implicit.extend(["encryption", "audit logging", "fraud detection"])
        if "user" in text and "login" in text:
            implicit.extend(["password recovery", "session management", "account lockout"])
        return implicit

    def _identify_gaps(self, messages: List[str]) -> List[str]:
        """Identify areas needing clarification"""
        gaps = []
        if not any("target" in msg.lower() or "goal" in msg.lower() for msg in messages):
            gaps.append("Success metrics not defined")
        if not any("user" in msg.lower() or "actor" in msg.lower() for msg in messages):
            gaps.append("User types not specified")
        if not any("deadline" in msg.lower() or "timeline" in msg.lower() for msg in messages):
            gaps.append("Timeline not discussed")
        return gaps

    def _assess_sentiment(self, message: str) -> str:
        """Assess user sentiment from last message"""
        # Simple keyword-based sentiment
        positive_words = ["great", "love", "excellent", "perfect", "good"]
        negative_words = ["bad", "hate", "terrible", "poor", "frustrated"]

        text = message.lower()
        if any(word in text for word in positive_words):
            return "positive"
        elif any(word in text for word in negative_words):
            return "negative"
        return "neutral"

    def _measure_momentum(self, state: GraphState) -> float:
        """Measure conversation engagement (0.0-1.0)"""
        if len(state.chat_history) < 2:
            return 0.5

        # Engagement indicators
        avg_message_length = sum(
            len(msg.content) for msg in state.chat_history if msg.role == "user"
        ) / len([m for m in state.chat_history if m.role == "user"])

        # Normalize to 0-1
        momentum = min(1.0, avg_message_length / 200)
        return momentum

    def _empty_context(self) -> Dict[str, Any]:
        """Return empty context dict"""
        return {
            "conversation_summary": "",
            "identified_domain": "unknown",
            "mentioned_actors": [],
            "mentioned_features": [],
            "implicit_needs": [],
            "clarification_gaps": ["Project requirements not yet gathered"],
            "turn_sentiment": "neutral",
            "conversation_momentum": 0.5
        }
```

---

## Testing Strategy

### Unit Tests

```python
# tests/unit/agents/test_conversational_agent.py
import pytest
from src.agents.conversational.agent import ConversationalAgent
from src.schemas.state import GraphState

@pytest.mark.asyncio
async def test_agent_initialization():
    agent = ConversationalAgent()
    assert agent.name == "conversational"
    assert agent.model == "gpt-4-turbo-preview"

@pytest.mark.asyncio
async def test_single_turn_conversation():
    agent = ConversationalAgent()
    state = create_test_state()
    result = await agent.invoke(state)
    assert len(result.chat_history) > len(state.chat_history)

@pytest.mark.asyncio
async def test_confidence_scoring():
    agent = ConversationalAgent()
    state = create_test_state()
    result = await agent.invoke(state)
    assert 0.0 <= result.confidence <= 1.0

@pytest.mark.asyncio
async def test_next_action_determination():
    agent = ConversationalAgent()
    state = create_test_state(turn=1)
    result = await agent.invoke(state)
    assert result.chat_history[-1].metadata["next_action"] in [
        "continue_eliciting",
        "extract_requirements",
        "clarify",
        "wait_for_input"
    ]
```

### Integration Tests

```python
# tests/integration/agents/test_conversational_flow.py
@pytest.mark.asyncio
async def test_three_turn_conversation():
    """Test complete 3-turn conversation"""
    agent = ConversationalAgent()

    # Turn 1
    state = create_initial_state()
    state = await agent.invoke(state)
    assert state.current_turn == 1

    # Turn 2
    state.chat_history.append(create_user_message("Both iOS and Android"))
    state.current_turn = 2
    state = await agent.invoke(state)
    assert state.current_turn == 2

    # Turn 3
    state.chat_history.append(create_user_message("2.5 seconds on 4G"))
    state.current_turn = 3
    state = await agent.invoke(state)
    assert state.chat_history[-1].metadata["next_action"] == "extract_requirements"
```

---

## Definition of Done

- [ ] All 12 acceptance criteria passed and verified
- [ ] Base agent class implemented with proper abstraction
- [ ] ConversationalAgent fully functional with LLM integration
- [ ] Dynamic prompt template system working with context adaptation
- [ ] Context manager extracting meaningful patterns
- [ ] Clarification detector identifying ambiguous statements
- [ ] Response formatter producing valid, structured output
- [ ] Token budget management enforcing limits
- [ ] Confidence scoring formula implemented and validated
- [ ] Error handling and fallbacks operational
- [ ] Comprehensive logging and instrumentation in place
- [ ] Unit tests: 100% pass rate (12+ tests)
- [ ] Integration tests: 100% pass rate (5+ tests)
- [ ] Code follows project conventions (Black, Ruff, MyPy)
- [ ] All docstrings present and clear
- [ ] Type hints complete throughout
- [ ] No console warnings or errors on startup

---

## Dependencies for Next Stories

Once Story 2 is complete, the following stories can begin in parallel:

- **STORY-003:** Extraction Agent Implementation
- **STORY-004:** Inference Agent Implementation
- **STORY-005:** LangGraph Orchestrator Integration

---

## Notes for Windsurf AI Implementation

### Key Implementation Priorities

1. **Start with base class** - Defines contract for all agents
2. **Implement conversational agent core logic** - Most complex agent
3. **Build context manager** - Foundation for intelligent prompting
4. **Add clarification detection** - Critical for dialog quality
5. **Implement response formatter** - Ensures structured output
6. **Token budget management** - Prevents API cost overruns
7. **Comprehensive error handling** - Production-ready reliability

### Critical Files to Create

- `src/agents/base.py` - Base agent class
- `src/agents/conversational/agent.py` - Main conversational agent
- `src/agents/conversational/prompt_builder.py` - Dynamic prompts
- `src/agents/conversational/context_manager.py` - Context extraction
- `src/agents/conversational/clarification_detector.py` - Ambiguity detection
- `src/agents/conversational/response_formatter.py` - Output validation
- `src/templates/prompts/conversational.txt` - System prompt template

### Testing Execution

```bash
# Run all agent tests
uv run pytest tests/unit/agents/ -v --tb=short

# Run integration tests
uv run pytest tests/integration/agents/ -v --tb=short

# Test specific agent
uv run pytest tests/unit/agents/test_conversational_agent.py::test_single_turn_conversation -v

# With coverage
uv run pytest tests/unit/agents/ --cov=src/agents --cov-report=html
```

### Common Pitfalls to Avoid

- ❌ Don't hardcode API keys - use config/environment
- ❌ Don't skip error handling - LLM calls are unreliable
- ❌ Don't ignore token budgets - costs grow quickly
- ❌ Don't over-engineer prompts initially - iterate based on results
- ❌ Don't skip logging - debugging without it is impossible
- ❌ Don't forget type hints - critical for maintainability

---

## References

- **Design Packet 1:** Agent-by-Agent Low Level Design (Section 8.1)
- **Design Packet 2:** Agent Implementation Blueprints (Section 6)
- **Story 1:** Project Foundation (prerequisite)
- **LangChain Documentation:** https://python.langchain.com/
- **OpenAI API Guide:** https://platform.openai.com/docs/

---

**End of Story 2 Document**

**Next Steps:**
1. Complete Story 1 acceptance criteria verification
2. Begin Story 2 implementation with base agent class
3. Iterate on prompt templates based on test results
4. Integrate with Story 1 infrastructure
5. Prepare for Story 3 (Extraction Agent) which depends on conversational agent
