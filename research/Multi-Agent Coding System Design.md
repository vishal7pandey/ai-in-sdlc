<img src="https://r2cdn.perplexity.ai/pplx-full-logo-primary-dark%402x.png" style="height:64px;margin-right:32px"/>

# Multi-Agent Coding System Design

Based on your requirements, here's a comprehensive design for a **LangGraph-based multi-agent coding system** with human-in-the-loop capabilities, modular architecture, and standardized templates.

## System Overview

An intelligent multi-agent system that transforms user ideas into production-ready code through a sequential workflow of specialized agents. Each agent generates specific artifacts (BRD, PRD, FRD, HLD, LLD, code, tests) with human review checkpoints at every stage.[^1][^2][^3]

## Agent Design: 8 Specialized Agents

### Agent 1: Requirements Analyzer

**Role:** Transform raw user input into structured business and functional requirements[^1][^2]

**Expected Input:**

- User's natural language description of project idea
- Rough documents, notes, or bullet points
- Optional reference documents or examples

**Expected Output:**

- Business Requirements Document (BRD)
- Product Requirements Document (PRD)
- Initial user stories and use cases

**High-Level Prompt:**

```
You are a Requirements Analysis Agent specialized in extracting and structuring
software requirements.

ROLE: Transform raw user ideas into clear, structured business and product
requirements.

INPUT: You will receive natural language descriptions, rough notes, or informal
documents about a project idea.

TASK:
1. Analyze input to identify:
   - Business objectives and goals
   - Target users and personas
   - Core features and functionality
   - Success criteria
   - Constraints and assumptions

2. Structure findings into:
   - Business Requirements Document (BRD) with business context, objectives,
     stakeholders
   - Product Requirements Document (PRD) with features, user flows, acceptance
     criteria

OUTPUT FORMAT: Use provided BRD and PRD templates. Fill all sections. Mark
unclear items as [TBD] for human review.

INTERACTION: When critical information is missing, pause and ask the user
specific questions before proceeding.
```

**State Schema:**

```python
{
    "user_input": str,              # Raw input from user
    "clarification_questions": list, # Questions for user
    "brd_draft": dict,              # Structured BRD content
    "prd_draft": dict,              # Structured PRD content
    "status": enum,                 # draft/under_review/approved
    "human_feedback": str           # User's review comments
}
```


### Agent 2: Requirements Decomposer

**Role:** Break down requirements into detailed functional and non-functional specifications[^2][^3]

**Expected Input:**

- Approved BRD from Agent 1
- Approved PRD from Agent 1

**Expected Output:**

- Functional Requirements Document (FRD)
- Non-Functional Requirements Document (NFR)
- Detailed use cases with flows

**High-Level Prompt:**

```
You are a Requirements Decomposition Agent that creates detailed technical
specifications.

ROLE: Transform high-level requirements into granular functional and
non-functional specifications.

INPUT: Approved Business Requirements Document (BRD) and Product Requirements
Document (PRD).

TASK:
1. Extract and detail functional requirements:
   - Specific system behaviors
   - Input/output specifications
   - Business rules and logic
   - Data requirements
   - Integration points

2. Define non-functional requirements:
   - Performance (response time, throughput)
   - Security (authentication, authorization, encryption)
   - Scalability and availability
   - Usability and accessibility
   - Compliance and regulatory needs

3. Create detailed use cases with actors, preconditions, main flow, alternative
   flows, postconditions, and exceptions

OUTPUT FORMAT: Use FRD and NFR templates. Each requirement must have unique ID,
description, priority, and acceptance criteria.

INTERACTION: Present requirements grouped by priority (P0-Critical, P1-High,
P2-Medium, P3-Low). Ask user to validate prioritization.
```


### Agent 3: Architecture Designer

**Role:** Design system architecture and create technical design documents[^1][^2][^3]

**Expected Input:**

- Approved FRD from Agent 2
- Approved NFR from Agent 2

**Expected Output:**

- High-Level Design (HLD)
- Low-Level Design (LLD)
- Architecture Decision Records (ADR)
- System architecture diagrams

**High-Level Prompt:**

```
You are a Software Architecture Agent that designs system architecture.

ROLE: Create comprehensive technical design based on functional and non-functional
requirements.

INPUT: Functional Requirements Document (FRD), Non-Functional Requirements Document
(NFR), optional technology stack preferences.

TASK:
1. Create High-Level Design (HLD):
   - System architecture (monolith/microservices/serverless)
   - Component diagram with interactions
   - Technology stack recommendations
   - Data flow diagrams
   - Integration patterns

2. Create Low-Level Design (LLD):
   - Module specifications
   - Class diagrams and relationships
   - Database schema (tables, relationships, indexes)
   - API specifications (endpoints, methods, payloads)
   - Sequence diagrams for critical flows

3. Document Architecture Decision Records (ADR):
   - Key technical decisions made
   - Context and reasoning
   - Alternatives considered
   - Consequences and trade-offs

OUTPUT FORMAT: Use HLD, LLD, and ADR templates. Include both textual descriptions
and mermaid diagrams.

INTERACTION: Present multiple architecture options (e.g., Option A: Microservices,
Option B: Modular Monolith). Ask user to select preferred approach.
```


### Agents 4-8: Additional Specialized Agents

The system includes five additional agents for complete SDLC coverage:

- **Agent 4: Project Structure Planner** - Repository structure and boilerplate
- **Agent 5: Code Planner** - Task breakdown and implementation roadmap
- **Agent 6: Code Generator** - Actual code implementation
- **Agent 7: Test Planner** - Test strategy and test case specifications
- **Agent 8: Test Code Generator** - Automated test code generation


## LangGraph Architecture Implementation

![Multi-agent coding system architecture with LangGraph, PostgreSQL, Redis, and React frontend](https://user-gen-media-assets.s3.amazonaws.com/seedream_images/8af6a93f-846f-4d53-8c06-76cd8ee381c8.png)

Multi-agent coding system architecture with LangGraph, PostgreSQL, Redis, and React frontend

### State Management with PostgreSQL + Redis

**State Schema (TypedDict):**

```python
from typing import TypedDict, Annotated
from langgraph.graph import add_messages

class AgentState(TypedDict):
    # Shared across all agents
    messages: Annotated[list, add_messages]
    current_agent: str
    current_phase: str

    # User input
    user_input: str
    user_preferences: dict

    # Document artifacts (incrementally built)
    brd: dict
    prd: dict
    frd: dict
    nfr: dict
    hld: dict
    lld: dict
    adr_records: list
    project_structure: dict
    implementation_plan: dict
    tasks: list
    code_files: list
    test_plan: dict
    test_files: list

    # Human-in-the-loop
    requires_human_review: bool
    human_feedback: str
    approval_status: str  # 'pending', 'approved', 'revision_needed'

    # Metadata
    iteration_count: int
    session_id: str
```

This schema stores all artifacts as structured objects, enabling incremental updates to specific sections without regenerating entire documents.[^4][^5][^6]

### Human-in-the-Loop Implementation

LangGraph provides built-in support for human-in-the-loop workflows through the `interrupt_before` mechanism and `Command` primitive:[^7][^8][^9][^10]

```python
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.graph import StateGraph, Command

def create_graph_with_hitl():
    # Initialize checkpointer for persistence
    checkpointer = PostgresSaver.from_conn_string(
        "postgresql://user:pass@localhost/agentdb"
    )

    # Build graph
    workflow = StateGraph(AgentState)

    # Add agent nodes
    workflow.add_node("requirements_analyzer", requirements_analyzer_node)
    workflow.add_node("human_review", human_review_node)

    # After each agent, route to human review
    workflow.add_conditional_edges(
        "requirements_analyzer",
        lambda state: "human_review" if state["requires_human_review"]
                     else "requirements_decomposer"
    )

    # After human review, decide next step based on approval
    workflow.add_conditional_edges(
        "human_review",
        route_after_human_review,
        {
            "approved": "requirements_decomposer",  # Continue to next agent
            "revision": "requirements_analyzer",     # Loop back for revision
            "abort": END
        }
    )

    # Compile with checkpointer and interrupt points
    app = workflow.compile(
        checkpointer=checkpointer,
        interrupt_before=["human_review"]  # Pause for human input
    )

    return app
```

When execution reaches a `human_review` node, the workflow pauses and waits for user feedback via the API. The user can approve, request revisions, or provide specific edits:[^8][^9][^10][^11]

```python
# Resume with approval decision
agent.invoke(
    Command(
        resume={"decisions": [{"type": "approve"}]}  # or "edit", "reject"
    ),
    config=config  # Same thread ID to resume paused conversation
)
```


### Modular Agent Design Pattern

Each agent follows a consistent base class pattern for plug-and-play modularity:[^1][^2]

```python
class BaseAgent:
    def __init__(self, llm, prompt_template, output_parser):
        self.llm = llm
        self.prompt_template = prompt_template
        self.output_parser = output_parser
        self.chain = prompt_template | llm | output_parser

    def process(self, state: AgentState) -> dict:
        '''Process state and return updated artifacts'''
        raise NotImplementedError

    def validate_output(self, output: dict) -> bool:
        '''Validate output meets requirements'''
        raise NotImplementedError

class RequirementsAnalyzerAgent(BaseAgent):
    def __init__(self):
        super().__init__(
            llm=ChatOpenAI(model="gpt-4-turbo"),
            prompt_template=load_template("requirements_analyzer_prompt.txt"),
            output_parser=JsonOutputParser(pydantic_object=BRDOutput)
        )

    def process(self, state: AgentState) -> dict:
        result = self.chain.invoke({
            "user_input": state["user_input"],
            "template": load_template("brd_template.md")
        })
        return {"brd": result["brd"], "prd": result["prd"]}
```


### React Frontend Integration

The React frontend uses the `useStream` hook from `@langchain/langgraph-sdk/react` for real-time state streaming:[^12]

```jsx
import { useStream } from "@langchain/langgraph-sdk/react";

function AgentWorkflow() {
    const thread = useStream({
        apiUrl: "http://localhost:8000",
        assistantId: "coding-agent",
        messagesKey: "messages",
    });

    const provideFeedback = async (approved, feedback) => {
        await thread.invoke(
            Command({
                resume: {
                    decisions: [{
                        type: approved ? "approve" : "edit",
                        feedback: feedback
                    }]
                }
            }),
            { configurable: { thread_id: thread.threadId } }
        );
    };

    return (
        <div>
            <h2>Current Agent: {thread.state?.current_agent}</h2>
            {thread.state?.requires_human_review && (
                <DocumentViewer document={thread.state.brd} />
                <textarea placeholder="Provide feedback..." />
                <button onClick={() => provideFeedback(true, feedback)}>
                    Approve
                </button>
            )}
        </div>
    );
}
```

This provides real-time updates as agents process information and pauses for human review when needed.[^12]

### Incremental State Updates

The system supports fine-grained updates to specific artifact sections without full regeneration:[^4][^5][^13]

```python
class StateManager:
    def update_artifact(self, thread_id: str, artifact_name: str,
                       updates: dict, merge: bool = True):
        '''Incrementally update a specific artifact'''
        state = self.load_state(thread_id)

        if merge and artifact_name in state:
            # Merge updates into existing artifact
            state[artifact_name].update(updates)
        else:
            # Replace artifact
            state[artifact_name] = updates

        self.save_state(thread_id, state)
```

This allows users to iteratively refine specific sections (e.g., "update only the security section of the HLD") without regenerating the entire document.[^5][^4]

### Template Management with Jinja2

All artifacts use standardized Markdown templates with Jinja2 for dynamic content:[^14][^15]

```python
class TemplateManager:
    def fill_template(self, name: str, data: dict) -> str:
        '''Fill template with data using Jinja2'''
        from jinja2 import Template
        template = Template(self.get_template(name))
        return template.render(**data)
```

Example BRD template structure:

```markdown
# Business Requirements Document (BRD)

**Project Name:** {{ project_name }}
**Date:** {{ date }}

## 1. Business Objectives
{{ business_objectives }}

## 2. Stakeholders
{% for stakeholder in stakeholders %}
- **{{ stakeholder.role }}:** {{ stakeholder.name }}
{% endfor %}

## 3. Success Criteria
{% for criterion in success_criteria %}
- {{ criterion }}
{% endfor %}
```


## Technology Stack

| Component | Technology | Purpose |
| :-- | :-- | :-- |
| State Schema | TypedDict + Annotated | Define shared data structure across agents[^16][^13] |
| Agent Node | Python function returning AgentState | Process state and generate artifacts[^1][^2] |
| HITL Node | interrupt_before + Command | Pause for human review and approval[^8][^9] |
| Checkpointer | PostgresSaver | Persist state for resume/recovery[^6][^13] |
| Template Manager | Jinja2 + Markdown | Standardize document formats[^14] |
| State Manager | PostgreSQL + Redis | Efficient state storage and retrieval[^4][^5] |
| React Hook | useStream from @langchain/langgraph-sdk/react | Real-time state streaming to UI[^12] |
| FastAPI Endpoint | FastAPI + LangServe | HTTP interface for agent invocation[^17] |

## POC Implementation Roadmap (6 Weeks)

**Week 1-2: Foundation \& Core Agents**

- Setup LangGraph with 3 agents: Requirements Analyzer, Architecture Designer, Code Generator
- Implement basic state schema and PostgreSQL persistence
- Create 5 essential templates (BRD, PRD, HLD, LLD, Code)
- Test basic end-to-end flow

**Week 3-4: Human-in-the-Loop \& React UI**

- Add interrupt points and approval workflow
- Build React frontend with document viewer
- Implement incremental state updates
- Enable resume from any checkpoint

**Week 5-6: Expansion \& Polish**

- Add 2 more agents (Requirements Decomposer, Test Planner)
- Enhance templates with more detail
- Add LangSmith monitoring for tracing and debugging
- Improve UI with workflow visualization
- Add error handling and retry logic


## Key Design Decisions

1. **Sequential Workflow:** Agents run sequentially with human-in-the-loop checkpoints after each agent[^1][^2][^3]
2. **State Persistence:** Hybrid approach using PostgreSQL for durability and Redis for session caching[^4][^5][^18]
3. **Modularity:** Base agent class pattern enables plug-and-play agent development[^2][^1]
4. **Templates:** Markdown with Jinja2 for easy customization and version control[^14][^15]
5. **Scope Control:** Incremental artifact updates using merge operations, not full regeneration[^5][^4]
6. **Error Handling:** Retry logic with human escalation for failures[^2]

This design provides a solid foundation for your POC that can incrementally scale to handle more complex workflows and additional agents as needed.
<span style="display:none">[^19][^20][^21][^22][^23][^24][^25][^26][^27][^28][^29][^30][^31][^32][^33][^34][^35][^36][^37][^38]</span>

<div align="center">⁂</div>

[^1]: https://aws.amazon.com/blogs/machine-learning/build-a-multi-agent-system-with-langgraph-and-mistral-on-aws/

[^2]: https://aws.amazon.com/blogs/machine-learning/build-multi-agent-systems-with-langgraph-and-amazon-bedrock/

[^3]: https://blog.langchain.com/langgraph-multi-agent-workflows/

[^4]: https://dev.to/jamesli/langgraph-state-machines-managing-complex-agent-task-flows-in-production-36f4

[^5]: https://aankitroy.com/blog/langgraph-state-management-memory-guide

[^6]: https://docs.langchain.com/oss/python/langgraph/persistence

[^7]: https://github.com/langchain-ai/langchain/issues/22649

[^8]: https://docs.langchain.com/oss/python/langchain/human-in-the-loop

[^9]: https://docs.langchain.com/oss/python/langchain/middleware

[^10]: https://docs.langchain.com/oss/python/deepagents/human-in-the-loop

[^11]: https://docs.langchain.com/langsmith/add-human-in-the-loop

[^12]: https://docs.langchain.com/langsmith/use-stream-react

[^13]: https://sparkco.ai/blog/mastering-langgraph-state-management-in-2025

[^14]: https://arxiv.org/html/2402.14871v1

[^15]: https://www.linkedin.com/pulse/building-multi-agent-systems-complete-blueprint-templates-zhang-cnzwe

[^16]: https://docs.langchain.com/oss/python/langgraph/graph-api

[^17]: https://datasciencedojo.com/blog/react-agent-with-langchain-toolkit/

[^18]: https://redis.io/blog/langgraph-redis-build-smarter-ai-agents-with-memory-persistence/

[^19]: https://www.youtube.com/watch?v=E0fQWFNqGgg

[^20]: https://kinde.com/learn/ai-for-software-engineering/ai-devops/orchestrating-multi-step-agents-temporal-dagster-langgraph-patterns-for-long-running-work/

[^21]: https://codelabs.developers.google.com/aidemy-multi-agent/instructions

[^22]: https://www.auxiliobits.com/blog/orchestrating-long-running-processes-using-langgraph-agents/

[^23]: https://docs.langchain.com/oss/python/langgraph/workflows-agents

[^24]: https://auth0.com/blog/async-ciba-python-langgraph-auth0/

[^25]: https://latenode.com/blog/langgraph-multi-agent-orchestration-complete-framework-guide-architecture-analysis-2025

[^26]: https://www.langchain.com/langgraph

[^27]: https://launchdarkly.com/docs/tutorials/agents-langgraph

[^28]: https://www.langchain.com/langchain

[^29]: https://docs.langchain.com/oss/python/langchain/multi-agent

[^30]: https://www.emse.fr/~boissier/enseignement/maop14/DOC/cartago/env-prog-cartago-tech-rep.pdf

[^31]: https://docs.langchain.com/oss/python/langchain/agents

[^32]: https://techcommunity.microsoft.com/blog/microsoft365copilotblog/office-agent-–-“taste-driven”-multi-agent-system-for-microsoft-365-copilot/4457397

[^33]: https://www.kaggle.com/code/ksmooi/langchain-tool-integrated-with-react-agent

[^34]: https://docs.langchain.com/oss/python/langgraph/memory

[^35]: https://www.productcompass.pm/p/multi-agent-research-system

[^36]: https://github.com/langchain-ai/react-agent-js

[^37]: https://airbyte.com/data-engineering-resources/using-langchain-react-agents

[^38]: https://github.com/langchain-ai/langgraph/discussions/350
