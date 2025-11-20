from __future__ import annotations

from datetime import datetime

from src.agents.synthesis.agent import SynthesisAgent
from src.schemas import GraphState, Requirement, RequirementType
from src.schemas.requirement import Priority


def _base_state() -> GraphState:
    return GraphState(
        session_id="sess-syn-1",
        project_name="Synthesis Demo",
        user_id="user-1",
    )


def _make_requirement(req_id: str, title: str, type_: RequirementType) -> Requirement:
    return Requirement(
        id=req_id,
        title=title,
        type=type_,
        actor="user",
        action="log in",
        acceptance_criteria=["User can log in with email and password"],
        priority=Priority.MEDIUM,
        confidence=0.9,
        inferred=False,
        rationale="This is a valid test rationale for synthesis agent.",
        source_refs=["chat:turn:0"],
        created_at=datetime.utcnow(),
    )


async def test_synthesis_agent_generates_markdown_document() -> None:
    state = _base_state().with_updates(
        requirements=[
            _make_requirement("REQ-001", "User can log in", RequirementType.FUNCTIONAL),
            _make_requirement(
                "REQ-002",
                "System meets performance requirements",
                RequirementType.NON_FUNCTIONAL,
            ),
        ]
    )

    agent = SynthesisAgent()
    result = await agent.execute(state)

    assert "rd_draft" in result["state_updates"]
    markdown = result["state_updates"]["rd_draft"]

    assert "# Requirements Document for Synthesis Demo" in markdown
    assert "## Functional Requirements" in markdown
    assert "REQ-001" in markdown
    assert "User can log in" in markdown
    assert (
        "## Non Functional Requirements" in markdown or "## Non-Functional Requirements" in markdown
    )
    assert "REQ-002" in markdown
    assert "Acceptance Criteria" in markdown
    assert "Source" in markdown
