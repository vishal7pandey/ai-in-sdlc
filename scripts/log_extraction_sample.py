"""Emit sample extraction agent logs for documentation."""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime

from src.agents.extraction import ExtractionAgent
from src.schemas import GraphState, Message

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)


RESPONSE = """
{
  "requirements": [
    {
      "id": "REQ-010",
      "title": "Users authenticate with email and password",
      "type": "functional",
      "actor": "user",
      "action": "authenticate using email and password credentials",
      "condition": "when accessing protected resources",
      "acceptance_criteria": [
        "User can enter valid email and password credentials",
        "System validates credentials before granting access"
      ],
      "priority": "must",
      "confidence": 0.9,
      "inferred": false,
      "rationale": "Stated explicitly by the user in the session",
      "source_refs": ["chat:turn:0"]
    }
  ],
  "confidence": 0.88,
  "ambiguous_items": ["Clarify maximum allowed login attempts"],
  "metadata": {
    "tokens_used": 512,
    "duration_ms": 120.5,
    "model": "gpt-4"
  }
}
"""


class StubLLM:
    async def ainvoke(self, *_args, **_kwargs):  # pragma: no cover - helper script only
        class Result:
            def __init__(self, content: str) -> None:
                self.content = content

        return Result(RESPONSE)


def _state() -> GraphState:
    return GraphState(
        session_id="demo-extraction",
        project_name="Story-003",
        user_id="user-demo",
        chat_history=[
            Message(
                id="m1",
                role="user",
                content="Users must be able to log in with email and password.",
                timestamp=datetime.utcnow(),
            ),
        ],
        current_turn=1,
    )


async def main() -> None:
    agent = ExtractionAgent(openai_api_key="test")
    agent.llm = StubLLM()

    state = await agent.invoke(_state())

    print("\nExtracted requirements:", len(state.requirements))
    if state.requirements:
        req = state.requirements[-1]
        print("Last requirement:", req.id, "-", req.title)
        print("Source refs:", req.source_refs)
    print("Overall confidence:", state.confidence)
    print("Ambiguous items:", state.ambiguous_items)
    print("Extraction metadata:", state.extraction_metadata)


if __name__ == "__main__":
    asyncio.run(main())
