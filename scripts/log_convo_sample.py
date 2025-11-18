"""Emit sample conversational agent logs for documentation."""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime

from src.agents.conversational.agent import ConversationalAgent
from src.schemas import GraphState, Message

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)


class StubLLM:
    async def ainvoke(self, *_args, **_kwargs):  # pragma: no cover - helper script only
        class Result:
            def __init__(self, content: str) -> None:
                self.content = content

        content = (
            "response: Thanks for the context! I'll keep eliciting more detail.\n"
            "nextAction: continue_eliciting\n"
            "confidence: 0.85\n"
            "extractedTopics:\n  - onboarding\n"
            "sentiment: positive\n"
        )
        return Result(content)


def _state() -> GraphState:
    return GraphState(
        session_id="demo",
        project_name="Story-002",
        user_id="user-demo",
        chat_history=[
            Message(
                id="m1",
                role="user",
                content="Need smoother onboarding",
                timestamp=datetime.utcnow(),
            ),
        ],
        current_turn=1,
    )


async def main() -> None:
    agent = ConversationalAgent(openai_api_key="test")
    agent.llm = StubLLM()
    state = await agent.invoke(_state())
    print("\nFinal confidence:", state.confidence)
    print("Last next action:", state.last_next_action)


if __name__ == "__main__":
    asyncio.run(main())
