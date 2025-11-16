"""Conversational agent orchestration."""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime
from typing import Any

try:
    from langchain.output_parsers import PydanticOutputParser
except ModuleNotFoundError:  # pragma: no cover - compatibility shim
    from langchain_core.output_parsers import PydanticOutputParser  # type: ignore

from src.agents.base import AgentResult, BaseAgent
from src.schemas import GraphState, Message
from src.utils.logging import get_logger, log_with_context

from .clarification_detector import ClarificationDetector
from .confidence_scorer import ConfidenceScorer
from .context_manager import ContextManager
from .prompt_builder import PromptBuilder
from .response_formatter import ResponseFormatter
from .schemas import ConversationalResponse
from .token_budget import TokenBudgetManager

logger = get_logger(__name__)


class ConversationalAgent(BaseAgent):
    """Conversational agent for requirements elicitation."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(name="conversational", **kwargs)
        self.prompt_builder = PromptBuilder()
        self.context_manager = ContextManager()
        self.clarification_detector = ClarificationDetector()
        self.response_formatter = ResponseFormatter()
        self.token_manager = TokenBudgetManager(model=self.model)
        self.parser = PydanticOutputParser(pydantic_object=ConversationalResponse)
        self.confidence_scorer = ConfidenceScorer()

    async def execute(self, state: GraphState) -> AgentResult:
        context = await self.context_manager.extract_context(state)
        correlation_id = state.correlation_id or "unknown"
        log_with_context(
            logger,
            "debug",
            "Context extracted",
            correlation_id=correlation_id,
            topics=context.get("mentioned_features", []),
            actors=context.get("mentioned_actors", []),
            sentiment=context.get("turn_sentiment"),
        )

        user_text = state.chat_history[-1].content if state.chat_history else ""
        ambiguity_result = self.clarification_detector.detect(user_text)
        ambiguity_payload = asdict(ambiguity_result)
        log_with_context(
            logger,
            "debug",
            "Ambiguity detection",
            correlation_id=correlation_id,
            is_ambiguous=ambiguity_result.is_ambiguous,
            score=ambiguity_result.ambiguity_score,
        )

        prompt_text = self.prompt_builder.build(
            project_name=state.project_name,
            current_turn=state.current_turn,
            requirements_count=len(state.requirements),
            context=context,
            ambiguity_result=ambiguity_payload,
            chat_history=state.chat_history,
        )

        log_with_context(
            logger,
            "debug",
            "LLM call start",
            correlation_id=correlation_id,
            model=self.model,
        )

        try:
            llm_response = await self.llm.ainvoke(
                prompt_text,
                max_tokens=self.token_manager.max_response_tokens,
                stop=None,
            )
        except Exception as exc:  # pragma: no cover
            log_with_context(
                logger,
                "warning",
                "Conversational agent fallback triggered",
                correlation_id=correlation_id,
                error=str(exc),
            )
            return self._fallback_result(state, context, ambiguity_payload)

        try:
            response_payload = self.response_formatter.parse_and_validate(llm_response.content)
        except Exception as exc:  # pragma: no cover
            log_with_context(
                logger,
                "warning",
                "Conversational agent parse fallback",
                correlation_id=correlation_id,
                error=str(exc),
            )
            return self._fallback_result(state, context, ambiguity_payload)

        log_with_context(
            logger,
            "debug",
            "LLM response parsed",
            correlation_id=correlation_id,
            next_action=response_payload.next_action,
            topics=response_payload.extracted_topics,
        )

        assistant_message = Message(
            id=f"msg-{state.current_turn + 1}",
            role="assistant",
            content=response_payload.message,
            timestamp=datetime.utcnow(),
        )

        clarifications = response_payload.clarifying_questions or []
        pending_clarifications = clarifications if response_payload.next_action == "clarify" else []
        merged_topics = sorted({*state.extracted_topics, *response_payload.extracted_topics})

        parsed_fields = {
            "message": bool(response_payload.message),
            "next_action": bool(response_payload.next_action),
            "confidence": True,
        }
        final_confidence = self.confidence_scorer.calculate(
            llm_confidence=response_payload.confidence,
            parsed_fields=parsed_fields,
            source_text=user_text,
        )

        agent_result: AgentResult = {
            "chat_history_update": [assistant_message],
            "state_updates": {
                "confidence": final_confidence,
                "last_agent": self.name,
                "conversation_context": context,
                "ambiguity_assessment": ambiguity_payload,
                "pending_clarifications": pending_clarifications,
                "extracted_topics": merged_topics,
                "last_next_action": response_payload.next_action,
                "last_sentiment": response_payload.sentiment,
            },
        }

        return agent_result

    def _fallback_result(
        self, state: GraphState, context: dict[str, Any], ambiguity: dict[str, Any]
    ) -> AgentResult:
        message = Message(
            id=f"msg-{state.current_turn + 1}",
            role="assistant",
            content=(
                "I'm experiencing a hiccup processing that request. Could you rephrase or provide more details "
                "while I stabilize?"
            ),
            timestamp=datetime.utcnow(),
        )

        return {
            "chat_history_update": [message],
            "state_updates": {
                "confidence": 0.3,
                "last_agent": self.name,
                "conversation_context": context,
                "ambiguity_assessment": ambiguity,
                "pending_clarifications": [],
                "last_next_action": "wait_for_input",
                "last_sentiment": "neutral",
            },
        }
