"""Abstract base class and shared helpers for all agents."""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime
from typing import TYPE_CHECKING, Any, TypedDict
from uuid import uuid4

from langchain_openai import ChatOpenAI
from openai import APIError, RateLimitError, Timeout
from tenacity import (
    RetryError,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from src.config import settings
from src.utils.logging import get_logger, log_with_context

if TYPE_CHECKING:
    from src.schemas import GraphState, Message, Requirement

LLM_EXCEPTIONS = (RateLimitError, APIError, Timeout)


class AgentResult(TypedDict, total=False):
    """Typed dictionary returned by concrete agent execute() implementations."""

    chat_history_update: list[Message]
    requirements_update: list[Requirement]
    inferred_requirements_update: list[Requirement]
    validation_issues_update: list[dict[str, Any]]
    state_updates: dict[str, Any]
    confidence: float


class BaseAgent(ABC):
    """Common functionality for all orchestrated agents."""

    def __init__(
        self,
        name: str,
        *,
        model: str = "gpt-4-turbo-preview",
        temperature: float = 0.2,
        max_tokens: int = 2000,
        openai_api_key: str | None = None,
        timeout: float | None = 30.0,
    ) -> None:
        self.name = name
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.logger = get_logger(f"agent.{self.name}")

        # Prefer explicit API key when provided, otherwise fall back to the
        # configured default from application settings. This avoids
        # construction-time errors in environments where OPENAI_API_KEY is
        # not set (e.g. tests that stub out agent behavior).
        api_key = openai_api_key or settings.OPENAI_API_KEY

        self.llm = self._wrap_llm(
            ChatOpenAI(
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                api_key=api_key,
                timeout=timeout,
            )
        )

        self._start_time: datetime | None = None

    @abstractmethod
    async def execute(self, state: GraphState) -> AgentResult:
        """Perform agent-specific work and return state update instructions."""

    async def invoke(self, state: GraphState) -> GraphState:
        """Public entry point with instrumentation and error handling."""

        self._start_time = datetime.utcnow()
        correlation_id = state.correlation_id or str(uuid4())

        try:
            log_with_context(
                self.logger,
                "info",
                f"{self.name} agent started",
                agent=self.name,
                session_id=state.session_id,
                turn=state.current_turn,
                correlation_id=correlation_id,
            )

            result = await self.execute(state)
            updated_state = self._merge_result(state, result)

            duration_ms = (datetime.utcnow() - self._start_time).total_seconds() * 1000
            log_with_context(
                self.logger,
                "info",
                f"{self.name} agent completed",
                agent=self.name,
                duration_ms=duration_ms,
                confidence=updated_state.confidence,
                correlation_id=correlation_id,
            )

            return updated_state

        except RetryError as retry_error:
            last_exc = retry_error.last_attempt.exception()
            if last_exc is None:  # pragma: no cover - defensive guard
                raise retry_error
            raise last_exc from retry_error  # pragma: no cover
        except Exception as exc:
            return self._handle_failure(state, exc, correlation_id)

    def _merge_result(self, state: GraphState, result: AgentResult) -> GraphState:
        updates: dict[str, Any] = result.get("state_updates", {}).copy()

        if "confidence" not in updates and "confidence" in result:
            updates["confidence"] = result["confidence"]

        chat_history_update = result.get("chat_history_update")
        if chat_history_update:
            updates["chat_history"] = [*state.chat_history, *chat_history_update]

        req_update = result.get("requirements_update")
        if req_update:
            updates["requirements"] = [*state.requirements, *req_update]

        inferred_update = result.get("inferred_requirements_update")
        if inferred_update:
            updates["inferred_requirements"] = [*state.inferred_requirements, *inferred_update]

        validation_update = result.get("validation_issues_update")
        if validation_update:
            updates["validation_issues"] = [*state.validation_issues, *validation_update]

        updates.setdefault("last_agent", self.name)
        updates.setdefault("iterations", state.iterations + 1)

        return state.with_updates(**updates)

    def _handle_failure(self, state: GraphState, exc: Exception, correlation_id: str) -> GraphState:
        error_count = state.error_count + 1
        log_with_context(
            self.logger,
            "error",
            f"{self.name} agent failed",
            agent=self.name,
            error=str(exc),
            error_type=type(exc).__name__,
            error_count=error_count,
            correlation_id=correlation_id,
        )

        degraded_state = state.with_updates(
            error_count=error_count,
            last_agent=self.name,
            confidence=max(0.0, state.confidence - 0.2),
        )

        if error_count >= 3:
            raise exc

        return degraded_state

    @staticmethod
    def llm_retry(**retry_kwargs):
        """Decorator factory to retry LLM calls with exponential backoff."""

        return retry(
            stop=stop_after_attempt(retry_kwargs.get("attempts", 3)),
            wait=wait_exponential(multiplier=1, min=2, max=10),
            retry=retry_if_exception_type(LLM_EXCEPTIONS),
            reraise=True,
        )

    def _wrap_llm(self, llm: ChatOpenAI) -> ChatOpenAI:
        """Wrap LLM invoke call with retry handler."""

        original_invoke = llm._generate  # type: ignore[attr-defined]

        @self.llm_retry()
        def safe_generate(*args, **kwargs):
            return original_invoke(*args, **kwargs)

        llm._generate = safe_generate  # type: ignore[assignment]
        return llm


__all__ = ["AgentResult", "BaseAgent"]
