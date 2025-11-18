"""Extraction Agent orchestrating structured requirement parsing."""

from __future__ import annotations

from typing import Any

try:
    from langchain.output_parsers import PydanticOutputParser
except ModuleNotFoundError:  # pragma: no cover - compatibility shim
    from langchain_core.output_parsers import PydanticOutputParser

from src.agents.base import AgentResult, BaseAgent
from src.schemas import GraphState, Requirement
from src.services.embedding_service import EmbeddingService
from src.storage.vectorstore import VectorStoreService
from src.utils.logging import get_logger, log_with_context

from .ambiguity_detector import AmbiguityDetector
from .criteria_generator import CriteriaGenerator
from .entity_extractor import EntityExtractor
from .prompt_builder import ExtractionPromptBuilder
from .schemas import ExtractionOutput
from .traceability_linker import TraceabilityLinker
from .type_classifier import TypeClassifier

logger = get_logger(__name__)


class ExtractionAgent(BaseAgent):
    """Transforms conversational context into structured requirements."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(name="extraction", temperature=0.1, **kwargs)
        self.prompt_builder = ExtractionPromptBuilder()
        self.entity_extractor = EntityExtractor()
        self.type_classifier = TypeClassifier()
        self.criteria_generator = CriteriaGenerator()
        self.traceability_linker = TraceabilityLinker()
        self.ambiguity_detector = AmbiguityDetector()
        self.parser = PydanticOutputParser(pydantic_object=ExtractionOutput)
        self.embedding_service = EmbeddingService()
        self.vector_store = VectorStoreService()

    async def execute(self, state: GraphState) -> AgentResult:
        correlation_id = state.correlation_id or "unknown"
        log_with_context(
            logger,
            "debug",
            "Extraction run started",
            correlation_id=correlation_id,
            turn=state.current_turn,
        )

        prompt = self.prompt_builder.build(
            state, format_instructions=self.parser.get_format_instructions()
        )
        try:
            llm_response = await self.llm.ainvoke(prompt)
            parsed = self.parser.parse(llm_response.content)
        except Exception as exc:  # pragma: no cover - refined in later ACs
            log_with_context(
                logger,
                "warning",
                "Extraction agent fallback",
                correlation_id=correlation_id,
                error=str(exc),
            )
            fallback_confidence = max(0.0, state.confidence - 0.2)
            return {
                "requirements_update": [],
                "state_updates": {
                    "last_agent": self.name,
                    "confidence": fallback_confidence,
                    "extraction_metadata": {
                        "tokens_used": 0,
                        "duration_ms": 0.0,
                        "model": self.model,
                    },
                },
            }

        agent_result = self._convert_output(parsed, state)
        # Store embeddings for semantic search; errors here should not
        # break core extraction behavior, so keep this best-effort.
        try:  # pragma: no cover - defensive; main logic covered via tests
            await self._store_embeddings(agent_result["requirements_update"], state.session_id)
        except Exception as _emb_exc:
            # Logged at debug level only in case vector store is unavailable.
            log_with_context(
                logger,
                "debug",
                "Embedding storage failed",
                correlation_id=correlation_id,
            )
        log_with_context(
            logger,
            "debug",
            "Extraction run complete",
            correlation_id=correlation_id,
            requirements=len(agent_result["requirements_update"]),
        )
        return agent_result

    def _convert_output(self, output: ExtractionOutput, state: GraphState) -> AgentResult:
        """Map ExtractionOutput into AgentResult structure."""

        new_requirements: list[Requirement] = []
        ambiguous_notes: list[str] = list(output.ambiguous_items)
        for item in output.requirements:
            req, suggestions = self._enrich_requirement(item, state)
            new_requirements.append(req)
            ambiguous_notes.extend(suggestions)

        deduped_ambiguous = self._dedupe(ambiguous_notes)
        overall_confidence = self._calculate_overall_confidence(
            llm_confidence=output.confidence,
            requirements=new_requirements,
            ambiguous_items=deduped_ambiguous,
        )

        return {
            "requirements_update": new_requirements,
            "state_updates": {
                "confidence": overall_confidence,
                "last_agent": self.name,
                "ambiguous_items": deduped_ambiguous,
                "extraction_metadata": output.metadata.model_dump(),
            },
        }

    def _enrich_requirement(
        self, item: Requirement, state: GraphState
    ) -> tuple[Requirement, list[str]]:
        payload = item.model_dump()
        actor = payload.get("actor") or self.entity_extractor.extract_actors(payload["action"])[0]
        payload["actor"] = actor

        if not payload.get("condition"):
            payload["condition"] = self.entity_extractor.extract_condition(payload["action"])

        payload["type"] = self.type_classifier.classify(payload["title"], action=payload["action"])

        if len(payload["acceptance_criteria"]) < 2:
            payload["acceptance_criteria"] = self.criteria_generator.generate(
                actor=actor,
                action=payload["action"],
                req_type=payload["type"],
            )

        payload["source_refs"] = self.traceability_linker.link(
            f"{payload['title']} {payload['action']}", state.chat_history
        )

        ambiguity = self.ambiguity_detector.detect(f"{payload['title']} {payload['action']}")
        if ambiguity["is_ambiguous"]:
            payload["confidence"] = min(payload.get("confidence", 1.0), 0.75)

        raw_suggestions = ambiguity.get("suggestions") or []
        suggestions: list[str] = (
            [str(s) for s in raw_suggestions] if isinstance(raw_suggestions, list) else []
        )

        return Requirement(**payload), suggestions

    async def _store_embeddings(self, requirements: list[Requirement], session_id: str) -> None:
        """Store embeddings for extracted requirements to support semantic search."""

        if not requirements:
            return

        for req in requirements:
            text = f"{req.title} {req.action}"
            embedding = await self.embedding_service.get_embedding(text)
            await self.vector_store.add_requirement(
                requirement_id=req.id,
                title=req.title,
                action=req.action,
                embedding=embedding,
                metadata={
                    "session_id": session_id,
                    "type": req.type.value,
                    "priority": req.priority.value,
                    "confidence": req.confidence,
                    "inferred": req.inferred,
                },
            )

    def _calculate_overall_confidence(
        self,
        *,
        llm_confidence: float,
        requirements: list[Requirement],
        ambiguous_items: list[str],
    ) -> float:
        """Combine LLM output and enrichment signals into a final confidence score."""

        llm_conf = self._clamp(llm_confidence)
        if requirements:
            avg_req_conf = sum(req.confidence for req in requirements) / len(requirements)
        else:
            avg_req_conf = llm_conf or 0.5

        base_score = 0.6 * llm_conf + 0.4 * avg_req_conf
        ambiguity_penalty = min(len(ambiguous_items) * 0.03, 0.25)
        return self._clamp(base_score - ambiguity_penalty)

    @staticmethod
    def _clamp(value: float) -> float:
        return max(0.0, min(value, 1.0))

    @staticmethod
    def _dedupe(items: list[str]) -> list[str]:
        seen = set()
        ordered: list[str] = []
        for item in items:
            if item and item not in seen:
                seen.add(item)
                ordered.append(item)
        return ordered
