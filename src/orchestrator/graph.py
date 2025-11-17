"""LangGraph workflow definition for the multi-agent orchestrator.

This module wires together the conversational and extraction agents (and
placeholder nodes for future agents) into a StateGraph. The compiled
``graph`` object can be used by the API layer to process a single turn of
user input.
"""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from src.orchestrator.checkpointer import DualCheckpointer
from src.orchestrator.nodes import (
    conversational_node,
    extraction_node,
    inference_node,
    review_node,
    synthesis_node,
    validation_node,
)
from src.orchestrator.routing import decide_next_step, review_router, validation_router
from src.schemas import GraphState


def build_graph() -> StateGraph:
    """Build and compile the requirements engineering workflow graph."""

    workflow: StateGraph = StateGraph(GraphState)

    # Nodes
    workflow.add_node("conversational", conversational_node)
    workflow.add_node("extraction", extraction_node)
    workflow.add_node("inference", inference_node)
    workflow.add_node("validation", validation_node)
    workflow.add_node("synthesis", synthesis_node)
    workflow.add_node("review", review_node)

    # Entry point
    workflow.add_edge(START, "conversational")

    # Conditional routing after conversation
    workflow.add_conditional_edges(
        "conversational",
        decide_next_step,
        {
            "extract": "extraction",
            "validate": "validation",
            "synthesis": "synthesis",
            "continue": END,
            "error_handler": END,
        },
    )

    # Extraction always flows to validation
    workflow.add_edge("extraction", "validation")

    # Validation -> synthesis / conversation / inference
    workflow.add_conditional_edges(
        "validation",
        validation_router,
        {
            "pass": "synthesis",
            "fail": "conversational",
            "needs_inference": "inference",
        },
    )

    # Inference loops back to validation
    workflow.add_edge("inference", "validation")

    # Synthesis goes to review
    workflow.add_edge("synthesis", "review")

    # Review decides whether to end or return to conversation
    workflow.add_conditional_edges(
        "review",
        review_router,
        {
            "approved": END,
            "revision": "conversational",
            "pending": END,
        },
    )

    # Compile with dual-layer checkpointing (Redis + Postgres) so that
    # orchestrator state is persisted across node executions.
    checkpointer = DualCheckpointer()
    graph = workflow.compile(checkpointer=checkpointer)
    return graph


# Singleton compiled graph instance used by the API layer.
graph = build_graph()
