"""LangGraph-centric orchestration with explicit branching, loop and retry."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

from langgraph.graph import END, START, StateGraph

from nodes.analytical import AnalyticalNode
from nodes.planner import PlannerNode
from nodes.preprocessing import PreprocessingNode
from nodes.semantic import SemanticNode
from nodes.technical import TechnicalNode
from nodes.training import TrainingNode
from state import AgenticState


@dataclass
class GraphNodes:
    """Container for callable graph nodes."""

    planner_signals: PlannerNode
    analytical: AnalyticalNode
    semantic: SemanticNode
    technical: TechnicalNode
    preprocessing: PreprocessingNode
    training: TrainingNode


def build_agentic_graph(nodes: GraphNodes) -> Any:
    """Build and compile the orchestration graph.

    The planner orchestrates most nodes. The technical agent triggers
    a direct chain: technical -> preprocessing -> training, without
    returning to the planner in between.
    """
    graph = StateGraph(AgenticState)

    graph.add_node("planner_signals", nodes.planner_signals)
    graph.add_node("analytical", nodes.analytical)
    graph.add_node("semantic", nodes.semantic)
    graph.add_node("technical", nodes.technical)
    graph.add_node("preprocessing", nodes.preprocessing)
    graph.add_node("training", nodes.training)
    graph.add_edge(START, "planner_signals")
    # Planner-driven routing.
    graph.add_conditional_edges(
        "planner_signals",
        planner_router,
        {
            "analytical": "analytical",
            "semantic": "semantic",
            "technical": "technical",
            "END": END,
        },
    )

    # Worker nodes that return control to planner.
    graph.add_edge("analytical", "planner_signals")
    graph.add_edge("semantic", "planner_signals")

    # Direct chain: technical -> preprocessing -> training -> planner.
    graph.add_conditional_edges(
        "technical",
        technical_router,
        {
            "preprocessing": "preprocessing",
            "replan": "planner_signals",
            "END": END,
        },
    )

    graph.add_edge("preprocessing", "training")
    graph.add_conditional_edges(
        "training",
        training_router,
        {"planner_signals": "planner_signals", "END": END},
    )

    graph.set_entry_point("planner_signals")
    return graph.compile()


def technical_router(state: Dict[str, Any]) -> str:
    """Return failures to Planner; workers never choose the retry action."""
    if (
        bool(state.get("technical_error", False))
        or bool(state.get("request_planner_replan", False))
        or state.get("technical_status") == "NO_VALID_NEW_CONFIGURATION"
    ):
        return "replan"
    return "preprocessing"


def training_router(state: Dict[str, Any]) -> str:
    """Return every completed validation trial to the authoritative Planner."""
    _ = state
    return "planner_signals"

def planner_router(state: Dict[str, Any]) -> str:
    """Route only the action already resolved and approved for the Planner."""
    planner_signals = state.get("planner_signals", {})
    if not isinstance(planner_signals, dict):
        return "END"
    if bool(state.get("done", False)):
        return "END"

    next_action = str(planner_signals.get("next_action", "END"))
    if next_action == "END":
        return "END"
    # training is no longer a direct planner target;
    # it is reached automatically via the technical chain.
    if next_action in {
        "analytical",
        "semantic",
        "technical",
    }:
        return next_action
    return "END"


