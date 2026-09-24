"""Typed shared state used by the LangGraph runtime."""

from __future__ import annotations

from typing import Annotated, Any, Dict, List, TypedDict

import pandas as pd
from utils.semantic_contract import create_empty_semantic_context
from core.framework_settings import SETTINGS


def _merge_dicts(left: Dict[str, Any], right: Dict[str, Any]) -> Dict[str, Any]:
    """Merge dict updates produced by parallel branches."""
    if not left:
        return dict(right or {})
    if not right:
        return dict(left)
    merged = dict(left)
    merged.update(right)
    return merged


def _merge_lists(left: List[Any], right: List[Any]) -> List[Any]:
    """Concatenate list updates produced by parallel branches."""
    if not left:
        return list(right or [])
    if not right:
        return list(left)
    return list(left) + list(right)




class AgenticState(TypedDict, total=False):
    """Single state object shared across all LangGraph nodes."""

    series_id: str
    iteration: int
    max_iterations: int
    maximum_valid_trainings: int
    minimum_valid_trainings_before_stop: int
    valid_training_count: int
    step: int
    done: bool
    stop_reason: str | None
    stop_metadata: Dict[str, Any] | None
    last_action: str | None
    target_mase: float
    next_node: str
    stop: bool
    orchestration_reason: str

    # Error and retry fields.
    technical_error: bool
    non_retryable_candidate_failure: bool
    technical_retry_count: int
    cumulative_technical_retry_count: int
    candidate_error_counts: Dict[str, int]
    rejected_technical_candidates: List[Dict[str, Any]]
    max_technical_retries: int
    last_technical_error: str | None
    technical_status: str | None
    request_planner_replan: bool
    technical_escalation: Dict[str, Any] | None
    technical_exhausted_models: List[str]
    technical_compliance_trace: List[Dict[str, Any]]

    # Core outputs requested by the orchestrator contract.
    data_summary: Dict[str, Any]
    dataset_length: int
    analytical_features: Dict[str, Any]
    extreme_outliers_detected: Dict[str, Any]
    log_transform_recommended: bool
    log_candidate_required: bool
    log_transform_supported: bool
    log_candidate_evaluated: bool
    raw_candidate_evaluated: bool
    semantic_signals: List[Dict[str, Any]]
    semantic_raw_signals: List[Dict[str, Any]]
    semantic_operational_signals: List[Dict[str, Any]]
    semantic_excluded_signals: List[Dict[str, Any]]
    semantic_agent_uses_analytical_output: bool
    analytical_agent_uses_semantic_output: bool
    model_results: List[Dict[str, Any]]
    best_score: float
    history: Annotated[List[Dict[str, Any]], _merge_lists]
    planner_signals: Dict[str, Any]
    need_semantic: bool
    parallel_mode: bool

    # Planner-to-agent guidance for targeted re-invocations.
    planner_guidance: Dict[str, Any] | str | None
    planner_instruction: str | None

    # Runtime context fields used by planner and agents.
    semantic_context: Dict[str, Any]
    context_regime: str
    model_trust_score: float
    tested_models: List[Dict[str, Any]]
    performance_history: List[Dict[str, Any]]
    best_model: Dict[str, Any] | None
    strategy: str
    exploration_score: float
    recommendations: Dict[str, Any]
    conflicting_recommendations: Annotated[List[Dict[str, Any] | str], _merge_lists]

    # Technical selection fields (set by TechnicalAgent, consumed by PreprocessingAgent and TrainingAgent).
    model_rationale: str
    model_confidence: float
    technical_mode: str

    current_candidate_config: Dict[str, Any]
    candidate_record: Dict[str, Any]
    candidate_ledger: List[Dict[str, Any]]
    candidate_seed: int
    experiment_seed: int
    dataset_fingerprint: str
    candidate_catalog_fingerprint: str
    determinism_audit: Dict[str, Any]
    candidate_request: Dict[str, Any]
    candidate_request_history: List[Dict[str, Any]]
    search_trace: List[Dict[str, Any]]

    # Preprocessing fields (used by TrainingAgent).
    replace_outliers: Dict[str, Any]
    processed_train_df: pd.DataFrame
    replace_outliers_log: Dict[str, Any]
    preprocessed_train_df: pd.DataFrame
    preprocessed_val_df: pd.DataFrame
    preprocessing_log: Dict[str, Any]
    preprocessing_candidate_id: str | None

    # Analysis and history tracking
    analysis_history: List[Dict[str, Any]]
    inferred_domain: str

    # Configuration and constraints
    forecast_horizon: int
    partition_sizes: Dict[str, int]
    enforce_full_valid_training_budget: bool
    model_prescription: Dict[str, Any]
    model_selection_signals: Dict[str, Any]
    semantic_completed: bool
    semantic_aggregation: Dict[str, Any]

    # Training fields.
    training_error: str | None
    mase_reference: Dict[str, Any]

    # Safety counter to prevent infinite loops when iteration is not incremented.
    planner_call_count: int

    # Dataset and semantic splits.
    train_df: pd.DataFrame
    raw_train_df: pd.DataFrame
    raw_validation_df: pd.DataFrame
    val_df: pd.DataFrame
    full_df: pd.DataFrame
    split_metadata: Dict[str, Any]
    holdout_boundary_applied: bool
    holdout_metadata: Dict[str, Any]
    timemmd_metadata: Dict[str, Any]
    external_numeric_columns_excluded: int
    winning_candidate_id: str
    semantic_docs_df: pd.DataFrame
    semantic_docs: List[Dict[str, Any]]
    semantic_docs_train_df: pd.DataFrame
    semantic_docs_val_df: pd.DataFrame
    semantic_split_meta: Dict[str, Any]
    semantic_document_filtering: Dict[str, Any]
    semantic_max_docs: int
    semantic_max_chunks: int
    semantic_chunk_size: int
    semantic_deduplicate_facts: bool
    semantic_near_duplicate_threshold: float


    # Authoritative influence, Planner and safety state.
    influence_mode: str
    semantic_enabled: bool
    controller_mode: str
    reliability_mode: str
    planner_policy: Dict[str, Any]
    max_replanning_attempts: int
    model_catalog: Dict[str, List[str]]
    planner_decision: Dict[str, Any]
    search_decision_state: Dict[str, Any]
    search_controller: Dict[str, Any]
    safety_guard_result: Dict[str, Any]
    replanning_attempts: int

    semantic_target_resolution: Dict[str, Any]
    diagnostics: Dict[str, Any]
    # Memory fields mirrored from AgentMemory.
    memory: Annotated[Dict[str, Any], _merge_dicts]


def create_initial_state(series_id: str, max_iterations: int) -> AgenticState:
    """Create an initial state ready to be used by StateGraph.invoke."""
    return {
        "series_id": series_id,
        "iteration": 0,
        "step": 0,
        "max_iterations": max_iterations,
        "maximum_valid_trainings": 0,
        "minimum_valid_trainings_before_stop": SETTINGS.minimum_valid_trainings_before_stop,
        "valid_training_count": 0,
        "done": False,
        "stop_reason": None,
        "stop_metadata": None,
        "last_action": None,
        "target_mase": SETTINGS.finalize_mase_threshold,
        "next_node": "technical_agent",
        "stop": False,
        "orchestration_reason": "",
        "technical_error": False,
        "non_retryable_candidate_failure": False,
        "technical_retry_count": 0,
        "cumulative_technical_retry_count": 0,
        "candidate_error_counts": {
            "invalid_configuration": 0,
            "duplicate_candidate": 0,
            "execution_failure": 0,
            "technical_generation_failure": 0,
        },
        "rejected_technical_candidates": [],
        "max_technical_retries": SETTINGS.max_technical_retries,
        "last_technical_error": None,
        "technical_exhausted_models": [],
        "technical_compliance_trace": [],
        "data_summary": {},
        "dataset_length": 0,
        "extreme_outliers_detected": {"found": False},
        "log_transform_recommended": False,
        "log_candidate_required": False,
        "log_transform_supported": False,
        "log_candidate_evaluated": False,
        "raw_candidate_evaluated": False,
        "analytical_features": {},
        "semantic_signals": [],
        "semantic_raw_signals": [],
        "semantic_operational_signals": [],
        "semantic_agent_uses_analytical_output": False,
        "analytical_agent_uses_semantic_output": False,
        "model_results": [],
        "history": [],
        "best_score": float("inf"),
        "model_selection_signals": {},
        "need_semantic": True,
        "parallel_mode": False,
        "semantic_context": create_empty_semantic_context(),
        "semantic_completed": False,
        "context_regime": "normal",
        "model_trust_score": 0.6,
        "tested_models": [],
        "performance_history": [],
        "best_model": None,
        "strategy": "exploration",
        "technical_mode": "explore",
        "exploration_score": 0.5,
        "recommendations": {},
        "conflicting_recommendations": [],
        "model_rationale": "",
        "model_confidence": 0.0,
        "preprocessing_log": {},
        "replace_outliers": {},
        "processed_train_df": pd.DataFrame(),
        "replace_outliers_log": {},
        "preprocessing_candidate_id": None,
        "preprocessed_train_df": pd.DataFrame(),
        "preprocessed_val_df": pd.DataFrame(),
        "raw_train_df": pd.DataFrame(),
        "raw_validation_df": pd.DataFrame(),
        "analysis_history": [],
        "inferred_domain": "unknown",
        "forecast_horizon": 1,
        "partition_sizes": {},
        "enforce_full_valid_training_budget": False,
        "split_metadata": {},
        "model_prescription": {},
        "planner_guidance": None,
        "planner_instruction": None,
        "current_candidate_config": {},
        "training_error": None,
        "mase_reference": {},
        "planner_call_count": 0,
        "planner_signals": {
            "next_action": "analytical",
            "reason": "Initial bootstrap: analytical prerequisite.",
            "technical_mode": "explore",
        },
        "influence_mode": "fuzzy",
        "semantic_enabled": True,
        "controller_mode": "fuzzy",
        "reliability_mode": "fuzzy",
        "planner_policy": {},
        "max_replanning_attempts": SETTINGS.max_replanning_attempts,
        "model_catalog": {
            "available_models": [],
            "unavailable_models": [],
            "disabled_models": [],
            "failed_models": [],
        },
        "planner_decision": {},
        "search_decision_state": {},
        "search_controller": {},
        "safety_guard_result": {},
        "replanning_attempts": 0,
        "memory": {
            "failed_models": [],
            "successful_patterns": [],
            "dataset_signatures": {},
        },
        "semantic_target_resolution": {},
        "diagnostics": {},
        "semantic_aggregation": {}
    }
