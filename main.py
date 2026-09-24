"""Entry point for the LangGraph-native AgenticAI runtime."""

from __future__ import annotations
import argparse
import copy
from datetime import datetime, timezone
import random
import re
import sys
from pathlib import Path
import uuid
from typing import Any, Dict
import numpy as np
import pandas as pd



from agents.planner_agent import PlannerAgent
from agents.analytical_agent import AnalyticalAgent
from agents.preprocessing_agent import PreprocessingAgent, PreprocessingPlan
from agents.semantic_agent import SemanticAgent
from agents.technical_agent import TechnicalAgent
from agents.training_agent import TrainingAgent


from core.memory import AgentMemory
from core.framework_settings import DataSplit, ExperimentInputs, SETTINGS
from core.candidate_catalog import (
    DEFAULT_CANDIDATE_CATALOG,
    dataframe_fingerprint,
)
from evaluation.holdout import FinalEvaluator, SearchState, TestHoldout
from evaluation.experiment import ExperimentManifest
from memory.memory_manager import MemoryManager
from models.availability import build_model_catalog
from models.trainer import Trainer
from nodes.analytical import AnalyticalNode
from nodes.planner import PlannerNode
from nodes.semantic import SemanticNode
from nodes.technical import TechnicalNode
from nodes.preprocessing import PreprocessingNode
from nodes.training import TrainingNode
from state import AgenticState, create_initial_state
from utils.data_loader import (
    load_dataset,
    load_semantic_documents_df,
    parse_document_dates,
)

from evaluation.final_reporting import evaluate_final_test, write_final_outputs
from utils.influence_mode import validate_experiment_config
from utils.model_family_state import MODEL_TO_FAMILY
from utils.numeric_input_policy import assert_raw_series_contract, numeric_input_audit
from utils.training_budget import adaptive_stop_summary
from utils.configuration_identity import (
    log_exploration_status,
    target_transform_label,
)
from utils.preprocessing_registry import (
    CANONICAL_PREPROCESSING_TRANSFORMATIONS,
    assert_preprocessing_registry_consistency,
    runtime_implemented_transformations,
)

PROJECT_ROOT = Path(__file__).resolve().parents[0]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def _new_run_id() -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{stamp}_{uuid.uuid4().hex[:8]}"


def _prepare_run_output_path(
    base_root: Path | str | None, explicit_run_id: str | None = None
) -> tuple[Path, str]:
    """Build the isolated output location for one execution."""
    root = Path(base_root or SETTINGS.output_root).expanduser()
    raw_run_id = explicit_run_id or _new_run_id()
    run_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(raw_run_id)).strip("._")
    if not run_id:
        raise ValueError("run_id must contain at least one safe character")
    return (root / "runs" / run_id if SETTINGS.isolate_runs else root), run_id


class _NodeTracer:
    """Wraps a node callable to record each invocation in a shared step log."""

    def __init__(self, name: str, node: Any, log: list) -> None:
        self._name = name
        self._node = node
        self._log = log

    def __call__(self, state: Dict[str, Any]) -> Dict[str, Any]:
        step = len(self._log) + 1
        msg = f"step {step}: agent {self._name}"
        self._log.append(msg)
        print(f"[STEP {step}] Running agent: {self._name.upper()}...")
        return self._node(state)


def _framework_runtime(
    inputs: ExperimentInputs,
    *,
    run_id: str | None,
    output_root: Path | str | None,
) -> Dict[str, Any]:
    """Adapt immutable framework settings to the existing graph contracts."""
    dataset_path = inputs.dataset_path.resolve()
    if not dataset_path.is_file():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")
    resolved_output, resolved_run_id = _prepare_run_output_path(output_root, run_id)
    return {
        "project_root": str(Path.cwd()),
        "seed": SETTINGS.seed,
        "data": {
            "source_path": str(dataset_path),
            "series_id": inputs.series_id,
            "domain": inputs.domain,
            "split": inputs.split,
        },
        "semantic": {**dict(SETTINGS.semantic), "csv_paths": [str(dataset_path)]},
        "semantic_aggregation": dict(SETTINGS.semantic_aggregation),
        "semantic_target_resolution": dict(SETTINGS.semantic_target_resolution),
        "replace_outliers": dict(SETTINGS.replace_outliers),
        "experiment": {
            "max_replanning_attempts": SETTINGS.max_replanning_attempts,
            "planner_policy": dict(SETTINGS.planner_policy),
        },
        "loop": {
            "max_iterations": SETTINGS.max_iterations,
            "maximum_valid_trainings": SETTINGS.maximum_valid_trainings,
            "minimum_valid_trainings_before_stop": SETTINGS.minimum_valid_trainings_before_stop,
            "enforce_full_valid_training_budget": SETTINGS.enforce_full_valid_training_budget,
            "finalize_mase_threshold": SETTINGS.finalize_mase_threshold,
            "max_technical_retries": SETTINGS.max_technical_retries,
        },
        "llm": {
            "model_name": SETTINGS.llm_model_name,
            "max_new_tokens": SETTINGS.llm_max_new_tokens,
            "temperature": SETTINGS.llm_temperature,
            "do_sample": SETTINGS.llm_do_sample,
            "device": SETTINGS.llm_device,
        },
        "output": {
            "root": str(resolved_output),
            "run_id": resolved_run_id,
            "forecast_horizon": inputs.forecast_horizon,
        },
        "inputs": inputs,
    }


def _prepare_initial_state(
    config: Dict[str, Any], debug_enabled: bool = True
) -> tuple[AgenticState, Trainer, MemoryManager, Path, str, TestHoldout]:
    """Load data and create initialized state and runtime dependencies."""
    influence = validate_experiment_config(config)
    assert_preprocessing_registry_consistency(
        CANONICAL_PREPROCESSING_TRANSFORMATIONS,
        TechnicalAgent.ALLOWED_TRANSFORMATIONS,
    )
    assert_preprocessing_registry_consistency(
        CANONICAL_PREPROCESSING_TRANSFORMATIONS,
        runtime_implemented_transformations(PreprocessingAgent),
    )
    base_dir = Path(config["project_root"]).resolve()
    data_cfg = config.get("data", {})
    data_path = Path(data_cfg["source_path"])
    requested_series_id = data_cfg.get("series_id")

    output_root = base_dir / config["output"]["root"]
    output_root.mkdir(parents=True, exist_ok=True)

    series_id, full_df = load_dataset(
        data_root=data_path,
        series_id=requested_series_id,
    )
    assert_raw_series_contract(full_df, context="Loaded numerical series")
    external_numeric_columns_excluded = int(
        full_df.attrs.get("external_numeric_columns_excluded", 0)
    )
    # Modes without semantics do not load or retain semantic documents.
    semantic_docs_df = pd.DataFrame()
    semantic_document_filtering: Dict[str, Any] = {
        "policy": "timestamped_documents_only",
        "input_rows": 0,
        "accepted_rows": 0,
        "selected_rows": 0,
        "dropped_rows": 0,
        "drop_reasons": {},
        "temporal_boundary_exclusions": {},
    }
    if influence.semantic_enabled:
        semantic_cfg = config.get("semantic", {})
        semantic_csv_paths = [data_path]
        semantic_parquet_paths: list[Path] = []
        semantic_docs_df = load_semantic_documents_df(
            parquet_paths=semantic_parquet_paths,
            csv_paths=semantic_csv_paths,
            entity_id=series_id,
            max_docs=int(config["semantic"]["max_docs"]),
            include_prediction_text=bool(semantic_cfg.get("include_prediction_text", True)),
        )
        semantic_document_filtering = dict(
            semantic_docs_df.attrs.get(
                "semantic_document_filtering", semantic_document_filtering
            )
        )
        semantic_document_filtering["temporal_boundary_exclusions"] = {}
        if semantic_docs_df.empty:
            raise RuntimeError(
                "No timestamp-valid semantic documents with text were found "
                "in the selected dataset."
            )
        _, parsed_document_dates, semantic_docs_df = parse_document_dates(
            semantic_docs_df
        )
        numeric_start = full_df["date"].min()
        numeric_end = full_df["date"].max()
        numeric_range_mask = (
            (parsed_document_dates >= numeric_start)
            & (parsed_document_dates <= numeric_end)
        )
        numeric_range_excluded = int((~numeric_range_mask).sum())
        semantic_document_filtering["temporal_boundary_exclusions"][
            "outside_numeric_series_range"
        ] = numeric_range_excluded
        semantic_docs_df = semantic_docs_df.loc[numeric_range_mask].reset_index(
            drop=True
        )
        if semantic_docs_df.empty:
            raise RuntimeError(
                "No timestamp-valid semantic documents overlap the numeric "
                "series time range."
            )

    forecast_horizon = int(config["inputs"].forecast_horizon)
    train_size, validation_size, test_size = config["inputs"].split.sizes(
        len(full_df), forecast_horizon
    )
    search_df, test_holdout = TestHoldout.separate(full_df, test_size=test_size)
    if len(search_df) != train_size + validation_size:
        raise AssertionError("Chronological split sizes do not cover the search partition.")
    train_preview = search_df.iloc[:train_size]
    validation_preview = search_df.iloc[train_size:]
    test_preview = test_holdout._copy_for_final_evaluator()
    print(
        "[DATA]\n"
        f"Observations: {len(full_df)}\n"
        f"Start: {full_df['date'].min()}\n"
        f"End: {full_df['date'].max()}\n\n"
        "[TEMPORAL SPLIT]\n"
        f"Total observations: {len(full_df)}\n"
        f"Train: {len(train_preview)} ({len(train_preview) / len(full_df):.2%})\n"
        f"Validation: {len(validation_preview)} ({len(validation_preview) / len(full_df):.2%})\n"
        f"Test: {len(test_preview)} ({len(test_preview) / len(full_df):.2%})\n"
        f"Validation windows: {len(validation_preview) // forecast_horizon}\n"
        f"Test windows: {len(test_preview) // forecast_horizon}\n"
        f"Train range: {train_preview['date'].min()} -> {train_preview['date'].max()}\n"
        f"Validation range: {validation_preview['date'].min()} -> {validation_preview['date'].max()}\n"
        f"Test range: {test_preview['date'].min()} -> {test_preview['date'].max()}"
    )
    if not semantic_docs_df.empty:
        search_document_dates = pd.to_datetime(
            semantic_docs_df["date"], errors="raise", utc=True
        ).dt.tz_localize(None)
        search_boundary_mask = search_document_dates <= search_df["date"].max()
        semantic_document_filtering["temporal_boundary_exclusions"][
            "after_search_boundary"
        ] = int((~search_boundary_mask).sum())
        semantic_docs_df = semantic_docs_df.loc[search_boundary_mask].reset_index(
            drop=True
        )
        if semantic_docs_df.empty:
            raise RuntimeError(
                "No timestamp-valid semantic documents are available before "
                "the test holdout boundary."
            )
        semantic_document_filtering["search_eligible_rows"] = int(
            len(semantic_docs_df)
        )

    drop_reasons = dict(semantic_document_filtering.get("drop_reasons", {}))
    accepted_rows = int(semantic_document_filtering.get("accepted_rows", 0))
    selected_rows = int(semantic_document_filtering.get("selected_rows", 0))
    drop_reasons["max_docs_selection"] = max(0, accepted_rows - selected_rows)
    drop_reasons.update(
        {
            str(reason): int(count)
            for reason, count in semantic_document_filtering.get(
                "temporal_boundary_exclusions", {}
            ).items()
        }
    )
    semantic_document_filtering["exclusion_reasons"] = drop_reasons
    semantic_document_filtering["total_excluded_rows"] = int(
        sum(drop_reasons.values())
    )

    state = create_initial_state(
        series_id=series_id, max_iterations=int(config["loop"]["max_iterations"])
    )

    state.update(influence.to_dict())
    state["replace_outliers"] = dict(
        config.get("replace_outliers", {}) or {}
    )
    state["planner_policy"] = dict(
        config.get("experiment", {}).get("planner_policy", {})
    )
    state["search_controller"] = dict(config.get("search_controller", {}) or {})
    state["semantic_aggregation"] = dict(config.get("semantic_aggregation", {}))
    state["semantic_target_resolution"] = dict(
        config.get("semantic_target_resolution", {})
    )
    state["target_mase"] = float(config["loop"]["finalize_mase_threshold"])
    state["maximum_valid_trainings"] = int(
        config.get("loop", {}).get("maximum_valid_trainings", 0)
    )
    state["minimum_valid_trainings_before_stop"] = int(
        config.get("loop", {}).get("minimum_valid_trainings_before_stop", 5)
    )
    state["forecast_horizon"] = int(config["inputs"].forecast_horizon)
    state["partition_sizes"] = {
        "train": int(train_size), "validation": int(validation_size), "test": int(test_size)
    }
    state["enforce_full_valid_training_budget"] = bool(config.get("loop", {}).get("enforce_full_valid_training_budget", False))
    state["max_technical_retries"] = int(
        config.get("loop", {}).get("max_technical_retries", 2)
    )
    state["max_replanning_attempts"] = int(
        config.get("experiment", {}).get("max_replanning_attempts", 3)
    )

    state["full_df"] = search_df
    state["external_numeric_columns_excluded"] = (
        external_numeric_columns_excluded
    )
    state["timemmd_metadata"] = {
        "enabled": False,
        "domain": data_cfg.get("domain"),
        "numeric_path": str(data_path),
        "text_paths": [str(data_path)],
    }
    state["run_id"] = config.get("output", {}).get("run_id")
    state["experiment_seed"] = int(config["seed"])
    state["dataset_fingerprint"] = dataframe_fingerprint(search_df)
    state["candidate_catalog_fingerprint"] = DEFAULT_CANDIDATE_CATALOG.fingerprint
    state["holdout_boundary_applied"] = True
    state["holdout_metadata"] = {
        "fingerprint": test_holdout.fingerprint,
        "size": test_holdout.size,
        "start_date": test_holdout.start_date,
        "end_date": test_holdout.end_date,
    }
    state["semantic_docs_df"] = semantic_docs_df
    state["semantic_document_filtering"] = semantic_document_filtering
    state["semantic_max_docs"] = int(config["semantic"]["max_docs"])
    state["semantic_max_chunks"] = int(config["semantic"].get("max_chunks", 0))
    state["semantic_chunk_size"] = int(config["semantic"].get("chunk_size", 100))
    state["semantic_deduplicate_facts"] = bool(
        config["semantic"].get("deduplicate_facts", True)
    )
    state["semantic_near_duplicate_threshold"] = float(
        config["semantic"].get("near_duplicate_threshold", 0.90)
    )

    model_cfg = config.get("models", {})
    if not isinstance(model_cfg, dict):
        model_cfg = {}
    model_catalog = build_model_catalog(
        configured_models=model_cfg.get("enabled"),
        disabled_models=model_cfg.get("disabled", []),
        failed_models=model_cfg.get("failed", []),
    )
    state["model_catalog"] = model_catalog

    memory = AgentMemory()
    memory_manager = MemoryManager(memory=memory)
    memory_manager.register_dataset_signature(state)

    trainer = Trainer(random_state=int(config["seed"]))


    search_state = SearchState.from_mapping(state)
    return search_state, trainer, memory_manager, output_root, series_id, test_holdout


def run_graph(
    inputs: ExperimentInputs,
    *,
    run_id: str | None = None,
    output_root: Path | str | None = None,
) -> Dict[str, Any]:
    """Run T-FUSE from explicit experiment inputs and fixed framework settings."""
    from graph import GraphNodes, build_agentic_graph
    from llm.hf_pipeline import HFLocalLLM
    import numpy as np
    import time

    config = _framework_runtime(inputs, run_id=run_id, output_root=output_root)
    start_time = time.time()
    debug_enabled = False

    if debug_enabled:
        print("[RUN] starting LangGraph-native execution")

    set_seed(int(config["seed"]))

    state, trainer, memory_manager, output_root, series_id, test_holdout = _prepare_initial_state(
        config, debug_enabled=debug_enabled
    )

    llm = HFLocalLLM(
        model_name=str(config["llm"]["model_name"]),
        max_new_tokens=int(config["llm"]["max_new_tokens"]),
        temperature=float(config["llm"]["temperature"]),
        do_sample=bool(config["llm"]["do_sample"]),
        device=(
            int(config["llm"]["device"])
            if str(config["llm"]["device"]).isdigit()
            else str(config["llm"]["device"])
        ),
        verbose=False,
    )

    analytical_agent = AnalyticalAgent()
    semantic_agent = SemanticAgent(
        llm=llm,
        max_docs=int(config["semantic"]["max_docs"]),
        max_chunks=int(config["semantic"].get("max_chunks", 0)),
        chunk_size=int(config["semantic"].get("chunk_size", 100)),
        max_chars_per_doc=int(config["semantic"].get("max_chars_per_doc", 2000)),
        max_total_chars=int(config["semantic"].get("max_total_chars", 24000)),
        max_prompt_tokens=int(config["semantic"].get("max_prompt_tokens", 12000)),
        deduplicate_facts=bool(config["semantic"].get("deduplicate_facts", True)),
        near_duplicate_threshold=float(
            config["semantic"].get("near_duplicate_threshold", 0.90)
        ),
    )
    technical_agent = TechnicalAgent(
        llm=llm,
        trainer=trainer,
        finalize_mase_threshold=float(config["loop"]["finalize_mase_threshold"]),
    )
    preprocessing_agent = PreprocessingAgent(llm=llm)
    training_agent = TrainingAgent(
        trainer=trainer,
        finalize_mase_threshold=float(config["loop"]["finalize_mase_threshold"]),
    )
    planner_agent = PlannerAgent(
        max_iterations=int(config["loop"]["max_iterations"]),
        policy_config=dict(
            config.get("experiment", {}).get("planner_policy", {})
        ),
    )

    agent_steps: list = []
    nodes = GraphNodes(
        planner_signals=_NodeTracer(
            "planner", PlannerNode(planner=planner_agent), agent_steps
        ),
        analytical=_NodeTracer(
            "analytical",
            AnalyticalNode(agent=analytical_agent, memory_manager=memory_manager),
            agent_steps,
        ),
        semantic=_NodeTracer(
            "semantic",
            SemanticNode(
                agent=semantic_agent,
                memory_manager=memory_manager,
                mode=state["influence_mode"],
            ),
            agent_steps,
        ),
        technical=_NodeTracer(
            "technical",
            TechnicalNode(agent=technical_agent, memory_manager=memory_manager),
            agent_steps,
        ),
        preprocessing=_NodeTracer(
            "preprocessing",
            PreprocessingNode(agent=preprocessing_agent, memory_manager=memory_manager),
            agent_steps,
        ),
        training=_NodeTracer(
            "training",
            TrainingNode(agent=training_agent, memory_manager=memory_manager),
            agent_steps,
        ),
    )

    app = build_agentic_graph(nodes)

    final_state = app.invoke(state)

    best_mase = float(final_state.get("best_model", {}).get("mase", float("inf")))
    target_mase = float(final_state.get("target_mase", float("inf")))
    final_state["target_mase_reached"] = bool(best_mase <= target_mase)

    if not bool(final_state.get("done", False)):
        raise RuntimeError("Graph terminated without a Planner-authored stop")

    best_tech = final_state.get("best_model")
    if not isinstance(best_tech, dict):
        raise RuntimeError("No technical result generated")
    if best_tech.get("candidate_seed") is None:
        raise RuntimeError("Winning candidate is missing candidate_seed")
    winning_candidate_seed = int(best_tech["candidate_seed"])
    winning_candidate_signature = str(
        best_tech.get("configuration_signature")
        or best_tech.get("candidate_id")
        or best_tech.get("training_id")
    )
    winning_candidate_id = str(
        best_tech.get("candidate_id")
        or best_tech.get("configuration_signature")
        or best_tech.get("training_id")
    )
    final_state["winning_candidate_id"] = winning_candidate_id
    final_evaluator = FinalEvaluator(
        holdout=test_holdout, winning_candidate_id=winning_candidate_id
    )

    print(
        "\n"
        + "=" * 50
        + "\n[FINAL EVALUATION] Starting final test phase...\n"
        + "=" * 50
    )

    raw_train = final_state.get("raw_train_df")
    raw_val = final_state.get("raw_validation_df")
    current_val = final_state.get("val_df")
    if not isinstance(raw_train, pd.DataFrame) or raw_train.empty:
        raise RuntimeError("Missing immutable raw_train_df")
    if not isinstance(raw_val, pd.DataFrame) or raw_val.empty:
        raise RuntimeError("Missing immutable raw_validation_df")
    if not isinstance(current_val, pd.DataFrame) or not np.array_equal(
        pd.to_numeric(raw_val["target"], errors="coerce").to_numpy(
            dtype=float
        ),
        pd.to_numeric(current_val["target"], errors="coerce").to_numpy(
            dtype=float
        ),
        equal_nan=True,
    ):
        raise RuntimeError("Validation targets changed after the split")

    raw_test = final_evaluator.acquire_once(
        winning_candidate_id=winning_candidate_id
    )
    raw_test_original = (
        raw_test.copy(deep=True) if isinstance(raw_test, pd.DataFrame) else raw_test
    )

    data_summary = final_state.get("data_summary", {})
    base_metrics = data_summary.get("base_metrics", {})
    inferred_freq = base_metrics.get("inferred_freq", "unknown")
    if raw_test is not None and not raw_test.empty:
        try:
            raw_test = AnalyticalAgent.canonicalize_evaluation_partition(
                raw_test,
                freq=str(inferred_freq),
                partition_name="test",
            )
            if not np.array_equal(
                pd.to_numeric(
                    raw_test_original["target"], errors="coerce"
                ).to_numpy(dtype=float),
                pd.to_numeric(raw_test["target"], errors="coerce").to_numpy(
                    dtype=float
                ),
                equal_nan=True,
            ):
                raise RuntimeError("Test targets were modified")
            print("[DATA] Test set validated and preserved without resampling")
        except Exception as exc:
            raise RuntimeError(
                "Final test failed immutable-partition validation: "
                f"{exc}"
            ) from exc

    combined_train_val = pd.concat([raw_train, raw_val], ignore_index=True)
    best_preprocessing_log = best_tech.get("preprocessing_log", {}) or {}
    best_transformations = best_preprocessing_log.get("transformations", [])
    refit_transformations = preprocessing_agent.specifications_for_refit(
        best_transformations
    )

    if best_transformations:
        replay_plan = PreprocessingPlan(
            transformations=refit_transformations,
            notes="Replay of winning preprocessing pipeline on train+val.",
        )
        print(
            f"[PREPROCESSING] Replaying winning pipeline: "
            f"{[t.get('name') for t in best_transformations if isinstance(t, dict)]}"
        )
        final_full_df, new_transformations_log = (
            preprocessing_agent._apply_transformations(combined_train_val, replay_plan)
        )
        final_replace_outliers_log = next(
            (
                copy.deepcopy(item)
                for item in new_transformations_log
                if item.get("name") == "replace_outliers"
            ),
            {},
        )
        new_preprocessing_log = {
            "model": best_tech["selected_model"],
            "transformations": new_transformations_log,
            "notes": "Winning preprocessing refitted on original train+validation.",
            "replace_outliers": final_replace_outliers_log,
        }
    else:
        print("[PREPROCESSING] No preprocessing in winning pipeline — using raw data.")
        final_full_df = combined_train_val
        new_preprocessing_log = {
            "model": best_tech["selected_model"],
            "transformations": [],
            "notes": "No preprocessing in winning pipeline.",
            "replace_outliers": {},
        }


    _final_transformations = new_preprocessing_log.get("transformations", [])

    if _final_transformations and raw_test is not None and not raw_test.empty:
        preprocessed_test_df = preprocessing_agent.apply_transformations_from_log(
            raw_test.copy(),
            _final_transformations,
            start_idx=len(combined_train_val),
            history_df=combined_train_val,
        )
    else:
        preprocessed_test_df = (
            raw_test.copy() if isinstance(raw_test, pd.DataFrame) else raw_test
        )

    final_evaluation = evaluate_final_test(
        trainer=trainer,
        preprocessing_agent=preprocessing_agent,
        best_tech=best_tech,
        winning_candidate_seed=winning_candidate_seed,
        winning_candidate_signature=winning_candidate_signature,
        final_full_df=final_full_df,
        preprocessed_test_df=preprocessed_test_df,
        combined_train_val=combined_train_val,
        raw_train=raw_train,
        raw_val=raw_val,
        current_val=current_val,
        raw_test=raw_test,
        raw_test_original=raw_test_original,
        new_preprocessing_log=new_preprocessing_log,
        final_state=final_state,
        forecast_horizon=int(config["output"]["forecast_horizon"]),
    )
    test_metrics_final = final_evaluation.test_metrics
    test_preds_final = final_evaluation.test_predictions
    _train_metrics_final = final_evaluation.train_metrics
    final_test_report = final_evaluation.final_test_report
    final_evaluation_status = final_evaluation.evaluation_status
    aligned_test_y_true = final_evaluation.aligned_test_y_true
    aligned_test_index = final_evaluation.aligned_test_index
    final_retraining_audit = final_evaluation.retraining_audit
    evaluation_integrity = final_evaluation.evaluation_integrity
    test_protocol_metrics = copy.deepcopy(test_metrics_final)
    perf_history = final_state.get("performance_history", [])
    winning_model_name = str(best_tech["selected_model"])
    final_preprocessing_info = dict(
        getattr(trainer.library, "last_preprocessing_info", {}) or {}
    )
    final_feature_columns = list(
        _train_metrics_final.get("feature_columns_used")
        or _train_metrics_final.get("feature_columns")
        or final_preprocessing_info.get("feature_columns_used")
        or []
    )
    final_numeric_input_audit = numeric_input_audit(
        final_feature_columns,
        textual_evidence_used=bool(final_state.get("semantic_completed", False)),
    )
    final_numeric_input_audit["external_numeric_columns_excluded_at_load"] = int(
        final_state.get("external_numeric_columns_excluded", 0)
    )
    final_numeric_input_audit["final_model_training_columns"] = (
        ["date", "target"]
        if winning_model_name == "Prophet"
        else ["target", *final_feature_columns]
    )
    final_retraining_audit["numeric_input_audit"] = copy.deepcopy(
        final_numeric_input_audit
    )
    log_transform_audit = log_exploration_status(final_state)

    winning_history_entry = {}
    winner_id = best_tech.get("training_id") or best_tech.get("technical_decision_id")
    winner_signature = best_tech.get("configuration_signature")
    for entry in perf_history:
        entry_id = entry.get("training_id") or entry.get("technical_decision_id")
        if (winner_id and entry_id == winner_id) or (
            not winner_id and winner_signature and entry.get("configuration_signature") == winner_signature
        ):
            winning_history_entry = entry
            break
    best_target_transform = target_transform_label(
        best_tech.get(
            "preprocessing_log",
            winning_history_entry.get("preprocessing_transformations", []),
        )
    )
    log_transform_audit["best_candidate_uses_log"] = (
        best_target_transform in {"log1p", "log"}
    )


    y_true_val: list = []
    if isinstance(best_tech, dict) and "y_true_val" in best_tech:
        y_true_val = [float(x) for x in best_tech["y_true_val"]]
    elif (
        isinstance(winning_history_entry, dict)
        and "y_true_val" in winning_history_entry
    ):
        y_true_val = [float(x) for x in winning_history_entry["y_true_val"]]
    else:
        val_df = final_state.get("val_df")
        if val_df is not None and hasattr(val_df, "get") and not val_df.empty:
            try:
                y_true_val = [float(x) for x in val_df["target"].tolist()]
            except Exception:
                y_true_val = []

    # Full model card for the winner
    winning_model_card = {
        "model_name": winning_model_name,
        "model_type": winning_model_name,
        "technical_decision_id": best_tech.get("technical_decision_id", winning_history_entry.get("technical_decision_id")),
        "training_id": best_tech.get("training_id", winning_history_entry.get("training_id")),
        "configuration_signature": best_tech.get("configuration_signature", winning_history_entry.get("configuration_signature")),
        "hyperparameters": best_tech.get("hyperparameters", {}),
        "validation_score": best_tech.get("validation_score"),
        "validation_metric": best_tech.get("validation_metric", "mse"),
        "mase": best_tech.get("mase", winning_history_entry.get("mase")),
        "rmse": best_tech.get("rmse", winning_history_entry.get("rmse")),
        "mse": best_tech.get("mse", winning_history_entry.get("mse")),
        "r2": best_tech.get("r2", winning_history_entry.get("r2")),
        "mae": best_tech.get("mae", winning_history_entry.get("mae")),
        "mape": best_tech.get("mape", winning_history_entry.get("mape")),
        "step": best_tech.get("step", winning_history_entry.get("step")),
        "preprocessing_log": new_preprocessing_log,
        "target_transform": best_target_transform,
        "metrics_scale": "original",
        "numeric_input_audit": copy.deepcopy(final_numeric_input_audit),
        # Per-split metrics (train / val / test)
        "train_metrics": best_tech.get(
            "train_metrics", winning_history_entry.get("train_metrics", {})
        ),
        "val_metrics": best_tech.get(
            "val_metrics", winning_history_entry.get("val_metrics", {})
        ),
    }
    validation_metrics = dict(winning_model_card.get("val_metrics", {}) or {})
    validation_metrics.update({
        "model_type": winning_model_name,
        "technical_decision_id": winning_model_card.get("technical_decision_id"),
        "configuration_signature": winning_model_card.get("configuration_signature"),
        "split": "validation",
        "metrics_computed_on": "validation",
        "evaluation_protocol": validation_metrics.pop("validation_protocol", "rolling_origin_refit"),
        "n_observations": validation_metrics.get("evaluated_points", len(y_true_val)),
        "forecast_horizon": int(config["output"]["forecast_horizon"]),
        "split_metadata": final_state.get("split_metadata", {}),
        "rolling_origin_count": final_state.get(
            "split_metadata", {}
        ).get("rolling_origin_count"),
        "validation_size": final_state.get("split_metadata", {}).get(
            "validation_size"
        ),
        "validation_ratio_effective": final_state.get(
            "split_metadata", {}
        ).get("validation_ratio_effective"),
        "train_size": final_state.get("split_metadata", {}).get("train_size"),
        "test_size": final_state.get("split_metadata", {}).get("test_size"),
        "train_end": final_state.get("split_metadata", {}).get("train_end"),
        "validation_start": final_state.get("split_metadata", {}).get(
            "validation_start"
        ),
        "validation_end": final_state.get("split_metadata", {}).get(
            "validation_end"
        ),
        "test_start": final_state.get("split_metadata", {}).get("test_start"),
        "test_end": final_state.get("split_metadata", {}).get("test_end"),
    })
    winning_model_card["val_metrics"] = validation_metrics

    val_preds_raw = (
        best_tech.get("val_predictions")
        or winning_history_entry.get("predictions")
        or []
    )
    val_predictions_out = [float(x) for x in val_preds_raw]
    forecast_strategy = "single_best_model"

    _val_metrics = validation_metrics or {}
    _test_metrics = test_protocol_metrics or {}
    configured_protocol_horizon = int(config["output"]["forecast_horizon"])
    validation_effective_horizons = list(
        _val_metrics.get("validation_effective_horizons", []) or []
    )
    final_effective_horizon = _test_metrics.get("effective_model_horizon")
    if final_effective_horizon is None and test_preds_final:
        final_effective_horizon = len(test_preds_final)
    forecast_protocol = {
        "configured_forecast_horizon": configured_protocol_horizon,
        "validation_protocol": _val_metrics.get(
            "evaluation_protocol",
            _val_metrics.get("validation_protocol", "rolling_origin_refit"),
        ),
        "number_of_validation_origins": int(
            _val_metrics.get(
                "number_of_validation_origins",
                len(validation_effective_horizons),
            )
        ),
        "validation_effective_horizons": validation_effective_horizons,
        "final_effective_horizon": final_effective_horizon,
        "horizon_consistency": bool(
            validation_effective_horizons
            and all(
                int(value) == configured_protocol_horizon
                for value in validation_effective_horizons
            )
            and final_effective_horizon is not None
            and int(final_effective_horizon) == configured_protocol_horizon
            and len(test_preds_final) == test_holdout.size
        ),
    }
    if (
        final_evaluation_status.get("run_status") == "final_evaluation_succeeded"
        and not forecast_protocol["horizon_consistency"]
    ):
        raise RuntimeError(
            "invalid_forecast_protocol: validation/final horizon mismatch"
        )
    fusion_output = final_state.get("diagnostics", {}).get(
        "evidence_fusion", {}
    )
    if not isinstance(fusion_output, dict):
        fusion_output = {}
    families_explored: list[str] = []
    for item in perf_history:
        model = str(item.get("model", "")) if isinstance(item, dict) else ""
        family = MODEL_TO_FAMILY.get(model)
        if family and family not in families_explored:
            families_explored.append(family)
    final_search_state = final_state.get("search_decision_state", {})
    if not isinstance(final_search_state, dict):
        final_search_state = {}
    experiment_manifest = ExperimentManifest.create(
        dataset_fingerprint=str(final_state.get("dataset_fingerprint", "")),
        candidate_catalog_fingerprint=str(
            final_state.get("candidate_catalog_fingerprint", "")
        ),
        seed=int(config["seed"]),
        influence_mode=str(final_state.get("influence_mode", "fuzzy")),
        maximum_valid_trainings=int(
            final_state.get("maximum_valid_trainings", 0)
        ),
        forecast_horizon=int(config["output"]["forecast_horizon"]),
        semantic_document_filtering=dict(
            final_state.get("semantic_document_filtering", {})
        ),
    )

    final_result = {
        "series_id": series_id,
        "winning_candidate_seed": winning_candidate_seed,
        "final_retraining_audit": final_retraining_audit,
        "forecast_horizon": int(config["output"]["forecast_horizon"]),
        "forecast_protocol": forecast_protocol,
        "dataset_fingerprint": final_state.get("dataset_fingerprint"),
        "candidate_catalog_fingerprint": final_state.get(
            "candidate_catalog_fingerprint"
        ),
        "split_metadata": final_state.get("split_metadata", {}),
        "rolling_origin_count": final_state.get(
            "split_metadata", {}
        ).get("rolling_origin_count"),
        "validation_size": final_state.get("split_metadata", {}).get(
            "validation_size"
        ),
        "validation_ratio_effective": final_state.get(
            "split_metadata", {}
        ).get("validation_ratio_effective"),
        "train_size": final_state.get("split_metadata", {}).get("train_size"),
        "test_size": final_state.get("split_metadata", {}).get("test_size"),
        "train_end": final_state.get("split_metadata", {}).get("train_end"),
        "validation_start": final_state.get("split_metadata", {}).get(
            "validation_start"
        ),
        "validation_end": final_state.get("split_metadata", {}).get(
            "validation_end"
        ),
        "test_start": final_state.get("split_metadata", {}).get("test_start"),
        "test_end": final_state.get("split_metadata", {}).get("test_end"),
        "semantic_split_meta": final_state.get("semantic_split_meta", {}),
        "semantic_document_filtering": final_state.get(
            "semantic_document_filtering", {}
        ),
        "timemmd_metadata": final_state.get("timemmd_metadata", {}),
        "influence_mode": final_state.get("influence_mode", "fuzzy"),
        "experiment_manifest": experiment_manifest.to_dict(),
        "mase_definition": "validation:raw_observed_training",
        **final_numeric_input_audit,
        "numeric_input_audit": final_numeric_input_audit,
        **log_transform_audit,
        "log_transform_audit": log_transform_audit,
        "semantic_reliability": fusion_output.get("semantic_trust", 0.0),
        "semantic_discriminability": fusion_output.get(
            "semantic_discriminability",
            fusion_output.get("discriminability", 0.0),
        ),
        "cross_modal_conflict": fusion_output.get("conflict", 0.0),
        "search_breadth": final_search_state.get("search_breadth", 1.0),
        "families_explored": families_explored,
        "family_coverage_count": len(families_explored),
        "best_validation_model": best_tech.get("selected_model"),
        "best_validation_mase": best_tech.get("mase"),
        "winning_model": winning_model_card,
        "winning_candidate_id": winning_candidate_id,
        "forecast_strategy": forecast_strategy,
        "best_model": winning_model_name,
        "best_hyperparameters": best_tech["hyperparameters"],
        "best_validation_score": best_tech.get("validation_score"),
        "best_rmse": best_tech.get("rmse", winning_history_entry.get("rmse")),
        "best_mase": best_tech.get("mase", winning_history_entry.get("mase")),
        "val_mase": _val_metrics.get("mase"),
        "val_rmse": _val_metrics.get("rmse"),
        "val_mse": _val_metrics.get("mse"),
        "val_mape": _val_metrics.get("mape"),
        "val_mae": _val_metrics.get("mae"),
        "val_r2": _val_metrics.get("r2"),
        "val_scaled_mse": _val_metrics.get("scaled_mse"),
        "mase_denominator_source": _val_metrics.get("mase_denominator_source"),
        "mase_denominator_common_across_candidates": _val_metrics.get(
            "mase_denominator_common_across_candidates"
        ),
        "mase_denominator_value": _val_metrics.get("mase_denominator_value"),
        "mase_reference_training_size": _val_metrics.get(
            "mase_reference_training_size"
        ),
        "search_mase_reference": _val_metrics.get("mase_reference"),
        **final_test_report,
        "validation_metrics": validation_metrics,
        "val_predictions": val_predictions_out,
        "y_true_val": y_true_val,
        "test_predictions": test_preds_final,
        "y_true_test": aligned_test_y_true,
        "test_index": aligned_test_index,
        "final_evaluation_status": final_evaluation_status,
        "performance_history": perf_history,
        "search_trace": final_state.get("search_trace", []),
        "planner_status": (
            "stop" if bool(final_state.get("done", False)) else "continue"
        ),
        "stop_reason": final_state.get("stop_reason"),
        "stop_metadata": final_state.get("stop_metadata"),
        "target_mase_reached": bool(final_state.get("target_mase_reached", False)),
        "iterations": int(final_state.get("valid_training_count", final_state.get("iteration", 0))),
        **adaptive_stop_summary(final_state),
        "valid_training_count": int(final_state.get("valid_training_count", 0)),
        "enforce_full_valid_training_budget": bool(final_state.get("enforce_full_valid_training_budget", False)),
        "budget_realization_status": (
            "completed"
            if int(final_state.get("maximum_valid_trainings", 0)) > 0
            and int(final_state.get("valid_training_count", 0))
            == int(final_state.get("maximum_valid_trainings", 0))
            else "incomplete"
        ),
        "evaluation_epsilon": float(
            config.get("evaluation", {}).get("epsilon", 0.01)
        ),
        "strategy": final_state.get("strategy", "exploration"),
        "context_regime": final_state.get("context_regime", "normal"),
        "semantic_context": final_state.get("semantic_context", {}),
        "semantic_raw_signals": final_state.get("semantic_raw_signals", []),
        "semantic_operational_signals": final_state.get(
            "semantic_operational_signals", []
        ),
        "semantic_agent_uses_analytical_output": False,
        "influence_components": {
            "semantic_enabled": bool(final_state.get("semantic_enabled", False)),
            "controller_mode": final_state.get("controller_mode", "fuzzy"),
            "reliability_mode": final_state.get("reliability_mode", "fuzzy"),
        },
        "evidence_fusion": fusion_output,
        "search_decision_state": final_state.get("search_decision_state", {}),
        "extreme_outliers_detected": final_state.get("data_summary", {}).get(
            "extreme_outliers_detected",
            final_state.get("extreme_outliers_detected", {}),
        ),
        "evaluation_integrity": evaluation_integrity,
        **evaluation_integrity,
        "llm_usage": llm.usage_metrics(),
        "seed": int(config["seed"]),
        "run_id": config.get("output", {}).get("run_id"),
        "run_output_root": str(output_root),
        "semantic_inferred_domain": final_state.get("semantic_context", {}).get(
            "inferred_domain"
        ),
        "memory": final_state.get("memory", {}),
        "technical_error": bool(final_state.get("technical_error", False)),
        "last_technical_error": final_state.get("last_technical_error"),
        "technical_retry_count": int(final_state.get("technical_retry_count", 0)),
        "technical_retry_summary": {
            "current_retry_count": int(final_state.get("technical_retry_count", 0)),
            "cumulative_retry_count": int(final_state.get("cumulative_technical_retry_count", 0)),
            "rejected_candidate_count": len(final_state.get("rejected_technical_candidates", [])),
            "rejected_before_training": sum(not bool(x.get("was_sent_to_training")) for x in final_state.get("rejected_technical_candidates", [])),
            "rejected_after_training": sum(bool(x.get("was_sent_to_training")) for x in final_state.get("rejected_technical_candidates", [])),
            "error_counts": final_state.get("candidate_error_counts", {}),
            "execution_failure_count": int(
                final_state.get("candidate_error_counts", {}).get(
                    "execution_failure", 0
                )
            ),
            "invalid_configuration_count": int(
                final_state.get("candidate_error_counts", {}).get(
                    "invalid_configuration", 0
                )
            ),
            "duplicate_candidate_count": int(
                final_state.get("candidate_error_counts", {}).get(
                    "duplicate_candidate", 0
                )
            ),
            "technical_generation_failure_count": int(
                final_state.get("candidate_error_counts", {}).get(
                    "technical_generation_failure", 0
                )
            ),
        },
        "candidate_error_counts": final_state.get(
            "candidate_error_counts", {}
        ),
        "rejected_technical_candidates": final_state.get("rejected_technical_candidates", []),
        "technical_compliance_trace": final_state.get(
            "technical_compliance_trace", []
        ),
        "final_test_evaluation_count": final_evaluator.evaluation_count,
        "test_holdout_fingerprint": test_holdout.fingerprint,
        "best_score": float(
            final_state.get(
                "best_score", best_tech.get("validation_score", float("inf"))
            )
        ),
        "exploration_score": float(final_state.get("exploration_score", 0.5)),
        "model_trust_score": float(final_state.get("model_trust_score", 0.6)),
        "planner_decision": final_state.get("planner_signals", {}),
        "conflicting_recommendations": final_state.get(
            "conflicting_recommendations", []
        ),
    }

    end_time = time.time()
    elapsed_time = end_time - start_time
    final_result["execution_time_seconds"] = elapsed_time
    write_final_outputs(
        output_root=output_root,
        final_result=final_result,
        performance_history=perf_history,
        search_trace=final_state.get("search_trace", []),
        validation_predictions=val_predictions_out,
        validation_ground_truth=y_true_val,
        test_predictions=test_preds_final,
        test_ground_truth=aligned_test_y_true,
        test_index=aligned_test_index,
        forecast_strategy=forecast_strategy,
    )

    return final_result


def main(argv: list[str] | None = None) -> int:
    """CLI entry point; importing this module never starts an experiment."""
    parser = argparse.ArgumentParser(
        description="Run the T-FUSE forecasting workflow."
    )
    parser.add_argument(
        "--dataset",
        required=True,
        type=Path,
        help="Path to the numerical/textual dataset used for this experiment.",
    )
    parser.add_argument(
        "--horizon",
        required=True,
        type=int,
        help="Positive forecasting horizon used by each forecast operation.",
    )
    parser.add_argument(
        "--split",
        required=True,
        nargs=3,
        type=float,
        metavar=("TRAIN", "VALIDATION", "TEST"),
        help="Chronological train, validation, and test proportions summing to 1.",
    )
    parser.add_argument(
        "--run-id",
        help="Optional stable run identifier. By default a unique UTC id is generated.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        help="Optional base directory for isolated run outputs.",
    )
    args = parser.parse_args(argv)
    try:
        inputs = ExperimentInputs(
            dataset_path=args.dataset,
            forecast_horizon=args.horizon,
            split=DataSplit(*args.split),
        )
        run_graph(
            inputs,
            run_id=args.run_id,
            output_root=args.output_root,
        )
    except (FileNotFoundError, ValueError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
