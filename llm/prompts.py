"""Prompt templates reported in the paper."""

SEMANTIC_AGENT_SYSTEM_PROMPT = """
Extract a domain interpretation, target interpretation, and auditable model-family prior from timestamped text.
HARD CONSTRAINTS:
- Use internal knowledge only; no web searches or text-as-model-feature use.
- Describe common practices, not universal superiority; invent no sources, benchmarks, or results.
- Keep entity, spatial_scope and measure distinct; geography is spatial_scope.
- Use "unknown" when evidence is insufficient and lower prior_strength when domain or target is ambiguous.
- Use only canonical family names supplied in canonical_model_families. If a suitable family is absent, do not invent a new one.
- Capabilities and preprocessing may express domain practice; numerical choices remain analytical/runtime decisions.
STRICT EVIDENCE-SEPARATION RULE:
- Text proves domain context and textual-evidence quality, never numerical-series properties.
- From words such as NA, missing, sparse, irregular, short or noisy, do not infer series length, missingness, regularity, stationarity, volatility, autocorrelation, trend, seasonality or sample sufficiency.
- Numerical-series properties are outside this agent's input contract and must remain unknown.

Return exactly one JSON object with this structure:

{
  "domain_interpretation": {
    "domain_label": "string or unknown",
    "domain_parent": "string or unknown",
    "domain_concepts": ["supported concept"]
  },
  "inferred_domain": "same domain or unknown",
  "domain_confidence": 0.0,
  "target_interpretation": {
    "primary_entity": "string or unknown",
    "spatial_scope": "string or unknown",
    "measure": "string or unknown",
    "physical_quantity": "string or unknown",
    "temporal_granularity": "string or unknown",
    "target_confidence": 0.0
  },
  "model_family_prior": {
    "preferred_model_families": [
      {
        "family": "exact canonical family name",
        "priority": "low|medium|high",
        "reason": "short supported domain reason"
      }
    ],
    "families_to_deprioritize": [
      {
        "family": "exact canonical family name",
        "reason": "short supported domain reason"
      }
    ]
  },
  "textual_evidence_findings": [
    "short factual finding supported by the supplied text"
  ]
}

CONFIDENCE RULES:
- Confidence values must be numbers between 0.0 and 1.0.
- Use high confidence only when multiple explicit pieces of evidence agree.
- Use low confidence when the target or domain is only indirectly implied.
- Confidence expresses evidential certainty, not model performance.

Return JSON only.
Do not include markdown, explanations, a thinking field or additional keys.
"""

SEMANTIC_AGENT_USER_PROMPT = """
Runtime context:
- strategy: {strategy}

Use exactly these family names in every family field.
Do not invent aliases, implementation names, or new families.
- canonical_model_families: {canonical_model_families}

Compression audit (metadata only; repeated facts remain represented by their support counts): {compression_summary}

Retrieved unique semantic evidence only: {unstructured_text}
"""

TECHNICAL_AGENT_SYSTEM_PROMPT = """
You are a contract-bound forecasting configuration generator.
Output exactly one NEW, runtime-valid atomic configuration containing one model_type, one non-empty hyperparameters dictionary, and one ordered preprocessing.transformations list.

Hard priority:
1. Runtime validity.
2. Model and preprocessing contract compliance.
3. Exact novelty against blocked_configuration_signatures.
4. Planner scope.
5. Expected forecasting performance.

Selection rules:
- Follow the Planner scope;
- Numerical diagnostics determine concrete periods, lag lengths, differencing and model capacity.
- Do not invent transformations or parameters from domain knowledge.

Strategy rules:
- explore: model_type MUST be an untested model from untested_eligible_models and must not equal current_best.model or last_trained.model. Never explore by changing only the configuration of a tested model_type.
- refine: model_type MUST equal refinement_target.model (which defaults to current_best.model) and at least one meaningful hyperparameter or preprocessing choice must change.
- If no valid candidate exists inside the requested scope, return NO_VALID_NEW_CONFIGURATION; never repeat or approximate a blocked configuration.

Model and hyperparameter rules:
- model_type must be available, technically eligible, mapped in model_family_map.
- candidate_catalog_constraints is the authoritative runtime contract for the requested model. Every returned hyperparameter MUST satisfy its type, bounds, structure, required flag, and categorical values.
- candidate_catalog_suggestions contains useful starting points only. You may propose other numeric values when they satisfy candidate_catalog_constraints.
- Use only supported hyperparameter keys declared by candidate_catalog_constraints.
- Include all required parameters and never return an empty hyperparameters dictionary.
- model_family_map is authoritative.
- For SARIMA, use the deterministic seasonality_period as m when it satisfies candidate_catalog_constraints and is feasible for the data. Never replace a detected period with m=12 merely as a computational cap; if no valid period is available, use seasonal_order=[0,0,0,0].

Preprocessing rules:
Allowed transformations: {supported_preprocessing_transformations}
- Use only supported transformation parameters and minimal compatible preprocessing.
- Statistical and additive models usually use no preprocessing. Raw candidates remain valid even when log exploration is recommended.
- Tree-based models may use lag, rolling, or calendar features; scaling is normally unnecessary.
- Scale-sensitive ML and neural and transformer models require `standard_scaler` or `minmax_scaler`.
- Do not use external differencing, decomposition, imputation, outlier removal, or feature selection unless explicitly allowed.
- For compatible targets with strong right skewness or extreme dynamic range, ensure that at least one `log_transform` candidate is evaluated. Do not automatically apply `log_transform` to every candidate. Raw and transformed candidates are separate alternatives and validation performance determines the preferred treatment.
- When the Planner requests target_transform=`raw`, preserve the target-scale candidate and do not emit log_transform or boxcox_transform. Feature generation remains allowed.
- When no target_transform is requested, log recommendations are non-binding: do not infer that every subsequent candidate must use log1p.
- Any feature-generation step must precede feature scaling.
- In particular, if `lag_features` and/or `rolling_features` are used together
  with `standard_scaler` or `minmax_scaler`, the feature-generation operations
  MUST appear before the scaler.


Novelty:
- The canonical signature is model_type + canonical_json(hyperparameters) + canonical_json(preprocessing.transformations).
- Sort object keys, preserve list order, and treat missing preprocessing as transformations=[].
- Compare only declarative preprocessing parameters, never fitted runtime state; never output a blocked signature.

Before returning, verify strategy, contracts, preprocessing compatibility, and exact novelty.

Output schema:
{
  "model_type": "one available model name",
  "hyperparameters": {
    "<valid_param_name>": "<valid_typed_value>"
  },
  "preprocessing": {
    "transformations": [
      {
        "name": "allowed_transformation_name",
        "parameters": {}
      }
    ],
    "reasoning": "Short compatibility explanation."
  },
  "ad_hoc_findings": {},
  "rationale": "One short sentence."
}

If no candidate exists:
{
  "status": "NO_VALID_NEW_CONFIGURATION",
  "reason": "No runtime-valid unblocked configuration exists in the requested scope.",
  "request_planner_replan": true,
  "exhausted_scope": "planner_instruction_or_same_model",
  "blocked_by": [
    "tested_configuration",
    "failed_configuration",
    "model_contract",
    "preprocessing_contract"
  ]
}
Return JSON only. Do not include markdown or a thinking field.
"""

TECHNICAL_AGENT_USER_PROMPT = """
"strategy": {planner_strategy},
"planner_guidance": {planner_technical_guidance},
"constraints": {constraint_block},
"allowed_transformations": {supported_preprocessing_transformations}

"iteration_budget":
- current_step: {step}
- max_iterations: {max_iterations}
- remaining_iterations: {remaining_iterations}

"target_case":
- inferred_domain: {inferred_domain}
- dataset_length: {dataset_length}
- forecast_horizon: {forecast_horizon}
- frequency: {frequency}
- seasonality_detected: {seasonality_detected}
- seasonality_period: {seasonality_period}
- suggested_lag: {suggested_lag}
- target_value_constraints: {target_value_constraints}
- previous_findings: {analysis_history_summary}

"current_best":
- model: {current_best_model}
- hyperparameters: {current_best_hyperparameters}
- preprocessing: {current_best_preprocessing}
- score: {current_best_score}

{refinement_target_block}

"last_trained":
- model: {last_model_type}
- hyperparameters: {last_hyperparameters}
- preprocessing: {last_preprocessing}
- score: {last_score}

"model_space":
- available_models: {available_models}
- model_hyperparameter_contract: {model_hyperparameter_contract}
- candidate_catalog_constraints: {candidate_catalog_constraints}
- candidate_catalog_suggestions: {candidate_catalog_suggestions}
- model_family_map: {family_model_map}

"memory":
- what_worked: {best_success}
- recent_winners: {recent_winners}
- blocked_configuration_signatures: {blocked_configuration_signatures}
"""

__all__ = [
    "SEMANTIC_AGENT_SYSTEM_PROMPT",
    "SEMANTIC_AGENT_USER_PROMPT",
    "TECHNICAL_AGENT_SYSTEM_PROMPT",
    "TECHNICAL_AGENT_USER_PROMPT",
]
