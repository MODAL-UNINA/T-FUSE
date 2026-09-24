import copy
from typing import Any
from utils.semantic_evidence_validator import is_operational_semantic_entry

def create_empty_semantic_context(
    *,
    status: str = "not_run",
    finding: str | None = None,
) -> dict[str, Any]:
    """Create an empty, safe default semantic context adhering to contract 3.0."""
    ctx = {
        "semantic_status": status,
        "inferred_domain": "unknown",
        "inferred_subdomain": "unknown",
        "domain_confidence": 0.0,
        "domain_interpretation": {
            "domain_label": "unknown",
            "domain_parent": "unknown",
            "domain_concepts": [],
            "domain_confidence": 0.0,
            "domain_agreement": 0.0,
            "alternative_domains": []
        },
        "target_interpretation": {
            "target_label": "unknown",
            "primary_entity": "unknown",
            "spatial_scope": "unknown",
            "measure": "unknown",
            "physical_quantity": "unknown",
            "value_type": "unknown",
            "task_type": "unknown",
            "temporal_granularity": "unknown",
            "unit_candidates": [],
            "alternative_entities": [],
            "alternative_spatial_scopes": [],
            "target_confidence": 0.0,
            "likely_target": "unknown",
            "quantity_type": "unknown",
            "target_resolution_status": "unavailable",
            "primary_hypothesis": None,
            "alternative_hypotheses": [],
            "target_quantity_conflict": 0.0,
        },
        "family_conflicts": [],
        "family_support": {},
        "model_family_prior": {
            "preferred_model_families": [],
            "families_to_deprioritize": [],
            "preferred_capabilities": [],
            "preprocessing_prior": [],
            "raw_prior_strength": "none",
            "prior_strength": "none"
        },
        "knowledge_limits": {
            "target_ambiguity": "high",
            "domain_prior_reliability": "low",
            "assumptions": []
        },
        "evidence_summary": {
            "documents_available": 0,
            "documents_selected": 0,
            "documents_seen": 0,
            "chunks_available": 0,
            "chunks_processed": 0,
            "chunks_succeeded": 0,
            "chunks_failed": 0,
            "evidence_coverage": 0.0,
            "selection_coverage": 0.0,
            "source_date_start": "",
            "source_date_end": "",
            "train_documents_only": True
        },
        "aggregation_summary": {
            "method": "semantic_clustering_and_weighted_voting",
            "chunk_agreement": 0.0,
            "aggregated_domain_confidence": 0.0,
            "domain_agreement": 0.0,
            "target_confidence": 0.0,
            "target_agreement": 0.0,
            "target_reliability": 0.0,
            "target_reliability_source": "confidence_times_agreement",
            "target_quantity_conflict": 0.0,
            "target_resolution_status": "unavailable",
            "target_resolution_score": 0.0,
            "target_resolution_audit": {},
            "target_resolution_consistency_audit": None,
            "raw_aggregation_reliability": 0.0,
            "reliability_score": 0.0,
            "family_conflict_ratio": 0.0,
            "embedding_backend": "tfidf",
            "similarity_threshold": 0.80
        },
        "ad_hoc_findings": [],
        "textual_evidence_findings": [],
        "numerical_series_claims": [],
        "unsupported_numerical_claims": [],
        "semantic_validation_summary": {
            "total_entries_checked": 0,
            "accepted_entries": 0,
            "partially_accepted_entries": 0,
            "excluded_entries": 0,
            "excluded_positive_family_votes": 0,
            "excluded_negative_family_votes": 0,
            "excluded_preprocessing_entries": 0,
        },
        "semantic_validation_status": "unavailable"
    }
    
    if finding:
        # Framework diagnostics are retained for audit but are never semantic
        # evidence and therefore cannot affect any operational aggregation.
        ctx["ad_hoc_findings"].append({
            "value": str(finding),
            "validation_status": "excluded",
            "operational": False,
            "exclusion_reason": "framework diagnostic, not semantic evidence",
            "claim_category": None,
            "action": "excluded_from_operational_support",
        })
        
    return ctx

def first_known_semantic_value(*values: object, default: str = "unknown") -> str:
    """Returns the first non-empty string that is not unknown/none/null."""
    for v in values:
        if v and isinstance(v, str):
            v_lower = v.lower()
            if v_lower not in ("", "unknown", "none", "null"):
                return v
    return default

def _priority_to_weight(priority: str) -> float:
    priority = str(priority).lower()
    if priority == "high":
        return 3.0
    if priority == "medium":
        return 2.0
    if priority == "low":
        return 1.0
    return 0.0

def _weight_to_priority(weight: float) -> str:
    if weight >= 2.5:
        return "high"
    if weight >= 1.5:
        return "medium"
    return "low"

def aggregate_semantic_signals(
    signals: list[Any],
    *,
    documents_available: int,
    documents_selected: int,
    documents_seen: int,
    chunks_available: int,
    chunks_processed: int,
    chunks_succeeded: int | None = None,
    chunks_failed: int | None = None,
    semantic_aggregation_config: dict[str, Any] | None = None,
    semantic_target_resolution_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Deterministically aggregate semantic signals."""
    from utils.model_family_state import CANONICAL_FAMILY_ORDER, canonicalize_family_label
    from utils.semantic_clustering import SemanticTextEncoder, weighted_semantic_field_vote, build_target_label, build_likely_target
    from utils.semantic_normalization import normalize_semantic_text
    
    TRANSFORMATION_CANONICAL = {
        "log": "log_transform",
        "log transform": "log_transform",
        "log transformation": "log_transform",
        "log_transform": "log_transform",
    
        "seasonal decomposition": "seasonal_decomposition",
        "seasonal_decompose": "seasonal_decomposition",
        "seasonal_decomposition": "seasonal_decomposition",
    
        "seasonal differencing": "seasonal_differencing",
        "seasonal_difference": "seasonal_differencing",
        "seasonal_differencing": "seasonal_differencing",
    
        "normalize": "normalization",
        "normalisation": "normalization",
        "normalization": "normalization",
    
        "difference": "differencing",
        "differencing": "differencing",
    
        "none": "none",
    }
    
    def canonicalize_transformation(t: str) -> str:
        norm = normalize_semantic_text(t)
        return TRANSFORMATION_CANONICAL.get(norm, t)
    
    VALUE_TYPE_CANONICAL = {
        "continuous": "continuous",
        "continuous value": "continuous",
        "numeric continuous": "continuous",
        "count": "count",
        "integer count": "count",
        "binary": "binary",
        "categorical": "categorical",
        "index": "index",
        "rate": "rate",
    }
    
    encoder = SemanticTextEncoder()
    ctx = create_empty_semantic_context(status="aggregated")
    ctx["aggregation_summary"]["embedding_backend"] = encoder.backend
    
    # Calculate Coverages
    # ``documents_seen`` is the number of eligible documents represented by
    # valid operational extraction results, not merely sent to the LLM.
    evidence_coverage = (documents_seen / documents_available) if documents_available > 0 else 0.0
    evidence_coverage = max(0.0, min(1.0, evidence_coverage))
    
    selection_coverage = (documents_seen / documents_selected) if documents_selected > 0 else 0.0
    selection_coverage = max(0.0, min(1.0, selection_coverage))
    
    if chunks_succeeded is None:
        chunks_succeeded = chunks_processed
    if chunks_failed is None:
        chunks_failed = 0
    
    if not signals:
        ctx["semantic_status"] = "unavailable"
        ctx["evidence_summary"].update({
            "documents_available": documents_available,
            "documents_selected": documents_selected,
            "documents_seen": documents_seen,
            "chunks_available": chunks_available,
            "chunks_processed": chunks_processed,
            "chunks_succeeded": chunks_succeeded,
            "chunks_failed": chunks_failed,
            "evidence_coverage": evidence_coverage,
            "selection_coverage": selection_coverage,
        })
        return ctx

    domain_labels = []
    domain_parents = []
    domain_confidences = []
    observation_weights = []
    
    domain_representations = []
    domain_concept_sets = []
    
    primary_entities = []
    spatial_scopes = []
    measures = []
    physical_quantities = []
    value_types = []
    task_types = []
    temporal_granularities = []
    target_confidences = []
    
    # family support
    positive_support = {}
    negative_support = {}
    
    capabilities = {}
    prep_priors = {}
    
    ad_hoc = []
    assumptions = []
    textual_findings = []
    numerical_claims = []
    unsupported_claims = []
    validation_summary = copy.deepcopy(ctx["semantic_validation_summary"])
    
    min_date = None
    max_date = None
    
    for sig in signals:
        if isinstance(sig, dict):
            s = sig
        else:
            s = sig.model_dump() if hasattr(sig, "model_dump") else getattr(sig, "__dict__", {})
            
        dom = s.get("domain_interpretation", {})
        tgt = s.get("target_interpretation", {})
            
        conf = float(s.get("domain_confidence", 0.0))
        conf = float(s.get("domain_confidence", 0.0))
        conf = max(0.0, min(1.0, conf))
        obs = float(s.get("text_obs", 1))
        obs = max(1.0, obs)
        weight = conf * obs
        
        # Domain (use normalized first_known_semantic_value as done in normalize)
        dom = s.get("domain_interpretation", {})
        dom_label = first_known_semantic_value(
            dom.get("domain_label"),
            s.get("inferred_domain"),
            default="unknown"
        )
        dom_parent = dom.get("domain_parent", "unknown")
        
        domain_labels.append(dom_label)
        domain_parents.append(dom_parent)
        domain_confidences.append(conf)
        observation_weights.append(obs)
        
        dom_concepts = set(normalize_semantic_text(c) for c in dom.get("domain_concepts", []))
        domain_concept_sets.append(dom_concepts)
        
        rep = f"{normalize_semantic_text(dom_label)} {normalize_semantic_text(dom_parent)} {' '.join(sorted(dom_concepts))}"
        domain_representations.append(rep)
        
        # Target
        tgt = s.get("target_interpretation", {})
        pe = tgt.get("primary_entity", "unknown")
        primary_entities.append(pe)
        
        ss = tgt.get("spatial_scope", "unknown")
        spatial_scopes.append(ss)
        
        me = tgt.get("measure", "unknown")
        measures.append(me)
        
        physical_quantities.append(tgt.get("physical_quantity", "unknown"))
        value_types.append(tgt.get("value_type", "unknown"))
        task_types.append(tgt.get("task_type", "unknown"))
        temporal_granularities.append(tgt.get("temporal_granularity", "unknown"))
        
        t_conf = float(tgt.get("target_confidence", 0.0))
        t_conf = max(0.0, min(1.0, t_conf))
        target_confidences.append(t_conf)
            
        # Prior extraction
        prior = s.get("model_family_prior") or s.get("internal_knowledge_prior") or {}
        
        for pref in prior.get("preferred_model_families", []):
            if not is_operational_semantic_entry(pref):
                continue
            if isinstance(pref, dict):
                f_raw = pref.get("family", "")
                priority_weight = {"low": 1.0, "medium": 2.0, "high": 3.0}.get(str(pref.get("priority", "")).lower(), 2.0)
            else:
                f_raw = pref
                priority_weight = 2.0
            f_can = canonicalize_family_label(f_raw)
            if f_can:
                positive_support[f_can] = positive_support.get(f_can, 0.0) + weight * priority_weight
                
        for dep in prior.get("families_to_deprioritize", []):
            if not is_operational_semantic_entry(dep):
                continue
            if isinstance(dep, dict):
                f_raw = dep.get("family", "")
            else:
                f_raw = dep
            f_can = canonicalize_family_label(f_raw)
            if f_can:
                negative_support[f_can] = negative_support.get(f_can, 0.0) + weight
                
        for cap in prior.get("preferred_capabilities", []):
            if not is_operational_semantic_entry(cap):
                continue
            cap_value = (
                cap.get("value", cap.get("capability", ""))
                if isinstance(cap, dict) else cap
            )
            if cap_value:
                capabilities[str(cap_value)] = capabilities.get(str(cap_value), 0.0) + weight
            
        for prep in prior.get("preprocessing_prior", []):
            if not is_operational_semantic_entry(prep):
                continue
            if isinstance(prep, dict):
                tr = prep.get("transformation", "")
                p_val = _priority_to_weight(prep.get("priority", "medium"))
                if tr:
                    can_tr = canonicalize_transformation(tr)
                    if can_tr not in prep_priors:
                        prep_priors[can_tr] = {"weight": 0.0, "families": set()}
                    prep_priors[can_tr]["weight"] += p_val * weight
                    prep_priors[can_tr]["families"].update(prep.get("applicable_families", []))

        # Ad hoc and limits
        limits = s.get("knowledge_limits", {})
        for a in limits.get("assumptions", []):
            if a not in assumptions:
                assumptions.append(a)
                
        for f in s.get("ad_hoc_findings", []):
            if not is_operational_semantic_entry(f):
                continue
            finding_value = (
                f.get("value", f.get("finding", ""))
                if isinstance(f, dict) else f
            )
            if finding_value and finding_value not in ad_hoc:
                ad_hoc.append(finding_value)
        for finding in s.get("textual_evidence_findings", []):
            if isinstance(finding, str) and finding not in textual_findings:
                textual_findings.append(finding)
        for claim in s.get("numerical_series_claims", []):
            if isinstance(claim, dict):
                numerical_claims.append(copy.deepcopy(claim))
        for violation in s.get("unsupported_numerical_claims", []):
            if isinstance(violation, dict):
                unsupported_claims.append(copy.deepcopy(violation))
        signal_summary = s.get("semantic_validation_summary", {})
        if isinstance(signal_summary, dict):
            for key in validation_summary:
                try:
                    validation_summary[key] += max(0, int(signal_summary.get(key, 0)))
                except (TypeError, ValueError):
                    continue
                
        # Dates
        d_start = s.get("source_date_start", "")
        if d_start:
            if not min_date or d_start < min_date:
                min_date = d_start
        d_end = s.get("source_date_end", "")
        if d_end:
            if not max_date or d_end > max_date:
                max_date = d_end

    # Resolution
    
    config = semantic_aggregation_config or {}
    label_weight = float(config.get("domain_label_weight", 0.60))
    context_weight = float(config.get("domain_context_weight", 0.40))
    min_concept_overlap = float(config.get("min_domain_concept_overlap", 0.15))
    alt_support_thresh = float(config.get("alternative_domain_support_threshold", 0.10))

    # 1. Domain
    dom_result = weighted_semantic_field_vote(
        domain_labels, confidences=domain_confidences, observation_weights=observation_weights, encoder=encoder,
        representations=domain_representations, parents=domain_parents, concept_sets=domain_concept_sets,
        label_weight=label_weight, context_weight=context_weight, 
        min_concept_overlap=min_concept_overlap, alternative_support_threshold=alt_support_thresh
    )
    
    dom_parent_result = weighted_semantic_field_vote(
        domain_parents, confidences=domain_confidences, observation_weights=observation_weights, encoder=encoder, exact_match=True
    )
    
    # Process concepts using only the winning cluster
    concept_support = {}
    total_winning_cluster_weight = 0.0
    
    for i in dom_result.winning_indices:
        s = signals[i]
        if not isinstance(s, dict):
            s = s.model_dump() if hasattr(s, "model_dump") else getattr(s, "__dict__", {})
        dom = s.get("domain_interpretation", {})
        
        conf = float(s.get("domain_confidence", 0.0))
        conf = max(0.0, min(1.0, conf))
        obs = float(s.get("text_obs", 1))
        obs = max(1.0, obs)
        weight = conf * obs
        
        total_winning_cluster_weight += weight
        
        unique_concepts = {
            normalize_semantic_text(c) 
            for c in dom.get("domain_concepts", []) 
            if normalize_semantic_text(c) != "unknown"
        }
        
        for c in unique_concepts:
            concept_support[c] = concept_support.get(c, 0.0) + weight
            
    valid_concepts = []
    if total_winning_cluster_weight > 0:
        for c, w in concept_support.items():
            if w / total_winning_cluster_weight >= 0.30:
                valid_concepts.append((c, w / total_winning_cluster_weight))
    # sort by weight desc, then string
    valid_concepts.sort(key=lambda x: (-x[1], x[0]))
    top_concepts = [c[0] for c in valid_concepts[:8]]

    ctx["domain_interpretation"].update({
        "domain_label": dom_result.value,
        "domain_parent": dom_parent_result.value,
        "domain_concepts": top_concepts,
        "domain_confidence": dom_result.confidence,
        "domain_agreement": dom_result.agreement,
        "alternative_domains": dom_result.alternative_values
    })
    
    # Inferred subdomain
    winning_subdomains = []
    winning_subdomain_confidences = []
    winning_subdomain_weights = []
    for i in dom_result.winning_indices:
        s = signals[i]
        if not isinstance(s, dict):
            s = s.model_dump() if hasattr(s, "model_dump") else getattr(s, "__dict__", {})
        
        conf = float(s.get("domain_confidence", 0.0))
        conf = max(0.0, min(1.0, conf))
        obs = float(s.get("text_obs", 1))
        obs = max(1.0, obs)
        
        sub = s.get("inferred_subdomain", "unknown")
        winning_subdomains.append(sub)
        winning_subdomain_confidences.append(conf)
        winning_subdomain_weights.append(obs)
        
    subdomain_result = weighted_semantic_field_vote(
        winning_subdomains, confidences=winning_subdomain_confidences, observation_weights=winning_subdomain_weights, encoder=encoder
    )
    
    ctx["inferred_domain"] = dom_result.value
    ctx["inferred_subdomain"] = subdomain_result.value if subdomain_result.value else "unknown"
    ctx["domain_confidence"] = dom_result.confidence
    
    # 2. Target
    pe_res = weighted_semantic_field_vote(
        primary_entities, confidences=target_confidences, observation_weights=observation_weights, encoder=encoder
    )
    ss_res = weighted_semantic_field_vote(
        spatial_scopes, confidences=target_confidences, observation_weights=observation_weights, encoder=encoder
    )
    me_res = weighted_semantic_field_vote(
        measures, confidences=target_confidences, observation_weights=observation_weights, encoder=encoder
    )
    pq_res = weighted_semantic_field_vote(
        physical_quantities, confidences=target_confidences, observation_weights=observation_weights, encoder=encoder
    )
    vt_res = weighted_semantic_field_vote(
        value_types, confidences=target_confidences, observation_weights=observation_weights, encoder=encoder, exact_match=True
    )
    tt_res = weighted_semantic_field_vote(
        task_types, confidences=target_confidences, observation_weights=observation_weights, encoder=encoder, exact_match=True
    )
    tg_res = weighted_semantic_field_vote(
        temporal_granularities, confidences=target_confidences, observation_weights=observation_weights, encoder=encoder, exact_match=True
    )
    
    final_target_label = build_target_label(pe_res.value, me_res.value, tg_res.value)
    
    likely_target = build_likely_target(
        primary_entity=pe_res.value,
        measure=me_res.value,
        temporal_granularity=tg_res.value
    )
    
    norm_val_type = normalize_semantic_text(vt_res.value)
    quantity_type = VALUE_TYPE_CANONICAL.get(norm_val_type, "unknown")
    
    # Approx target agreement as mean of core fields
    target_agreement = (pe_res.agreement + me_res.agreement + tt_res.agreement) / 3.0
    
    # Collect units and alternatives from winning target signals
    target_winning_indices = pe_res.winning_indices
    total_target_weight = 0.0
    unit_support = {}
    alternative_entities_support = {}
    alternative_scopes_support = {}
    
    pe_res_norm = normalize_semantic_text(pe_res.value)
    ss_res_norm = normalize_semantic_text(ss_res.value)
    likely_target_norm = normalize_semantic_text(likely_target)
    
    for i in target_winning_indices:
        s = signals[i]
        if not isinstance(s, dict):
            s = s.model_dump() if hasattr(s, "model_dump") else getattr(s, "__dict__", {})
            
        tgt = s.get("target_interpretation", {})
        t_conf = float(tgt.get("target_confidence", 0.0))
        t_conf = max(0.0, min(1.0, t_conf))
        obs = float(s.get("text_obs", 1))
        obs = max(1.0, obs)
        w = t_conf * obs
        
        total_target_weight += w
        
        for u in tgt.get("unit_candidates", []):
            nu = normalize_semantic_text(u)
            if nu and nu != "unknown":
                unit_support[nu] = unit_support.get(nu, 0.0) + w
                
        for ae in tgt.get("alternative_entities", []):
            nae = normalize_semantic_text(ae)
            if nae and nae != "unknown" and nae != pe_res_norm and nae != likely_target_norm:
                alternative_entities_support[nae] = alternative_entities_support.get(nae, 0.0) + w
                
        for asc in tgt.get("alternative_spatial_scopes", []):
            nasc = normalize_semantic_text(asc)
            if nasc and nasc != "unknown" and nasc != ss_res_norm:
                alternative_scopes_support[nasc] = alternative_scopes_support.get(nasc, 0.0) + w

    final_units = []
    final_alt_entities = []
    final_alt_scopes = []
    
    if total_target_weight > 0:
        for k, sw in unit_support.items():
            if sw / total_target_weight >= 0.20:
                final_units.append((k, sw))
        for k, sw in alternative_entities_support.items():
            if sw / total_target_weight >= 0.20:
                final_alt_entities.append((k, sw))
        for k, sw in alternative_scopes_support.items():
            if sw / total_target_weight >= 0.20:
                final_alt_scopes.append((k, sw))
                
    final_units.sort(key=lambda x: (-x[1], x[0]))
    final_alt_entities.sort(key=lambda x: (-x[1], x[0]))
    final_alt_scopes.sort(key=lambda x: (-x[1], x[0]))
    
    ctx["target_interpretation"].update({
        "target_label": final_target_label,
        "primary_entity": pe_res.value,
        "spatial_scope": ss_res.value,
        "measure": me_res.value,
        "physical_quantity": pq_res.value,
        "value_type": vt_res.value,
        "task_type": tt_res.value,
        "temporal_granularity": tg_res.value,
        "target_confidence": pe_res.confidence,
        "unit_candidates": [u[0] for u in final_units],
        "alternative_entities": [ae[0] for ae in final_alt_entities[:5]],
        "alternative_spatial_scopes": [asc[0] for asc in final_alt_scopes[:5]],
        
        "likely_target": likely_target,
        "quantity_type": quantity_type
    })
    
    # 3. Family Support
    final_pref = []
    final_dep = []
    conflicts = []
    
    all_families = set(positive_support.keys()) | set(negative_support.keys())
    raw_family_support = {
        family: float(
            positive_support.get(family, 0.0)
            - negative_support.get(family, 0.0)
        )
        for family in CANONICAL_FAMILY_ORDER
    }
    from utils.semantic_normalization import normalize_family_support

    normalized_family_support = normalize_family_support(raw_family_support)

    for f in CANONICAL_FAMILY_ORDER:
        p = positive_support.get(f, 0.0)
        n = negative_support.get(f, 0.0)
        raw = raw_family_support[f]
        normalized = normalized_family_support[f]
        
        ctx["family_support"][f] = {
            "positive": float(p),
            "negative": float(n),
            "raw_support": float(raw),
            "normalized_support": float(normalized),
            # Compatibility alias for legacy audit consumers. Unlike the old
            # per-family ratio, this preserves cross-family discrimination.
            "net": float(normalized),
            "evidence_available": f in all_families,
            "normalization": "cross_family_max",
        }

        if f not in all_families:
            continue
        
        if normalized >= 0.25:
            final_pref.append({
                "family": f,
                "priority": "medium",
                "reason": "Net positive validated support",
                "validation_status": "accepted",
                "operational": True,
                "exclusion_reason": None,
                "claim_category": None,
            })
        elif raw < 0.0:
            final_dep.append({
                "family": f,
                "reason": "Net negative validated support",
                "validation_status": "accepted",
                "operational": True,
                "exclusion_reason": None,
                "claim_category": None,
            })
        else:
            conflicts.append(f)
            
    ctx["family_conflicts"] = conflicts
    ctx["model_family_prior"]["preferred_model_families"] = final_pref
    ctx["model_family_prior"]["families_to_deprioritize"] = final_dep
    
    family_conflict_ratio = len(conflicts) / len(all_families) if all_families else 0.0
    ctx["aggregation_summary"]["family_conflict_ratio"] = float(family_conflict_ratio)
    
    # 4. Misc
    ctx["model_family_prior"]["preferred_capabilities"] = [{
        "value": key,
        "validation_status": "accepted",
        "operational": True,
        "exclusion_reason": None,
        "claim_category": None,
    } for key, _ in sorted(
        capabilities.items(), key=lambda item: (-item[1], item[0])
    )[:5]]
    
    norm_factor = sum(observation_weights) if sum(observation_weights) > 0 else 1.0
    final_prep = []
    for k, v in prep_priors.items():
        w = v["weight"] / norm_factor
        if w > 0:
            final_prep.append({
                "transformation": k,
                "applicable_families": list(v["families"]),
                "priority": _weight_to_priority(w),
                "reason": "Aggregated from validated semantic chunks",
                "validation_status": "accepted",
                "operational": True,
                "exclusion_reason": None,
                "claim_category": None,
            })
    ctx["model_family_prior"]["preprocessing_prior"] = final_prep
    
    if final_pref and documents_seen > 0:
        base_score = (
            0.50 * dom_result.confidence
            + 0.30 * dom_result.agreement
            + 0.20 * evidence_coverage
        )
        
        if family_conflict_ratio > 0.5:
            base_score *= 0.5
            
        reliability_score = base_score * dom_result.confidence
        
        # Invariants
        if dom_result.confidence == 0.0 or evidence_coverage == 0.0:
            raw_prior_strength = "none"
        elif pe_res.agreement < 0.3:
            # high ambiguity
            raw_prior_strength = "none"
        elif reliability_score >= 0.70:
            raw_prior_strength = "high"
        elif reliability_score >= 0.40:
            raw_prior_strength = "medium"
        elif reliability_score > 0.0:
            raw_prior_strength = "low"
        else:
            raw_prior_strength = "none"
            
        ctx["model_family_prior"]["raw_prior_strength"] = raw_prior_strength
        ctx["model_family_prior"]["prior_strength"] = raw_prior_strength
    else:
        reliability_score = 0.0
        ctx["model_family_prior"]["raw_prior_strength"] = "none"
        ctx["model_family_prior"]["prior_strength"] = "none"
        
    ctx["aggregation_summary"]["raw_aggregation_reliability"] = float(reliability_score)
    ctx["aggregation_summary"]["reliability_score"] = float(reliability_score)
    
    ctx["aggregation_summary"].update({
        "aggregated_domain_confidence": dom_result.confidence,
        "domain_agreement": dom_result.agreement,
        "chunk_agreement": dom_result.agreement,
        "target_confidence": pe_res.confidence,
        "target_agreement": target_agreement
    })
    
    ctx["ad_hoc_findings"] = [{
        "value": value,
        "validation_status": "accepted",
        "operational": True,
        "exclusion_reason": None,
        "claim_category": None,
    } for value in ad_hoc]
    ctx["textual_evidence_findings"] = textual_findings
    ctx["numerical_series_claims"] = numerical_claims
    ctx["unsupported_numerical_claims"] = sorted(
        unsupported_claims,
        key=lambda item: (
            int(item.get("source_chunk_index", -1)), str(item.get("field", "")),
            str(item.get("family", "")), str(item.get("claim_category", "")),
        ),
    )
    ctx["semantic_validation_summary"] = validation_summary
    ctx["semantic_validation_status"] = "filtered" if unsupported_claims else "validated"
    ctx["knowledge_limits"]["assumptions"] = assumptions
    
    ctx["evidence_summary"].update({
        "documents_available": documents_available,
        "documents_selected": documents_selected,
        "documents_seen": documents_seen,
        "chunks_available": chunks_available,
        "chunks_processed": chunks_processed,
        "chunks_succeeded": chunks_succeeded,
        "chunks_failed": chunks_failed,
        "evidence_coverage": evidence_coverage,
        "selection_coverage": selection_coverage,
        "source_date_start": min_date or "",
        "source_date_end": max_date or "",
        "train_documents_only": True
    })
    
    if reliability_score >= 0.70:
        domain_prior_reliability = "high"
    elif reliability_score >= 0.40:
        domain_prior_reliability = "medium"
    else:
        domain_prior_reliability = "low"
    ctx["knowledge_limits"]["domain_prior_reliability"] = domain_prior_reliability

    target_reliability = max(0.0, min(1.0, pe_res.confidence * target_agreement))
    ctx["aggregation_summary"]["target_reliability"] = target_reliability
    ctx["aggregation_summary"]["target_reliability_source"] = "confidence_times_agreement"

    from utils.semantic_target_resolution import (
        resolve_target_quantity,
        validate_target_resolution_consistency,
    )

    target_resolution = resolve_target_quantity(
        signals,
        target_confidence=pe_res.confidence,
        target_agreement=target_agreement,
        config=semantic_target_resolution_config,
        encoder=encoder,
    )
    primary_hypothesis = target_resolution["primary_hypothesis"]
    ctx["target_interpretation"].update({
        "target_resolution_status": target_resolution["target_resolution_status"],
        "primary_hypothesis": primary_hypothesis,
        "alternative_hypotheses": target_resolution["alternative_hypotheses"],
        "target_quantity_conflict": target_resolution["target_quantity_conflict"],
    })
    if isinstance(primary_hypothesis, dict):
        primary_measure = str(primary_hypothesis.get("measure", "unknown"))
        primary_quantity = str(
            primary_hypothesis.get("physical_quantity", "unknown")
        )
        primary_likely_target = build_likely_target(
            primary_entity=pe_res.value,
            measure=primary_measure,
            temporal_granularity=tg_res.value,
        )
        ctx["target_interpretation"].update({
            "measure": primary_measure,
            "physical_quantity": primary_quantity,
            "likely_target": primary_likely_target,
            "target_label": build_target_label(
                pe_res.value, primary_measure, tg_res.value
            ),
        })
    ctx["aggregation_summary"].update({
        "target_quantity_conflict": target_resolution["target_quantity_conflict"],
        "target_resolution_status": target_resolution["target_resolution_status"],
        "target_resolution_score": target_resolution["target_resolution_score"],
        "target_resolution_audit": {
            key: copy.deepcopy(target_resolution[key])
            for key in (
                "primary_hypothesis_support", "second_hypothesis_support",
                "alternative_total_support", "hypothesis_margin",
                "number_of_material_alternatives",
                "material_alternative_threshold", "classification_config",
                "quantity_resolution_method", "quantity_evidence_chunk_count",
                "quantity_total_chunk_count", "quantity_evidence_coverage",
            )
        },
    })
    target_ambiguity, consistency_audit = validate_target_resolution_consistency(
        target_resolution, ctx["knowledge_limits"].get("target_ambiguity", "high")
    )
    ctx["knowledge_limits"]["target_ambiguity"] = target_ambiguity
    ctx["aggregation_summary"]["target_resolution_consistency_audit"] = (
        consistency_audit
    )

    return ctx
