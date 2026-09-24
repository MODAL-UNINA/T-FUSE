"""Semantic agent – dataset-aware semantic signal auditor and extractor.

"""

from __future__ import annotations

import json
import logging
import re
import copy
from dataclasses import dataclass
from typing import Any, Dict, List, Literal

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from llm.prompts import SEMANTIC_AGENT_SYSTEM_PROMPT, SEMANTIC_AGENT_USER_PROMPT
from utils.semantic_contract import (
    aggregate_semantic_signals,
    create_empty_semantic_context,
)
from utils.semantic_family_space import (
    derive_canonical_family_space,
    normalize_signal_families,
)

logger = logging.getLogger(__name__)

class _PaperOutputModel(BaseModel):
    """Base model that rejects every field not declared in the paper prompt."""

    model_config = ConfigDict(extra="forbid")


class TargetInterpretation(_PaperOutputModel):
    primary_entity: str
    spatial_scope: str
    measure: str
    physical_quantity: str
    temporal_granularity: str
    target_confidence: float = Field(ge=0.0, le=1.0)


class DomainInterpretation(_PaperOutputModel):
    domain_label: str
    domain_parent: str
    domain_concepts: list[str]


class PreferredModelFamily(_PaperOutputModel):
    family: str
    priority: Literal["low", "medium", "high"]
    reason: str


class DeprioritizedModelFamily(_PaperOutputModel):
    family: str
    reason: str


class ModelFamilyPrior(_PaperOutputModel):
    preferred_model_families: List[PreferredModelFamily]
    families_to_deprioritize: List[DeprioritizedModelFamily]


class SemanticSignal(_PaperOutputModel):
    """Exact Semantic Agent JSON contract reported in the paper."""

    domain_interpretation: DomainInterpretation
    inferred_domain: str
    domain_confidence: float = Field(ge=0.0, le=1.0)
    target_interpretation: TargetInterpretation
    model_family_prior: ModelFamilyPrior
    textual_evidence_findings: List[str]


@dataclass(frozen=True)
class ChunkExtractionResult:
    """Explicit result of one semantic extraction call."""

    success: bool
    operational_signals: tuple[SemanticSignal, ...]
    excluded_signals: tuple[Dict[str, Any], ...]
    documents_covered: int
    error: str | None
    source_reliability: float
    evidence_specificity: float
    entity_consistency: float
    domain_target_quantity_coherence: float


@dataclass(frozen=True)
class SemanticCompressionAudit:
    documents_input: int
    documents_with_unique_evidence: int
    facts_input: int
    unique_facts: int
    exact_duplicates_removed: int
    near_duplicates_removed: int
    original_characters: int
    compacted_characters: int
    compression_ratio: float
    near_duplicate_backend: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "documents_input": self.documents_input,
            "documents_with_unique_evidence": self.documents_with_unique_evidence,
            "facts_input": self.facts_input,
            "unique_facts": self.unique_facts,
            "exact_duplicates_removed": self.exact_duplicates_removed,
            "near_duplicates_removed": self.near_duplicates_removed,
            "duplicate_facts_removed": (
                self.exact_duplicates_removed + self.near_duplicates_removed
            ),
            "original_characters": self.original_characters,
            "compacted_characters": self.compacted_characters,
            "compression_ratio": self.compression_ratio,
            "near_duplicate_backend": self.near_duplicate_backend,
        }


def _json_dumps_safe(value: Any) -> str:
    try:
        return json.dumps(value, ensure_ascii=False, default=str)
    except Exception:
        return str(value)


def _first_present(rec: Dict[str, Any], keys: List[str], default: str = "") -> str:
    for key in keys:
        value = rec.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    return default


def _date_col(df: pd.DataFrame) -> str | None:
    return "date" if "date" in df.columns else None


def _is_operational_signal(signal: SemanticSignal) -> bool:
    prior = signal.model_family_prior
    entries = [
        *prior.preferred_model_families,
        *prior.families_to_deprioritize,
    ]
    return bool(
        signal.domain_confidence > 0.0
        or signal.target_interpretation.target_confidence > 0.0
        or signal.textual_evidence_findings
        or any(bool(entry.family.strip()) for entry in entries)
    )


def _excluded_entries(signal: SemanticSignal) -> list[Dict[str, Any]]:
    # Schema-invalid chunks are excluded before a SemanticSignal is created.
    return []


def _chunk_quality(
    chunk_df: pd.DataFrame, signals: list[SemanticSignal]
) -> tuple[float, float, float, float]:
    if "source_reliability" in chunk_df:
        reliability_values = pd.to_numeric(
            chunk_df["source_reliability"], errors="coerce"
        ).dropna()
        source_reliability = (
            float(reliability_values.clip(0.0, 1.0).mean())
            if not reliability_values.empty
            else 0.5
        )
    else:
        source_reliability = 0.5

    if not signals:
        return source_reliability, 0.0, 0.0, 0.0
    specificity_values: list[float] = []
    coherence_values: list[float] = []
    entities: list[str] = []
    for signal in signals:
        target = signal.target_interpretation
        values = [
            target.primary_entity,
            target.measure,
            target.physical_quantity,
            target.temporal_granularity,
        ]
        specificity_values.append(
            sum(str(value).strip().lower() not in {"", "unknown"} for value in values)
            / len(values)
        )
        coherence_values.append(
            sum(
                (
                    signal.inferred_domain,
                    target.primary_entity,
                    target.physical_quantity,
                )[index].strip().lower()
                not in {"", "unknown"}
                for index in range(3)
            )
            / 3.0
        )
        entities.append(target.primary_entity.strip().lower())

    expected_entities = {
        str(value).strip().lower()
        for column in ("entity_id", "entity", "series_id")
        if column in chunk_df
        for value in chunk_df[column].dropna().tolist()
        if str(value).strip()
    }
    resolved_entities = {
        entity for entity in entities if entity not in {"", "unknown"}
    }
    if expected_entities and resolved_entities:
        entity_consistency = float(
            bool(expected_entities.intersection(resolved_entities))
        )
    elif len(resolved_entities) > 1:
        entity_consistency = 1.0 / len(resolved_entities)
    else:
        entity_consistency = 0.5
    return (
        source_reliability,
        sum(specificity_values) / len(specificity_values),
        entity_consistency,
        sum(coherence_values) / len(coherence_values),
    )


_SEMANTIC_TEXT_COLUMNS = (
    "_semantic_compacted_text",
    "facts",
    "fact",
    "text",
    "content",
    "summary",
    "description",
    "body",
)
_FACT_SPLIT_RE = re.compile(r"\s*(?:\||;|\n+)\s*")
_SOURCE_TAG_RE = re.compile(r"\[source(?:\s*:[^\]]*)?\]", re.IGNORECASE)
_FACT_PREFIX_RE = re.compile(
    r"^available facts are as follows\s*:\s*", re.IGNORECASE
)


def _split_semantic_facts(text: str) -> list[str]:
    cleaned = str(text).replace("\\n", "\n").strip()
    pieces = _FACT_SPLIT_RE.split(cleaned)
    facts: list[str] = []
    for piece in pieces:
        fact = _FACT_PREFIX_RE.sub("", piece).strip(" -.;")
        if fact:
            facts.append(fact)
    return facts


def _normalize_semantic_fact(text: str) -> str:
    value = _SOURCE_TAG_RE.sub(" ", str(text).lower())
    value = re.sub(r"[^\w]+", " ", value, flags=re.UNICODE)
    return " ".join(value.split())


def _compress_semantic_documents(
    docs_df: pd.DataFrame,
    *,
    near_duplicate_threshold: float = 0.90,
) -> tuple[pd.DataFrame, SemanticCompressionAudit]:
    """Represent repeated facts once while retaining every source document row."""
    result = docs_df.copy().reset_index(drop=True)
    occurrences: list[tuple[int, str, str]] = []
    original_characters = 0
    for doc_idx, rec in enumerate(result.to_dict(orient="records")):
        raw_text = _first_present(rec, list(_SEMANTIC_TEXT_COLUMNS[1:]), default="")
        original_characters += len(raw_text)
        for fact in _split_semantic_facts(raw_text):
            normalized = _normalize_semantic_fact(fact)
            if normalized:
                occurrences.append((doc_idx, fact, normalized))

    unique_index: dict[str, int] = {}
    unique_facts: list[str] = []
    unique_occurrences: list[list[tuple[int, str]]] = []
    for doc_idx, fact, normalized in occurrences:
        index = unique_index.get(normalized)
        if index is None:
            index = len(unique_facts)
            unique_index[normalized] = index
            unique_facts.append(normalized)
            unique_occurrences.append([])
        unique_occurrences[index].append((doc_idx, fact))

    parent = list(range(len(unique_facts)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[max(left_root, right_root)] = min(left_root, right_root)

    backend = "exact_match"
    threshold = max(0.0, min(1.0, float(near_duplicate_threshold)))
    if len(unique_facts) > 1 and threshold < 1.0:
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.neighbors import NearestNeighbors

            matrix = TfidfVectorizer(
                analyzer="char_wb", ngram_range=(3, 5), max_features=25_000
            ).fit_transform(unique_facts)
            neighbors = NearestNeighbors(
                metric="cosine", algorithm="brute", n_jobs=1
            ).fit(matrix)
            distances, indices = neighbors.radius_neighbors(
                matrix, radius=max(0.0, 1.0 - threshold)
            )
            for left, (row_distances, row_indices) in enumerate(
                zip(distances, indices)
            ):
                left_length = max(1, len(unique_facts[left]))
                for distance, right in zip(row_distances, row_indices):
                    right = int(right)
                    if right <= left or float(distance) > (1.0 - threshold):
                        continue
                    length_ratio = min(left_length, len(unique_facts[right])) / max(
                        left_length, len(unique_facts[right]), 1
                    )
                    if length_ratio >= 0.75:
                        union(left, right)
            backend = "tfidf_char_ngrams"
        except Exception as exc:
            logger.warning("Near-duplicate semantic compression unavailable: %s", exc)
            backend = "exact_match_fallback"

    clusters: dict[int, list[int]] = {}
    for index in range(len(unique_facts)):
        clusters.setdefault(find(index), []).append(index)

    compacted_by_doc: list[list[str]] = [[] for _ in range(len(result))]
    date_col = _date_col(result)
    for members in clusters.values():
        member_occurrences = [
            occurrence
            for member in members
            for occurrence in unique_occurrences[member]
        ]
        representative_doc, representative_text = min(
            member_occurrences, key=lambda item: item[0]
        )
        supporting_docs = sorted({doc_idx for doc_idx, _ in member_occurrences})
        support_note = ""
        if len(supporting_docs) > 1:
            support_note = f" [represented in {len(supporting_docs)} documents"
            if date_col:
                dates = pd.to_datetime(
                    result.iloc[supporting_docs][date_col], errors="coerce"
                ).dropna()
                if not dates.empty:
                    support_note += (
                        f"; {dates.min().date().isoformat()} to "
                        f"{dates.max().date().isoformat()}"
                    )
            support_note += "]"
        compacted_by_doc[representative_doc].append(
            representative_text + support_note
        )

    compacted_text = ["; ".join(facts) for facts in compacted_by_doc]
    result["_semantic_compacted_text"] = compacted_text
    compacted_characters = sum(len(text) for text in compacted_text)
    cluster_count = len(clusters)
    exact_removed = max(0, len(occurrences) - len(unique_facts))
    near_removed = max(0, len(unique_facts) - cluster_count)
    ratio = (
        1.0 - (compacted_characters / original_characters)
        if original_characters > 0
        else 0.0
    )
    audit = SemanticCompressionAudit(
        documents_input=len(result),
        documents_with_unique_evidence=sum(bool(text) for text in compacted_text),
        facts_input=len(occurrences),
        unique_facts=cluster_count,
        exact_duplicates_removed=exact_removed,
        near_duplicates_removed=near_removed,
        original_characters=original_characters,
        compacted_characters=compacted_characters,
        compression_ratio=max(0.0, min(1.0, ratio)),
        near_duplicate_backend=backend,
    )
    return result, audit


def _build_chunk_payload(
    chunk_df: pd.DataFrame,
    max_chars_per_doc: int,
    max_total_chars: int,
) -> str:
    """Serialize every document in a chunk without exposing source metadata."""
    compact: List[Dict[str, str]] = []
    text_cols = list(_SEMANTIC_TEXT_COLUMNS)
    records: List[tuple[Dict[str, Any], str]] = []

    for rec in chunk_df.to_dict(orient="records"):
        if "_semantic_compacted_text" in rec:
            compacted_value = rec.get("_semantic_compacted_text")
            raw_text = "" if pd.isna(compacted_value) else str(compacted_value)
        else:
            raw_text = _first_present(rec, text_cols, default="")
        text = raw_text.replace("\\n", " ").strip()
        if text:
            records.append((rec, text))

    if not records:
        return "[]"

    total_cap = max(1, int(max_total_chars))
    per_doc_cap = max(
        1,
        min(int(max_chars_per_doc), total_cap // len(records)),
    )
    for rec, text in records:
        compact.append(
            {
                "date": _first_present(rec, ["date"]),
                "text": text[:per_doc_cap],
            }
        )
    return json.dumps(compact, ensure_ascii=False)


def _disabled_signal(
    source_date_start: str = "",
    source_date_end: str = "",
    ad_hoc_findings: List[str] | None = None,
    text_obs: int = 0,
    source_chunk_index: int = -1,
    source_chunk_size: int = 0,
) -> SemanticSignal:

    _ = (
        source_date_start,
        source_date_end,
        ad_hoc_findings,
        text_obs,
        source_chunk_index,
        source_chunk_size,
    )
    return SemanticSignal(
        domain_interpretation=DomainInterpretation(
            domain_label="unknown", domain_parent="unknown", domain_concepts=[]
        ),
        inferred_domain="unknown",
        domain_confidence=0.0,
        target_interpretation=TargetInterpretation(
            primary_entity="unknown",
            spatial_scope="unknown",
            measure="unknown",
            physical_quantity="unknown",
            temporal_granularity="unknown",
            target_confidence=0.0,
        ),
        model_family_prior=ModelFamilyPrior(
            preferred_model_families=[], families_to_deprioritize=[]
        ),
        textual_evidence_findings=[],
    )


def _coerce_signal(
    raw: Dict[str, Any],
    source_date_start: str,
    source_date_end: str,
    text_obs: int,
    source_chunk_index: int = -1,
    source_chunk_size: int = 0,
) -> SemanticSignal:
    _ = (
        source_date_start,
        source_date_end,
        text_obs,
        source_chunk_index,
        source_chunk_size,
    )
    try:
        return SemanticSignal(**raw)
    except Exception as exc:
        logger.warning("Failed to coerce semantic signal: %s | raw=%s", exc, raw)
        return _disabled_signal(
            source_date_start=source_date_start,
            source_date_end=source_date_end,
            text_obs=text_obs,
            source_chunk_index=source_chunk_index,
            source_chunk_size=source_chunk_size,
            ad_hoc_findings=[
                f"Semantic schema validation failed: {exc} | raw={_json_dumps_safe(raw)}"
            ],
        )


def _normalize_signal_families(
    raw: Dict[str, Any], canonical_families: List[str]
) -> List[Dict[str, Any]]:
    return normalize_signal_families(raw, canonical_families)


@dataclass
class SemanticAgent:
    llm: Any

    max_docs: int = 0  # 0 = process all available documents
    max_chunks: int = 0  # 0 = process all chunks
    chunk_size: int = 100  # original documents per LLM call

    max_chars_per_doc: int = 2_000
    max_total_chars: int = 24_000
    max_prompt_tokens: int = 12_000
    deduplicate_facts: bool = True
    near_duplicate_threshold: float = 0.90

    prefer_recent_docs: bool = True

    def _build_user_prompt(
        self,
        chunk_df: pd.DataFrame,
        context: Dict[str, Any],
    ) -> str:
        """Build a bounded semantic prompt before it reaches the LLM wrapper.

        """
        char_budget = max(1, int(self.max_total_chars))
        requested_token_budget = max(1, int(self.max_prompt_tokens))
        llm_token_budget = getattr(self.llm, "input_token_budget", None)
        token_budget = (
            min(requested_token_budget, int(llm_token_budget))
            if llm_token_budget is not None
            else requested_token_budget
        )

        def render(total_chars: int) -> str:
            prompt_json = _build_chunk_payload(
                chunk_df,
                max_chars_per_doc=self.max_chars_per_doc,
                max_total_chars=total_chars,
            )
            return SEMANTIC_AGENT_USER_PROMPT.format(
                strategy=context.get("strategy", "exploration"),
                canonical_model_families=context.get(
                    "canonical_model_families", "unknown"
                ),
                compression_summary=_json_dumps_safe(
                    context.get("compression_summary", {"enabled": False})
                ),
                unstructured_text=prompt_json,
            )

        user_prompt = render(char_budget)
        counter = getattr(self.llm, "count_chat_tokens", None)
        if not callable(counter):
            return user_prompt

        for _ in range(8):
            token_count = int(counter(SEMANTIC_AGENT_SYSTEM_PROMPT, user_prompt))
            if token_count <= token_budget:
                return user_prompt
            reduced_budget = max(
                1,
                int(char_budget * (token_budget / token_count) * 0.90),
            )
            if reduced_budget >= char_budget:
                reduced_budget = char_budget - 1
            char_budget = max(1, reduced_budget)
            user_prompt = render(char_budget)

        token_count = int(counter(SEMANTIC_AGENT_SYSTEM_PROMPT, user_prompt))
        if token_count > token_budget:
            raise ValueError(
                "Semantic prompt instructions exceed the configured token budget: "
                f"{token_count} > {token_budget}"
            )
        return user_prompt

    def _extract_chunk(
        self,
        chunk_df: pd.DataFrame,
        step: int,
        chunk_idx: int,
        context: Dict[str, Any] | None = None,
    ) -> ChunkExtractionResult:
        if context is None:
            context = {}

        # Calculate chunk dates
        source_date_start = ""
        source_date_end = ""
        date_col = _date_col(chunk_df)
        if date_col:
            dates = pd.to_datetime(chunk_df[date_col], errors="coerce").dropna()
            if not dates.empty:
                source_date_start = dates.min().isoformat()
                source_date_end = dates.max().isoformat()

        text_obs = len(chunk_df)

        try:
            user_prompt = self._build_user_prompt(chunk_df, context)
        except Exception as exc:
            logger.warning("Failed to format semantic user prompt: %s", exc)
            disabled = _disabled_signal(
                    source_date_start=source_date_start,
                    source_date_end=source_date_end,
                    text_obs=text_obs,
                    source_chunk_index=chunk_idx,
                    source_chunk_size=text_obs,
                    ad_hoc_findings=[
                        f"Semantic user prompt preparation failed: {exc}"
                    ],
                )
            return ChunkExtractionResult(
                False, (), (disabled.model_dump(),), 0,
                f"prompt_format_error: {exc}", 0.0, 0.0, 0.0, 0.0,
            )

        try:
            raw: Any = self.llm.generate_json(
                system_prompt=SEMANTIC_AGENT_SYSTEM_PROMPT,
                user_prompt=user_prompt,
            )
        except Exception as exc:
            logger.warning("LLM call failed for semantic chunk %d: %s", chunk_idx, exc)
            disabled = _disabled_signal(
                    source_date_start=source_date_start,
                    source_date_end=source_date_end,
                    text_obs=text_obs,
                    source_chunk_index=chunk_idx,
                    source_chunk_size=text_obs,
                    ad_hoc_findings=[
                        f"Semantic LLM call failed: {exc} | user_prompt={_json_dumps_safe(user_prompt)}"
                    ],
                )
            return ChunkExtractionResult(
                False, (), (disabled.model_dump(),), 0,
                f"llm_error: {exc}", 0.0, 0.0, 0.0, 0.0,
            )

        if not isinstance(raw, dict):
            disabled = _disabled_signal(
                    source_date_start=source_date_start,
                    source_date_end=source_date_end,
                    text_obs=text_obs,
                    source_chunk_index=chunk_idx,
                    source_chunk_size=text_obs,
                    ad_hoc_findings=[
                        f"Semantic LLM returned unexpected type: {type(raw)} | raw={_json_dumps_safe(raw)}"
                    ],
                )
            return ChunkExtractionResult(
                False, (), (disabled.model_dump(),), 0,
                f"unexpected_output_type: {type(raw).__name__}",
                0.0, 0.0, 0.0, 0.0,
            )

        from utils.semantic_evidence_validator import validate_semantic_output

        context.setdefault("family_normalization_trace", []).extend(
            _normalize_signal_families(
                raw, context.get("canonical_family_list", [])
            )
        )
        clean, violations = validate_semantic_output(
            raw, source_chunk_index=chunk_idx
        )

        context.setdefault("raw_signals", []).append(copy.deepcopy(clean))
        signal = _coerce_signal(
            clean,
            source_date_start,
            source_date_end,
            text_obs,
            source_chunk_index=chunk_idx,
            source_chunk_size=text_obs,
        )
        signals = [signal]

        operational = [signal for signal in signals if _is_operational_signal(signal)]
        excluded = [
            entry for signal in signals for entry in _excluded_entries(signal)
        ]
        excluded.extend(violations)
        excluded.extend(
            signal.model_dump()
            for signal in signals
            if not _is_operational_signal(signal)
        )
        quality = _chunk_quality(chunk_df, operational)
        return ChunkExtractionResult(
            success=bool(operational),
            operational_signals=tuple(operational),
            excluded_signals=tuple(excluded),
            documents_covered=text_obs if operational else 0,
            error=None if operational else "no_operational_semantic_evidence",
            source_reliability=quality[0],
            evidence_specificity=quality[1],
            entity_consistency=quality[2],
            domain_target_quantity_coherence=quality[3],
        )

    def _load_docs(self, state: Dict[str, Any]) -> pd.DataFrame:
        """Load ONLY the training semantic documents to prevent temporal leakage."""
        df = state.get("semantic_docs_train_df")
        if isinstance(df, pd.DataFrame) and not df.empty:
            return df.copy()

        raise ValueError(
            "semantic_docs_train_df is empty or missing. SemanticAgent requires training documents."
        )

    def _prepare_docs(
        self, docs_df: pd.DataFrame, state: Dict[str, Any]
    ) -> pd.DataFrame:
        """Sort canonical, timestamp-valid training documents for extraction."""
        docs_df = docs_df.copy()

        col = _date_col(docs_df)
        if col is None:
            raise ValueError(
                "Semantic timestamp policy violation: canonical date is required."
            )
        parsed_dates = pd.to_datetime(docs_df[col], errors="coerce", utc=True)
        invalid_count = int(len(parsed_dates) - parsed_dates.notna().sum())
        if invalid_count:
            raise ValueError(
                "Semantic timestamp policy violation: "
                f"{invalid_count} training documents have invalid dates."
            )
        docs_df = (
            docs_df.assign(_parsed_date=parsed_dates)
            .sort_values("_parsed_date", ascending=True)
            .drop(columns=["_parsed_date"], errors="ignore")
            .reset_index(drop=True)
        )

        docs_limit = int(state.get("semantic_max_docs", self.max_docs))

        if docs_limit > 0:
            if self.prefer_recent_docs:
                docs_df = docs_df.tail(docs_limit).reset_index(drop=True)
            else:
                docs_df = docs_df.head(docs_limit).reset_index(drop=True)

        return docs_df

    def _build_extraction_context(self, state: Dict[str, Any]) -> Dict[str, Any]:
        supplied_families = state.get("semantic_canonical_families")
        canonical_families, _ = derive_canonical_family_space(None)
        if isinstance(supplied_families, list):
            supplied_set = {
                str(family) for family in supplied_families
                if str(family) in canonical_families
            }
            search_space_families = [
                family for family in canonical_families
                if family in supplied_set
            ]
            status = str(
                state.get("semantic_family_space_status", "runtime_available")
            )
        else:
            search_space_families, status = derive_canonical_family_space(None)
        if status == "fallback_registry":
            logger.warning(
                "SemanticAgent could not derive runtime family space; using registry fallback."
            )

        search_space_text = "\n".join(f"- {family}" for family in search_space_families)

        return {
            "strategy": state.get("strategy", "exploration"),
            "canonical_model_families": search_space_text,
            "canonical_family_list": search_space_families,
            "semantic_family_space_status": status,
            "family_normalization_trace": [],
            "raw_signals": [],
        }

    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        step = int(state.get("step", 0))

        try:
            original_docs_df = self._load_docs(state)
            documents_available = len(original_docs_df)
        except ValueError as e:
            logger.warning(str(e))
            original_docs_df = pd.DataFrame()
            documents_available = 0

        if original_docs_df.empty:
            semantic_context = create_empty_semantic_context(
                status="unavailable",
                finding="No training documents available for semantic extraction.",
            )
            semantic_context["evidence_summary"]["documents_available"] = documents_available
            state["semantic_signals"] = []
            state["semantic_raw_signals"] = []
            state["semantic_operational_signals"] = []
            state["semantic_context"] = semantic_context
            return state

        docs_df = self._prepare_docs(original_docs_df, state)
        documents_selected = len(docs_df)
        deduplicate_facts = bool(
            state.get("semantic_deduplicate_facts", self.deduplicate_facts)
        )
        near_duplicate_threshold = float(
            state.get(
                "semantic_near_duplicate_threshold",
                self.near_duplicate_threshold,
            )
        )
        if deduplicate_facts:
            docs_df, compression_audit = _compress_semantic_documents(
                docs_df,
                near_duplicate_threshold=near_duplicate_threshold,
            )
            compression_summary = {
                "enabled": True,
                **compression_audit.to_dict(),
            }
        else:
            compression_summary = {
                "enabled": False,
                "documents_input": documents_selected,
                "documents_with_unique_evidence": documents_selected,
                "facts_input": None,
                "unique_facts": None,
                "duplicate_facts_removed": 0,
                "compression_ratio": 0.0,
                "near_duplicate_backend": "disabled",
            }
        chunk_size = max(1, int(state.get("semantic_chunk_size", self.chunk_size)))
        max_chunks = int(state.get("semantic_max_chunks", self.max_chunks))

        extraction_context = self._build_extraction_context(state)
        extraction_context["compression_summary"] = compression_summary
        all_signals: List[SemanticSignal] = []
        excluded_signals: List[Dict[str, Any]] = []
        chunk_quality: List[ChunkExtractionResult] = []

        chunks_available = len(range(0, documents_selected, chunk_size))
        
        documents_processed = 0
        documents_covered = 0
        chunks_processed = 0
        chunks_succeeded = 0
        chunks_failed = 0

        for chunk_idx, start in enumerate(range(0, documents_selected, chunk_size)):
            if max_chunks > 0 and chunk_idx >= max_chunks:
                break

            chunk_df = docs_df.iloc[start : start + chunk_size].reset_index(drop=True)
            
            chunks_processed += 1
            documents_processed += len(chunk_df)
            
            try:
                extraction = self._extract_chunk(
                    chunk_df, step=step, chunk_idx=chunk_idx, context=extraction_context
                )
                
                excluded_signals.extend(extraction.excluded_signals)
                chunk_quality.append(extraction)
                if extraction.success:
                    chunks_succeeded += 1
                    documents_covered += extraction.documents_covered
                    all_signals.extend(extraction.operational_signals)
                else:
                    chunks_failed += 1
                    
            except Exception as e:
                chunks_failed += 1
                logger.error("Failed to extract chunk: %s", e)
                continue

        semantic_context = aggregate_semantic_signals(
            all_signals,
            documents_available=documents_available,
            documents_selected=documents_selected,
            documents_seen=documents_covered,
            chunks_available=chunks_available,
            chunks_processed=chunks_processed,
            chunks_succeeded=chunks_succeeded,
            chunks_failed=chunks_failed,
            semantic_aggregation_config=state.get("semantic_aggregation", {}),
            semantic_target_resolution_config=state.get(
                "semantic_target_resolution", {}
            ),
        )
        semantic_context["semantic_family_space_status"] = extraction_context[
            "semantic_family_space_status"
        ]
        semantic_context["semantic_family_space"] = list(
            extraction_context["canonical_family_list"]
        )
        semantic_context["family_normalization_trace"] = list(
            extraction_context["family_normalization_trace"]
        )
        valid_quality = [item for item in chunk_quality if item.success]
        def _quality_mean(field: str) -> float:
            return (
                sum(float(getattr(item, field)) for item in valid_quality)
                / len(valid_quality)
                if valid_quality
                else 0.0
            )
        semantic_context["evidence_summary"].update(
            {
                "documents_processed": documents_processed,
                "documents_covered": documents_covered,
                "valid_chunk_count": chunks_succeeded,
                "failed_chunk_count": chunks_failed,
                "source_reliability": _quality_mean("source_reliability"),
                "evidence_specificity": _quality_mean("evidence_specificity"),
                "entity_consistency": _quality_mean("entity_consistency"),
                "domain_target_quantity_coherence": _quality_mean(
                    "domain_target_quantity_coherence"
                ),
                "compression": compression_summary,
                "documents_represented": documents_selected,
                "unique_facts_sent": compression_summary.get("unique_facts"),
                "duplicate_facts_removed": compression_summary.get(
                    "duplicate_facts_removed", 0
                ),
                "compression_ratio": compression_summary.get(
                    "compression_ratio", 0.0
                ),
            }
        )
        operational_signals = [s.model_dump() for s in all_signals]
        state["semantic_raw_signals"] = copy.deepcopy(
            extraction_context["raw_signals"]
        )
        state["semantic_operational_signals"] = operational_signals
        state["semantic_excluded_signals"] = copy.deepcopy(excluded_signals)
        # Compatibility alias: semantic_signals is always operational.
        state["semantic_signals"] = operational_signals
        state["semantic_context"] = semantic_context

        logger.info(
            "SemanticAgent: completed with domain=%s, strength=%s",
            semantic_context.get("inferred_domain"),
            semantic_context.get("model_family_prior", {}).get("prior_strength"),
        )

        return state
