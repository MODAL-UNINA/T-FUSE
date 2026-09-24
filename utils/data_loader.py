"""Load numerical series and timestamped Time-MMD semantic documents."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd


SEMANTIC_TIMESTAMP_POLICY = "timestamped_documents_only"
_SEMANTIC_DROP_REASONS = (
    "missing_temporal_interval_columns",
    "invalid_start_date",
    "invalid_end_date",
    "invalid_interval_order",
    "missing_semantic_text",
)


def _empty_semantic_audit() -> dict:
    return {
        "policy": SEMANTIC_TIMESTAMP_POLICY,
        "input_rows": 0,
        "accepted_rows": 0,
        "selected_rows": 0,
        "dropped_rows": 0,
        "drop_reasons": {reason: 0 for reason in _SEMANTIC_DROP_REASONS},
        "sources": [],
    }


def _empty_semantic_documents(audit: Optional[dict] = None) -> pd.DataFrame:
    frame = pd.DataFrame(
        columns=["start_date", "end_date", "date", "source", "facts"]
    )
    frame.attrs["semantic_document_filtering"] = audit or _empty_semantic_audit()
    return frame


def _merge_semantic_audits(audits: List[dict], selected_rows: int) -> dict:
    merged = _empty_semantic_audit()
    for audit in audits:
        merged["input_rows"] += int(audit.get("input_rows", 0))
        merged["accepted_rows"] += int(audit.get("accepted_rows", 0))
        merged["dropped_rows"] += int(audit.get("dropped_rows", 0))
        for reason in _SEMANTIC_DROP_REASONS:
            merged["drop_reasons"][reason] += int(
                audit.get("drop_reasons", {}).get(reason, 0)
            )
        merged["sources"].extend(list(audit.get("sources", [])))
    merged["selected_rows"] = int(selected_rows)
    return merged


def _parse_timemmd_date(series: pd.Series) -> pd.Series:
    """Parse a Time-MMD interval endpoint deterministically as ``YYYY-MM-DD``."""
    if pd.api.types.is_datetime64_any_dtype(series):
        parsed = pd.to_datetime(series, errors="coerce", utc=True)
    else:
        cleaned = series.astype("string").str.strip()
        parsed = pd.to_datetime(
            cleaned, format="%Y-%m-%d", errors="coerce", utc=True
        )
    return parsed.dt.tz_localize(None)


def _normalize_timestamped_semantic_rows(
    frame: pd.DataFrame,
    facts: pd.Series,
    *,
    source: str,
) -> pd.DataFrame:
    """Keep only rows with valid Time-MMD intervals and non-empty text.

    A source must expose ``start_date`` or ``end_date``. When both columns are
    present, both endpoints are mandatory and ``start_date <= end_date``. A
    single available endpoint represents a point interval. The canonical
    assignment timestamp is always ``end_date`` when present, otherwise
    ``start_date``.
    """
    normalized = _normalize_columns(frame)
    facts = facts.reindex(normalized.index).fillna("").astype(str).str.strip()
    audit = _empty_semantic_audit()
    audit["input_rows"] = int(len(normalized))
    reasons = pd.Series("", index=normalized.index, dtype="object")

    has_start = "start_date" in normalized.columns
    has_end = "end_date" in normalized.columns
    if not has_start and not has_end:
        reasons.loc[:] = "missing_temporal_interval_columns"
        start_dates = pd.Series(
            pd.NaT, index=normalized.index, dtype="datetime64[ns]"
        )
        end_dates = start_dates.copy()
    else:
        start_dates = (
            _parse_timemmd_date(normalized["start_date"])
            if has_start
            else pd.Series(pd.NaT, index=normalized.index, dtype="datetime64[ns]")
        )
        end_dates = (
            _parse_timemmd_date(normalized["end_date"])
            if has_end
            else pd.Series(pd.NaT, index=normalized.index, dtype="datetime64[ns]")
        )
        if has_start:
            reasons.loc[(reasons == "") & ~start_dates.notna()] = "invalid_start_date"
        if has_end:
            reasons.loc[(reasons == "") & ~end_dates.notna()] = "invalid_end_date"
        if has_start and has_end:
            reasons.loc[
                (reasons == "") & (start_dates > end_dates)
            ] = "invalid_interval_order"

    reasons.loc[(reasons == "") & facts.eq("")] = "missing_semantic_text"
    accepted = reasons.eq("")
    assignment_dates = end_dates if has_end else start_dates

    docs = pd.DataFrame(index=normalized.index)
    if has_start:
        docs["start_date"] = start_dates
    if has_end:
        docs["end_date"] = end_dates
    docs["date"] = assignment_dates
    docs["source"] = source
    docs["facts"] = facts
    docs = docs.loc[accepted].reset_index(drop=True)

    for reason in _SEMANTIC_DROP_REASONS:
        audit["drop_reasons"][reason] = int(reasons.eq(reason).sum())
    audit["accepted_rows"] = int(accepted.sum())
    audit["selected_rows"] = int(len(docs))
    audit["dropped_rows"] = int((~accepted).sum())
    audit["sources"] = [
        {
            "source": source,
            "input_rows": int(len(normalized)),
            "accepted_rows": int(accepted.sum()),
            "dropped_rows": int((~accepted).sum()),
            "drop_reasons": dict(audit["drop_reasons"]),
        }
    ]
    docs.attrs["semantic_document_filtering"] = audit
    return docs


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    seen = {}
    new_cols = []
    for c in df.columns:
        lowered = str(c).strip().lower()
        if lowered in seen:
            seen[lowered] += 1
            new_cols.append(f"{lowered}_{seen[lowered]}")
        else:
            seen[lowered] = 0
            new_cols.append(lowered)
    df.columns = new_cols
    return df


def parse_date_series(series: pd.Series) -> pd.Series:
    if pd.api.types.is_datetime64_any_dtype(series):
        return series
    
    def clean_val(val):
        if pd.isna(val):
            return None
        try:
            if isinstance(val, (int, float)):
                if not np.isnan(val):
                    return str(int(val))
            elif isinstance(val, str):
                val_str = val.strip()
                if val_str.endswith(".0"):
                    val_str = val_str[:-2]
                return val_str
        except Exception:
            pass
        return str(val)
    
    cleaned = series.apply(clean_val)
    try:
        parsed = pd.to_datetime(cleaned, errors="coerce", format="mixed")
    except TypeError:  # pandas < 2.0
        parsed = pd.to_datetime(cleaned, errors="coerce")
    
    # Try parsing YYYYMM format if standard parsing yielded NaT
    mask = parsed.isna() & cleaned.notna()
    if mask.any():
        for idx in cleaned[mask].index:
            val = str(cleaned.loc[idx]).strip()
            if len(val) == 6 and val.isdigit():
                try:
                    parsed.loc[idx] = pd.to_datetime(val, format="%Y%m")
                except Exception:
                    pass
    return parsed


def _read_single_series(path: Path) -> pd.DataFrame:
    """Read one target series and its temporal column.

    Column names are normalized first, then the first supported temporal and
    target column are selected. All other columns, including Time-MMD D0-D4
    values and other external numerical regressors, are deliberately excluded.
    """
    df = _normalize_columns(pd.read_csv(path))

    # Drop only unambiguous index-artifact columns, never a real Date column.
    artifact_columns = [
        c for c in df.columns
        if str(c).startswith("unnamed:") or str(c) == "index"
    ]
    if artifact_columns:
        df = df.drop(columns=artifact_columns, errors="ignore")

    target_candidates = ["ot", "target", "value", "measurement", "series_value"]
    date_candidates = ["end_date", "date", "ds", "timestamp", "time", "datetime", "start_date"]

    date_col = next((c for c in date_candidates if c in df.columns), None)
    if date_col is None:
        raise ValueError(f"Missing date/temporal column in {path}")

    target_col = next((c for c in target_candidates if c in df.columns), None)
    if target_col is None:
        raise ValueError(
            f"Missing target column in {path}; expected one of {target_candidates}"
        )

    df[date_col] = parse_date_series(df[date_col])

    excluded_numeric_columns = [
        str(column)
        for column in df.columns
        if column not in {date_col, target_col}
        and pd.api.types.is_numeric_dtype(df[column])
    ]
    out_dict = {
        "date": df[date_col],
        "target": pd.to_numeric(df[target_col], errors="coerce"),
    }

    out = pd.DataFrame(out_dict)
    out = (
        out.dropna(subset=["date", "target"])
        .sort_values("date")
        .reset_index(drop=True)
    )
    if out.empty:
        raise ValueError("Dataset skipped: no valid dated target observations.")
    out.attrs["external_numeric_columns_excluded"] = len(
        excluded_numeric_columns
    )
    return out


def list_series_files(data_root: str | Path) -> List[Path]:
    """List CSV files for the dataset."""
    root = Path(data_root)
    if root.is_file():
        return [root]
    files = sorted(root.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No CSV files found under {root}")
    return files


def load_dataset(
    data_root: str | Path,
    series_id: str | None = None,
) -> Tuple[str, pd.DataFrame]:
    """Load one series from dataset."""
    files = list_series_files(data_root)
    requested_id = series_id
    if requested_id:
        matched = [p for p in files if p.stem.lower() == requested_id.lower()]
        if not matched:
            raise FileNotFoundError(
                f"Series id {requested_id} not found in {data_root}"
            )
        file_path = matched[0]
    else:
        file_path = files[0]
    return file_path.stem, _read_single_series(file_path)






def _pick_first_column(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    lowered = {str(c).strip().lower(): c for c in df.columns}
    for cand in candidates:
        if cand in lowered:
            return str(lowered[cand])
    return None






def load_semantic_documents_df_from_parquet(
    parquet_paths: List[str | Path],
    entity_id: Optional[str] = None,
    max_docs: int = 0,
) -> pd.DataFrame:
    """Load only timestamped Time-MMD semantic rows from parquet files."""
    _ = entity_id  # Kept for API compatibility; documents are not entity-filtered.
    frames: List[pd.DataFrame] = []
    audits: List[dict] = []

    for raw_path in parquet_paths:
        path = Path(raw_path)
        if not path.exists():
            continue

        df = pd.read_parquet(path)
        if df.empty:
            continue
        normalized = _normalize_columns(df)
        facts_cols = [
            column
            for column in normalized.columns
            if str(column).startswith("final_search")
        ]
        if not facts_cols:
            facts_cols = [
                column
                for column in normalized.columns
                if column
                in {
                    "facts", "fact", "pred", "prediction", "text", "content",
                    "summary", "headline", "title", "description",
                }
            ]
        facts = _join_semantic_text(normalized, facts_cols)
        docs = _normalize_timestamped_semantic_rows(
            normalized, facts, source=path.stem
        )
        audits.append(dict(docs.attrs["semantic_document_filtering"]))
        if not docs.empty:
            frames.append(docs)

    if not frames:
        return _empty_semantic_documents(_merge_semantic_audits(audits, 0))

    combined = pd.concat(frames, ignore_index=True)
    selected = select_recent_balanced_semantic_docs(combined, int(max_docs))
    selected.attrs["semantic_document_filtering"] = _merge_semantic_audits(
        audits, len(selected)
    )
    return selected


def select_recent_balanced_semantic_docs(
    docs_df: pd.DataFrame,
    max_docs: int,
) -> pd.DataFrame:
    """Select recent semantic rows after a leakage-safe temporal split.

    If multiple ``source`` values are present, every source receives a recent
    quota before unused capacity is filled by the most recent remaining rows.
    """
    if not isinstance(docs_df, pd.DataFrame) or docs_df.empty or max_docs <= 0:
        return docs_df.copy().reset_index(drop=True)
    if len(docs_df) <= int(max_docs):
        return docs_df.copy().reset_index(drop=True)

    work = docs_df.copy().reset_index(drop=True)
    if "date" not in work.columns:
        raise ValueError("Semantic documents require a canonical date column.")
    work["_sort_date"] = pd.to_datetime(work["date"], errors="raise")
    work["_row_order"] = np.arange(len(work))
    work = work.sort_values(["_sort_date", "_row_order"], na_position="first")

    sources = []
    if "source" in work.columns:
        sources = [str(x) for x in work["source"].dropna().unique().tolist()]
    selected_indices: List[int] = []
    if len(sources) > 1:
        quota = max(1, int(max_docs) // len(sources))
        for source in sources:
            group = work[work["source"].astype(str) == source]
            selected_indices.extend(group.tail(quota).index.tolist())
    selected_indices = list(dict.fromkeys(selected_indices))
    selected = work.loc[selected_indices] if selected_indices else work.iloc[0:0]
    remaining = max(0, int(max_docs) - len(selected))
    if remaining:
        leftovers = work.drop(index=selected_indices, errors="ignore").tail(remaining)
        selected = pd.concat([selected, leftovers], axis=0)
    selected = selected.sort_values(["_sort_date", "_row_order"], na_position="first").tail(int(max_docs))
    return selected.drop(columns=["_sort_date", "_row_order"]).reset_index(drop=True)


def _load_semantic_documents_df_from_csv(
    csv_paths: List[str | Path],
    max_docs: int = 0,
    include_prediction_text: bool = True,
) -> pd.DataFrame:
    """Load CSV semantic documents with recent, source-balanced sampling.

    Time-MMD contains report and search streams with very different densities.
    Reading files sequentially and truncating at ``max_docs`` can silently drop
    one source and over-represent very old context. We therefore normalize all
    requested CSVs first, then keep recent rows with a balanced per-source floor.
    The later numeric train-boundary split remains the authoritative leakage
    barrier.
    """
    frames: List[pd.DataFrame] = []
    audits: List[dict] = []

    for raw_path in csv_paths:
        path = Path(raw_path)
        if not path.exists():
            continue
        df = pd.read_csv(path)
        if df.empty:
            continue

        normalized = _normalize_columns(df)
        facts_candidates = [
            "facts", "fact",
            *(["pred", "prediction"] if include_prediction_text else []),
            "text", "content", "summary", "headline", "title", "description",
        ]
        facts_cols = [
            c for c in normalized.columns
            if str(c).strip().lower().startswith("final_search")
        ]
        if not facts_cols:
            facts_cols = [
                c for c in normalized.columns
                if str(c).strip().lower() in facts_candidates
            ]
        if not facts_cols:
            exclude_cols = {"start_date", "end_date", "date"}
            for target in ("ot", "target", "value", "measurement", "series_value"):
                if target in normalized.columns:
                    exclude_cols.add(target)
            facts_cols = [
                c for c in normalized.columns
                if c not in exclude_cols
                and (pd.api.types.is_object_dtype(normalized[c]) or pd.api.types.is_string_dtype(normalized[c]))
            ]
        facts = _join_semantic_text(normalized, facts_cols)
        docs = _normalize_timestamped_semantic_rows(
            normalized, facts, source=path.stem
        )
        audits.append(dict(docs.attrs["semantic_document_filtering"]))
        if not docs.empty:
            frames.append(docs)

    if not frames:
        return _empty_semantic_documents(_merge_semantic_audits(audits, 0))

    combined = pd.concat(frames, ignore_index=True)
    selected = select_recent_balanced_semantic_docs(combined, int(max_docs))
    selected.attrs["semantic_document_filtering"] = _merge_semantic_audits(
        audits, len(selected)
    )
    return selected


def _join_semantic_text(frame: pd.DataFrame, facts_cols: List[str]) -> pd.Series:
    """Join configured semantic fields while treating null-like values as empty."""
    if not facts_cols:
        return pd.Series("", index=frame.index, dtype="object")

    def _join_non_empty(row: pd.Series) -> str:
        parts: List[str] = []
        for column in facts_cols:
            if pd.isna(row[column]):
                continue
            value = str(row[column]).strip()
            if not value or value.lower() == "nan":
                continue
            key = str(column).strip().lower()
            if key in {"fact", "facts"}:
                value = f"FACT: {value}"
            elif key in {"pred", "prediction"}:
                value = f"PREDICTION: {value}"
            parts.append(value)
        return " | ".join(parts)

    return frame.apply(_join_non_empty, axis=1)

def load_semantic_documents_df(
    parquet_paths: Optional[List[str | Path]] = None,
    csv_paths: Optional[List[str | Path]] = None,
    entity_id: Optional[str] = None,
    max_docs: int = 0,
    include_prediction_text: bool = True,
) -> pd.DataFrame:
    """Load timestamp-valid semantic documents and attach filtering metadata."""
    _ = entity_id  # Unused by design for raw document ingestion.
    frames: List[pd.DataFrame] = []
    audits: List[dict] = []

    if csv_paths:
        csv_df = _load_semantic_documents_df_from_csv(
            csv_paths=csv_paths, max_docs=max_docs,
            include_prediction_text=include_prediction_text,
        )
        audits.append(dict(csv_df.attrs.get("semantic_document_filtering", {})))
        if not csv_df.empty:
            frames.append(csv_df)

    if parquet_paths:
        consumed = sum(len(df) for df in frames)
        remaining = max_docs if max_docs <= 0 else max(0, max_docs - consumed)
        if max_docs <= 0 or remaining > 0:
            pq_df = load_semantic_documents_df_from_parquet(
                parquet_paths=parquet_paths,
                entity_id=entity_id,
                max_docs=remaining if max_docs > 0 else max_docs,
            )
            audits.append(dict(pq_df.attrs.get("semantic_document_filtering", {})))
            if not pq_df.empty:
                frames.append(pq_df)

    if not frames:
        return _empty_semantic_documents(_merge_semantic_audits(audits, 0))

    out = pd.concat(frames, ignore_index=True)
    if max_docs > 0:
        out = out.head(max_docs)
    out.attrs["semantic_document_filtering"] = _merge_semantic_audits(
        audits, len(out)
    )
    return out



def parse_document_dates(docs_df: pd.DataFrame) -> Tuple[Optional[str], pd.Series, pd.DataFrame]:
    """Require and normalize a valid assignment timestamp for every document."""
    if docs_df.empty:
        return None, pd.Series(dtype="datetime64[ns]"), docs_df

    date_candidates = ["end_date", "date", "start_date"]
    lowered = {str(column).strip().lower(): column for column in docs_df.columns}
    raw_column = next(
        (lowered[candidate] for candidate in date_candidates if candidate in lowered),
        None,
    )
    if raw_column is None:
        raise ValueError(
            "Semantic timestamp policy violation: no Time-MMD interval endpoint."
        )
    parsed_dates = pd.to_datetime(
        docs_df[raw_column], errors="coerce", utc=True
    ).dt.tz_localize(None)
    invalid_count = int(len(parsed_dates) - parsed_dates.notna().sum())
    if invalid_count:
        raise ValueError(
            "Semantic timestamp policy violation: "
            f"{invalid_count} document rows have an invalid assignment timestamp."
        )
    normalized = docs_df.copy()
    normalized["date"] = parsed_dates
    normalized.attrs.update(docs_df.attrs)
    return str(raw_column), parsed_dates, normalized
