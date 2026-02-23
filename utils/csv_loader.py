"""Robust CSV loader with encoding/delimiter auto-detection and structured profiling.

Handles all common CSV variants: different encodings, delimiters, quoted fields,
bad lines, BOM, headerless files, duplicate columns, and large files.
Works with both Streamlit UploadedFile and FastAPI UploadFile objects.
"""

import csv
import io
import logging
from typing import Any, BinaryIO, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

logger = logging.getLogger("csv_loader")

TARGET_HINTS = {"target", "label", "class", "y", "outcome", "result", "churn",
                "survived", "default", "fraud", "spam", "diagnosis", "species"}

ID_HINTS = {"id", "index", "idx", "row_id", "row_number", "record_id",
            "uid", "uuid", "guid", "key", "pk", "primary_key"}

DATETIME_HINTS = {"date", "time", "datetime", "timestamp", "created_at",
                  "updated_at", "created", "modified", "day", "month", "year"}

ENCODINGS_TO_TRY = ["utf-8-sig", "utf-8", "latin-1", "cp1252", "iso-8859-1"]
DELIMITERS_TO_TRY = [",", ";", "\t", "|"]


class CSVLoadError(Exception):
    """Raised when CSV loading fails with an actionable message."""

    def __init__(self, message: str, cause: str = "", suggestion: str = ""):
        self.cause = cause
        self.suggestion = suggestion
        super().__init__(message)


def _extract_bytes(source: Any) -> bytes:
    """Extract raw bytes from various upload sources.

    Args:
        source: Streamlit UploadedFile, FastAPI UploadFile, file path string,
                bytes, or BytesIO object.

    Returns:
        Raw bytes of the file content.
    """
    if isinstance(source, bytes):
        return source

    if isinstance(source, str):
        with open(source, "rb") as f:
            return f.read()

    if isinstance(source, io.BytesIO):
        source.seek(0)
        return source.read()

    if hasattr(source, "getbuffer"):
        return bytes(source.getbuffer())

    if hasattr(source, "read"):
        if hasattr(source, "seek"):
            source.seek(0)
        data = source.read()
        if isinstance(data, str):
            return data.encode("utf-8")
        return data

    raise CSVLoadError(
        "Cannot read the uploaded file.",
        cause="unsupported_source",
        suggestion="Please upload a valid CSV file.",
    )


def _detect_encoding(raw_bytes: bytes) -> str:
    """Detect encoding by trying multiple encodings on a sample."""
    sample = raw_bytes[:8192]

    if sample[:3] == b"\xef\xbb\xbf":
        return "utf-8-sig"

    for enc in ENCODINGS_TO_TRY:
        try:
            sample.decode(enc)
            return enc
        except (UnicodeDecodeError, LookupError):
            continue

    return "latin-1"


def _detect_delimiter(text_sample: str) -> str:
    """Auto-detect delimiter using csv.Sniffer, with fallback heuristics."""
    try:
        lines = text_sample.strip().split("\n")[:20]
        sample_text = "\n".join(lines)
        dialect = csv.Sniffer().sniff(sample_text, delimiters=",;\t|")
        return dialect.delimiter
    except csv.Error:
        pass

    lines = text_sample.strip().split("\n")[:10]
    best_delim = ","
    best_count = 0
    for delim in DELIMITERS_TO_TRY:
        counts = [line.count(delim) for line in lines if line.strip()]
        if not counts:
            continue
        avg = sum(counts) / len(counts)
        consistency = min(counts) / max(counts) if max(counts) > 0 else 0
        score = avg * consistency
        if score > best_count:
            best_count = score
            best_delim = delim

    return best_delim


def _has_header(text_sample: str, delimiter: str) -> bool:
    """Check whether the first row looks like a header using multiple heuristics."""
    lines = text_sample.strip().split("\n")
    if len(lines) < 2:
        return True

    first_fields = [f.strip().strip('"') for f in lines[0].split(delimiter)]
    second_fields = [f.strip().strip('"') for f in lines[1].split(delimiter)]

    first_numeric = sum(1 for f in first_fields if _is_numeric_string(f))
    second_numeric = sum(1 for f in second_fields if _is_numeric_string(f))
    if first_numeric < second_numeric:
        return True

    all_first_alpha = all(
        f.replace("_", "").replace(" ", "").isalpha() for f in first_fields if f
    )
    if all_first_alpha and len(first_fields) > 1:
        return True

    try:
        result = csv.Sniffer().has_header(text_sample[:4096])
        return result
    except csv.Error:
        pass

    return True


def _is_numeric_string(s: str) -> bool:
    """Check if a string looks numeric."""
    try:
        float(s.replace(",", ""))
        return True
    except (ValueError, AttributeError):
        return False


def load_csv(
    source: Any,
    *,
    sample_rows: int = 20000,
    delimiter: Optional[str] = None,
    encoding: Optional[str] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Load a CSV from any source with robust error handling.

    Args:
        source: File path (str), bytes, BytesIO, Streamlit UploadedFile,
                or FastAPI UploadFile.
        sample_rows: Max rows to load. 0 or None for all rows.
        delimiter: Force a specific delimiter. Auto-detected if None.
        encoding: Force a specific encoding. Auto-detected if None.

    Returns:
        Tuple of (DataFrame, profile_dict).

    Raises:
        CSVLoadError: With actionable cause and suggestion.
    """
    warnings: List[str] = []

    try:
        raw_bytes = _extract_bytes(source)
    except CSVLoadError:
        raise
    except Exception as exc:
        raise CSVLoadError(
            f"Failed to read uploaded file: {exc}",
            cause="read_error",
            suggestion="The file may be corrupted. Try re-uploading it.",
        )

    if len(raw_bytes) == 0:
        raise CSVLoadError(
            "The uploaded file is empty.",
            cause="empty_file",
            suggestion="Please upload a CSV file with data.",
        )

    if encoding is None:
        encoding = _detect_encoding(raw_bytes)
        logger.info("Detected encoding: %s", encoding)

    try:
        text = raw_bytes.decode(encoding)
    except (UnicodeDecodeError, LookupError):
        for fallback_enc in ENCODINGS_TO_TRY:
            try:
                text = raw_bytes.decode(fallback_enc)
                encoding = fallback_enc
                warnings.append(f"Fell back to encoding: {fallback_enc}")
                break
            except (UnicodeDecodeError, LookupError):
                continue
        else:
            raise CSVLoadError(
                "Cannot decode the file with any supported encoding.",
                cause="encoding_error",
                suggestion="Try saving the file as UTF-8 in your spreadsheet application.",
            )

    stripped = text.strip()
    if not stripped:
        raise CSVLoadError(
            "The file contains no data (only whitespace/BOM).",
            cause="empty_file",
            suggestion="Please upload a CSV with actual data rows.",
        )

    if delimiter is None:
        delimiter = _detect_delimiter(stripped)
        logger.info("Detected delimiter: %r", delimiter)

    has_hdr = _has_header(stripped, delimiter)
    header_arg = 0 if has_hdr else None
    if not has_hdr:
        warnings.append("No header row detected; columns auto-named.")

    nrows = sample_rows if sample_rows and sample_rows > 0 else None

    df = None
    parse_errors: List[str] = []

    for engine in ["c", "python"]:
        try:
            buf = io.StringIO(text)
            df = pd.read_csv(
                buf,
                sep=delimiter,
                encoding=encoding,
                header=header_arg,
                nrows=nrows,
                on_bad_lines="warn" if engine == "c" else "skip",
                engine=engine,
                skipinitialspace=True,
                quotechar='"',
                low_memory=False,
            )
            break
        except Exception as exc:
            parse_errors.append(f"Engine '{engine}': {exc}")
            continue

    if df is None:
        raise CSVLoadError(
            f"Failed to parse CSV with all engines. Errors: {'; '.join(parse_errors)}",
            cause="parse_error",
            suggestion="Check that the file is a valid CSV. Try opening it in a spreadsheet app and re-saving.",
        )

    if df.empty:
        raise CSVLoadError(
            "The CSV parsed successfully but contains no data rows.",
            cause="empty_data",
            suggestion="Make sure your CSV has data below the header row.",
        )

    df = _clean_columns(df, warnings)

    profile = _build_profile(df, warnings, encoding, delimiter)

    logger.info(
        "CSV loaded: %d rows x %d cols, encoding=%s, delimiter=%r",
        len(df), len(df.columns), encoding, delimiter,
    )

    return df, profile


def _clean_columns(df: pd.DataFrame, warnings: List[str]) -> pd.DataFrame:
    """Fix duplicate, unnamed, and whitespace column names."""
    import re

    cols = []
    seen = {}
    for i, col in enumerate(df.columns):
        col_str = str(col).strip()

        if col_str == "" or col_str.startswith("Unnamed"):
            col_str = f"column_{i}"
            warnings.append(f"Unnamed column at position {i} renamed to '{col_str}'.")

        pandas_dup = re.match(r"^(.+)\.\d+$", col_str)
        if pandas_dup:
            base = pandas_dup.group(1)
            if base in seen or any(str(c).strip() == base for c in df.columns if str(c).strip() != col_str):
                warnings.append(f"Duplicate column '{base}' detected (renamed by parser to '{col_str}').")

        if col_str in seen:
            seen[col_str] += 1
            new_name = f"{col_str}_{seen[col_str]}"
            warnings.append(f"Duplicate column '{col_str}' renamed to '{new_name}'.")
            col_str = new_name
        else:
            seen[col_str] = 0

        cols.append(col_str)

    df.columns = cols
    return df


def _build_profile(
    df: pd.DataFrame,
    warnings: List[str],
    encoding: str,
    delimiter: str,
) -> Dict[str, Any]:
    """Compute comprehensive dataset profile."""
    schema = _infer_schema(df)
    stats = _compute_stats(df)
    quality = _compute_quality(df, stats)
    target_col, problem_type = _detect_problem(df, schema)
    id_columns = _detect_id_columns(df, schema)
    datetime_candidates = _detect_datetime_candidates(df, schema)
    leakage_columns = _detect_leakage_columns(df, schema, target_col)

    return {
        "schema": schema,
        "stats": stats,
        "quality_score": quality,
        "target_column": target_col,
        "problem_type": problem_type,
        "id_columns": id_columns,
        "datetime_candidates": datetime_candidates,
        "leakage_columns": leakage_columns,
        "encoding": encoding,
        "delimiter": delimiter,
        "warnings": warnings,
    }


def _infer_schema(df: pd.DataFrame) -> Dict[str, str]:
    """Classify each column as numeric, categorical, datetime, boolean, or text."""
    schema: Dict[str, str] = {}
    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            schema[col] = "datetime"
        elif pd.api.types.is_bool_dtype(df[col]):
            schema[col] = "boolean"
        elif pd.api.types.is_numeric_dtype(df[col]):
            schema[col] = "numeric"
        elif isinstance(df[col].dtype, pd.CategoricalDtype) or (
            df[col].dtype == "object" and df[col].nunique() < max(20, len(df) * 0.05)
        ):
            schema[col] = "categorical"
        else:
            schema[col] = "text"
    return schema


def _compute_stats(df: pd.DataFrame) -> Dict[str, Any]:
    """Compute basic statistics for the dataset."""
    missing = df.isnull().sum().to_dict()
    unique = df.nunique().to_dict()

    missing_pct = {}
    for col in df.columns:
        missing_pct[col] = round(missing[col] / len(df) * 100, 2) if len(df) > 0 else 0.0

    cardinality = {}
    for col in df.columns:
        cardinality[col] = round(unique[col] / len(df) * 100, 2) if len(df) > 0 else 0.0

    numeric_stats = {}
    for col in df.select_dtypes(include=[np.number]).columns:
        numeric_stats[col] = {
            "mean": round(float(df[col].mean()), 4) if not df[col].isna().all() else None,
            "std": round(float(df[col].std()), 4) if not df[col].isna().all() else None,
            "min": float(df[col].min()) if not df[col].isna().all() else None,
            "max": float(df[col].max()) if not df[col].isna().all() else None,
            "median": float(df[col].median()) if not df[col].isna().all() else None,
        }

    return {
        "row_count": len(df),
        "column_count": len(df.columns),
        "missing_values": missing,
        "missing_pct": missing_pct,
        "unique_values": unique,
        "cardinality_pct": cardinality,
        "dtypes": {c: str(d) for c, d in df.dtypes.items()},
        "numeric_stats": numeric_stats,
    }


def _compute_quality(df: pd.DataFrame, stats: Dict[str, Any]) -> float:
    """Compute a data quality score (0-100)."""
    total_cells = stats["row_count"] * stats["column_count"]
    if total_cells == 0:
        return 0.0
    total_missing = sum(stats["missing_values"].values())
    completeness = 1.0 - (total_missing / total_cells)

    uniqueness_ratios = [
        stats["unique_values"][c] / stats["row_count"]
        for c in df.columns
        if stats["row_count"] > 0
    ]
    avg_uniqueness = float(np.mean(uniqueness_ratios)) if uniqueness_ratios else 0.0

    score = (completeness * 70) + (avg_uniqueness * 30)
    return round(min(max(score, 0.0), 100.0), 2)


def _detect_problem(
    df: pd.DataFrame, schema: Dict[str, str]
) -> Tuple[Optional[str], str]:
    """Detect target column and problem type."""
    target_col = _find_target_column(df, schema)

    has_datetime = any(v == "datetime" for v in schema.values())
    if has_datetime and target_col and schema.get(target_col) == "numeric":
        return target_col, "time_series"

    if target_col is None:
        all_numeric = all(v == "numeric" for v in schema.values())
        if all_numeric and len(df.columns) >= 3:
            return None, "clustering"
        return None, "eda_only"

    nunique = df[target_col].nunique()
    if schema.get(target_col) == "numeric" and nunique > 20:
        return target_col, "regression"
    if nunique == 2:
        return target_col, "binary_classification"
    if nunique <= 20:
        return target_col, "multiclass_classification"
    return target_col, "regression"


def _find_target_column(
    df: pd.DataFrame, schema: Dict[str, str]
) -> Optional[str]:
    """Find the most likely target column using heuristics."""
    for col in df.columns:
        if col.lower().strip() in TARGET_HINTS:
            return col

    for col in df.columns:
        if any(hint in col.lower() for hint in TARGET_HINTS):
            return col

    last_col = df.columns[-1]
    if schema.get(last_col) in ("numeric", "categorical", "boolean"):
        nunique = df[last_col].nunique()
        if nunique <= 30:
            return last_col

    return None


def _detect_id_columns(df: pd.DataFrame, schema: Dict[str, str]) -> List[str]:
    """Detect columns that look like IDs (near-unique, non-informative)."""
    id_cols = []
    for col in df.columns:
        if col.lower().strip() in ID_HINTS:
            id_cols.append(col)
            continue

        if any(hint in col.lower() for hint in ID_HINTS):
            id_cols.append(col)
            continue

        if len(df) > 10:
            ratio = df[col].nunique() / len(df)
            if ratio > 0.95 and schema.get(col) in ("numeric", "text"):
                if schema.get(col) == "numeric":
                    vals = df[col].dropna()
                    if len(vals) > 1:
                        diffs = vals.sort_values().diff().dropna()
                        if (diffs == 1).mean() > 0.9:
                            id_cols.append(col)

    return id_cols


def _detect_datetime_candidates(
    df: pd.DataFrame, schema: Dict[str, str]
) -> List[str]:
    """Detect columns that might be datetime."""
    candidates = []

    for col in df.columns:
        if schema.get(col) == "datetime":
            candidates.append(col)
            continue

        if any(hint in col.lower() for hint in DATETIME_HINTS):
            if schema.get(col) != "numeric":
                try:
                    sample = df[col].dropna().head(20)
                    pd.to_datetime(sample, infer_datetime_format=True)
                    candidates.append(col)
                except (ValueError, TypeError):
                    pass

    return candidates


def _detect_leakage_columns(
    df: pd.DataFrame, schema: Dict[str, str], target_col: Optional[str]
) -> List[str]:
    """Detect columns that might cause data leakage."""
    leakage = []
    if target_col is None:
        return leakage

    target_series = df[target_col] if target_col in df.columns else None
    if target_series is None:
        return leakage

    for col in df.columns:
        if col == target_col:
            continue

        if len(df) > 10:
            ratio = df[col].nunique() / len(df)
            if ratio > 0.95 and schema.get(col) == "text":
                leakage.append(col)
                continue

        if schema.get(col) == "numeric" and target_series is not None:
            try:
                corr = df[col].corr(target_series.astype(float))
                if abs(corr) > 0.995:
                    leakage.append(col)
            except (ValueError, TypeError):
                pass

    return leakage
