"""
Feature Engineer Agent - Automated Feature Engineering

Handles arbitrary datasets by:
- Skipping ID-like and free-text columns
- Safely detecting datetime columns
- Creating meaningful interaction features
- Encoding categoricals based on cardinality
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime

from .base_agent import BaseAgent, TaskResult

logger = logging.getLogger(__name__)


def _is_id_column(df: pd.DataFrame, col: str) -> bool:
    """Detect ID-like columns that should be excluded from feature engineering."""
    col_lower = col.lower().strip()
    if col_lower in ("id", "index", "row_id", "row_number", "unnamed: 0"):
        return True
    if col_lower.endswith(("_id", " id")):
        return True
    if col_lower.startswith(("id_", "id ")):
        return True
    if pd.api.types.is_integer_dtype(df[col]) and len(df) > 20:
        if df[col].nunique() >= len(df) * 0.9:
            return True
    return False


def _is_text_column(df: pd.DataFrame, col: str) -> bool:
    """Detect free-text columns (emails, names, descriptions) that shouldn't be encoded."""
    if not (df[col].dtype == "object" or pd.api.types.is_string_dtype(df[col])):
        return False
    sample = df[col].dropna()
    if len(sample) == 0:
        return False
    # Don't treat date-like columns as text
    if _looks_like_date(df[col]):
        return False
    # Average string length > 50 -> likely free text (descriptions, etc.)
    avg_len = sample.astype(str).str.len().mean()
    if avg_len > 50:
        return True
    # Very high cardinality text (> 80% unique with 20+ unique values) -> likely names/emails
    if sample.nunique() > 20 and sample.nunique() / len(sample) > 0.8:
        return True
    # Common free-text column name patterns
    col_lower = col.lower()
    if any(kw in col_lower for kw in ["email", "name", "description", "text", "comment",
                                       "address", "url", "subject", "body", "note"]):
        if sample.nunique() > 20:
            return True
    return False


def _looks_like_date(series: pd.Series) -> bool:
    """Check if a string column likely contains dates."""
    sample = series.dropna().head(20)
    if len(sample) == 0:
        return False
    date_pattern_count = sample.astype(str).str.contains(
        r"\d{1,4}[-/]\d{1,2}[-/]\d{1,4}", regex=True
    ).sum()
    return date_pattern_count >= len(sample) * 0.5


class FeatureEngineerAgent(BaseAgent):
    """Agent for automated feature engineering."""

    def __init__(self):
        super().__init__(
            name="FeatureEngineerAgent",
            description="Automated feature engineering and transformation",
            capabilities=["feature_creation", "encoding", "scaling", "interaction_features"]
        )
        self.feature_report: Dict[str, Any] = {}
        self.transformers: Dict[str, Any] = {}

    def get_system_prompt(self) -> str:
        return "You are an expert Feature Engineering Agent."

    async def execute(self, task: Dict[str, Any]) -> TaskResult:
        action = task.get("action", "engineer_features")

        try:
            if action == "engineer_features":
                return await self._engineer_features(task)
            elif action == "encode_features":
                return await self._encode_features(task)
            elif action == "scale_features":
                return await self._scale_features(task)
            return TaskResult(success=False, error=f"Unknown action: {action}")
        except Exception as e:
            return TaskResult(success=False, error=str(e))

    async def _engineer_features(self, task: Dict[str, Any]) -> TaskResult:
        df = task.get("dataframe")
        if df is None:
            df_path = task.get("dataframe_path")
            if df_path:
                df = pd.read_csv(df_path) if df_path.endswith('.csv') else pd.read_excel(df_path)
            else:
                return TaskResult(success=False, error="No dataframe provided")

        df = df.copy()
        # Convert pandas 3.x StringDtype to object for numpy compatibility
        for col in df.columns:
            if pd.api.types.is_string_dtype(df[col]) and df[col].dtype != "object":
                df[col] = df[col].astype(object)

        target = task.get("target_column")
        original_cols = list(df.columns)

        self.feature_report = {
            "timestamp": datetime.now().isoformat(),
            "original_features": original_cols,
            "created_features": [],
            "encoded_features": [],
            "scaled_features": [],
            "dropped_columns": [],
            "steps": []
        }

        # ---- Identify columns to skip ----
        skip_cols = set()
        if target:
            skip_cols.add(target)
        for col in df.columns:
            if col in skip_cols:
                continue
            if _is_id_column(df, col):
                skip_cols.add(col)
                self.feature_report["dropped_columns"].append({"column": col, "reason": "ID-like"})
            elif _is_text_column(df, col):
                skip_cols.add(col)
                self.feature_report["dropped_columns"].append({"column": col, "reason": "free-text"})

        # Drop ID/text columns (except target)
        cols_to_drop = [c for c in skip_cols if c != target and c in df.columns]
        if cols_to_drop:
            df = df.drop(columns=cols_to_drop)
            self.feature_report["steps"].append({
                "step": "drop_non_feature_columns",
                "columns": cols_to_drop,
            })

        # ---- Extract datetime features ----
        for col in df.columns:
            if col == target:
                continue
            if (df[col].dtype == 'object' or pd.api.types.is_string_dtype(df[col])) and _looks_like_date(df[col]):
                try:
                    parsed = pd.to_datetime(df[col], errors='coerce')
                    if parsed.notna().sum() > len(parsed) * 0.5:
                        df[col] = parsed
                except Exception:
                    pass

        for col in df.select_dtypes(include=['datetime64']).columns:
            if col == target:
                continue
            df[f'{col}_year'] = df[col].dt.year
            df[f'{col}_month'] = df[col].dt.month
            df[f'{col}_day'] = df[col].dt.day
            df[f'{col}_dayofweek'] = df[col].dt.dayofweek
            self.feature_report["created_features"].extend([
                f'{col}_year', f'{col}_month', f'{col}_day', f'{col}_dayofweek'
            ])
            df = df.drop(columns=[col])

        # ---- Create interaction features ----
        if task.get("create_interactions", True):
            numeric_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c != target]
            for i, c1 in enumerate(numeric_cols[:5]):
                for c2 in numeric_cols[i+1:6]:
                    df[f'{c1}_x_{c2}'] = df[c1] * df[c2]
                    self.feature_report["created_features"].append(f'{c1}_x_{c2}')

        # ---- Transform skewed features ----
        for col in [c for c in df.select_dtypes(include=[np.number]).columns if c != target]:
            skew = df[col].skew()
            if pd.notna(skew) and abs(skew) > 1 and (df[col] > 0).all():
                df[f'{col}_log'] = np.log1p(df[col])
                self.feature_report["created_features"].append(f'{col}_log')

        # ---- Encode categorical ----
        cat_cols = [c for c in df.select_dtypes(include=['object', 'category', 'str']).columns if c != target]
        for col in cat_cols:
            n_unique = df[col].nunique()
            if n_unique == 2:
                df[col] = pd.factorize(df[col])[0]
            elif n_unique <= 10:
                dummies = pd.get_dummies(df[col], prefix=col, drop_first=True)
                df = pd.concat([df.drop(columns=[col]), dummies], axis=1)
            else:
                freq_map = df[col].value_counts(normalize=True).to_dict()
                df[f'{col}_freq'] = df[col].map(freq_map).fillna(0)
                df = df.drop(columns=[col])
            self.feature_report["encoded_features"].append(col)

        # ---- Scale numerical ----
        if task.get("scaling", "standard") != "none":
            numeric_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c != target]
            for col in numeric_cols:
                mean, std = df[col].mean(), df[col].std()
                if std > 0:
                    df[col] = (df[col] - mean) / std
            self.feature_report["scaled_features"] = numeric_cols

        self.feature_report["final_features"] = list(df.columns)

        return TaskResult(
            success=True,
            data={"dataframe": df, "feature_report": self.feature_report},
            metrics={"original_features": len(original_cols), "final_features": len(df.columns)}
        )

    async def _encode_features(self, task: Dict[str, Any]) -> TaskResult:
        df = task.get("dataframe")
        if df is None:
            return TaskResult(success=False, error="No dataframe provided")

        df = df.copy()
        for col in df.columns:
            if pd.api.types.is_string_dtype(df[col]) and df[col].dtype != "object":
                df[col] = df[col].astype(object)

        target = task.get("target_column")
        for col in [c for c in df.select_dtypes(include=['object', 'category', 'str']).columns if c != target]:
            if df[col].nunique() <= 10:
                dummies = pd.get_dummies(df[col], prefix=col, drop_first=True)
                df = pd.concat([df.drop(columns=[col]), dummies], axis=1)
            else:
                df[col] = pd.factorize(df[col])[0]

        return TaskResult(success=True, data={"dataframe": df})

    async def _scale_features(self, task: Dict[str, Any]) -> TaskResult:
        df = task.get("dataframe")
        if df is None:
            return TaskResult(success=False, error="No dataframe provided")

        target = task.get("target_column")
        for col in [c for c in df.select_dtypes(include=[np.number]).columns if c != target]:
            mean, std = df[col].mean(), df[col].std()
            if std > 0:
                df[col] = (df[col] - mean) / std

        return TaskResult(success=True, data={"dataframe": df})
