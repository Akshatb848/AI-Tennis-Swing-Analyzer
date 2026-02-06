"""
Data Cleaner Agent - Handles data preprocessing and cleaning

Improvements:
- Skips ID columns for outlier handling
- Better missing value reporting
- Detailed cleaning report
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime

from .base_agent import BaseAgent, TaskResult

logger = logging.getLogger(__name__)


def _is_id_column(df: pd.DataFrame, col: str) -> bool:
    """Detect ID-like columns that shouldn't be cleaned/clipped."""
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


class DataCleanerAgent(BaseAgent):
    """Agent for data cleaning and preprocessing."""

    def __init__(self):
        super().__init__(
            name="DataCleanerAgent",
            description="Data cleaning, preprocessing, and quality assurance",
            capabilities=["missing_value_handling", "outlier_detection", "duplicate_removal", "type_inference"]
        )
        self.cleaning_report: Dict[str, Any] = {}

    def get_system_prompt(self) -> str:
        return "You are an expert Data Cleaning Agent."

    async def execute(self, task: Dict[str, Any]) -> TaskResult:
        action = task.get("action", "clean_data")

        try:
            if action == "clean_data":
                return await self._clean_data(task)
            elif action == "handle_missing":
                return await self._handle_missing_values(task)
            elif action == "handle_outliers":
                return await self._handle_outliers(task)
            elif action == "remove_duplicates":
                return await self._remove_duplicates(task)
            elif action == "validate_data":
                return await self._validate_data(task)
            return TaskResult(success=False, error=f"Unknown action: {action}")
        except Exception as e:
            return TaskResult(success=False, error=str(e))

    async def _clean_data(self, task: Dict[str, Any]) -> TaskResult:
        df = task.get("dataframe")
        if df is None:
            df_path = task.get("dataframe_path")
            if df_path:
                df = pd.read_csv(df_path) if df_path.endswith('.csv') else pd.read_excel(df_path)
            else:
                return TaskResult(success=False, error="No dataframe provided")

        original_shape = df.shape
        original_missing_total = int(df.isnull().sum().sum())
        self.cleaning_report = {
            "original_shape": original_shape,
            "original_missing": df.isnull().sum().to_dict(),
            "original_missing_total": original_missing_total,
            "original_duplicates": int(df.duplicated().sum()),
            "steps": [],
            "timestamp": datetime.now().isoformat()
        }

        # Identify ID columns to skip for outlier handling
        id_cols = {c for c in df.columns if _is_id_column(df, c)}

        # Remove duplicates
        if task.get("remove_duplicates", True):
            before = len(df)
            df = df.drop_duplicates()
            removed = before - len(df)
            self.cleaning_report["steps"].append({"step": "remove_duplicates", "removed": removed})

        # Handle missing values
        missing_handled = 0
        if task.get("handle_missing", True):
            for col in df.columns:
                n_missing = int(df[col].isnull().sum())
                if n_missing > 0:
                    if df[col].dtype in ['int64', 'float64']:
                        df[col] = df[col].fillna(df[col].median())
                    else:
                        mode = df[col].mode()
                        df[col] = df[col].fillna(mode[0] if len(mode) > 0 else "Unknown")
                    missing_handled += n_missing
            self.cleaning_report["steps"].append({"step": "handle_missing", "values_filled": missing_handled})

        # Handle outliers (skip ID columns)
        outliers_clipped = 0
        if task.get("handle_outliers", True):
            numeric_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c not in id_cols]
            for col in numeric_cols:
                Q1, Q3 = df[col].quantile([0.25, 0.75])
                IQR = Q3 - Q1
                if IQR > 0:
                    lower = Q1 - 1.5 * IQR
                    upper = Q3 + 1.5 * IQR
                    n_outliers = int(((df[col] < lower) | (df[col] > upper)).sum())
                    outliers_clipped += n_outliers
                    df[col] = df[col].clip(lower=lower, upper=upper)
            self.cleaning_report["steps"].append({"step": "handle_outliers", "clipped": outliers_clipped})

        self.cleaning_report["final_shape"] = df.shape
        self.cleaning_report["final_missing"] = df.isnull().sum().to_dict()
        self.cleaning_report["final_missing_total"] = int(df.isnull().sum().sum())

        return TaskResult(
            success=True,
            data={"dataframe": df, "cleaning_report": self.cleaning_report},
            metrics={"rows_before": original_shape[0], "rows_after": df.shape[0]}
        )

    async def _handle_missing_values(self, task: Dict[str, Any]) -> TaskResult:
        df = task.get("dataframe")
        if df is None:
            return TaskResult(success=False, error="No dataframe provided")

        for col in df.columns:
            if df[col].isnull().sum() > 0:
                if df[col].dtype in ['int64', 'float64']:
                    df[col] = df[col].fillna(df[col].median())
                else:
                    mode = df[col].mode()
                    df[col] = df[col].fillna(mode[0] if len(mode) > 0 else "Unknown")

        return TaskResult(success=True, data={"dataframe": df})

    async def _handle_outliers(self, task: Dict[str, Any]) -> TaskResult:
        df = task.get("dataframe")
        if df is None:
            return TaskResult(success=False, error="No dataframe provided")

        id_cols = {c for c in df.columns if _is_id_column(df, c)}
        numeric_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c not in id_cols]
        for col in numeric_cols:
            Q1, Q3 = df[col].quantile([0.25, 0.75])
            IQR = Q3 - Q1
            if IQR > 0:
                df[col] = df[col].clip(lower=Q1 - 1.5 * IQR, upper=Q3 + 1.5 * IQR)

        return TaskResult(success=True, data={"dataframe": df})

    async def _remove_duplicates(self, task: Dict[str, Any]) -> TaskResult:
        df = task.get("dataframe")
        if df is None:
            return TaskResult(success=False, error="No dataframe provided")

        before = len(df)
        df = df.drop_duplicates()

        return TaskResult(success=True, data={"dataframe": df, "removed": before - len(df)})

    async def _validate_data(self, task: Dict[str, Any]) -> TaskResult:
        df = task.get("dataframe")
        if df is None:
            return TaskResult(success=False, error="No dataframe provided")

        return TaskResult(success=True, data={
            "total_rows": len(df),
            "total_columns": len(df.columns),
            "missing_values": df.isnull().sum().to_dict(),
            "duplicates": int(df.duplicated().sum()),
            "data_types": df.dtypes.astype(str).to_dict()
        })
