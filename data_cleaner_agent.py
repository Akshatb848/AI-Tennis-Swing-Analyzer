"""
Data Cleaner Agent - Handles data preprocessing and cleaning
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime

from .base_agent import BaseAgent, TaskResult

logger = logging.getLogger(__name__)


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
        self.cleaning_report = {
            "original_shape": original_shape,
            "original_missing": df.isnull().sum().to_dict(),
            "original_duplicates": int(df.duplicated().sum()),
            "steps": [],
            "timestamp": datetime.now().isoformat()
        }
        
        # Remove duplicates
        if task.get("remove_duplicates", True):
            before = len(df)
            df = df.drop_duplicates()
            self.cleaning_report["steps"].append({"step": "remove_duplicates", "removed": before - len(df)})
        
        # Handle missing values
        if task.get("handle_missing", True):
            for col in df.columns:
                if df[col].isnull().sum() > 0:
                    if df[col].dtype in ['int64', 'float64']:
                        df[col] = df[col].fillna(df[col].median())
                    else:
                        mode = df[col].mode()
                        df[col] = df[col].fillna(mode[0] if len(mode) > 0 else "Unknown")
        
        # Handle outliers
        if task.get("handle_outliers", True):
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                Q1, Q3 = df[col].quantile([0.25, 0.75])
                IQR = Q3 - Q1
                df[col] = df[col].clip(lower=Q1 - 1.5 * IQR, upper=Q3 + 1.5 * IQR)
        
        self.cleaning_report["final_shape"] = df.shape
        self.cleaning_report["final_missing"] = df.isnull().sum().to_dict()
        
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
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            Q1, Q3 = df[col].quantile([0.25, 0.75])
            IQR = Q3 - Q1
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
