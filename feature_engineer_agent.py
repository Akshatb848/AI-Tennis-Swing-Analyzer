"""
Feature Engineer Agent - Automated Feature Engineering
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime

from .base_agent import BaseAgent, TaskResult

logger = logging.getLogger(__name__)


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
        target = task.get("target_column")
        original_cols = list(df.columns)
        
        self.feature_report = {
            "timestamp": datetime.now().isoformat(),
            "original_features": original_cols,
            "created_features": [],
            "encoded_features": [],
            "scaled_features": [],
            "steps": []
        }
        
        # Extract datetime features
        for col in df.columns:
            if df[col].dtype == 'object':
                try:
                    df[col] = pd.to_datetime(df[col])
                except:
                    pass
        
        for col in df.select_dtypes(include=['datetime64']).columns:
            df[f'{col}_year'] = df[col].dt.year
            df[f'{col}_month'] = df[col].dt.month
            df[f'{col}_day'] = df[col].dt.day
            df[f'{col}_dayofweek'] = df[col].dt.dayofweek
            self.feature_report["created_features"].extend([
                f'{col}_year', f'{col}_month', f'{col}_day', f'{col}_dayofweek'
            ])
            df = df.drop(columns=[col])
        
        # Create interaction features
        if task.get("create_interactions", True):
            numeric_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c != target]
            for i, c1 in enumerate(numeric_cols[:5]):
                for c2 in numeric_cols[i+1:6]:
                    df[f'{c1}_x_{c2}'] = df[c1] * df[c2]
                    self.feature_report["created_features"].append(f'{c1}_x_{c2}')
        
        # Transform skewed features
        for col in [c for c in df.select_dtypes(include=[np.number]).columns if c != target]:
            if abs(df[col].skew()) > 1 and (df[col] > 0).all():
                df[f'{col}_log'] = np.log1p(df[col])
                self.feature_report["created_features"].append(f'{col}_log')
        
        # Encode categorical
        encoding = task.get("encoding", "auto")
        for col in [c for c in df.select_dtypes(include=['object', 'category']).columns if c != target]:
            n_unique = df[col].nunique()
            if n_unique == 2:
                df[col] = pd.factorize(df[col])[0]
            elif n_unique <= 10:
                dummies = pd.get_dummies(df[col], prefix=col, drop_first=True)
                df = pd.concat([df.drop(columns=[col]), dummies], axis=1)
            else:
                freq_map = df[col].value_counts(normalize=True).to_dict()
                df[f'{col}_freq'] = df[col].map(freq_map)
                df = df.drop(columns=[col])
            self.feature_report["encoded_features"].append(col)
        
        # Scale numerical
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
        
        target = task.get("target_column")
        for col in [c for c in df.select_dtypes(include=['object', 'category']).columns if c != target]:
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
