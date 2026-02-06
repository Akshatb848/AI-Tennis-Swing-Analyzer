"""
EDA Agent - Exploratory Data Analysis
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List
from datetime import datetime
from scipy import stats

from .base_agent import BaseAgent, TaskResult

logger = logging.getLogger(__name__)


class EDAAgent(BaseAgent):
    """Agent for Exploratory Data Analysis."""
    
    def __init__(self):
        super().__init__(
            name="EDAAgent",
            description="Exploratory data analysis and statistical profiling",
            capabilities=["statistical_profiling", "distribution_analysis", "correlation_analysis", "insight_generation"]
        )
        self.eda_report: Dict[str, Any] = {}
    
    def get_system_prompt(self) -> str:
        return "You are an expert EDA Agent."

    async def execute(self, task: Dict[str, Any]) -> TaskResult:
        action = task.get("action", "full_eda")
        
        try:
            if action == "full_eda":
                return await self._full_eda(task)
            elif action == "statistical_profile":
                return await self._statistical_profile(task)
            elif action == "correlation_analysis":
                return await self._correlation_analysis(task)
            return TaskResult(success=False, error=f"Unknown action: {action}")
        except Exception as e:
            return TaskResult(success=False, error=str(e))
    
    async def _full_eda(self, task: Dict[str, Any]) -> TaskResult:
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

        target_column = task.get("target_column")

        self.eda_report = {
            "timestamp": datetime.now().isoformat(),
            "dataset_info": self._get_dataset_info(df),
            "statistical_profile": self._compute_statistical_profile(df),
            "distributions": self._analyze_distributions(df),
            "correlations": self._compute_correlations(df),
            "insights": [],
            "recommendations": []
        }
        
        if target_column and target_column in df.columns:
            self.eda_report["target_analysis"] = self._analyze_target(df, target_column)
        
        insights = self._generate_insights(df, target_column)
        self.eda_report["insights"] = insights["insights"]
        self.eda_report["recommendations"] = insights["recommendations"]
        
        return TaskResult(
            success=True,
            data=self.eda_report,
            metrics={"columns_analyzed": len(df.columns), "insights_generated": len(self.eda_report["insights"])}
        )
    
    def _get_dataset_info(self, df: pd.DataFrame) -> Dict[str, Any]:
        return {
            "shape": {"rows": df.shape[0], "columns": df.shape[1]},
            "columns": list(df.columns),
            "dtypes": df.dtypes.astype(str).to_dict(),
            "memory_usage_mb": float(df.memory_usage(deep=True).sum() / 1024 / 1024),
            "missing_values": df.isnull().sum().to_dict(),
            "duplicates": int(df.duplicated().sum())
        }
    
    def _compute_statistical_profile(self, df: pd.DataFrame) -> Dict[str, Any]:
        profile = {"numeric": {}, "categorical": {}}
        
        for col in df.select_dtypes(include=[np.number]).columns:
            data = df[col].dropna()
            if len(data) == 0:
                continue
            profile["numeric"][col] = {
                "count": int(len(data)), "mean": float(data.mean()), "std": float(data.std()),
                "min": float(data.min()), "25%": float(data.quantile(0.25)),
                "50%": float(data.quantile(0.50)), "75%": float(data.quantile(0.75)),
                "max": float(data.max()), "skewness": float(data.skew()), "kurtosis": float(data.kurtosis())
            }
        
        for col in df.select_dtypes(include=['object', 'category', 'string']).columns:
            data = df[col].dropna()
            if len(data) == 0:
                continue
            vc = data.value_counts()
            profile["categorical"][col] = {
                "count": int(len(data)), "unique": int(data.nunique()),
                "top": str(vc.index[0]) if len(vc) > 0 else None,
                "top_freq": int(vc.iloc[0]) if len(vc) > 0 else 0
            }
        
        return profile
    
    def _analyze_distributions(self, df: pd.DataFrame) -> Dict[str, Any]:
        distributions = {}
        for col in df.select_dtypes(include=[np.number]).columns:
            data = df[col].dropna()
            if len(data) < 10:
                continue
            skewness = data.skew()
            distributions[col] = {
                "skewness": float(skewness),
                "kurtosis": float(data.kurtosis()),
                "skew_type": "symmetric" if abs(skewness) < 0.5 else ("right_skewed" if skewness > 0 else "left_skewed")
            }
        return distributions
    
    def _compute_correlations(self, df: pd.DataFrame) -> Dict[str, Any]:
        numeric_df = df.select_dtypes(include=[np.number])
        if numeric_df.shape[1] < 2:
            return {"message": "Not enough numeric columns"}
        
        corr = numeric_df.corr()
        high_corr = []
        for i in range(len(corr.columns)):
            for j in range(i + 1, len(corr.columns)):
                if abs(corr.iloc[i, j]) > 0.7:
                    high_corr.append({
                        "feature1": corr.columns[i],
                        "feature2": corr.columns[j],
                        "correlation": float(corr.iloc[i, j])
                    })
        
        return {"correlation_matrix": corr.to_dict(), "high_correlations": high_corr}
    
    def _analyze_target(self, df: pd.DataFrame, target: str) -> Dict[str, Any]:
        target_data = df[target]
        analysis = {"column": target, "dtype": str(target_data.dtype), "unique": int(target_data.nunique())}

        is_numeric = pd.api.types.is_numeric_dtype(target_data)
        if target_data.nunique() <= 10 or not is_numeric:
            analysis["task_type"] = "classification"
            analysis["class_distribution"] = target_data.value_counts().to_dict()
        else:
            analysis["task_type"] = "regression"
            analysis["statistics"] = {
                "mean": float(target_data.mean()), "std": float(target_data.std()),
                "min": float(target_data.min()), "max": float(target_data.max())
            }

        return analysis
    
    def _generate_insights(self, df: pd.DataFrame, target: Optional[str] = None) -> Dict[str, Any]:
        insights, recommendations = [], []
        
        total_cells = len(df) * len(df.columns)
        missing_pct = df.isnull().sum().sum() / total_cells * 100 if total_cells > 0 else 0
        if missing_pct > 10:
            insights.append(f"⚠️ {missing_pct:.1f}% missing values detected")
            recommendations.append("Implement sophisticated imputation")
        
        if df.duplicated().sum() > len(df) * 0.05:
            insights.append(f"🔄 {df.duplicated().sum() / len(df) * 100:.1f}% duplicate rows")
        
        for col in df.select_dtypes(include=[np.number]).columns[:5]:
            if abs(df[col].skew()) > 2:
                insights.append(f"📈 '{col}' is highly skewed")
        
        return {"insights": insights, "recommendations": recommendations}
    
    async def _statistical_profile(self, task: Dict[str, Any]) -> TaskResult:
        df = task.get("dataframe")
        if df is None:
            return TaskResult(success=False, error="No dataframe provided")
        df = df.copy()
        for col in df.columns:
            if pd.api.types.is_string_dtype(df[col]) and df[col].dtype != "object":
                df[col] = df[col].astype(object)
        return TaskResult(success=True, data=self._compute_statistical_profile(df))

    async def _correlation_analysis(self, task: Dict[str, Any]) -> TaskResult:
        df = task.get("dataframe")
        if df is None:
            return TaskResult(success=False, error="No dataframe provided")
        df = df.copy()
        for col in df.columns:
            if pd.api.types.is_string_dtype(df[col]) and df[col].dtype != "object":
                df[col] = df[col].astype(object)
        return TaskResult(success=True, data=self._compute_correlations(df))
