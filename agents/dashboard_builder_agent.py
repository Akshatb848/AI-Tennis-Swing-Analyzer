"""
Dashboard Builder Agent - Professional Dashboard Generation

Produces PowerBI/Tableau-quality dashboard data structures that include:
- KPI metric cards with delta indicators
- Data quality section
- Distribution and correlation charts
- Target analysis (classification/regression)
- Model comparison charts with feature importance
- Insights and recommendations panel
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List
from datetime import datetime

from .base_agent import BaseAgent, TaskResult

logger = logging.getLogger(__name__)


class DashboardBuilderAgent(BaseAgent):
    """Agent for building professional interactive dashboards."""

    def __init__(self):
        super().__init__(
            name="DashboardBuilderAgent",
            description="Interactive dashboard generation",
            capabilities=["dashboard_creation", "chart_generation", "kpi_cards", "report_generation"]
        )
        self.dashboard_config: Dict[str, Any] = {}

    def get_system_prompt(self) -> str:
        return "You are an expert Dashboard Building Agent."

    async def execute(self, task: Dict[str, Any]) -> TaskResult:
        action = task.get("action", "build_dashboard")

        try:
            if action == "build_dashboard":
                return await self._build_dashboard(task)
            elif action == "create_chart":
                return await self._create_chart(task)
            return TaskResult(success=False, error=f"Unknown action: {action}")
        except Exception as e:
            return TaskResult(success=False, error=str(e))

    async def _build_dashboard(self, task: Dict[str, Any]) -> TaskResult:
        project_id = task.get("project_id", "default")
        eda_report = task.get("eda_report", {})
        model_results = task.get("model_results", {})
        cleaning_report = task.get("cleaning_report", {})
        df = task.get("dataframe")
        target_column = task.get("target_column")
        title = task.get("title", "Data Science Dashboard")

        # Convert pandas 3.x StringDtype to object for numpy compatibility
        if df is not None:
            df = df.copy()
            for col in df.columns:
                if pd.api.types.is_string_dtype(df[col]) and df[col].dtype != "object":
                    df[col] = df[col].astype(object)

        sections: List[Dict[str, Any]] = []

        # ---- Section 1: KPI Cards ----
        if df is not None:
            n_rows = len(df)
            n_cols = len(df.columns)
            n_numeric = len(df.select_dtypes(include=[np.number]).columns)
            n_categorical = len(df.select_dtypes(include=['object', 'category', 'string']).columns)
            missing_pct = df.isnull().sum().sum() / (n_rows * n_cols) * 100 if (n_rows * n_cols) > 0 else 0
            dup_count = int(df.duplicated().sum())

            kpis = [
                {"title": "Total Records", "value": f"{n_rows:,}", "color": "#3498db"},
                {"title": "Features", "value": str(n_cols), "color": "#2ecc71"},
                {"title": "Missing %", "value": f"{missing_pct:.1f}%", "color": "#e74c3c" if missing_pct > 5 else "#2ecc71"},
                {"title": "Numeric", "value": str(n_numeric), "color": "#9b59b6"},
                {"title": "Categorical", "value": str(n_categorical), "color": "#f39c12"},
                {"title": "Duplicates", "value": str(dup_count), "color": "#e74c3c" if dup_count > 0 else "#2ecc71"},
            ]
            sections.append({"type": "kpi_section", "title": "Dataset Overview", "data": kpis})

        # ---- Section 2: Data Quality ----
        if df is not None:
            # Per-column missing values (top 10 worst)
            missing_by_col = df.isnull().sum()
            missing_cols = missing_by_col[missing_by_col > 0].sort_values(ascending=False).head(10)
            quality_data = {
                "quality_score": round(100 - missing_pct, 1),
                "total_cells": int(n_rows * n_cols),
                "total_missing": int(df.isnull().sum().sum()),
                "missing_by_column": {str(k): int(v) for k, v in missing_cols.items()},
                "dtypes_summary": {
                    "numeric": n_numeric,
                    "categorical": n_categorical,
                    "datetime": len(df.select_dtypes(include=['datetime64']).columns),
                },
            }
            sections.append({"type": "data_quality_section", "title": "Data Quality", "data": quality_data})

        # ---- Section 3: Distribution Charts ----
        if df is not None:
            charts = []
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            categorical_cols = df.select_dtypes(include=['object', 'category', 'string']).columns.tolist()

            # Numeric distributions (top 6)
            for col in numeric_cols[:6]:
                values = df[col].dropna()
                if len(values) == 0:
                    continue
                charts.append({
                    "type": "histogram",
                    "title": f"Distribution: {col}",
                    "column": col,
                    "data": {
                        "values": values.tolist()[:2000],
                        "mean": float(values.mean()),
                        "median": float(values.median()),
                        "std": float(values.std()),
                    }
                })

            # Correlation heatmap
            if len(numeric_cols) > 1:
                corr = df[numeric_cols].corr()
                charts.append({
                    "type": "heatmap",
                    "title": "Feature Correlations",
                    "data": {
                        "matrix": corr.values.tolist(),
                        "labels": corr.columns.tolist(),
                    }
                })

            # Categorical bar charts (top 4)
            for col in categorical_cols[:4]:
                vc = df[col].value_counts().head(10)
                charts.append({
                    "type": "bar",
                    "title": f"Distribution: {col}",
                    "column": col,
                    "data": {
                        "labels": [str(l) for l in vc.index.tolist()],
                        "values": vc.values.tolist(),
                    }
                })

            if charts:
                sections.append({"type": "chart_section", "title": "Visual Analytics", "data": charts})

        # ---- Section 4: Target Analysis ----
        if df is not None and target_column and target_column in df.columns:
            target_data = df[target_column]
            target_info: Dict[str, Any] = {
                "column": target_column,
                "dtype": str(target_data.dtype),
                "unique": int(target_data.nunique()),
            }

            is_numeric_target = pd.api.types.is_numeric_dtype(target_data)
            if target_data.nunique() <= 10 or not is_numeric_target:
                target_info["task_type"] = "classification"
                vc = target_data.value_counts()
                target_info["class_distribution"] = {
                    "labels": [str(l) for l in vc.index.tolist()],
                    "values": vc.values.tolist(),
                }
                if len(vc) > 0:
                    target_info["class_balance"] = (
                        "Balanced" if vc.max() / max(vc.min(), 1) < 2 else "Imbalanced"
                    )
                else:
                    target_info["class_balance"] = "Unknown (all values missing)"
            else:
                target_info["task_type"] = "regression"
                target_info["statistics"] = {
                    "mean": float(target_data.mean()) if np.issubdtype(target_data.dtype, np.number) else None,
                    "std": float(target_data.std()) if np.issubdtype(target_data.dtype, np.number) else None,
                    "min": float(target_data.min()) if np.issubdtype(target_data.dtype, np.number) else None,
                    "max": float(target_data.max()) if np.issubdtype(target_data.dtype, np.number) else None,
                }
                if np.issubdtype(target_data.dtype, np.number):
                    target_info["distribution_values"] = target_data.dropna().tolist()[:2000]

            sections.append({"type": "target_analysis_section", "title": "Target Variable Analysis", "data": target_info})

        # ---- Section 5: Model Comparison ----
        if model_results:
            results = model_results.get("results", model_results)
            task_type = model_results.get("task_type", "classification")
            best_model = model_results.get("best_model", "")
            primary_metric = "accuracy" if task_type == "classification" else "r2"

            model_comparison = {
                "task_type": task_type,
                "best_model": best_model,
                "primary_metric": primary_metric,
                "results": {},
                "feature_importance": {},
            }

            for name, info in results.items():
                if isinstance(info, dict) and "metrics" in info:
                    model_comparison["results"][name] = info["metrics"]
                    if info.get("feature_importance"):
                        model_comparison["feature_importance"][name] = info["feature_importance"]

            if model_comparison["results"]:
                sections.append({"type": "model_comparison_section", "title": "Model Performance", "data": model_comparison})

        # ---- Section 6: Insights & Recommendations ----
        insights = []
        recommendations = []

        if df is not None:
            missing_pct_val = df.isnull().sum().sum() / (len(df) * len(df.columns)) * 100 if (len(df) * len(df.columns)) > 0 else 0
            if missing_pct_val > 10:
                insights.append(f"High missing data rate: {missing_pct_val:.1f}%")
                recommendations.append("Consider advanced imputation methods (KNN, iterative)")
            elif missing_pct_val > 0:
                insights.append(f"Missing data present: {missing_pct_val:.1f}%")

            dup_pct = df.duplicated().sum() / len(df) * 100 if len(df) > 0 else 0
            if dup_pct > 5:
                insights.append(f"High duplicate rate: {dup_pct:.1f}%")
                recommendations.append("Review and remove duplicate rows")

            numeric_cols = df.select_dtypes(include=[np.number]).columns
            for col in numeric_cols[:5]:
                skew = df[col].skew()
                if pd.notna(skew) and abs(skew) > 2:
                    insights.append(f"'{col}' is highly skewed (skewness: {skew:.2f})")

            if len(numeric_cols) > 1:
                corr = df[numeric_cols].corr()
                for i in range(len(corr.columns)):
                    for j in range(i + 1, len(corr.columns)):
                        if abs(corr.iloc[i, j]) > 0.9:
                            insights.append(
                                f"Strong correlation between '{corr.columns[i]}' and '{corr.columns[j]}' ({corr.iloc[i, j]:.2f})"
                            )

        if model_results:
            best = model_results.get("best_model", "")
            best_metrics = model_results.get("best_metrics", {})
            if best:
                acc = best_metrics.get("accuracy") or best_metrics.get("r2")
                if acc is not None:
                    metric_name = "accuracy" if "accuracy" in best_metrics else "R2"
                    insights.append(f"Best model: {best} ({metric_name}: {acc:.4f})")
                    if isinstance(acc, float) and acc < 0.6:
                        recommendations.append("Model performance is low — try more feature engineering or gather more data")

        if not recommendations:
            recommendations.append("Run the full pipeline for comprehensive analysis")

        if insights or recommendations:
            sections.append({
                "type": "insights_section",
                "title": "Insights & Recommendations",
                "data": {"insights": insights, "recommendations": recommendations},
            })

        self.dashboard_config = {
            "title": title,
            "sections": len(sections),
            "created_at": datetime.now().isoformat(),
        }

        return TaskResult(
            success=True,
            data={"config": self.dashboard_config, "components": sections},
            metrics={"sections_count": len(sections)},
        )

    async def _create_chart(self, task: Dict[str, Any]) -> TaskResult:
        chart_type = task.get("chart_type", "histogram")
        data = task.get("data")
        title = task.get("title", "Chart")

        return TaskResult(success=True, data={
            "type": chart_type,
            "title": title,
            "data": data
        })
