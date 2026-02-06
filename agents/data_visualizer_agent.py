"""
Data Visualizer Agent - Data Visualization
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List
from datetime import datetime

from .base_agent import BaseAgent, TaskResult

logger = logging.getLogger(__name__)


class DataVisualizerAgent(BaseAgent):
    """Agent for data visualization."""
    
    def __init__(self):
        super().__init__(
            name="DataVisualizerAgent",
            description="Data visualization and charting",
            capabilities=["chart_generation", "distribution_plots", "correlation_plots", "custom_visualizations"]
        )
        self.charts: List[Dict[str, Any]] = []
    
    def get_system_prompt(self) -> str:
        return "You are an expert Data Visualization Agent."

    async def execute(self, task: Dict[str, Any]) -> TaskResult:
        action = task.get("action", "generate_visualizations")
        
        try:
            if action == "generate_visualizations":
                return await self._generate_visualizations(task)
            elif action == "create_chart":
                return await self._create_chart(task)
            return TaskResult(success=False, error=f"Unknown action: {action}")
        except Exception as e:
            return TaskResult(success=False, error=str(e))
    
    async def _generate_visualizations(self, task: Dict[str, Any]) -> TaskResult:
        df = task.get("dataframe")
        if df is None:
            return TaskResult(success=False, error="No dataframe provided")
        
        chart_types = task.get("chart_types", ["distribution", "correlation", "scatter", "boxplot"])
        self.charts = []
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object', 'category', 'str']).columns.tolist()
        
        # Distribution plots
        if "distribution" in chart_types:
            for col in numeric_cols[:6]:
                self.charts.append({
                    "type": "histogram",
                    "title": f"Distribution: {col}",
                    "column": col,
                    "data": {"values": df[col].dropna().tolist()[:1000]}
                })
        
        # Correlation heatmap
        if "correlation" in chart_types and len(numeric_cols) > 1:
            corr = df[numeric_cols].corr()
            self.charts.append({
                "type": "heatmap",
                "title": "Correlation Matrix",
                "data": {"matrix": corr.values.tolist(), "labels": corr.columns.tolist()}
            })
        
        # Scatter plots
        if "scatter" in chart_types and len(numeric_cols) >= 2:
            self.charts.append({
                "type": "scatter",
                "title": f"{numeric_cols[0]} vs {numeric_cols[1]}",
                "data": {
                    "x": df[numeric_cols[0]].dropna().tolist()[:500],
                    "y": df[numeric_cols[1]].dropna().tolist()[:500],
                    "x_label": numeric_cols[0],
                    "y_label": numeric_cols[1]
                }
            })
        
        # Box plots
        if "boxplot" in chart_types:
            for col in numeric_cols[:4]:
                q1, median, q3 = df[col].quantile([0.25, 0.5, 0.75])
                self.charts.append({
                    "type": "boxplot",
                    "title": f"Boxplot: {col}",
                    "column": col,
                    "data": {
                        "min": float(df[col].min()),
                        "q1": float(q1),
                        "median": float(median),
                        "q3": float(q3),
                        "max": float(df[col].max())
                    }
                })
        
        # Bar charts for categorical
        for col in categorical_cols[:3]:
            vc = df[col].value_counts().head(10)
            self.charts.append({
                "type": "bar",
                "title": f"Distribution: {col}",
                "column": col,
                "data": {"labels": vc.index.tolist(), "values": vc.values.tolist()}
            })
        
        return TaskResult(
            success=True,
            data={"charts": self.charts},
            metrics={"charts_generated": len(self.charts)}
        )
    
    async def _create_chart(self, task: Dict[str, Any]) -> TaskResult:
        chart_type = task.get("chart_type")
        df = task.get("dataframe")
        column = task.get("column")
        
        if df is None or column is None:
            return TaskResult(success=False, error="Missing dataframe or column")
        
        if chart_type == "histogram":
            data = {"values": df[column].dropna().tolist()[:1000]}
        elif chart_type == "bar":
            vc = df[column].value_counts().head(10)
            data = {"labels": vc.index.tolist(), "values": vc.values.tolist()}
        else:
            data = {"values": df[column].dropna().tolist()[:1000]}
        
        chart = {"type": chart_type, "title": f"{chart_type.title()}: {column}", "column": column, "data": data}
        self.charts.append(chart)
        
        return TaskResult(success=True, data={"chart": chart})
