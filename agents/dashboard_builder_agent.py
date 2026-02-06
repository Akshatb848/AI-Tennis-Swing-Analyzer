"""
Dashboard Builder Agent - Interactive Dashboard Generation
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List
from datetime import datetime
import os

from .base_agent import BaseAgent, TaskResult

logger = logging.getLogger(__name__)


class DashboardBuilderAgent(BaseAgent):
    """Agent for building interactive dashboards."""
    
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
        df = task.get("dataframe")
        title = task.get("title", "Data Science Dashboard")
        
        components = []
        
        if df is not None:
            # KPI cards
            kpis = [
                {"title": "Total Records", "value": f"{len(df):,}", "icon": "📊", "color": "#3498db"},
                {"title": "Features", "value": str(len(df.columns)), "icon": "🔢", "color": "#2ecc71"},
                {"title": "Missing %", "value": f"{df.isnull().sum().sum() / (len(df) * len(df.columns)) * 100:.1f}%", "icon": "⚠️", "color": "#e74c3c"},
                {"title": "Numeric Cols", "value": str(len(df.select_dtypes(include=[np.number]).columns)), "icon": "🔢", "color": "#9b59b6"}
            ]
            components.append({"type": "kpi_section", "data": kpis})
            
            # Charts
            charts = []
            for col in df.select_dtypes(include=[np.number]).columns[:4]:
                charts.append({
                    "type": "histogram",
                    "title": f"Distribution of {col}",
                    "data": {"values": df[col].dropna().tolist()[:1000], "column": col}
                })
            
            if len(df.select_dtypes(include=[np.number]).columns) > 1:
                corr = df.select_dtypes(include=[np.number]).corr()
                charts.append({
                    "type": "heatmap",
                    "title": "Correlations",
                    "data": {"matrix": corr.values.tolist(), "labels": corr.columns.tolist()}
                })
            
            components.append({"type": "chart_section", "data": charts})
        
        if model_results:
            components.append({"type": "model_section", "data": model_results})
        
        self.dashboard_config = {
            "title": title,
            "components": len(components),
            "created_at": datetime.now().isoformat()
        }
        
        return TaskResult(
            success=True,
            data={"config": self.dashboard_config, "components": components},
            metrics={"components_count": len(components)}
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
