"""
Report Generator Agent - Multi-Format Report Export

Generates professional reports in multiple formats:
- HTML (styled, dark-themed, self-contained, downloadable)
- Markdown (structured text for docs/wikis/README)
- CSV summary (tabular export of insights and metrics)
"""

import logging
import pandas as pd
from typing import Dict, Any, List
from datetime import datetime

from .base_agent import (
    BaseAgent, TaskResult, _sanitize_dataframe,
    get_numeric_cols, get_categorical_cols,
)

logger = logging.getLogger(__name__)


class ReportGeneratorAgent(BaseAgent):
    """Agent for generating exportable reports."""

    def __init__(self):
        super().__init__(
            name="ReportGeneratorAgent",
            description="Multi-format report generation (HTML, Markdown, CSV)",
            capabilities=["html_report", "markdown_report", "csv_summary"],
        )

    def get_system_prompt(self) -> str:
        return "You are an expert Report Generation Agent."

    async def execute(self, task: Dict[str, Any]) -> TaskResult:
        action = task.get("action", "generate_report")
        try:
            if action == "generate_report":
                return await self._generate_report(task)
            return TaskResult(success=False, error=f"Unknown action: {action}")
        except Exception as e:
            logger.error(f"Report error: {e}")
            return TaskResult(success=False, error=str(e))

    async def _generate_report(self, task: Dict[str, Any]) -> TaskResult:
        df = task.get("dataframe")
        if df is None:
            return TaskResult(success=False, error="No dataframe provided")

        df = _sanitize_dataframe(df.copy())

        insights = task.get("insights", [])
        recommendations = task.get("recommendations", [])
        model_results = task.get("model_results", {})
        eda_report = task.get("eda_report", {})
        target_column = task.get("target_column")
        title = task.get("title", "AI Analytics Report")
        fmt = task.get("format", "all")  # "html", "markdown", "csv", "all"

        result: Dict[str, Any] = {"title": title, "generated_at": datetime.now().isoformat()}

        if fmt in ("markdown", "all"):
            result["markdown"] = self._build_markdown(df, insights, recommendations, model_results, eda_report, target_column, title)

        if fmt in ("html", "all"):
            result["html"] = self._build_html(df, insights, recommendations, model_results, title)

        if fmt in ("csv", "all"):
            result["csv_summary"] = self._build_csv_summary(df, insights, recommendations)

        return TaskResult(success=True, data=result, metrics={"formats_generated": len(result) - 2})

    # ------------------------------------------------------------------
    # Markdown
    # ------------------------------------------------------------------
    def _build_markdown(self, df, insights, recommendations, model_results, eda_report, target, title) -> str:
        lines = [f"# {title}", f"*Generated on {datetime.now().strftime('%Y-%m-%d %H:%M')}*", ""]

        # Overview
        lines.append("## Dataset Overview")
        lines.append(f"- **Records**: {df.shape[0]:,}")
        lines.append(f"- **Features**: {df.shape[1]}")
        lines.append(f"- **Numeric**: {len(get_numeric_cols(df))}")
        lines.append(f"- **Categorical**: {len(get_categorical_cols(df))}")
        missing_pct = df.isnull().sum().sum() / max(df.shape[0] * df.shape[1], 1) * 100
        lines.append(f"- **Missing**: {missing_pct:.1f}%")
        if target:
            lines.append(f"- **Target**: {target}")
        lines.append("")

        # Insights
        if insights:
            lines.append("## Key Findings")
            high = [i for i in insights if i.get("priority") == "high"]
            medium = [i for i in insights if i.get("priority") == "medium"]
            low = [i for i in insights if i.get("priority") not in ("high", "medium")]

            for label, group in [("Critical", high), ("Important", medium), ("Note", low)]:
                if not group:
                    continue
                lines.append(f"### {label}")
                for item in group[:5]:
                    narrative = item.get("narrative", item.get("title", ""))
                    lines.append(f"- {narrative}")
                lines.append("")

        # Model results
        if model_results:
            lines.append("## Model Performance")
            results = model_results.get("results", model_results)
            best = model_results.get("best_model", "")
            task_type = model_results.get("task_type", "classification")
            metric_key = "accuracy" if task_type == "classification" else "r2"

            lines.append(f"| Model | {metric_key.upper()} | CV Mean |")
            lines.append("| --- | --- | --- |")
            for name, info in results.items():
                if not isinstance(info, dict) or "metrics" not in info:
                    continue
                m = info["metrics"]
                marker = " **BEST**" if name == best else ""
                lines.append(f"| {name}{marker} | {m.get(metric_key, 0):.4f} | {m.get('cv_mean', 0):.4f} |")
            lines.append("")

        # Recommendations
        if recommendations:
            lines.append("## Recommendations")
            for r in recommendations[:8]:
                action = r.get("action", r.get("title", ""))
                detail = r.get("detail", "")
                lines.append(f"- **{action}**: {detail}")
            lines.append("")

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # HTML
    # ------------------------------------------------------------------
    def _build_html(self, df, insights, recommendations, model_results, title) -> str:
        high = [i for i in insights if i.get("priority") == "high"]
        medium = [i for i in insights if i.get("priority") == "medium"]
        ts = datetime.now().strftime("%Y-%m-%d %H:%M")

        insights_html = ""
        for item in (high + medium)[:10]:
            prio = item.get("priority", "low")
            color = "#ef4444" if prio == "high" else "#f59e0b"
            narrative = item.get("narrative", item.get("title", ""))
            insights_html += f"""
            <div style="border-left:4px solid {color}; padding:12px 16px; margin:8px 0;
                 background:rgba(30,41,59,0.9); border-radius:0 8px 8px 0;">
              <strong style="color:{color}">[{prio.upper()}]</strong>
              <span style="color:#f1f5f9"> {narrative}</span>
            </div>"""

        recs_html = ""
        for r in recommendations[:8]:
            action = r.get("action", "")
            detail = r.get("detail", "")
            recs_html += f"""
            <div style="border-left:4px solid #10b981; padding:12px 16px; margin:8px 0;
                 background:rgba(30,41,59,0.9); border-radius:0 8px 8px 0;">
              <strong style="color:#10b981">{action}</strong>
              <p style="color:#94a3b8; margin:4px 0 0 0;">{detail}</p>
            </div>"""

        return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{title}</title>
<style>
  body {{ font-family:'DM Sans','Inter',sans-serif; background:#0f172a; color:#f1f5f9; padding:40px; }}
  .container {{ max-width:900px; margin:0 auto; }}
  h1 {{ background:linear-gradient(135deg,#6366f1,#10b981,#f59e0b);
       -webkit-background-clip:text; -webkit-text-fill-color:transparent; }}
  h2 {{ color:#818cf8; border-bottom:1px solid rgba(99,102,241,0.3); padding-bottom:8px; }}
  .stat-row {{ display:flex; gap:16px; margin:16px 0; flex-wrap:wrap; }}
  .stat-card {{ flex:1; min-width:120px; background:rgba(30,41,59,0.9);
       border:1px solid rgba(99,102,241,0.2); border-radius:12px; padding:16px; text-align:center; }}
  .stat-value {{ font-size:1.5rem; font-weight:700; color:#6366f1; }}
  .stat-label {{ font-size:0.85rem; color:#94a3b8; }}
</style>
</head>
<body>
<div class="container">
  <h1>{title}</h1>
  <p style="color:#94a3b8;">Generated {ts}</p>

  <h2>Dataset Overview</h2>
  <div class="stat-row">
    <div class="stat-card"><div class="stat-value">{df.shape[0]:,}</div><div class="stat-label">Records</div></div>
    <div class="stat-card"><div class="stat-value">{df.shape[1]}</div><div class="stat-label">Features</div></div>
    <div class="stat-card"><div class="stat-value">{len(get_numeric_cols(df))}</div><div class="stat-label">Numeric</div></div>
    <div class="stat-card"><div class="stat-value">{len(get_categorical_cols(df))}</div><div class="stat-label">Categorical</div></div>
  </div>

  <h2>Key Findings ({len(high)} critical, {len(medium)} important)</h2>
  {insights_html if insights_html else '<p style="color:#94a3b8;">No significant findings.</p>'}

  <h2>Recommendations</h2>
  {recs_html if recs_html else '<p style="color:#94a3b8;">Run the full pipeline for recommendations.</p>'}
</div>
</body>
</html>"""

    # ------------------------------------------------------------------
    # CSV summary
    # ------------------------------------------------------------------
    def _build_csv_summary(self, df, insights, recommendations) -> str:
        rows = ["category,priority,title,detail"]
        for i in insights:
            cat = i.get("category", "")
            prio = i.get("priority", "")
            title = i.get("title", "").replace(",", ";")
            narr = i.get("narrative", "").replace(",", ";").replace("\n", " ")
            rows.append(f"{cat},{prio},{title},{narr}")
        for r in recommendations:
            action = r.get("action", "").replace(",", ";")
            detail = r.get("detail", "").replace(",", ";")
            rows.append(f"recommendation,{r.get('priority','')},{action},{detail}")
        return "\n".join(rows)
