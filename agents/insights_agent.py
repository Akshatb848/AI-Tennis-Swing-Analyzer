"""
Insights Agent - Enhanced Business Intelligence & Narrative Generation

Combines capabilities from both platforms:
- Trend analysis (time-based and overall)
- Correlation discovery with narrative explanations
- Anomaly detection (IQR-based)
- Statistical hypothesis testing (t-tests between groups)
- Seasonality detection (autocorrelation)
- Distribution analysis (skewness/kurtosis)
- Performance gap analysis across categories
- Executive summary generation
- Business recommendations
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime
from scipy import stats

from .base_agent import (
    BaseAgent, TaskResult, _sanitize_dataframe,
    get_numeric_cols, get_categorical_cols, get_datetime_cols,
)

logger = logging.getLogger(__name__)


def _detect_date_column(df: pd.DataFrame) -> Optional[str]:
    """Auto-detect the best date/time column."""
    for col in get_datetime_cols(df):
        return col
    for col in df.columns:
        if pd.api.types.is_object_dtype(df[col]) or pd.api.types.is_string_dtype(df[col]):
            name = col.lower()
            if any(kw in name for kw in ["date", "time", "timestamp"]):
                try:
                    parsed = pd.to_datetime(df[col], errors="coerce")
                    if parsed.notna().sum() > len(parsed) * 0.5:
                        return col
                except Exception:
                    pass
    return None


class InsightsAgent(BaseAgent):
    """Agent for automated business insight generation with narratives."""

    def __init__(self):
        super().__init__(
            name="InsightsAgent",
            description="Business intelligence, insight discovery, and narrative generation",
            capabilities=[
                "trend_analysis", "correlation_discovery", "anomaly_detection",
                "hypothesis_testing", "seasonality_detection", "executive_summary",
                "recommendations",
            ],
        )

    def get_system_prompt(self) -> str:
        return "You are an expert Business Intelligence Agent."

    async def execute(self, task: Dict[str, Any]) -> TaskResult:
        action = task.get("action", "generate_insights")
        try:
            if action == "generate_insights":
                return await self._generate_insights(task)
            elif action == "executive_summary":
                return await self._executive_summary(task)
            return TaskResult(success=False, error=f"Unknown action: {action}")
        except Exception as e:
            logger.error(f"Insights error: {e}")
            return TaskResult(success=False, error=str(e))

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------
    async def _generate_insights(self, task: Dict[str, Any]) -> TaskResult:
        df = task.get("dataframe")
        if df is None:
            return TaskResult(success=False, error="No dataframe provided")

        df = _sanitize_dataframe(df.copy())
        target = task.get("target_column")

        numeric_cols = get_numeric_cols(df)
        categorical_cols = get_categorical_cols(df)
        date_col = _detect_date_column(df)

        insights: List[Dict[str, Any]] = []
        recommendations: List[Dict[str, Any]] = []

        # 1. Trend analysis
        insights.extend(self._analyze_trends(df, numeric_cols, date_col))

        # 2. Correlation discovery
        insights.extend(self._analyze_correlations(df, numeric_cols))

        # 3. Anomaly detection
        insights.extend(self._detect_anomalies(df, numeric_cols))

        # 4. Distribution analysis
        insights.extend(self._analyze_distributions(df, numeric_cols))

        # 5. Statistical tests
        insights.extend(self._perform_statistical_tests(df, numeric_cols, categorical_cols))

        # 6. Seasonality detection
        if date_col:
            insights.extend(self._detect_seasonality(df, numeric_cols, date_col))

        # 7. Performance gaps
        insights.extend(self._analyze_performance_gaps(df, numeric_cols, categorical_cols))

        # 8. Recommendations
        recommendations = self._generate_recommendations(df, insights, target)

        # Sort by priority
        priority_order = {"high": 0, "medium": 1, "low": 2}
        insights.sort(key=lambda x: priority_order.get(x.get("priority", "low"), 3))

        # Generate executive summary
        exec_summary = self._build_executive_summary(df, insights, recommendations)

        return TaskResult(
            success=True,
            data={
                "insights": insights,
                "recommendations": recommendations,
                "executive_summary": exec_summary,
                "summary": {
                    "total_insights": len(insights),
                    "high_priority": sum(1 for i in insights if i.get("priority") == "high"),
                    "categories": list({i.get("category", "general") for i in insights}),
                },
            },
            metrics={"insights_count": len(insights), "recommendations_count": len(recommendations)},
        )

    async def _executive_summary(self, task: Dict[str, Any]) -> TaskResult:
        result = await self._generate_insights(task)
        if not result.success:
            return result

        insights = result.data["insights"]
        recs = result.data["recommendations"]
        df = task.get("dataframe")

        summary_lines = [f"Dataset: {df.shape[0]:,} records x {df.shape[1]} features"]
        high = [i for i in insights if i.get("priority") == "high"]
        if high:
            summary_lines.append(f"Critical Findings: {len(high)}")
            for i, h in enumerate(high[:5], 1):
                summary_lines.append(f"  {i}. {h.get('narrative', h.get('title', ''))}")

        if recs:
            summary_lines.append(f"Top Recommendations:")
            for i, r in enumerate(recs[:5], 1):
                summary_lines.append(f"  {i}. {r.get('action', r.get('title', ''))}")

        return TaskResult(
            success=True,
            data={
                "executive_summary": "\n".join(summary_lines),
                "insights": insights,
                "recommendations": recs,
            },
        )

    # ------------------------------------------------------------------
    # Analysis methods
    # ------------------------------------------------------------------
    def _analyze_trends(self, df, numeric_cols, date_col) -> List[Dict]:
        insights = []
        if not date_col or not numeric_cols:
            return insights

        try:
            ts = df.copy()
            ts[date_col] = pd.to_datetime(ts[date_col], errors="coerce")
            ts = ts.dropna(subset=[date_col]).sort_values(date_col)
        except Exception:
            return insights

        n = len(ts)
        if n < 10:
            return insights

        recent = ts.tail(max(1, n // 5))
        early = ts.head(max(1, n // 5))

        for col in numeric_cols[:10]:
            early_mean = early[col].mean()
            recent_mean = recent[col].mean()
            if pd.isna(early_mean) or pd.isna(recent_mean) or early_mean == 0:
                continue
            change_pct = ((recent_mean - early_mean) / abs(early_mean)) * 100

            if abs(change_pct) > 10:
                direction = "increased" if change_pct > 0 else "decreased"
                insights.append({
                    "category": "trend",
                    "priority": "high" if abs(change_pct) > 30 else "medium",
                    "title": f"{col} {direction} by {abs(change_pct):.1f}%",
                    "narrative": (
                        f"'{col}' has {direction} by {abs(change_pct):.1f}% comparing "
                        f"the earliest period (avg {early_mean:,.2f}) to the most "
                        f"recent period (avg {recent_mean:,.2f})."
                    ),
                    "metric": col,
                    "change_pct": round(change_pct, 2),
                })
        return insights

    def _analyze_correlations(self, df, numeric_cols) -> List[Dict]:
        insights = []
        if len(numeric_cols) < 2:
            return insights

        corr = df[numeric_cols].corr()
        seen = set()
        for i, c1 in enumerate(numeric_cols):
            for c2 in numeric_cols[i + 1:]:
                pair = (c1, c2)
                if pair in seen:
                    continue
                seen.add(pair)
                r = corr.loc[c1, c2]
                if pd.isna(r):
                    continue
                if abs(r) >= 0.7:
                    direction = "positive" if r > 0 else "negative"
                    strength = "very strong" if abs(r) >= 0.9 else "strong"
                    insights.append({
                        "category": "correlation",
                        "priority": "medium",
                        "title": f"{strength} {direction} correlation between {c1} & {c2}",
                        "narrative": (
                            f"There is a {strength} {direction} correlation (r={r:.2f}) "
                            f"between '{c1}' and '{c2}'. "
                            + (f"As '{c1}' increases, '{c2}' tends to increase as well."
                               if r > 0 else
                               f"As '{c1}' increases, '{c2}' tends to decrease.")
                        ),
                        "correlation": round(float(r), 3),
                        "columns": [c1, c2],
                    })
        return insights

    def _detect_anomalies(self, df, numeric_cols) -> List[Dict]:
        insights = []
        for col in numeric_cols[:10]:
            data = df[col].dropna()
            if len(data) < 20:
                continue
            q1, q3 = data.quantile(0.25), data.quantile(0.75)
            iqr = q3 - q1
            if iqr == 0:
                continue
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
            outliers = data[(data < lower) | (data > upper)]
            pct = len(outliers) / len(data) * 100
            if pct > 2:
                insights.append({
                    "category": "anomaly",
                    "priority": "high" if pct > 10 else "medium",
                    "title": f"{len(outliers)} anomalies in {col} ({pct:.1f}%)",
                    "narrative": (
                        f"'{col}' has {len(outliers)} data points ({pct:.1f}%) "
                        f"outside the expected range [{lower:,.2f}, {upper:,.2f}]. "
                        f"These outliers may indicate data quality issues or "
                        f"exceptional events."
                    ),
                    "metric": col,
                    "anomaly_count": len(outliers),
                    "anomaly_pct": round(pct, 2),
                })
        return insights

    def _analyze_distributions(self, df, numeric_cols) -> List[Dict]:
        insights = []
        for col in numeric_cols[:10]:
            data = df[col].dropna()
            if len(data) < 20:
                continue
            skew = float(data.skew())
            kurt = float(data.kurtosis())
            if abs(skew) > 1.5:
                direction = "right" if skew > 0 else "left"
                insights.append({
                    "category": "distribution",
                    "priority": "low",
                    "title": f"{col} is {direction}-skewed (skew={skew:.2f})",
                    "narrative": (
                        f"'{col}' shows significant {direction} skewness "
                        f"(skew={skew:.2f}, kurtosis={kurt:.2f}). "
                        f"Consider log transformation for modeling."
                    ),
                    "metric": col,
                    "skewness": round(skew, 3),
                    "kurtosis": round(kurt, 3),
                })
        return insights

    def _perform_statistical_tests(self, df, numeric_cols, categorical_cols) -> List[Dict]:
        insights = []
        for cat_col in categorical_cols[:3]:
            groups = df[cat_col].dropna().unique()
            if len(groups) < 2 or len(groups) > 10:
                continue
            for num_col in numeric_cols[:5]:
                group_data = [df.loc[df[cat_col] == g, num_col].dropna() for g in groups[:2]]
                if any(len(g) < 30 for g in group_data):
                    continue
                try:
                    t_stat, p_val = stats.ttest_ind(group_data[0], group_data[1], equal_var=False)
                except Exception:
                    continue
                if p_val < 0.05:
                    g0_mean = group_data[0].mean()
                    g1_mean = group_data[1].mean()
                    diff_pct = ((g0_mean - g1_mean) / abs(g1_mean) * 100) if g1_mean != 0 else 0
                    insights.append({
                        "category": "statistical_test",
                        "priority": "medium",
                        "title": f"Significant difference in {num_col} by {cat_col}",
                        "narrative": (
                            f"'{num_col}' is significantly different between "
                            f"'{groups[0]}' (avg {g0_mean:,.2f}) and "
                            f"'{groups[1]}' (avg {g1_mean:,.2f}), "
                            f"a {abs(diff_pct):.1f}% gap (p={p_val:.4f})."
                        ),
                        "p_value": round(float(p_val), 5),
                        "columns": [cat_col, num_col],
                    })
        return insights

    def _detect_seasonality(self, df, numeric_cols, date_col) -> List[Dict]:
        insights = []
        try:
            ts = df.copy()
            ts[date_col] = pd.to_datetime(ts[date_col], errors="coerce")
            ts = ts.dropna(subset=[date_col]).sort_values(date_col)
        except Exception:
            return insights

        for col in numeric_cols[:5]:
            values = ts[col].dropna().values
            if len(values) < 14:
                continue
            try:
                autocorr = np.corrcoef(values[:-7], values[7:])[0, 1]
            except Exception:
                continue
            if pd.notna(autocorr) and abs(autocorr) > 0.4:
                strength = "strong" if abs(autocorr) > 0.7 else "moderate"
                insights.append({
                    "category": "seasonality",
                    "priority": "medium",
                    "title": f"{strength} weekly seasonality in {col}",
                    "narrative": (
                        f"'{col}' shows {strength} weekly seasonality "
                        f"(autocorrelation at lag-7 = {autocorr:.2f}). "
                        f"Values tend to repeat in weekly cycles."
                    ),
                    "metric": col,
                    "autocorrelation": round(float(autocorr), 3),
                })
        return insights

    def _analyze_performance_gaps(self, df, numeric_cols, categorical_cols) -> List[Dict]:
        insights = []
        for cat_col in categorical_cols[:3]:
            for num_col in numeric_cols[:3]:
                try:
                    grouped = df.groupby(cat_col)[num_col].mean().dropna()
                except Exception:
                    continue
                if len(grouped) < 2:
                    continue
                top = grouped.idxmax()
                bottom = grouped.idxmin()
                if grouped[bottom] == 0:
                    continue
                gap = ((grouped[top] - grouped[bottom]) / abs(grouped[bottom])) * 100
                if gap > 20:
                    insights.append({
                        "category": "performance_gap",
                        "priority": "high" if gap > 50 else "medium",
                        "title": f"{gap:.0f}% gap in {num_col} across {cat_col}",
                        "narrative": (
                            f"'{top}' leads in '{num_col}' with avg {grouped[top]:,.2f}, "
                            f"while '{bottom}' trails at {grouped[bottom]:,.2f} — "
                            f"a {gap:.1f}% performance gap."
                        ),
                        "columns": [cat_col, num_col],
                        "top_performer": str(top),
                        "bottom_performer": str(bottom),
                        "gap_pct": round(gap, 2),
                    })
        return insights

    def _build_executive_summary(self, df, insights, recommendations) -> str:
        """Build a concise executive summary from insights."""
        lines = [f"Analysis of {df.shape[0]:,} records across {df.shape[1]} features."]
        high = [i for i in insights if i.get("priority") == "high"]
        if high:
            lines.append(f"{len(high)} critical finding(s) detected:")
            for h in high[:3]:
                lines.append(f"  - {h.get('title', '')}")
        else:
            lines.append("No critical issues detected.")
        if recommendations:
            lines.append(f"Top recommendation: {recommendations[0].get('action', '')}")
        return " ".join(lines)

    def _generate_recommendations(self, df, insights, target) -> List[Dict]:
        recs = []
        categories = {i.get("category") for i in insights}

        if "anomaly" in categories:
            recs.append({
                "priority": "high",
                "category": "data_quality",
                "action": "Investigate and address detected anomalies",
                "detail": "Review outlier records for data entry errors or exceptional events.",
            })

        high_corr = [i for i in insights if i.get("category") == "correlation" and abs(i.get("correlation", 0)) > 0.9]
        if high_corr:
            recs.append({
                "priority": "medium",
                "category": "feature_engineering",
                "action": "Consider removing highly correlated features before modeling",
                "detail": "Multicollinearity can destabilise model coefficients.",
            })

        if "trend" in categories:
            declining = [i for i in insights if i.get("category") == "trend" and i.get("change_pct", 0) < -10]
            if declining:
                cols = ", ".join(i["metric"] for i in declining[:3])
                recs.append({
                    "priority": "high",
                    "category": "business",
                    "action": f"Investigate declining metrics: {cols}",
                    "detail": "Significant downward trends detected — root cause analysis recommended.",
                })

        if "performance_gap" in categories:
            recs.append({
                "priority": "medium",
                "category": "business",
                "action": "Benchmark low-performing segments against top performers",
                "detail": "Significant performance gaps detected across categories.",
            })

        missing_pct = df.isnull().sum().sum() / (len(df) * len(df.columns)) * 100 if len(df) > 0 else 0
        if missing_pct > 5:
            recs.append({
                "priority": "medium",
                "category": "data_quality",
                "action": f"Address {missing_pct:.1f}% missing data",
                "detail": "Consider advanced imputation (KNN, iterative) for better model accuracy.",
            })

        if not recs:
            recs.append({
                "priority": "low",
                "category": "general",
                "action": "Run the full analysis pipeline for comprehensive insights",
                "detail": "Data looks clean — proceed with feature engineering and modeling.",
            })

        return recs
