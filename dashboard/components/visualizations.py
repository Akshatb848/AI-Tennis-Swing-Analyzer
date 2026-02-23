"""Chart builders using Plotly for the dashboard."""

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

CHART_COLORS = {
    "primary": "#6366f1",
    "secondary": "#8b5cf6",
    "accent": "#a78bfa",
    "success": "#10b981",
    "warning": "#f59e0b",
    "danger": "#ef4444",
    "muted": "#94a3b8",
    "bg": "#f8fafc",
    "text": "#1e293b",
}

CHART_TEMPLATE = "plotly_white"

def _base_layout(title: str = "", height: int = 400) -> dict:
    return dict(
        title=dict(text=title, font=dict(size=14, color=CHART_COLORS["text"])),
        template=CHART_TEMPLATE,
        height=height,
        margin=dict(l=40, r=20, t=50, b=40),
        font=dict(family="Inter, system-ui, sans-serif", size=12, color=CHART_COLORS["text"]),
        plot_bgcolor="white",
        paper_bgcolor="white",
    )


def render_viz_from_spec(spec: dict, df: pd.DataFrame) -> Optional[go.Figure]:
    chart_type = spec.get("type", spec.get("chart_type", ""))
    x_col = spec.get("x", spec.get("x_col"))
    y_col = spec.get("y", spec.get("y_col"))
    color_by = spec.get("color", spec.get("color_by"))
    title = spec.get("title", "")

    try:
        if chart_type == "histogram":
            if x_col and x_col in df.columns:
                fig = px.histogram(
                    df, x=x_col,
                    color=color_by if color_by and color_by in df.columns else None,
                    title=title, opacity=0.85,
                    color_discrete_sequence=[CHART_COLORS["primary"]],
                )
                fig.update_layout(**_base_layout(title))
                return fig
        elif chart_type == "bar":
            if x_col and x_col in df.columns:
                counts = df[x_col].value_counts().head(20)
                fig = go.Figure(go.Bar(
                    x=counts.index.astype(str), y=counts.values,
                    marker_color=CHART_COLORS["primary"],
                ))
                fig.update_layout(**_base_layout(title), xaxis_title=x_col, yaxis_title="Count")
                return fig
        elif chart_type == "scatter":
            if x_col and y_col and x_col in df.columns and y_col in df.columns:
                fig = px.scatter(
                    df.head(2000), x=x_col, y=y_col,
                    color=color_by if color_by and color_by in df.columns else None,
                    title=title, opacity=0.6,
                    color_discrete_sequence=[CHART_COLORS["primary"]],
                    trendline="ols" if len(df) > 10 else None,
                )
                fig.update_layout(**_base_layout(title))
                return fig
        elif chart_type == "box":
            if x_col and y_col and x_col in df.columns and y_col in df.columns:
                fig = px.box(df, x=x_col, y=y_col, title=title,
                             color_discrete_sequence=[CHART_COLORS["primary"]])
                fig.update_layout(**_base_layout(title))
                return fig
            elif y_col and y_col in df.columns:
                fig = px.box(df, y=y_col, title=title,
                             color_discrete_sequence=[CHART_COLORS["primary"]])
                fig.update_layout(**_base_layout(title))
                return fig
        elif chart_type == "pie":
            if x_col and x_col in df.columns:
                counts = df[x_col].value_counts().head(10)
                fig = px.pie(values=counts.values, names=counts.index.astype(str), title=title,
                             color_discrete_sequence=px.colors.qualitative.Set3)
                fig.update_layout(**_base_layout(title))
                return fig
    except Exception:
        return None

    return None


def plot_correlation_heatmap(corr_matrix: dict, title: str = "Correlation Matrix") -> Optional[go.Figure]:
    if not corr_matrix:
        return None
    try:
        df_corr = pd.DataFrame(corr_matrix)
        fig = go.Figure(data=go.Heatmap(
            z=df_corr.values,
            x=df_corr.columns.tolist(),
            y=df_corr.index.tolist(),
            colorscale="RdBu_r",
            zmid=0,
            text=np.round(df_corr.values, 2),
            texttemplate="%{text}",
            textfont={"size": 10},
        ))
        fig.update_layout(**_base_layout(title, height=500))
        return fig
    except Exception:
        return None


def plot_leaderboard(leaderboard: list) -> Optional[go.Figure]:
    if not leaderboard:
        return None

    names = [m.get("name", "?") for m in leaderboard]
    scores = [m.get("score", 0) for m in leaderboard]

    sorted_data = sorted(zip(names, scores), key=lambda x: x[1])
    names, scores = zip(*sorted_data) if sorted_data else ([], [])

    fig = go.Figure(go.Bar(
        x=list(scores),
        y=list(names),
        orientation="h",
        marker_color=[CHART_COLORS["primary"] if s == max(scores) else CHART_COLORS["muted"] for s in scores],
        text=[f"{s:.4f}" for s in scores],
        textposition="outside",
    ))
    layout = _base_layout("Model Leaderboard", max(300, len(names) * 50))
    layout["xaxis_title"] = "Score"
    fig.update_layout(**layout)
    return fig


def plot_feature_importance(importance_dict: dict, top_n: int = 15) -> Optional[go.Figure]:
    if not importance_dict:
        return None
    sorted_features = sorted(importance_dict.items(), key=lambda x: abs(x[1]), reverse=True)[:top_n]
    if not sorted_features:
        return None
    names, values = zip(*sorted_features)
    fig = go.Figure(go.Bar(
        x=list(values), y=list(names), orientation="h",
        marker_color=CHART_COLORS["primary"],
    ))
    layout = _base_layout(f"Top {top_n} Feature Importances", max(300, len(names) * 30))
    layout["xaxis_title"] = "Importance"
    fig.update_layout(**layout)
    return fig


def plot_missing_values(missing_pct: dict) -> Optional[go.Figure]:
    if not missing_pct:
        return None
    non_zero = {k: v for k, v in missing_pct.items() if v > 0}
    if not non_zero:
        return None
    sorted_items = sorted(non_zero.items(), key=lambda x: x[1], reverse=True)
    names, values = zip(*sorted_items)
    fig = go.Figure(go.Bar(
        x=list(values), y=list(names), orientation="h",
        marker_color=[CHART_COLORS["danger"] if v > 30 else CHART_COLORS["warning"] if v > 10 else CHART_COLORS["success"] for v in values],
    ))
    layout = _base_layout("Missing Values (%)", max(300, len(names) * 25))
    layout["xaxis_title"] = "Missing %"
    fig.update_layout(**layout)
    return fig


def plot_distribution_analysis(dist_data: list) -> Optional[go.Figure]:
    if not dist_data:
        return None
    features = [d["feature"] for d in dist_data]
    skewness = [d["skewness"] for d in dist_data]
    colors = [
        CHART_COLORS["danger"] if abs(s) > 1.5 else
        CHART_COLORS["warning"] if abs(s) > 0.75 else
        CHART_COLORS["success"]
        for s in skewness
    ]
    fig = go.Figure(go.Bar(
        x=features, y=skewness, marker_color=colors,
        text=[f"{s:.2f}" for s in skewness], textposition="outside",
    ))
    layout = _base_layout("Feature Skewness", 350)
    layout["xaxis_title"] = "Feature"
    layout["yaxis_title"] = "Skewness"
    fig.add_hline(y=0.75, line_dash="dash", line_color=CHART_COLORS["warning"], annotation_text="Threshold")
    fig.add_hline(y=-0.75, line_dash="dash", line_color=CHART_COLORS["warning"])
    fig.update_layout(**layout)
    return fig


def plot_vif_chart(vif_data: list) -> Optional[go.Figure]:
    if not vif_data:
        return None
    features = [d["feature"] for d in vif_data]
    vif_values = [min(d["vif"], 50) for d in vif_data]
    colors = [
        CHART_COLORS["danger"] if d["vif"] > 10 else
        CHART_COLORS["warning"] if d["vif"] > 5 else
        CHART_COLORS["success"]
        for d in vif_data
    ]
    fig = go.Figure(go.Bar(
        x=features, y=vif_values, marker_color=colors,
        text=[f"{v:.1f}" for v in vif_values], textposition="outside",
    ))
    layout = _base_layout("Variance Inflation Factor (VIF)", 350)
    layout["xaxis_title"] = "Feature"
    layout["yaxis_title"] = "VIF"
    fig.add_hline(y=5, line_dash="dash", line_color=CHART_COLORS["warning"], annotation_text="Moderate (VIF=5)")
    fig.add_hline(y=10, line_dash="dash", line_color=CHART_COLORS["danger"], annotation_text="High (VIF=10)")
    fig.update_layout(**layout)
    return fig


def plot_before_after_skewness(skew_corrections: dict) -> Optional[go.Figure]:
    if not skew_corrections:
        return None
    features = list(skew_corrections.keys())[:15]
    before = [skew_corrections[f]["original_skewness"] for f in features]
    after = [skew_corrections[f]["new_skewness"] for f in features]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        name="Before", x=features, y=before,
        marker_color=CHART_COLORS["danger"], opacity=0.7,
    ))
    fig.add_trace(go.Bar(
        name="After", x=features, y=after,
        marker_color=CHART_COLORS["success"], opacity=0.7,
    ))
    layout = _base_layout("Skewness Correction: Before vs After", 350)
    layout["barmode"] = "group"
    layout["yaxis_title"] = "Skewness"
    fig.update_layout(**layout)
    return fig


def plot_hypothesis_summary(tests: list) -> Optional[go.Figure]:
    if not tests:
        return None
    sig = sum(1 for t in tests if t.get("significant", False))
    non_sig = len(tests) - sig
    fig = go.Figure(go.Pie(
        values=[sig, non_sig],
        labels=["Significant (p < 0.05)", "Not Significant"],
        marker=dict(colors=[CHART_COLORS["primary"], CHART_COLORS["muted"]]),
        hole=0.5,
        textinfo="value+percent",
    ))
    layout = _base_layout(f"Hypothesis Test Results ({len(tests)} tests)", 300)
    fig.update_layout(**layout)
    return fig
