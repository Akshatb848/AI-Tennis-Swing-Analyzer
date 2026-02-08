"""
Data Science Agent Platform - Chat-Based Application

Conversational interface powered by LLM-based coordinator agent.
Users interact via natural language; the coordinator dispatches
specialized agents and interprets results.
"""

import streamlit as st
import pandas as pd
import numpy as np

# CRITICAL: Disable pandas 3.x StringDtype globally.
# Without this, pd.read_csv() creates StringDtype columns which crash
# numpy operations (corr, get_dummies, factorize) across all agents.
pd.set_option("future.infer_string", False)
import asyncio
import json
from datetime import datetime
from pathlib import Path
import sys
import os

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from agents.coordinator_agent import CoordinatorAgent
from agents.data_cleaner_agent import DataCleanerAgent
from agents.eda_agent import EDAAgent
from agents.feature_engineer_agent import FeatureEngineerAgent
from agents.model_trainer_agent import ModelTrainerAgent
from agents.automl_agent import AutoMLAgent
from agents.dashboard_builder_agent import DashboardBuilderAgent
from agents.data_visualizer_agent import DataVisualizerAgent
from agents.forecast_agent import ForecastAgent
from agents.insights_agent import InsightsAgent
from agents.report_generator_agent import ReportGeneratorAgent
from llm.client import get_llm_client, validate_llm_client, FallbackClient
from llm.prompts import PromptTemplates
from utils.helpers import generate_sample_data

# Page config
st.set_page_config(
    page_title="Data Science Agent Platform",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS — glassmorphism dark theme
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&family=Inter:wght@400;500;600;700&display=swap');

    :root {
        --primary: #6366f1;
        --primary-light: #818cf8;
        --secondary: #10b981;
        --accent: #f59e0b;
        --danger: #ef4444;
        --background: #0f172a;
        --surface: #1e293b;
        --text: #f1f5f9;
        --text-muted: #94a3b8;
        --border: rgba(99, 102, 241, 0.2);
    }

    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 30%, #0f172a 70%, #1e1b4b 100%);
        font-family: 'DM Sans', 'Inter', sans-serif;
    }

    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(135deg, #6366f1 0%, #10b981 50%, #f59e0b 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        text-align: center;
        margin-bottom: 0.5rem;
    }

    .sub-header {
        text-align: center;
        color: #94a3b8;
        margin-bottom: 1rem;
        font-size: 1rem;
    }

    /* Metric cards */
    .metric-card {
        background: linear-gradient(145deg, rgba(30, 41, 59, 0.9), rgba(51, 65, 85, 0.7));
        border: 1px solid rgba(99, 102, 241, 0.2);
        border-radius: 16px;
        padding: 1.5rem;
        text-align: center;
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
        overflow: hidden;
    }

    .metric-card::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 3px;
        background: linear-gradient(90deg, #6366f1, #10b981);
        opacity: 0;
        transition: opacity 0.3s ease;
    }

    .metric-card:hover {
        border-color: rgba(99, 102, 241, 0.5);
        transform: translateY(-4px);
        box-shadow: 0 20px 40px rgba(99, 102, 241, 0.25);
    }

    .metric-card:hover::before { opacity: 1; }

    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        background: linear-gradient(135deg, #6366f1, #818cf8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-family: 'JetBrains Mono', monospace;
    }

    .metric-label {
        color: #94a3b8;
        font-size: 0.85rem;
        margin-top: 0.5rem;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    /* Narrative card */
    .narrative-card {
        background: linear-gradient(145deg, rgba(16, 185, 129, 0.1), rgba(30, 41, 59, 0.9));
        border: 1px solid rgba(16, 185, 129, 0.3);
        border-left: 4px solid #10b981;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        line-height: 1.7;
    }

    .narrative-card h4 { color: #10b981; margin-bottom: 0.75rem; }
    .narrative-card p { color: #e2e8f0; font-size: 1rem; }

    /* Insight cards */
    .insight-card {
        background: linear-gradient(145deg, rgba(99, 102, 241, 0.08), rgba(16, 185, 129, 0.04));
        border: 1px solid rgba(99, 102, 241, 0.25);
        border-radius: 16px;
        padding: 1.25rem;
        margin-bottom: 1rem;
        transition: all 0.3s ease;
    }

    .insight-card:hover {
        border-color: rgba(99, 102, 241, 0.5);
        transform: translateX(4px);
    }

    .insight-card.high-priority { border-left: 4px solid #ef4444; }
    .insight-card.medium-priority { border-left: 4px solid #f59e0b; }
    .insight-card.low-priority { border-left: 4px solid #10b981; }

    .insight-icon { font-size: 1.5rem; margin-right: 0.75rem; }
    .insight-title { color: #f1f5f9; font-weight: 600; font-size: 1rem; }

    .insight-narrative {
        color: #e2e8f0;
        font-size: 0.95rem;
        margin-top: 0.75rem;
        padding: 0.75rem;
        background: rgba(30, 41, 59, 0.5);
        border-radius: 8px;
        line-height: 1.6;
        border-left: 3px solid #6366f1;
    }

    /* Recommendation cards */
    .recommendation-card {
        background: linear-gradient(145deg, rgba(16, 185, 129, 0.1), rgba(52, 211, 153, 0.05));
        border: 1px solid rgba(16, 185, 129, 0.3);
        border-radius: 16px;
        padding: 1.25rem;
        margin-bottom: 1rem;
        transition: all 0.3s ease;
    }

    .recommendation-card:hover {
        border-color: rgba(16, 185, 129, 0.6);
        box-shadow: 0 8px 30px rgba(16, 185, 129, 0.15);
    }

    .recommendation-title { color: #10b981; font-weight: 600; font-size: 1rem; }
    .recommendation-description { color: #d1d5db; font-size: 0.95rem; margin-top: 0.75rem; line-height: 1.6; }

    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #6366f1 0%, #4f46e5 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 0.75rem 1.5rem !important;
        font-weight: 600 !important;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
        box-shadow: 0 4px 15px rgba(99, 102, 241, 0.3) !important;
    }

    .stButton > button:hover {
        background: linear-gradient(135deg, #818cf8 0%, #6366f1 100%) !important;
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 25px rgba(99, 102, 241, 0.4) !important;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1e293b 0%, #0f172a 100%) !important;
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0.5rem;
        background: rgba(30, 41, 59, 0.7);
        border-radius: 16px;
        padding: 0.5rem;
        border: 1px solid rgba(99, 102, 241, 0.2);
    }

    .stTabs [data-baseweb="tab"] {
        border-radius: 12px;
        color: #94a3b8;
        font-weight: 500;
    }

    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #6366f1, #4f46e5) !important;
        color: white !important;
    }

    /* Expander */
    .streamlit-expanderHeader {
        background: rgba(30, 41, 59, 0.7) !important;
        border-radius: 12px !important;
        border: 1px solid rgba(99, 102, 241, 0.2) !important;
    }

    /* DataFrames */
    .stDataFrame {
        border: 1px solid rgba(99, 102, 241, 0.2) !important;
        border-radius: 16px !important;
        overflow: hidden !important;
    }

    /* Progress bar */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #6366f1, #10b981, #f59e0b) !important;
        border-radius: 10px !important;
    }

    /* Animations */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(15px); }
        to { opacity: 1; transform: translateY(0); }
    }

    .animate-fade-in { animation: fadeIn 0.6s ease-out; }
</style>
""", unsafe_allow_html=True)


# ---------- helpers ----------

def get_event_loop():
    """Get or create an asyncio event loop."""
    try:
        loop = asyncio.get_event_loop()
        if loop.is_closed():
            raise RuntimeError
        return loop
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        return loop


def run_async(coro):
    """Run an async coroutine from synchronous Streamlit code."""
    loop = get_event_loop()
    return loop.run_until_complete(coro)


def get_dataset_summary() -> str:
    """Build a concise text summary of the loaded dataset."""
    df = st.session_state.get("df")
    if df is None:
        return ""
    info = {
        "shape": {"rows": df.shape[0], "columns": df.shape[1]},
        "dtypes": df.dtypes.astype(str).to_dict(),
        "missing_values": df.isnull().sum().to_dict(),
    }
    return PromptTemplates.dataset_summary(info)


# ---------- initialisation ----------

def init_session_state():
    """Initialise all session state keys."""
    if "coordinator" not in st.session_state:
        llm_client = get_llm_client(allow_fallback=False)
        coordinator = CoordinatorAgent(llm_client=llm_client)
        coordinator.register_agent(DataCleanerAgent())
        coordinator.register_agent(EDAAgent())
        coordinator.register_agent(FeatureEngineerAgent())
        coordinator.register_agent(ModelTrainerAgent())
        coordinator.register_agent(AutoMLAgent())
        coordinator.register_agent(DashboardBuilderAgent())
        coordinator.register_agent(DataVisualizerAgent())
        coordinator.register_agent(ForecastAgent())
        coordinator.register_agent(InsightsAgent())
        coordinator.register_agent(ReportGeneratorAgent())
        st.session_state.coordinator = coordinator

    defaults = {
        "messages": [],
        "df": None,
        "target_column": None,
        "current_df": None,  # working copy after cleaning / FE
        "analysis_results": {},
        "pipeline_ran": False,
        "llm_status": {"state": "unverified", "message": "LLM not validated yet."},
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val

    # Add welcome message on first load
    if not st.session_state.messages:
        welcome = st.session_state.coordinator.get_welcome_message()
        st.session_state.messages.append({"role": "assistant", "content": welcome})

    if (
        st.session_state.llm_status.get("state") == "unverified"
        and st.session_state.coordinator.llm_client
    ):
        validation = run_async(validate_llm_client(st.session_state.coordinator.llm_client))
        st.session_state.llm_status = validation.to_dict()


# ---------- sidebar ----------

def render_sidebar():
    with st.sidebar:
        st.markdown(
            '<h2 class="main-header">🔬 DS Agent</h2>', unsafe_allow_html=True
        )

        # LLM provider config
        with st.expander("🤖 LLM Configuration", expanded=False):
            provider = st.selectbox(
                "Provider",
                ["auto", "openai", "anthropic", "ollama", "fallback"],
                help="auto = detect from env vars",
            )
            api_key = st.text_input("API Key (optional)", type="password")
            model = st.text_input("Model (optional)", placeholder="e.g. gpt-4o-mini")
            if st.button("Apply LLM Config"):
                from llm.client import get_llm_client as _get
                p = None if provider == "auto" else provider
                client = _get(
                    provider=p,
                    api_key=api_key or None,
                    model=model or None,
                    allow_fallback=False,
                )
                if provider == "fallback":
                    client = FallbackClient()
                validation = run_async(validate_llm_client(client))
                st.session_state.llm_status = validation.to_dict()
                if validation.state == "connected":
                    st.session_state.coordinator.llm_client = client
                    st.success(f"LLM validated: **{validation.provider}** / **{validation.model}**")
                elif validation.state == "rate_limited":
                    st.session_state.coordinator.llm_client = client
                    st.warning(validation.message)
                else:
                    st.session_state.coordinator.llm_client = None
                    st.error(validation.message)

            _render_llm_status_panel()

        st.markdown("---")

        # Dataset upload
        st.markdown("### 📂 Dataset")
        uploaded = st.file_uploader(
            "Upload", type=["csv", "xlsx", "xls", "json", "parquet"], label_visibility="collapsed"
        )
        if uploaded:
            _load_uploaded_file(uploaded)

        # Sample datasets
        with st.expander("📝 Sample datasets"):
            samples = {
                "Iris (Classification)": "iris",
                "Housing (Regression)": "housing",
                "Titanic (Classification)": "titanic",
                "Random Data": "random",
            }
            for label, key in samples.items():
                if st.button(label, key=f"sample_{key}"):
                    df = generate_sample_data(key)
                    _set_dataframe(df, label)

        # Target column
        if st.session_state.df is not None:
            st.markdown("### 🎯 Target Column")
            cols = ["None (Unsupervised)"] + list(st.session_state.df.columns)
            idx = 0
            if st.session_state.target_column and st.session_state.target_column in cols:
                idx = cols.index(st.session_state.target_column)
            target = st.selectbox("Target", cols, index=idx, label_visibility="collapsed")
            new_target = None if target == "None (Unsupervised)" else target
            if new_target != st.session_state.target_column:
                st.session_state.target_column = new_target

        st.markdown("---")

        # Status panel
        st.markdown("### 📊 Status")
        if st.session_state.df is not None:
            df = st.session_state.df
            st.success(f"**Dataset**: {df.shape[0]:,} × {df.shape[1]}")
            if st.session_state.target_column:
                st.info(f"**Target**: {st.session_state.target_column}")
        else:
            st.info("No dataset loaded")

        ar = st.session_state.analysis_results
        if ar:
            completed = [k for k in ar]
            st.markdown(f"**Analyses**: {', '.join(completed)}")

        st.markdown("---")
        if st.button("🔄 Reset All"):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()


def _load_uploaded_file(uploaded):
    """Parse an uploaded file into a DataFrame."""
    try:
        name = uploaded.name
        if name.endswith(".csv"):
            df = pd.read_csv(uploaded)
        elif name.endswith((".xlsx", ".xls")):
            df = pd.read_excel(uploaded)
        elif name.endswith(".json"):
            df = pd.read_json(uploaded)
        elif name.endswith(".parquet"):
            df = pd.read_parquet(uploaded)
        else:
            st.sidebar.error("Unsupported file type")
            return
        _set_dataframe(df, name)
    except Exception as e:
        st.sidebar.error(f"Error loading file: {e}")


def _render_llm_status_panel():
    """Render LLM readiness status with truthful messaging."""
    status = st.session_state.get("llm_status", {"state": "unverified", "message": ""})
    state = status.get("state", "unverified")
    message = status.get("message", "")
    provider = status.get("provider")
    model = status.get("model")
    detail = message
    if provider and model:
        detail = f"{detail} (Provider: {provider}, Model: {model})"

    if state == "connected":
        st.success(f"Connected ✅ {detail}")
    elif state == "rate_limited":
        st.warning(f"Rate limited ⚠️ {detail}")
    elif state == "unverified":
        st.info(f"Not validated ℹ️ {detail}")
    else:
        st.error(f"Not connected ❌ {detail}")


def _sanitize_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Convert pandas 3.x StringDtype columns to plain object for numpy compatibility.

    pandas 3.0 with future.infer_string=True makes all string columns use
    StringDtype, which breaks numpy operations (corr, get_dummies, factorize).
    This must be called once when data is first loaded.
    """
    for col in df.columns:
        if pd.api.types.is_string_dtype(df[col]) and df[col].dtype != "object":
            df[col] = df[col].astype(object)
    return df


def _set_dataframe(df: pd.DataFrame, name: str):
    """Store a dataframe in session state and add a system message."""
    df = _sanitize_dtypes(df)
    st.session_state.df = df
    st.session_state.current_df = df.copy()
    st.session_state.analysis_results = {}
    st.session_state.pipeline_ran = False
    msg = (
        f"Dataset **{name}** loaded: **{df.shape[0]:,}** rows × **{df.shape[1]}** columns.\n\n"
        f"Columns: {', '.join(str(c) for c in df.columns[:20])}"
        + (" ..." if len(df.columns) > 20 else "")
    )
    st.session_state.messages.append({"role": "assistant", "content": msg})
    # Auto-detect common target column names
    for candidate in ["target", "Target", "label", "Label", "Survived", "price", "class"]:
        if candidate in df.columns:
            st.session_state.target_column = candidate
            st.session_state.messages.append(
                {"role": "assistant", "content": f"Auto-detected target column: **{candidate}**. You can change it in the sidebar."}
            )
            break


# ---------- chat message rendering ----------

def render_chat():
    """Render the full chat history."""
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

            # Render inline visualisations if attached
            if "charts" in msg:
                _render_charts(msg["charts"])
            if "dataframe" in msg:
                st.dataframe(msg["dataframe"], use_container_width=True)
            if "metrics_table" in msg:
                st.dataframe(pd.DataFrame(msg["metrics_table"]), use_container_width=True)
            if "dashboard" in msg:
                _render_professional_dashboard(msg["dashboard"])
            if "insights_panel" in msg:
                _render_insights_panel(msg["insights_panel"])
            if "report_downloads" in msg:
                _render_report_downloads(msg["report_downloads"])


def _render_insights_panel(panel_data):
    """Render business insights and recommendations."""
    summary = panel_data.get("executive_summary", "")
    if summary:
        st.markdown(f"""<div class="narrative-card"><h4>Executive Summary</h4><p>{summary}</p></div>""", unsafe_allow_html=True)

    insights = panel_data.get("insights", [])
    if insights:
        st.markdown("#### Key Findings")
        for ins in insights[:10]:
            prio = ins.get("priority", "low")
            prio_class = f"{prio}-priority"
            icon = {"high": "🔴", "medium": "🟡"}.get(prio, "🟢")
            title = ins.get("title", "")
            narrative = ins.get("narrative", "")
            st.markdown(
                f"""<div class="insight-card {prio_class}">
                    <span class="insight-icon">{icon}</span>
                    <span class="insight-title">{title}</span>
                    <div class="insight-narrative">{narrative}</div>
                </div>""",
                unsafe_allow_html=True,
            )

    recs = panel_data.get("recommendations", [])
    if recs:
        st.markdown("#### Recommendations")
        for r in recs[:6]:
            action = r.get("action", "")
            detail = r.get("detail", "")
            st.markdown(
                f"""<div class="recommendation-card">
                    <div class="recommendation-title">💡 {action}</div>
                    <div class="recommendation-description">{detail}</div>
                </div>""",
                unsafe_allow_html=True,
            )


def _render_report_downloads(report_data):
    """Render download buttons for generated reports."""
    st.markdown("#### Export Report")
    c1, c2, c3 = st.columns(3)
    html_content = report_data.get("html")
    md_content = report_data.get("markdown")
    csv_content = report_data.get("csv_summary")
    title = report_data.get("title", "report")
    safe_title = title.lower().replace(" ", "_")

    if html_content:
        c1.download_button(
            "📄 Download HTML",
            data=html_content,
            file_name=f"{safe_title}.html",
            mime="text/html",
        )
    if md_content:
        c2.download_button(
            "📝 Download Markdown",
            data=md_content,
            file_name=f"{safe_title}.md",
            mime="text/markdown",
        )
    if csv_content:
        c3.download_button(
            "📊 Download CSV",
            data=csv_content,
            file_name=f"{safe_title}_summary.csv",
            mime="text/csv",
        )


def _render_charts(charts):
    """Render plotly charts stored in a message."""
    import plotly.express as px
    import plotly.graph_objects as go

    for chart in charts:
        ctype = chart.get("type")
        if ctype == "histogram":
            fig = px.histogram(
                x=chart["data"]["values"],
                title=chart.get("title", ""),
                marginal="box",
            )
            st.plotly_chart(fig, use_container_width=True)
        elif ctype == "heatmap":
            matrix = chart["data"]["matrix"]
            labels = chart["data"]["labels"]
            fig = px.imshow(
                matrix,
                x=labels,
                y=labels,
                text_auto=".2f",
                title=chart.get("title", ""),
                color_continuous_scale="RdBu_r",
            )
            st.plotly_chart(fig, use_container_width=True)
        elif ctype == "scatter":
            fig = px.scatter(
                x=chart["data"]["x"],
                y=chart["data"]["y"],
                title=chart.get("title", ""),
                labels={
                    "x": chart["data"].get("x_label", "X"),
                    "y": chart["data"].get("y_label", "Y"),
                },
            )
            st.plotly_chart(fig, use_container_width=True)
        elif ctype == "bar":
            fig = px.bar(
                x=chart["data"]["labels"],
                y=chart["data"]["values"],
                title=chart.get("title", ""),
            )
            st.plotly_chart(fig, use_container_width=True)
        elif ctype == "forecast_line":
            data = chart["data"]
            fig = go.Figure()
            # Historical values
            if data.get("historical_values") and data.get("dates"):
                n_hist = len(data["historical_values"])
                fig.add_trace(go.Scatter(
                    x=data["dates"][:n_hist],
                    y=data["historical_values"],
                    name="Historical",
                    mode="lines",
                    line=dict(color="#6366f1", width=2),
                ))
            # Forecast values
            if data.get("forecast_values") and data.get("dates"):
                n_hist = len(data.get("historical_values", []))
                fig.add_trace(go.Scatter(
                    x=data["dates"][n_hist:] if n_hist else data["dates"],
                    y=data["forecast_values"],
                    name="Forecast",
                    mode="lines",
                    line=dict(color="#10b981", width=2, dash="dash"),
                ))
            # Confidence interval
            if data.get("upper_bound") and data.get("lower_bound"):
                n_hist = len(data.get("historical_values", []))
                forecast_dates = data["dates"][n_hist:] if n_hist else data["dates"]
                fig.add_trace(go.Scatter(
                    x=list(forecast_dates) + list(reversed(forecast_dates)),
                    y=list(data["upper_bound"]) + list(reversed(data["lower_bound"])),
                    fill="toself",
                    fillcolor="rgba(16, 185, 129, 0.1)",
                    line=dict(color="rgba(16, 185, 129, 0)"),
                    name="95% Confidence",
                ))
            fig.update_layout(
                title=chart.get("title", "Forecast"),
                template="plotly_white",
                height=400,
            )
            st.plotly_chart(fig, use_container_width=True)


def _render_professional_dashboard(sections):
    """Render a professional PowerBI/Tableau-style dashboard from section data."""
    import plotly.express as px
    import plotly.graph_objects as go

    for section in sections:
        sec_type = section.get("type", "")
        sec_title = section.get("title", "")
        data = section.get("data", {})

        if sec_type == "kpi_section":
            st.markdown(f"#### {sec_title}")
            cols = st.columns(len(data))
            for col, kpi in zip(cols, data):
                with col:
                    st.metric(label=kpi["title"], value=kpi["value"])

        elif sec_type == "data_quality_section":
            with st.expander(f"📋 {sec_title}", expanded=False):
                q_score = data.get("quality_score", 100)
                st.progress(min(q_score / 100, 1.0), text=f"Quality Score: {q_score}%")
                c1, c2, c3 = st.columns(3)
                dtypes = data.get("dtypes_summary", {})
                c1.metric("Numeric", dtypes.get("numeric", 0))
                c2.metric("Categorical", dtypes.get("categorical", 0))
                c3.metric("Total Missing", data.get("total_missing", 0))
                missing_by_col = data.get("missing_by_column", {})
                if missing_by_col:
                    st.markdown("**Missing by column (top 10):**")
                    fig = go.Figure(go.Bar(
                        x=list(missing_by_col.values()),
                        y=list(missing_by_col.keys()),
                        orientation='h',
                        marker_color='#e74c3c',
                    ))
                    fig.update_layout(height=max(200, len(missing_by_col) * 30), margin=dict(l=0, r=0, t=0, b=0), template="plotly_white")
                    st.plotly_chart(fig, use_container_width=True)

        elif sec_type == "chart_section":
            st.markdown(f"#### {sec_title}")
            charts = data if isinstance(data, list) else []
            _render_charts(charts)

        elif sec_type == "target_analysis_section":
            st.markdown(f"#### {sec_title}")
            task_type = data.get("task_type", "")
            col_name = data.get("column", "")
            st.markdown(f"**Target**: `{col_name}` — **Task**: {task_type.title()} — **Unique values**: {data.get('unique', '?')}")

            if task_type == "classification":
                dist = data.get("class_distribution", {})
                if dist:
                    fig = px.pie(
                        names=dist.get("labels", []),
                        values=dist.get("values", []),
                        title=f"Class Distribution: {col_name}",
                        color_discrete_sequence=px.colors.qualitative.Set2,
                    )
                    fig.update_layout(template="plotly_white")
                    st.plotly_chart(fig, use_container_width=True)
                balance = data.get("class_balance", "")
                if balance:
                    st.info(f"Class balance: **{balance}**")
            elif task_type == "regression":
                vals = data.get("distribution_values")
                if vals:
                    fig = px.histogram(x=vals, title=f"Target Distribution: {col_name}", marginal="box")
                    fig.update_layout(template="plotly_white")
                    st.plotly_chart(fig, use_container_width=True)
                stats = data.get("statistics", {})
                if stats:
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("Mean", f"{stats.get('mean', 0):.2f}" if stats.get("mean") is not None else "N/A")
                    c2.metric("Std", f"{stats.get('std', 0):.2f}" if stats.get("std") is not None else "N/A")
                    c3.metric("Min", f"{stats.get('min', 0):.2f}" if stats.get("min") is not None else "N/A")
                    c4.metric("Max", f"{stats.get('max', 0):.2f}" if stats.get("max") is not None else "N/A")

        elif sec_type == "model_comparison_section":
            st.markdown(f"#### {sec_title}")
            results = data.get("results", {})
            best = data.get("best_model", "")
            primary_metric = data.get("primary_metric", "accuracy")

            if results:
                # Model comparison bar chart
                model_names = list(results.keys())
                metric_values = [m.get(primary_metric, 0) for m in results.values()]
                colors = ['#2ecc71' if n == best else '#3498db' for n in model_names]

                fig = go.Figure(data=[go.Bar(
                    x=model_names,
                    y=metric_values,
                    marker_color=colors,
                    text=[f'{v:.4f}' for v in metric_values],
                    textposition='outside',
                )])
                fig.update_layout(
                    title=f"Model Comparison ({primary_metric.upper()})",
                    xaxis_title="Model", yaxis_title=primary_metric.title(),
                    template="plotly_white", height=400,
                )
                st.plotly_chart(fig, use_container_width=True)

                # Metrics table
                rows = []
                for name, metrics in results.items():
                    row = {"Model": ("✅ " + name) if name == best else name}
                    row.update({k: round(v, 4) if isinstance(v, float) else v for k, v in metrics.items()})
                    rows.append(row)
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

            # Feature importance for best model
            fi = data.get("feature_importance", {}).get(best, {})
            if fi:
                sorted_fi = dict(sorted(fi.items(), key=lambda x: abs(x[1]), reverse=True)[:15])
                with st.expander("Feature Importance (Top 15)", expanded=False):
                    fig = go.Figure(go.Bar(
                        x=list(sorted_fi.values()),
                        y=list(sorted_fi.keys()),
                        orientation='h',
                        marker_color='#9b59b6',
                    ))
                    fig.update_layout(height=max(300, len(sorted_fi) * 25), margin=dict(l=0, r=0, t=0, b=0), template="plotly_white", yaxis=dict(autorange="reversed"))
                    st.plotly_chart(fig, use_container_width=True)

        elif sec_type == "insights_section":
            st.markdown(f"#### {sec_title}")
            insights = data.get("insights", [])
            recommendations = data.get("recommendations", [])
            if insights:
                for insight in insights:
                    st.markdown(f"- {insight}")
            if recommendations:
                st.markdown("**Recommendations:**")
                for rec in recommendations:
                    st.markdown(f"- {rec}")


# ---------- agent dispatch ----------

async def handle_user_message(user_input: str):
    """Process a user message through the coordinator agent."""
    llm_status = st.session_state.get("llm_status", {})
    if llm_status.get("state") != "connected":
        return (
            "LLM is not validated yet. Please configure a valid provider/API key in the sidebar. "
            f"Status: {llm_status.get('state', 'unverified')}."
        )
    coordinator: CoordinatorAgent = st.session_state.coordinator
    coordinator.add_to_memory("user", user_input)

    # Build context for intent analysis
    context = {
        "has_dataset": st.session_state.df is not None,
        "has_target": st.session_state.target_column is not None,
        "dataset_summary": get_dataset_summary(),
    }

    # Analyse intent
    intent_result = await coordinator.analyze_user_intent(user_input, context)
    intent = intent_result.get("intent", "general")

    # Route based on intent
    if intent == "help":
        response = coordinator.get_help_message()
    elif intent == "status":
        response = _build_status_message()
    elif intent == "upload_data":
        response = intent_result.get("explanation", "Please upload a dataset using the sidebar.")
    elif intent == "set_target":
        response = "You can set the target column in the sidebar dropdown."
    elif intent == "run_pipeline":
        response = await _run_full_pipeline(user_input)
    elif intent == "run_agent":
        # Guardrail: check if agent needs a target column
        agent_name = intent_result.get("agent", "")
        if agent_name in ("ModelTrainerAgent", "AutoMLAgent") and not st.session_state.target_column:
            response = (
                f"To use **{agent_name}**, you need a target column. "
                "Please select one from the sidebar dropdown, then try again."
            )
        else:
            response = await _run_single_agent(intent_result, user_input)
    else:
        # General conversation / data science Q&A — GPT-like interactive reply
        enriched_context = {
            **context,
            "completed_analyses": st.session_state.analysis_results,
        }
        response = await coordinator.generate_conversational_reply(user_input, enriched_context)

    coordinator.add_to_memory("assistant", response if isinstance(response, str) else response[0])
    return response


async def _run_single_agent(intent_result: dict, user_context: str):
    """Dispatch a single agent based on intent analysis."""
    agent_name = intent_result.get("agent", "")
    action = intent_result.get("action", "")
    coordinator: CoordinatorAgent = st.session_state.coordinator
    df = st.session_state.get("current_df")
    if df is None:
        df = st.session_state.get("df")

    if df is None:
        return "Please upload a dataset first (use the sidebar)."

    agent = coordinator.agent_registry.get(agent_name)
    if agent is None:
        return f"Agent '{agent_name}' is not available. {intent_result.get('explanation', '')}"

    target = st.session_state.target_column

    # Build task
    task = {"action": action, "dataframe": df, "target_column": target}
    if action == "build_dashboard":
        task["eda_report"] = st.session_state.analysis_results.get("eda", {})
        task["model_results"] = st.session_state.analysis_results.get("modeling", {})
        task["cleaning_report"] = st.session_state.analysis_results.get("cleaning", {})
    if action == "generate_report":
        task["eda_report"] = st.session_state.analysis_results.get("eda", {})
        task["model_results"] = st.session_state.analysis_results.get("modeling", {})
        # Collect insights from previous InsightsAgent run if available
        prev_insights = st.session_state.analysis_results.get("insights", {})
        task["insights"] = prev_insights.get("insights", [])
        task["recommendations"] = prev_insights.get("recommendations", [])

    # Execute
    result = await agent.run(task)

    if not result.success:
        return f"**{agent_name}** encountered an error: {result.error}"

    # Store results and update working dataframe
    result_data = result.data if isinstance(result.data, dict) else {}
    _store_agent_result(agent_name, action, result_data)

    # Interpret results via LLM
    interpretation = await coordinator.interpret_results(
        agent_name, action, result_data, user_context
    )
    coordinator.record_analysis(agent_name, action, {"success": True})

    # Build response message with optional inline artefacts
    msg_extras = _extract_message_extras(agent_name, result_data)

    if msg_extras:
        return interpretation, msg_extras
    return interpretation


async def _run_full_pipeline(user_context: str) -> str:
    """Run the full data science pipeline."""
    coordinator: CoordinatorAgent = st.session_state.coordinator
    df = st.session_state.get("df")

    if df is None:
        return "Please upload a dataset first (use the sidebar)."

    target = st.session_state.target_column
    current_df = df.copy()
    results_parts = []

    steps = [
        ("DataCleanerAgent", "clean_data", "🧹 Cleaning data..."),
        ("EDAAgent", "full_eda", "📊 Running EDA..."),
        ("FeatureEngineerAgent", "engineer_features", "⚙️ Engineering features..."),
    ]
    if target:
        steps.extend([
            ("ModelTrainerAgent", "train_models", "🤖 Training models..."),
            ("AutoMLAgent", "auto_select_models", "🔮 Running AutoML..."),
        ])

    for agent_name, action, status_msg in steps:
        agent = coordinator.agent_registry.get(agent_name)
        if agent is None:
            continue

        task = {"action": action, "dataframe": current_df, "target_column": target}
        result = await agent.run(task)

        if result.success:
            result_data = result.data if isinstance(result.data, dict) else {}
            _store_agent_result(agent_name, action, result_data)

            # Update working dataframe if agent produces one
            if "dataframe" in result_data and isinstance(result_data["dataframe"], pd.DataFrame):
                current_df = _sanitize_dtypes(result_data["dataframe"])
                st.session_state.current_df = current_df

            coordinator.record_analysis(agent_name, action, {"success": True})
        else:
            results_parts.append(f"**{agent_name}** failed: {result.error}")

    st.session_state.pipeline_ran = True

    # Auto-build dashboard at the end of pipeline
    dashboard_agent = coordinator.agent_registry.get("DashboardBuilderAgent")
    if dashboard_agent:
        dash_task = {
            "action": "build_dashboard",
            "dataframe": current_df,
            "target_column": target,
            "eda_report": st.session_state.analysis_results.get("eda", {}),
            "model_results": st.session_state.analysis_results.get("modeling", {}),
            "cleaning_report": st.session_state.analysis_results.get("cleaning", {}),
        }
        dash_result = await dashboard_agent.run(dash_task)
        if dash_result.success:
            dash_data = dash_result.data if isinstance(dash_result.data, dict) else {}
            _store_agent_result("DashboardBuilderAgent", "build_dashboard", dash_data)

    # Compile a summary of everything
    summary_data = {}
    for key, val in st.session_state.analysis_results.items():
        if isinstance(val, dict):
            summary_data[key] = {k: v for k, v in val.items() if not isinstance(v, pd.DataFrame)}

    interpretation = await coordinator.interpret_results(
        "Pipeline", "full_analysis", summary_data, user_context
    )

    if results_parts:
        interpretation += "\n\n**Issues:**\n" + "\n".join(results_parts)

    # Attach dashboard sections to response
    dash_components = st.session_state.analysis_results.get("dashboard", {}).get("components", [])
    if dash_components:
        return interpretation, {"dashboard": dash_components}

    return interpretation


def _store_agent_result(agent_name: str, action: str, result_data: dict):
    """Store agent result in session state for later use."""
    key_map = {
        "DataCleanerAgent": "cleaning",
        "EDAAgent": "eda",
        "FeatureEngineerAgent": "feature_engineering",
        "ModelTrainerAgent": "modeling",
        "AutoMLAgent": "automl",
        "DataVisualizerAgent": "visualization",
        "DashboardBuilderAgent": "dashboard",
        "ForecastAgent": "forecast",
        "InsightsAgent": "insights",
        "ReportGeneratorAgent": "report",
    }
    key = key_map.get(agent_name, agent_name)
    st.session_state.analysis_results[key] = result_data

    # Update working dataframe when relevant, sanitize dtypes to prevent
    # StringDtype leaking between agents
    if "dataframe" in result_data and isinstance(result_data["dataframe"], pd.DataFrame):
        st.session_state.current_df = _sanitize_dtypes(result_data["dataframe"])


def _extract_message_extras(agent_name: str, result_data: dict) -> dict:
    """Extract charts / dataframes to display inline in the chat message."""
    extras = {}

    if agent_name == "DataVisualizerAgent":
        extras["charts"] = result_data.get("charts", [])[:6]

    elif agent_name == "EDAAgent":
        # Attach a preview table of the statistical profile
        profile = result_data.get("statistical_profile", {}).get("numeric", {})
        if profile:
            rows = []
            for col, stats in list(profile.items())[:10]:
                rows.append({"column": col, **{k: round(v, 4) if isinstance(v, float) else v for k, v in stats.items()}})
            extras["metrics_table"] = rows

    elif agent_name == "ModelTrainerAgent":
        all_results = result_data.get("results", {})
        if all_results:
            rows = [{"Model": name, **d.get("metrics", {})} for name, d in all_results.items()]
            extras["metrics_table"] = rows

    elif agent_name == "DashboardBuilderAgent":
        extras["dashboard"] = result_data.get("components", [])

    elif agent_name == "ForecastAgent":
        forecast = result_data.get("forecast", {})
        if forecast and forecast.get("dates") and forecast.get("forecast_values"):
            extras["charts"] = [{
                "type": "forecast_line",
                "title": f"Forecast: {forecast.get('metric', 'Metric')}",
                "data": forecast,
            }]

    elif agent_name == "InsightsAgent":
        extras["insights_panel"] = {
            "insights": result_data.get("insights", []),
            "recommendations": result_data.get("recommendations", []),
            "executive_summary": result_data.get("executive_summary", ""),
        }

    elif agent_name == "ReportGeneratorAgent":
        extras["report_downloads"] = {
            "html": result_data.get("html"),
            "markdown": result_data.get("markdown"),
            "csv_summary": result_data.get("csv_summary"),
            "title": result_data.get("title", "Report"),
        }

    return extras


def _build_status_message() -> str:
    """Build a status summary message."""
    lines = ["**Current Status:**\n"]
    if st.session_state.df is not None:
        df = st.session_state.df
        lines.append(f"- **Dataset**: {df.shape[0]:,} rows × {df.shape[1]} columns")
    else:
        lines.append("- **Dataset**: Not loaded")

    if st.session_state.target_column:
        lines.append(f"- **Target**: {st.session_state.target_column}")
    else:
        lines.append("- **Target**: Not set")

    ar = st.session_state.analysis_results
    if ar:
        lines.append(f"- **Completed analyses**: {', '.join(ar.keys())}")
    else:
        lines.append("- **Completed analyses**: None yet")

    lines.append(f"- **Pipeline run**: {'Yes' if st.session_state.pipeline_ran else 'No'}")
    return "\n".join(lines)


# ---------- main ----------

def main():
    init_session_state()
    render_sidebar()

    # Header
    st.markdown(
        '<h1 class="main-header">🔬 AI Data Science Platform</h1>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<p class="sub-header">Chat with your AI Data Scientist — upload data, ask questions, get insights, forecast trends</p>',
        unsafe_allow_html=True,
    )

    # Render chat history
    render_chat()

    # Chat input
    if user_input := st.chat_input("Ask me anything about your data..."):
        # Display user message
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        # Process and display assistant response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = run_async(handle_user_message(user_input))

            # handle_user_message returns either a string or (string, extras) tuple
            extras = {}
            if isinstance(response, tuple):
                response_text, extras = response
            else:
                response_text = response

            st.markdown(response_text)

            if "charts" in extras:
                _render_charts(extras["charts"])
            if "metrics_table" in extras:
                st.dataframe(pd.DataFrame(extras["metrics_table"]), use_container_width=True)
            if "dashboard" in extras:
                _render_professional_dashboard(extras["dashboard"])
            if "insights_panel" in extras:
                _render_insights_panel(extras["insights_panel"])
            if "report_downloads" in extras:
                _render_report_downloads(extras["report_downloads"])

        # Store in message history
        msg = {"role": "assistant", "content": response_text}
        msg.update(extras)
        st.session_state.messages.append(msg)


if __name__ == "__main__":
    main()
