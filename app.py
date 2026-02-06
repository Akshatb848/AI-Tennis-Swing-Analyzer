"""
Data Science Agent Platform - Chat-Based Application

Conversational interface powered by LLM-based coordinator agent.
Users interact via natural language; the coordinator dispatches
specialized agents and interprets results.
"""

import streamlit as st
import pandas as pd
import numpy as np
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
from llm.client import get_llm_client
from llm.prompts import PromptTemplates
from utils.helpers import generate_sample_data

# Page config
st.set_page_config(
    page_title="Data Science Agent Platform",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown(
    """
<style>
    .main-header {
        font-size: 2rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        text-align: center;
        color: #666;
        margin-bottom: 1rem;
        font-size: 0.95rem;
    }
</style>
""",
    unsafe_allow_html=True,
)


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
        llm_client = get_llm_client()
        coordinator = CoordinatorAgent(llm_client=llm_client)
        coordinator.register_agent(DataCleanerAgent())
        coordinator.register_agent(EDAAgent())
        coordinator.register_agent(FeatureEngineerAgent())
        coordinator.register_agent(ModelTrainerAgent())
        coordinator.register_agent(AutoMLAgent())
        coordinator.register_agent(DashboardBuilderAgent())
        coordinator.register_agent(DataVisualizerAgent())
        st.session_state.coordinator = coordinator

    defaults = {
        "messages": [],
        "df": None,
        "target_column": None,
        "current_df": None,  # working copy after cleaning / FE
        "analysis_results": {},
        "pipeline_ran": False,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val

    # Add welcome message on first load
    if not st.session_state.messages:
        welcome = st.session_state.coordinator.get_welcome_message()
        st.session_state.messages.append({"role": "assistant", "content": welcome})


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
                p = None if provider == "auto" else (None if provider == "fallback" else provider)
                client = _get(provider=p, api_key=api_key or None, model=model or None)
                st.session_state.coordinator.llm_client = client
                st.success(f"LLM set to **{type(client).__name__}**")

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


def _set_dataframe(df: pd.DataFrame, name: str):
    """Store a dataframe in session state and add a system message."""
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


def _render_charts(charts):
    """Render plotly charts stored in a message."""
    import plotly.express as px

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


# ---------- agent dispatch ----------

async def handle_user_message(user_input: str):
    """Process a user message through the coordinator agent."""
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
        response = await _run_single_agent(intent_result, user_input)
    else:
        # General conversation / data science Q&A
        explanation = intent_result.get("explanation", "")
        if explanation:
            response = explanation
        else:
            response = "I'm not sure what you'd like me to do. Try asking me to analyze your data, train models, or run a full pipeline!"

    coordinator.add_to_memory("assistant", response if isinstance(response, str) else response[0])
    return response


async def _run_single_agent(intent_result: dict, user_context: str):
    """Dispatch a single agent based on intent analysis."""
    agent_name = intent_result.get("agent", "")
    action = intent_result.get("action", "")
    coordinator: CoordinatorAgent = st.session_state.coordinator
    df = st.session_state.get("current_df") or st.session_state.get("df")

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
                current_df = result_data["dataframe"]
                st.session_state.current_df = current_df

            coordinator.record_analysis(agent_name, action, {"success": True})
        else:
            results_parts.append(f"**{agent_name}** failed: {result.error}")

    st.session_state.pipeline_ran = True

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
    }
    key = key_map.get(agent_name, agent_name)
    st.session_state.analysis_results[key] = result_data

    # Update working dataframe when relevant
    if "dataframe" in result_data and isinstance(result_data["dataframe"], pd.DataFrame):
        st.session_state.current_df = result_data["dataframe"]


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
        charts = []
        for comp in result_data.get("components", []):
            if comp.get("type") == "chart_section":
                charts.extend(comp.get("data", []))
        if charts:
            extras["charts"] = charts[:6]

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
        '<h1 class="main-header">🔬 Data Science Agent Platform</h1>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<p class="sub-header">Chat with your AI Data Scientist — upload data, ask questions, get insights</p>',
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

        # Store in message history
        msg = {"role": "assistant", "content": response_text}
        msg.update(extras)
        st.session_state.messages.append(msg)


if __name__ == "__main__":
    main()
