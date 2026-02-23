"""Main Streamlit dashboard for the AI Data Science Agent Platform."""

import os
import sys
import tempfile

import numpy as np
import pandas as pd
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.orchestrator import Orchestrator
from dashboard.components.chat_interface import render_chat
from dashboard.components.visualizations import (
    plot_before_after_skewness,
    plot_correlation_heatmap,
    plot_distribution_analysis,
    plot_feature_importance,
    plot_hypothesis_summary,
    plot_leaderboard,
    plot_missing_values,
    plot_vif_chart,
    render_viz_from_spec,
)
from utils.csv_loader import CSVLoadError, load_csv

st.set_page_config(
    page_title="AI Data Scientist",
    page_icon="",
    layout="wide",
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', system-ui, -apple-system, sans-serif;
    }

    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 1rem;
        max-width: 1200px;
    }

    [data-testid="stSidebar"] {
        background: #0f172a;
        color: #e2e8f0;
    }
    [data-testid="stSidebar"] * {
        color: #e2e8f0 !important;
    }
    [data-testid="stSidebar"] .stMarkdown p {
        color: #cbd5e1 !important;
    }
    [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
        color: #f1f5f9 !important;
        font-weight: 600;
    }
    [data-testid="stSidebar"] hr {
        border-color: #1e293b !important;
    }
    [data-testid="stSidebar"] .stSelectbox label,
    [data-testid="stSidebar"] .stCheckbox label {
        color: #94a3b8 !important;
        font-size: 0.85rem;
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: 0;
        border-bottom: 2px solid #e2e8f0;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 10px 20px;
        font-weight: 500;
        font-size: 0.9rem;
        color: #64748b;
        border-bottom: 2px solid transparent;
        margin-bottom: -2px;
    }
    .stTabs [aria-selected="true"] {
        color: #6366f1 !important;
        border-bottom: 2px solid #6366f1 !important;
        background: transparent !important;
    }

    div[data-testid="stMetricValue"] {
        font-size: 1.5rem;
        font-weight: 700;
        color: #1e293b;
    }
    div[data-testid="stMetricLabel"] {
        font-size: 0.8rem;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    .main-title {
        font-size: 1.6rem;
        font-weight: 700;
        color: #1e293b;
        margin-bottom: 0.25rem;
        letter-spacing: -0.02em;
    }
    .main-subtitle {
        font-size: 0.85rem;
        color: #64748b;
        margin-bottom: 1.5rem;
    }

    .section-header {
        font-size: 1rem;
        font-weight: 600;
        color: #1e293b;
        border-bottom: 1px solid #e2e8f0;
        padding-bottom: 0.5rem;
        margin-bottom: 1rem;
        margin-top: 1.5rem;
    }

    .card {
        background: white;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 16px;
        margin-bottom: 12px;
    }
    .card-muted {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 16px;
        margin-bottom: 12px;
    }

    .status-chip {
        display: inline-block;
        padding: 2px 10px;
        border-radius: 12px;
        font-size: 0.75rem;
        font-weight: 500;
    }
    .status-completed { background: #dcfce7; color: #166534; }
    .status-failed { background: #fee2e2; color: #991b1b; }
    .status-pending { background: #f1f5f9; color: #475569; }

    .sig-yes { color: #059669; font-weight: 600; }
    .sig-no { color: #64748b; }

    .welcome-container {
        text-align: center;
        padding: 80px 40px;
    }
    .welcome-container h2 {
        font-size: 1.4rem;
        font-weight: 600;
        color: #1e293b;
        margin-bottom: 0.5rem;
    }
    .welcome-container p {
        color: #64748b;
        font-size: 0.95rem;
    }
    .pipeline-steps {
        display: flex;
        justify-content: center;
        gap: 8px;
        margin: 24px 0;
        flex-wrap: wrap;
    }
    .pipeline-step {
        background: #f1f5f9;
        border: 1px solid #e2e8f0;
        border-radius: 6px;
        padding: 8px 14px;
        font-size: 0.8rem;
        color: #475569;
        font-weight: 500;
    }
    .pipeline-arrow {
        color: #cbd5e1;
        line-height: 2.2;
    }
</style>
""", unsafe_allow_html=True)


def _init_clients():
    if "llm_client" not in st.session_state:
        try:
            from core.llm_client import LLMClient
            st.session_state["llm_client"] = LLMClient()
        except Exception:
            st.session_state["llm_client"] = None

    if "rag_client" not in st.session_state:
        try:
            from core.rag_client import RAGClient
            rag = RAGClient(persist_dir="./chroma_db")
            if not rag.is_indexed():
                docs_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "rag_docs")
                if os.path.isdir(docs_dir):
                    rag.index_documents(docs_dir)
            st.session_state["rag_client"] = rag
        except Exception:
            st.session_state["rag_client"] = None


_init_clients()


@st.cache_data
def generate_sample_dataset(name: str) -> tuple:
    if name == "iris":
        from sklearn.datasets import load_iris
        data = load_iris()
        df = pd.DataFrame(data.data, columns=data.feature_names)
        df["target"] = data.target
        return df, "target"
    elif name == "housing":
        from sklearn.datasets import fetch_california_housing
        data = fetch_california_housing()
        df = pd.DataFrame(data.data, columns=data.feature_names)
        df["MedHouseVal"] = data.target
        df = df.sample(500, random_state=42).reset_index(drop=True)
        return df, "MedHouseVal"
    elif name == "titanic":
        np.random.seed(42)
        n = 500
        df = pd.DataFrame({
            "Pclass": np.random.choice([1, 2, 3], n, p=[0.2, 0.3, 0.5]),
            "Sex": np.random.choice(["male", "female"], n, p=[0.65, 0.35]),
            "Age": np.random.normal(30, 14, n).clip(1, 80).round(1),
            "SibSp": np.random.choice([0, 1, 2, 3, 4], n, p=[0.6, 0.2, 0.1, 0.05, 0.05]),
            "Parch": np.random.choice([0, 1, 2, 3], n, p=[0.7, 0.15, 0.1, 0.05]),
            "Fare": np.random.exponential(30, n).round(2),
            "Embarked": np.random.choice(["S", "C", "Q"], n, p=[0.7, 0.2, 0.1]),
        })
        survival_prob = 0.3 + 0.2 * (df["Sex"] == "female").astype(float) - 0.1 * (df["Pclass"] / 3)
        df["Survived"] = (np.random.random(n) < survival_prob).astype(int)
        return df, "Survived"
    else:
        np.random.seed(42)
        df = pd.DataFrame({
            "feature_a": np.random.randn(200),
            "feature_b": np.random.uniform(0, 100, 200),
            "feature_c": np.random.choice(["cat", "dog", "bird"], 200),
            "feature_d": np.random.randint(1, 50, 200),
            "feature_e": np.random.exponential(10, 200).round(2),
            "target": np.random.randn(200) * 10 + 50,
        })
        return df, "target"


def save_df_to_temp(df: pd.DataFrame) -> str:
    os.makedirs("data/uploads", exist_ok=True)
    tmp = tempfile.NamedTemporaryFile(suffix=".csv", delete=False, dir="data/uploads")
    df.to_csv(tmp.name, index=False)
    tmp.close()
    return tmp.name


def run_pipeline(file_path: str, target_col: str = None, use_optuna: bool = True):
    llm = st.session_state.get("llm_client")
    rag = st.session_state.get("rag_client")
    orchestrator = Orchestrator(llm_client=llm, rag_client=rag)
    results = orchestrator.run_pipeline(file_path, target_col, use_optuna=use_optuna)

    st.session_state["pipeline_results"] = results
    st.session_state["pipeline_state"] = results.get("pipeline_state", {})

    strategy = orchestrator.get_step_result("strategy")
    if strategy and strategy.success:
        st.session_state["df"] = strategy.data.get("df")
        st.session_state["profile_data"] = strategy.data.get("profile", {})
        st.session_state["strategy_data"] = strategy.data.get("strategy", {})

    eng = orchestrator.get_step_result("engineering")
    if eng and eng.success:
        st.session_state["engineering_data"] = eng.data

    exploration = orchestrator.get_step_result("exploration")
    if exploration and exploration.success:
        st.session_state["exploration_data"] = exploration.data

    model_res = orchestrator.get_step_result("modeling")
    if model_res and model_res.success:
        st.session_state["model_results"] = model_res.data

    mlops = orchestrator.get_step_result("mlops")
    if mlops and mlops.success:
        st.session_state["mlops_data"] = mlops.data


def render_data_overview():
    profile = st.session_state.get("profile_data")
    df = st.session_state.get("df")
    strategy = st.session_state.get("strategy_data", {})

    if profile is None and df is None:
        preview = st.session_state.get("preview_df")
        preview_profile = st.session_state.get("preview_profile")
        if preview is not None:
            st.markdown('<p class="section-header">Data Preview</p>', unsafe_allow_html=True)
            st.dataframe(preview.head(100), width="stretch", hide_index=True)
            if preview_profile and preview_profile.get("stats"):
                stats = preview_profile["stats"]
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Rows", f"{stats['row_count']:,}")
                c2.metric("Columns", stats["column_count"])
                c3.metric("Quality", f"{preview_profile.get('quality_score', 0):.0f}%")
                c4.metric("Problem", preview_profile.get("problem_type", "--").replace("_", " ").title())
        else:
            st.info("Upload a dataset or select a sample to get started.")
        return

    stats = profile.get("stats", {})
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", f"{stats.get('row_count', 0):,}")
    c2.metric("Columns", stats.get("column_count", 0))
    c3.metric("Quality Score", f"{profile.get('quality_score', 0):.0f}%")
    c4.metric("Problem Type", profile.get("problem_type", "--").replace("_", " ").title())

    if strategy:
        objective = strategy.get("objective", {})
        recommendations = strategy.get("recommendations", [])
        if objective or recommendations:
            st.markdown('<p class="section-header">Strategy & Objectives</p>', unsafe_allow_html=True)
            if objective:
                obj_cols = st.columns(4)
                obj_cols[0].markdown(f"**Problem**<br>{objective.get('problem_type', '?').replace('_', ' ').title()}", unsafe_allow_html=True)
                obj_cols[1].markdown(f"**Target**<br>`{objective.get('target', '?')}`", unsafe_allow_html=True)
                obj_cols[2].markdown(f"**Primary KPI**<br>{objective.get('kpi', '?')}", unsafe_allow_html=True)
                obj_cols[3].markdown(f"**Constraints**<br>{len(objective.get('constraints', []))}", unsafe_allow_html=True)
            if recommendations:
                with st.expander("Recommendations", expanded=False):
                    for r in recommendations:
                        st.markdown(f"- {r}")

    col_left, col_right = st.columns(2)
    with col_left:
        st.markdown('<p class="section-header">Schema</p>', unsafe_allow_html=True)
        schema = profile.get("schema", {})
        if schema:
            schema_df = pd.DataFrame([
                {"Column": k, "Type": v, "Missing %": round(stats.get("missing_pct", {}).get(k, 0), 1)}
                for k, v in schema.items()
            ])
            st.dataframe(schema_df, width="stretch", hide_index=True, height=300)

    with col_right:
        st.markdown('<p class="section-header">Data Quality</p>', unsafe_allow_html=True)
        missing_fig = plot_missing_values(stats.get("missing_pct", {}))
        if missing_fig:
            st.plotly_chart(missing_fig, use_container_width=True)
        else:
            st.markdown('<div class="card-muted">No missing values detected.</div>', unsafe_allow_html=True)

        if profile.get("id_columns"):
            st.caption(f"ID columns: {', '.join(profile['id_columns'])}")
        if profile.get("leakage_columns"):
            st.warning(f"Potential leakage: {', '.join(profile['leakage_columns'])}")

    if df is not None:
        st.markdown('<p class="section-header">Data Sample</p>', unsafe_allow_html=True)
        st.dataframe(df.head(50), width="stretch", hide_index=True)


def render_feature_engineering():
    eng_data = st.session_state.get("engineering_data")
    if eng_data is None:
        st.info("Run the pipeline to see feature engineering results.")
        return

    report = eng_data.get("engineering_report", {})

    transformations = report.get("transformations_summary", [])
    if transformations:
        st.markdown('<p class="section-header">Transformation Pipeline</p>', unsafe_allow_html=True)
        trans_df = pd.DataFrame(transformations)
        st.dataframe(trans_df, width="stretch", hide_index=True)

    col1, col2 = st.columns(2)
    with col1:
        before = report.get("before_stats", {})
        before_features = before.get("features", {})
        if before_features:
            st.markdown('<p class="section-header">Before Engineering</p>', unsafe_allow_html=True)
            before_df = pd.DataFrame(before_features).T
            before_df.index.name = "Feature"
            st.dataframe(before_df.round(4), width="stretch", height=300)

    with col2:
        after = report.get("after_stats", {})
        after_features = after.get("features", {})
        if after_features:
            st.markdown('<p class="section-header">After Engineering</p>', unsafe_allow_html=True)
            after_df = pd.DataFrame(after_features).T
            after_df.index.name = "Feature"
            st.dataframe(after_df.round(4), width="stretch", height=300)

    skew_corrections = report.get("skewness_correction", {}).get("corrections", {})
    if skew_corrections:
        st.markdown('<p class="section-header">Skewness Corrections</p>', unsafe_allow_html=True)
        fig = plot_before_after_skewness(skew_corrections)
        if fig:
            st.plotly_chart(fig, use_container_width=True)

        skew_df = pd.DataFrame([
            {
                "Feature": col,
                "Method": info.get("method", ""),
                "Before": info.get("original_skewness", ""),
                "After": info.get("new_skewness", ""),
            }
            for col, info in skew_corrections.items()
        ])
        st.dataframe(skew_df, width="stretch", hide_index=True)

    details_col1, details_col2 = st.columns(2)
    with details_col1:
        impute = report.get("imputation", {})
        details = impute.get("details", {})
        if details:
            st.markdown('<p class="section-header">Imputation Details</p>', unsafe_allow_html=True)
            imp_df = pd.DataFrame([
                {"Column": col, "Method": info.get("method", ""), "Value": str(info.get("fill_value", "--"))}
                for col, info in details.items()
            ])
            st.dataframe(imp_df, width="stretch", hide_index=True)

    with details_col2:
        encoding = report.get("encoding", {})
        encoded = encoding.get("encoded", {})
        if encoded:
            st.markdown('<p class="section-header">Encoding Details</p>', unsafe_allow_html=True)
            enc_df = pd.DataFrame([
                {"Column": col, "Method": info.get("method", ""), "Categories": info.get("categories", "")}
                for col, info in encoded.items()
            ])
            st.dataframe(enc_df, width="stretch", hide_index=True)

    outliers = report.get("outlier_handling", {})
    outlier_details = outliers.get("details", {})
    if outlier_details:
        st.markdown('<p class="section-header">Outlier Handling</p>', unsafe_allow_html=True)
        out_df = pd.DataFrame([
            {
                "Column": col,
                "Method": info.get("method", ""),
                "Outliers": info.get("outlier_count", 0),
                "% of Data": info.get("outlier_pct", 0),
            }
            for col, info in outlier_details.items()
        ])
        st.dataframe(out_df, width="stretch", hide_index=True)

    interactions = report.get("interactions", {})
    created = interactions.get("created", [])
    if created:
        st.markdown('<p class="section-header">Interaction Features</p>', unsafe_allow_html=True)
        st.markdown(f"Created {len(created)} interaction features: `{'`, `'.join(created)}`")


def render_exploration():
    exploration = st.session_state.get("exploration_data")
    df = st.session_state.get("df")

    if exploration is None:
        st.info("Run the pipeline to see exploratory analysis.")
        return

    dist_analysis = exploration.get("distribution_analysis", [])
    if dist_analysis:
        st.markdown('<p class="section-header">Distribution Analysis</p>', unsafe_allow_html=True)
        dist_fig = plot_distribution_analysis(dist_analysis)
        if dist_fig:
            st.plotly_chart(dist_fig, use_container_width=True)

        dist_df = pd.DataFrame(dist_analysis)
        cols_to_show = ["feature", "skewness", "kurtosis", "skew_label", "kurtosis_label", "mean", "median"]
        available = [c for c in cols_to_show if c in dist_df.columns]
        st.dataframe(dist_df[available], width="stretch", hide_index=True)

    normality = exploration.get("normality_tests", [])
    hypothesis = exploration.get("hypothesis_tests", [])

    if normality or hypothesis:
        st.markdown('<p class="section-header">Hypothesis Testing</p>', unsafe_allow_html=True)

    test_col1, test_col2 = st.columns(2)

    with test_col1:
        if normality:
            st.markdown("**Normality Tests**")
            norm_df = pd.DataFrame(normality)
            cols_show = ["feature", "test", "statistic", "p_value", "conclusion"]
            available = [c for c in cols_show if c in norm_df.columns]
            st.dataframe(norm_df[available], width="stretch", hide_index=True, height=250)

    with test_col2:
        if hypothesis:
            fig = plot_hypothesis_summary(hypothesis)
            if fig:
                st.plotly_chart(fig, use_container_width=True)

    if hypothesis:
        st.markdown("**Group Comparison & Independence Tests**")
        hyp_df = pd.DataFrame(hypothesis)
        cols_show = ["feature", "test", "statistic", "p_value", "conclusion", "effect_size"]
        available = [c for c in cols_show if c in hyp_df.columns]
        st.dataframe(hyp_df[available], width="stretch", hide_index=True)

    corr = exploration.get("correlation_matrix")
    vif = exploration.get("vif_results", [])

    corr_col, vif_col = st.columns(2)
    with corr_col:
        if corr:
            st.markdown('<p class="section-header">Correlation Matrix</p>', unsafe_allow_html=True)
            fig = plot_correlation_heatmap(corr)
            if fig:
                st.plotly_chart(fig, use_container_width=True)

    with vif_col:
        if vif:
            st.markdown('<p class="section-header">Multicollinearity (VIF)</p>', unsafe_allow_html=True)
            vif_fig = plot_vif_chart(vif)
            if vif_fig:
                st.plotly_chart(vif_fig, use_container_width=True)
            vif_df = pd.DataFrame(vif)
            st.dataframe(vif_df, width="stretch", hide_index=True)

    viz_specs = exploration.get("viz_specs", [])
    if viz_specs and df is not None:
        st.markdown('<p class="section-header">Visualizations</p>', unsafe_allow_html=True)
        viz_cols = st.columns(2)
        rendered = 0
        for spec in viz_specs[:8]:
            if spec.get("plot_kind") == "correlation_heatmap":
                continue
            with viz_cols[rendered % 2]:
                fig = render_viz_from_spec(spec, df)
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
                    rendered += 1

    summary = exploration.get("summary", "")
    if summary:
        with st.expander("Full Analysis Summary", expanded=False):
            st.text(summary)


def render_modeling():
    model_data = st.session_state.get("model_results")
    if model_data is None:
        st.info("Run the pipeline to see modeling results.")
        return

    c1, c2, c3 = st.columns(3)
    c1.metric("Champion Model", model_data.get("champion_name", "--"))
    c2.metric("Best Score", f"{model_data.get('champion_score', 0):.4f}")
    c3.metric("Models Trained", len(model_data.get("leaderboard", [])))

    lb = model_data.get("leaderboard", [])
    if lb:
        st.markdown('<p class="section-header">Model Leaderboard</p>', unsafe_allow_html=True)
        fig = plot_leaderboard(lb)
        if fig:
            st.plotly_chart(fig, use_container_width=True)

        st.markdown('<p class="section-header">Detailed Results</p>', unsafe_allow_html=True)
        lb_df = pd.DataFrame(lb)
        for col in lb_df.select_dtypes(include=[np.number]).columns:
            lb_df[col] = lb_df[col].round(4)
        st.dataframe(lb_df, width="stretch", hide_index=True)

    champion_params = model_data.get("champion_params", {})
    if champion_params:
        with st.expander("Champion Hyperparameters"):
            params_df = pd.DataFrame([
                {"Parameter": k, "Value": str(v)} for k, v in champion_params.items()
            ])
            st.dataframe(params_df, width="stretch", hide_index=True)


def render_deployment():
    mlops = st.session_state.get("mlops_data")
    if mlops is None:
        st.info("Run the pipeline to see deployment information.")
        return

    st.markdown('<p class="section-header">Deployment Status</p>', unsafe_allow_html=True)
    ready = mlops.get("deployment_ready", False)
    if ready:
        st.markdown('<div class="card" style="border-left: 3px solid #10b981;">Model is ready for deployment.</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="card" style="border-left: 3px solid #f59e0b;">Model saved. Review deployment checklist before production use.</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"**Model path:** `{mlops.get('model_path', 'N/A')}`")

        recommendations = mlops.get("deployment_recommendations", [])
        if recommendations:
            st.markdown('<p class="section-header">Deployment Options</p>', unsafe_allow_html=True)
            for rec in recommendations:
                st.markdown(f'<div class="card-muted">{rec}</div>', unsafe_allow_html=True)

    with col2:
        monitoring = mlops.get("monitoring_config", {})
        if monitoring:
            st.markdown('<p class="section-header">Monitoring Configuration</p>', unsafe_allow_html=True)
            for k, v in monitoring.items():
                if isinstance(v, dict):
                    st.markdown(f"**{k.replace('_', ' ').title()}:**")
                    for k2, v2 in v.items():
                        st.markdown(f"&nbsp;&nbsp;&nbsp;&nbsp;{k2}: {v2}")
                else:
                    st.markdown(f"**{k.replace('_', ' ').title()}:** {v}")

    script = mlops.get("inference_script", "")
    if script:
        with st.expander("Inference Script"):
            st.code(script, language="python")


# --- Sidebar ---
with st.sidebar:
    st.markdown("## AI Data Scientist")
    st.markdown("---")

    st.markdown("### Upload Data")
    uploaded_file = st.file_uploader(
        "Upload a dataset",
        type=["csv", "xlsx", "xls", "parquet", "json"],
        help="CSV, Excel, Parquet, JSON supported",
    )

    st.markdown("### Sample Datasets")
    sample_cols = st.columns(2)
    with sample_cols[0]:
        iris_btn = st.button("Iris", key="btn_iris", use_container_width=True)
        titanic_btn = st.button("Titanic", key="btn_titanic", use_container_width=True)
    with sample_cols[1]:
        housing_btn = st.button("Housing", key="btn_housing", use_container_width=True)
        random_btn = st.button("Random", key="btn_random", use_container_width=True)

    sample_choice = None
    if iris_btn:
        sample_choice = "iris"
    elif housing_btn:
        sample_choice = "housing"
    elif titanic_btn:
        sample_choice = "titanic"
    elif random_btn:
        sample_choice = "random"

    if sample_choice:
        df_sample, default_target = generate_sample_dataset(sample_choice)
        file_path = save_df_to_temp(df_sample)
        st.session_state["pending_file"] = file_path
        st.session_state["pending_target"] = default_target
        st.session_state["pending_columns"] = df_sample.columns.tolist()
        st.session_state["preview_df"] = df_sample
        st.session_state["preview_profile"] = None

    if uploaded_file is not None:
        ext = os.path.splitext(uploaded_file.name)[1].lower()
        if ext == ".csv":
            try:
                df_up, profile_up = load_csv(uploaded_file)
                os.makedirs("data/uploads", exist_ok=True)
                fpath = os.path.join("data/uploads", uploaded_file.name)
                df_up.to_csv(fpath, index=False)

                st.session_state["pending_file"] = fpath
                st.session_state["pending_target"] = profile_up.get("target_column")
                st.session_state["pending_columns"] = df_up.columns.tolist()
                st.session_state["preview_df"] = df_up
                st.session_state["preview_profile"] = profile_up

                if profile_up.get("warnings"):
                    with st.expander("Parsing warnings"):
                        for w in profile_up["warnings"]:
                            st.warning(w)

                st.success(f"Loaded {len(df_up):,} rows, {len(df_up.columns)} columns")
                problem = profile_up.get("problem_type", "unknown")
                target = profile_up.get("target_column", "None detected")
                st.markdown(f"**Detected:** {problem.replace('_', ' ').title()} | Target: `{target}`")

            except CSVLoadError as e:
                st.error(f"Could not read CSV: {e}")
                st.info(f"{e.suggestion}" if e.suggestion else "Try re-saving as UTF-8 CSV.")
            except Exception as e:
                st.error(f"Error: {e}")
        else:
            try:
                os.makedirs("data/uploads", exist_ok=True)
                fpath = os.path.join("data/uploads", uploaded_file.name)
                with open(fpath, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                if ext in (".xlsx", ".xls"):
                    df_up = pd.read_excel(fpath)
                elif ext == ".parquet":
                    df_up = pd.read_parquet(fpath)
                else:
                    df_up = pd.read_json(fpath)

                st.session_state["pending_file"] = fpath
                st.session_state["pending_target"] = None
                st.session_state["pending_columns"] = df_up.columns.tolist()
                st.session_state["preview_df"] = df_up
                st.session_state["preview_profile"] = None
                st.success(f"Loaded {len(df_up):,} rows, {len(df_up.columns)} columns")
            except Exception as e:
                st.error(f"Error reading {ext} file: {e}")

    st.markdown("---")

    if "pending_columns" in st.session_state:
        st.markdown("### Configuration")
        cols = st.session_state["pending_columns"]
        default_idx = 0
        if st.session_state.get("pending_target") in cols:
            default_idx = cols.index(st.session_state["pending_target"])
        target_col = st.selectbox("Target column", cols, index=default_idx)
        use_optuna = st.checkbox("Optuna tuning", value=True, help="Bayesian hyperparameter optimization")

        if st.button("Run Pipeline", type="primary", use_container_width=True):
            file_path = st.session_state.get("pending_file", "")
            with st.status("Running pipeline...", expanded=True) as status:
                progress = st.progress(0)
                try:
                    st.write("Analyzing strategy...")
                    progress.progress(10)
                    run_pipeline(file_path, target_col, use_optuna=use_optuna)
                    progress.progress(100)
                    ps = st.session_state.get("pipeline_state", {})
                    for step, state in ps.items():
                        chip = "status-completed" if state == "completed" else "status-failed" if state == "failed" else "status-pending"
                        st.markdown(f'<span class="status-chip {chip}">{state}</span> {step}', unsafe_allow_html=True)
                    status.update(label="Pipeline complete", state="complete")
                except Exception as e:
                    progress.progress(100)
                    status.update(label="Pipeline failed", state="error")
                    st.error(f"Error: {e}")

    pipeline_state = st.session_state.get("pipeline_state", {})
    if pipeline_state:
        st.markdown("### Pipeline Status")
        for step, state in pipeline_state.items():
            chip = "status-completed" if state == "completed" else "status-failed" if state == "failed" else "status-pending"
            st.markdown(f'<span class="status-chip {chip}">{state}</span> {step}', unsafe_allow_html=True)


# --- Main Content ---
st.markdown('<h1 class="main-title">AI Data Scientist</h1>', unsafe_allow_html=True)
st.markdown('<p class="main-subtitle">Automated machine learning pipeline with intelligent analysis</p>', unsafe_allow_html=True)

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "Overview",
    "Feature Engineering",
    "Exploration & Testing",
    "Modeling",
    "Deployment",
    "Assistant",
])

with tab1:
    render_data_overview()

with tab2:
    render_feature_engineering()

with tab3:
    render_exploration()

with tab4:
    render_modeling()

with tab5:
    render_deployment()

with tab6:
    context = {}
    if st.session_state.get("profile_data"):
        context["profile"] = st.session_state["profile_data"]
    if st.session_state.get("model_results"):
        context["model_results"] = st.session_state["model_results"]
    if st.session_state.get("engineering_data"):
        context["engineering_data"] = st.session_state["engineering_data"]
    if st.session_state.get("exploration_data"):
        context["exploration_data"] = st.session_state["exploration_data"]
    if st.session_state.get("mlops_data"):
        context["mlops_data"] = st.session_state["mlops_data"]
    if st.session_state.get("strategy_data"):
        context["strategy_data"] = st.session_state["strategy_data"]
    render_chat(
        llm_client=st.session_state.get("llm_client"),
        rag_client=st.session_state.get("rag_client"),
        pipeline_context=context,
    )

if st.session_state.get("df") is None and st.session_state.get("preview_df") is None:
    st.markdown("""
<div class="welcome-container">
<h2>Welcome to the AI Data Scientist</h2>
<p>Upload a dataset or select a sample to begin automated analysis.</p>
<div class="pipeline-steps">
    <span class="pipeline-step">Strategy</span>
    <span class="pipeline-arrow">&rarr;</span>
    <span class="pipeline-step">Engineering</span>
    <span class="pipeline-arrow">&rarr;</span>
    <span class="pipeline-step">Exploration</span>
    <span class="pipeline-arrow">&rarr;</span>
    <span class="pipeline-step">Modeling</span>
    <span class="pipeline-arrow">&rarr;</span>
    <span class="pipeline-step">Deployment</span>
</div>
<p style="font-size: 0.8rem; color: #94a3b8;">Powered by XGBoost, LightGBM, Optuna, ChromaDB</p>
</div>
""", unsafe_allow_html=True)
