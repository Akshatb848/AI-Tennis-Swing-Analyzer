"""Chat interface component with LLM+RAG powered analysis."""

import streamlit as st
from datetime import datetime


def render_chat(llm_client=None, rag_client=None, pipeline_context: dict = None):
    """Render the AI chat interface with pipeline-aware analysis."""
    st.markdown('<p class="section-header" style="margin-top: 0;">Assistant</p>', unsafe_allow_html=True)

    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []

    context = pipeline_context or {}
    has_llm = llm_client is not None and llm_client.is_available()
    has_rag = rag_client is not None and rag_client.is_ready

    if has_llm:
        st.caption("Connected to LLM (Llama 3.3 70B) -- AI-powered analysis available")
    else:
        st.caption("LLM not connected -- using rule-based analysis")

    quick_col1, quick_col2, quick_col3 = st.columns(3)
    with quick_col1:
        if st.button("Analyze Results", key="qa_results", use_container_width=True):
            _inject_quick_action("Analyze the pipeline results and tell me what was achieved.", context, llm_client, rag_client)
    with quick_col2:
        if st.button("Suggest Improvements", key="qa_improve", use_container_width=True):
            _inject_quick_action("What are the specific areas for improvement and how can I get better results?", context, llm_client, rag_client)
    with quick_col3:
        if st.button("Explain Features", key="qa_features", use_container_width=True):
            _inject_quick_action("Explain the most important features and what transformations were applied.", context, llm_client, rag_client)

    chat_container = st.container(height=400)
    with chat_container:
        for msg in st.session_state["chat_history"]:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

    user_input = st.chat_input("Ask about your data, model performance, improvements...")

    if user_input:
        st.session_state["chat_history"].append({
            "role": "user",
            "content": user_input,
            "timestamp": datetime.now().isoformat(),
        })

        with st.spinner("Thinking..."):
            response = _generate_response(user_input, context, llm_client, rag_client)

        st.session_state["chat_history"].append({
            "role": "assistant",
            "content": response,
            "timestamp": datetime.now().isoformat(),
        })
        st.rerun()

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Clear Chat", type="secondary"):
            st.session_state["chat_history"] = []
            st.rerun()
    with col2:
        st.caption(f"{len(st.session_state['chat_history'])} messages")


def _inject_quick_action(query: str, context: dict, llm_client, rag_client):
    st.session_state["chat_history"].append({
        "role": "user",
        "content": query,
        "timestamp": datetime.now().isoformat(),
    })
    response = _generate_response(query, context, llm_client, rag_client)
    st.session_state["chat_history"].append({
        "role": "assistant",
        "content": response,
        "timestamp": datetime.now().isoformat(),
    })
    st.rerun()


def _generate_response(query: str, context: dict, llm_client, rag_client) -> str:
    rag_context = ""
    if rag_client and hasattr(rag_client, "is_ready") and rag_client.is_ready:
        try:
            docs = rag_client.retrieve(query, top_k=3)
            if docs:
                rag_context = "\n\n".join(docs)
        except Exception:
            pass

    if llm_client and llm_client.is_available():
        try:
            response = llm_client.answer_with_context(
                query=query,
                pipeline_context=context,
                rag_context=rag_context,
            )
            if response and not response.startswith("LLM request failed"):
                return response
        except Exception:
            pass

    return _rule_based_response(query, context, rag_context)


def _rule_based_response(query: str, context: dict, rag_context: str) -> str:
    q = query.lower()
    profile = context.get("profile", {})
    model_data = context.get("model_results", {})
    eng_data = context.get("engineering_data", {})
    exploration = context.get("exploration_data", {})

    if rag_context and any(kw in q for kw in ["how", "guide", "best practice", "deploy", "help"]):
        return f"Here's what I found in the knowledge base:\n\n{rag_context[:2000]}"

    if any(kw in q for kw in ["result", "achiev", "summary", "analyz"]):
        return _summarize_results(profile, model_data, eng_data, exploration)

    if any(kw in q for kw in ["improv", "better", "scope", "enhance", "optim"]):
        return _suggest_improvements(profile, model_data, eng_data)

    if any(kw in q for kw in ["feature", "engineer", "transform"]):
        return _explain_features(eng_data, profile)

    if any(kw in q for kw in ["model", "algorithm", "best", "champion"]):
        return _handle_model_query(model_data, profile)

    if any(kw in q for kw in ["drop", "remove", "delete"]):
        return _handle_drop_query(profile)

    if any(kw in q for kw in ["target", "predict", "label"]):
        return _handle_target_query(profile)

    if any(kw in q for kw in ["quality", "data", "clean", "missing"]):
        return _handle_quality_query(profile)

    if any(kw in q for kw in ["test", "hypothesis", "normal", "statistic"]):
        return _handle_stats_query(exploration)

    if any(kw in q for kw in ["vif", "multicollinear", "correlat"]):
        return _handle_collinearity_query(exploration)

    return (
        "I can help with questions about:\n"
        "- **Results analysis** -- what was achieved in the pipeline\n"
        "- **Improvement suggestions** -- how to get better performance\n"
        "- **Feature engineering** -- what transformations were applied\n"
        "- **Model comparison** -- which models performed best and why\n"
        "- **Statistical tests** -- normality, hypothesis testing, VIF\n"
        "- **Data quality** -- missing values, cleaning recommendations\n"
        "- **Deployment** -- production readiness and next steps\n\n"
        "Try asking something specific about your dataset or model!"
    )


def _summarize_results(profile, model_data, eng_data, exploration):
    parts = ["## Pipeline Results Summary\n"]

    if profile:
        stats = profile.get("stats", {})
        parts.append(f"**Dataset:** {stats.get('row_count', '?')} rows x {stats.get('column_count', '?')} columns")
        parts.append(f"**Problem:** {profile.get('problem_type', 'unknown').replace('_', ' ').title()}")
        parts.append(f"**Quality:** {profile.get('quality_score', 0):.0f}/100\n")

    if eng_data:
        report = eng_data.get("engineering_report", {})
        transforms = report.get("transformations_summary", [])
        if transforms:
            parts.append(f"**Feature Engineering:** {len(transforms)} transformation steps applied")
        skew = report.get("skewness_correction", {}).get("corrections", {})
        if skew:
            parts.append(f"**Skewness Corrections:** {len(skew)} features corrected for distribution normality")

    if exploration:
        normality = exploration.get("normality_tests", [])
        hypothesis = exploration.get("hypothesis_tests", [])
        if normality:
            parts.append(f"**Normality Tests:** {len(normality)} features tested")
        if hypothesis:
            significant = [t for t in hypothesis if t.get("p_value", 1) < 0.05]
            parts.append(f"**Hypothesis Tests:** {len(significant)}/{len(hypothesis)} statistically significant")

    if model_data:
        parts.append(f"\n**Champion Model:** {model_data.get('champion_name', 'N/A')}")
        parts.append(f"**Best Score:** {model_data.get('champion_score', 'N/A')}")
        lb = model_data.get("leaderboard", [])
        if lb:
            parts.append("\n**Model Leaderboard:**")
            for entry in lb[:5]:
                score = entry.get("score", 0)
                parts.append(f"- {entry.get('name', '?')}: {score:.4f}")

    if not model_data and not profile:
        parts.append("No pipeline results available yet. Run the pipeline first.")

    return "\n".join(parts)


def _suggest_improvements(profile, model_data, eng_data):
    parts = ["## Improvement Suggestions\n"]
    problem_type = profile.get("problem_type", "unknown") if profile else "unknown"

    if model_data:
        score = model_data.get("champion_score", 0)
        champion = model_data.get("champion_name", "unknown")
        lb = model_data.get("leaderboard", [])

        if isinstance(score, (int, float)):
            if problem_type in ("binary_classification", "multiclass_classification"):
                if score < 0.7:
                    parts.append("### Performance is below typical benchmarks\n")
                    parts.append("- Try **class balancing** (SMOTE, class weights) if target is imbalanced")
                    parts.append("- **Engineer domain-specific features** that capture business logic")
                    parts.append("- Consider **removing noisy features** that add variance without signal")
                    parts.append("- Try **ensemble methods** like stacking the top 3 models")
                elif score < 0.85:
                    parts.append("### Moderate performance with room for improvement\n")
                    parts.append("- **Increase Optuna trials** (50-100) for better hyperparameter search")
                    parts.append("- Try **feature interactions** between top predictors")
                    parts.append("- Experiment with **target encoding** for high-cardinality categoricals")
                    parts.append("- Consider **model stacking** to combine diverse model strengths")
                elif score < 0.95:
                    parts.append("### Good performance -- fine-tuning opportunities\n")
                    parts.append("- Focus on **misclassified samples** to understand failure modes")
                    parts.append("- Try **blending predictions** from top 2-3 models")
                    parts.append("- Look for **data leakage** if score seems too high")
                else:
                    parts.append("### Excellent performance\n")
                    parts.append("- Verify no **data leakage** (score may be artificially high)")
                    parts.append("- Focus on **model interpretability** (SHAP values, feature importance)")
                    parts.append("- Consider **model compression** for faster inference in production")
            elif problem_type == "regression":
                parts.append("### Regression improvements\n")
                parts.append("- Try **polynomial features** or interaction terms for non-linear patterns")
                parts.append("- Apply **target transformations** (log, Box-Cox) if residuals are skewed")
                parts.append("- Consider **quantile regression** if predicting ranges is useful")
                parts.append("- Try **regularization tuning** (L1/L2 mix via ElasticNet)")

        if lb and len(lb) > 1:
            top_score = lb[0].get("score", 0) if lb else 0
            bottom_score = lb[-1].get("score", 0) if lb else 0
            gap = top_score - bottom_score
            if gap < 0.02:
                parts.append("\n**Note:** Models are very close in performance. Ensembling them could help.")
            else:
                parts.append(f"\n**Note:** Score range is {gap:.4f}. The champion significantly outperforms others.")

    quality = profile.get("quality_score", 100) if profile else 100
    if quality < 80:
        parts.append("\n### Data Quality\n")
        parts.append(f"- Quality score is {quality:.0f}/100 -- **collecting more/cleaner data** would help most")

    parts.append("\n### General Recommendations\n")
    parts.append("- **Cross-validation:** Use stratified k-fold (5-10 folds) for robust evaluation")
    parts.append("- **Feature selection:** Try recursive feature elimination or SHAP-based selection")
    parts.append("- **External data:** Consider enriching with domain-relevant external features")

    return "\n".join(parts)


def _explain_features(eng_data, profile):
    if not eng_data:
        return "No feature engineering results available yet. Run the pipeline first."

    parts = ["## Feature Engineering Details\n"]
    report = eng_data.get("engineering_report", {})

    transforms = report.get("transformations_summary", [])
    if transforms:
        parts.append("### Transformation Pipeline\n")
        for t in transforms:
            parts.append(f"- **{t.get('step', '?')}:** {t.get('description', '')}")

    skew = report.get("skewness_correction", {}).get("corrections", {})
    if skew:
        parts.append("\n### Skewness Corrections\n")
        for col, info in list(skew.items())[:10]:
            parts.append(f"- **{col}:** {info.get('method', '?')} (skew {info.get('original_skewness', '?'):.2f} -> {info.get('new_skewness', '?'):.2f})")

    impute = report.get("imputation", {}).get("details", {})
    if impute:
        parts.append("\n### Imputation\n")
        for col, info in list(impute.items())[:10]:
            parts.append(f"- **{col}:** {info.get('method', '?')} (fill value: {info.get('fill_value', '?')})")

    encoding = report.get("encoding", {}).get("encoded", {})
    if encoding:
        parts.append("\n### Encoding\n")
        for col, info in list(encoding.items())[:10]:
            parts.append(f"- **{col}:** {info.get('method', '?')}")

    return "\n".join(parts)


def _handle_drop_query(profile):
    if not profile:
        return "No data profile available. Upload a dataset first."
    stats = profile.get("stats", {})
    missing = stats.get("missing_pct", {})
    high_missing = {k: v for k, v in missing.items() if v > 30}
    id_cols = profile.get("id_columns", [])
    leakage = profile.get("leakage_columns", [])

    parts = ["**Column drop analysis:**\n"]
    if id_cols:
        parts.append(f"- ID columns (safe to drop): {', '.join(id_cols)}")
    if high_missing:
        parts.append(f"- High missing values (>30%): {', '.join(f'{k} ({v:.0f}%)' for k, v in high_missing.items())}")
    if leakage:
        parts.append(f"- Potential leakage (should drop): {', '.join(leakage)}")
    if len(parts) == 1:
        parts.append("No columns are obviously safe to drop. All seem informative.")
    return "\n".join(parts)


def _handle_model_query(model_data, profile):
    if model_data and model_data.get("champion_name"):
        lb = model_data.get("leaderboard", [])
        parts = [f"**Champion model:** {model_data['champion_name']} (score: {model_data.get('champion_score', 'N/A')})\n"]
        if lb:
            parts.append("**Leaderboard:**")
            for entry in lb[:5]:
                score = entry.get("score", 0)
                train_time = entry.get("train_time", 0)
                parts.append(f"- {entry.get('name', '?')}: {score:.4f} (trained in {train_time:.1f}s)")
        return "\n".join(parts)

    problem = profile.get("problem_type", "unknown") if profile else "unknown"
    return f"No model trained yet. For **{problem}** problems, I'd recommend starting with GradientBoosting or XGBoost."


def _handle_target_query(profile):
    if not profile:
        return "No data profile available. Upload a dataset first."
    target = profile.get("target_column", "Not detected")
    problem = profile.get("problem_type", "unknown")
    return f"**Target column:** {target}\n**Problem type:** {problem.replace('_', ' ').title()}"


def _handle_quality_query(profile):
    if not profile:
        return "No data profile available. Upload a dataset first."
    quality = profile.get("quality_score", 0)
    stats = profile.get("stats", {})
    missing = stats.get("missing_pct", {})
    total_missing = sum(missing.values()) / max(len(missing), 1)

    parts = [f"**Data quality score:** {quality:.0f}/100\n"]
    parts.append(f"**Average missing:** {total_missing:.1f}%")
    high = {k: v for k, v in missing.items() if v > 10}
    if high:
        parts.append(f"**Columns with >10% missing:** {', '.join(f'{k} ({v:.0f}%)' for k, v in high.items())}")
    return "\n".join(parts)


def _handle_stats_query(exploration):
    if not exploration:
        return "No statistical test results available. Run the pipeline first."
    parts = ["## Statistical Test Results\n"]

    normality = exploration.get("normality_tests", [])
    if normality:
        parts.append("### Normality Tests\n")
        for t in normality[:10]:
            parts.append(f"- **{t.get('feature', '?')}:** {t.get('test', '?')} -- {t.get('conclusion', '?')} (p={t.get('p_value', '?'):.4f})")

    hypothesis = exploration.get("hypothesis_tests", [])
    if hypothesis:
        parts.append("\n### Hypothesis Tests\n")
        for t in hypothesis[:10]:
            sig = "Significant" if t.get("p_value", 1) < 0.05 else "Not significant"
            parts.append(f"- **{t.get('feature', '?')}:** {t.get('test', '?')} -- {sig} (p={t.get('p_value', '?'):.4f})")

    return "\n".join(parts)


def _handle_collinearity_query(exploration):
    if not exploration:
        return "No multicollinearity analysis available. Run the pipeline first."

    parts = ["## Multicollinearity Analysis\n"]
    vif = exploration.get("vif_results", [])
    if vif:
        parts.append("### VIF (Variance Inflation Factor)\n")
        parts.append("VIF > 5 indicates moderate multicollinearity, VIF > 10 indicates severe.\n")
        for v in sorted(vif, key=lambda x: x.get("VIF", 0), reverse=True)[:15]:
            vif_val = v.get("VIF", 0)
            flag = " (HIGH)" if vif_val > 5 else ""
            parts.append(f"- **{v.get('feature', '?')}:** VIF = {vif_val:.2f}{flag}")
    else:
        parts.append("No VIF data available.")

    corr = exploration.get("correlation_matrix")
    if corr:
        parts.append("\nCorrelation matrix is available in the Exploration & Testing tab.")

    return "\n".join(parts)
