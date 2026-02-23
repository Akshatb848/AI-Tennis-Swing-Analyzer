"""LLM client using OpenRouter via Replit AI Integrations.

Uses open-source models (Meta Llama 3.3 70B, Mistral Small 3.1) through OpenRouter.
Falls back to rule-based analysis when the LLM service is unavailable.
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Any, Dict, List, Optional

from config.settings import LOG_FORMAT

logger = logging.getLogger("llm_client")
if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter(LOG_FORMAT))
    logger.addHandler(_handler)
    logger.setLevel(logging.INFO)

PRIMARY_MODEL = "meta-llama/llama-3.3-70b-instruct"
FALLBACK_MODEL = "mistralai/mistral-small-3.1-24b-instruct"


class LLMClient:
    """LLM client powered by OpenRouter via Replit AI Integrations.

    Uses open-source models: Meta Llama 3.3 70B (primary) and Mistral Small 3.1 (fallback).
    No API key management needed - handled automatically by Replit.
    """

    def __init__(self, model: str = PRIMARY_MODEL) -> None:
        self.model = model
        self._client = None
        self._available: Optional[bool] = None
        self._init_client()

    def _init_client(self) -> None:
        base_url = os.environ.get("AI_INTEGRATIONS_OPENROUTER_BASE_URL")
        api_key = os.environ.get("AI_INTEGRATIONS_OPENROUTER_API_KEY")

        if not base_url or not api_key:
            logger.warning("OpenRouter env vars not set - LLM will be unavailable")
            self._available = False
            return

        try:
            from openai import OpenAI
            self._client = OpenAI(api_key=api_key, base_url=base_url)
            logger.info("LLM client initialised (model=%s)", self.model)
        except Exception as exc:
            logger.error("Failed to init OpenAI client: %s", exc)
            self._available = False

    def is_available(self) -> bool:
        return self._client is not None

    def is_connected(self) -> bool:
        return self.is_available()

    def chat(
        self,
        system_prompt: str,
        user_message: str,
        temperature: float = 0.7,
        max_tokens: int = 8192,
    ) -> str:
        if not self.is_available():
            return ""

        for model in [self.model, FALLBACK_MODEL]:
            try:
                response = self._client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_message},
                    ],
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                content = response.choices[0].message.content or ""
                content = _strip_thinking_tags(content)
                logger.info("LLM chat completed (model=%s, chars=%d)", model, len(content))
                return content
            except Exception as exc:
                logger.warning("LLM call failed with %s: %s", model, exc)
                if model == FALLBACK_MODEL:
                    return f"LLM request failed: {exc}"
                continue

        return "LLM request failed after all retries."

    def analyze_dataset(self, profile_dict: Dict[str, Any]) -> Dict[str, Any]:
        if self.is_available():
            result = self._llm_analyze_dataset(profile_dict)
            if result.get("analysis_source") != "llm_parse_fallback":
                return result
        logger.info("Using rule-based dataset analysis")
        return self._rule_based_analysis(profile_dict)

    def _llm_analyze_dataset(self, profile_dict: Dict[str, Any]) -> Dict[str, Any]:
        profile_summary = {
            "row_count": profile_dict.get("stats", {}).get("row_count", 0),
            "column_count": profile_dict.get("stats", {}).get("column_count", 0),
            "schema": profile_dict.get("schema", {}),
            "missing_pct": profile_dict.get("stats", {}).get("missing_pct", {}),
            "quality_score": profile_dict.get("quality_score", 0),
            "target_column": profile_dict.get("target_column"),
            "problem_type": profile_dict.get("problem_type"),
        }

        system_prompt = (
            "You are an expert data scientist. Analyze the dataset profile and return "
            "a JSON object with exactly these keys: problem_type, target_column, "
            "suggested_kpis (list of strings), recommendations (list of strings). "
            "Return ONLY valid JSON, no markdown or explanation."
        )
        user_message = f"Dataset profile:\n{json.dumps(profile_summary, indent=2, default=str)}"

        response = self.chat(system_prompt, user_message, temperature=0.3)

        try:
            start = response.find("{")
            end = response.rfind("}") + 1
            if start >= 0 and end > start:
                result = json.loads(response[start:end])
                result["analysis_source"] = "llm"
                return result
        except (json.JSONDecodeError, ValueError) as exc:
            logger.warning("Failed to parse LLM analysis: %s", exc)

        fallback = self._rule_based_analysis(profile_dict)
        fallback["analysis_source"] = "llm_parse_fallback"
        return fallback

    def analyze_results(self, pipeline_context: Dict[str, Any]) -> str:
        if not self.is_available():
            return self._rule_based_results_analysis(pipeline_context)

        system_prompt = (
            "You are an expert data scientist reviewing ML pipeline results. "
            "Analyze the results thoroughly and provide:\n"
            "1. A summary of what was achieved\n"
            "2. Key strengths of the current approach\n"
            "3. Specific areas for improvement with actionable suggestions\n"
            "4. Next steps the user should consider\n\n"
            "Be specific, reference actual metrics and feature names. "
            "Format with markdown headers and bullet points."
        )

        context_str = _build_results_context(pipeline_context)
        response = self.chat(system_prompt, context_str, temperature=0.4)

        if response and not response.startswith("LLM request failed"):
            return response
        return self._rule_based_results_analysis(pipeline_context)

    def suggest_improvements(self, pipeline_context: Dict[str, Any]) -> str:
        if not self.is_available():
            return self._rule_based_improvements(pipeline_context)

        system_prompt = (
            "You are a senior ML engineer reviewing a completed pipeline. "
            "Based on the results, provide detailed improvement suggestions:\n\n"
            "1. Feature Engineering improvements (new features, transformations)\n"
            "2. Model selection alternatives to try\n"
            "3. Hyperparameter tuning strategies\n"
            "4. Data quality improvements\n"
            "5. Production deployment considerations\n\n"
            "Be very specific and actionable. Reference actual metrics, features, "
            "and model names from the results. Format with markdown."
        )

        context_str = _build_results_context(pipeline_context)
        response = self.chat(system_prompt, context_str, temperature=0.5)

        if response and not response.startswith("LLM request failed"):
            return response
        return self._rule_based_improvements(pipeline_context)

    def what_if_analysis(self, query: str, current_state: Dict[str, Any]) -> str:
        if self.is_available():
            return self._llm_what_if(query, current_state)
        return self._rule_based_what_if(query, current_state)

    def answer_with_context(
        self,
        query: str,
        pipeline_context: Dict[str, Any],
        rag_context: str = "",
    ) -> str:
        if not self.is_available():
            return self._rule_based_context_answer(query, pipeline_context, rag_context)

        system_prompt = (
            "You are an expert AI data scientist assistant embedded in an automated ML platform. "
            "You have access to the user's complete pipeline results including data profiles, "
            "feature engineering details, statistical tests, model performance, and deployment status.\n\n"
            "Guidelines:\n"
            "- Answer questions using the actual pipeline results provided\n"
            "- When asked about improvements, analyze the specific metrics and suggest concrete actions\n"
            "- Reference specific feature names, model scores, and test results\n"
            "- If the user asks about scope of improvement, compare current metrics against typical benchmarks\n"
            "- Keep answers focused, structured, and actionable\n"
            "- Use markdown formatting for readability"
        )

        context_parts = [_build_results_context(pipeline_context)]
        if rag_context:
            context_parts.append(f"\n\nRelevant knowledge base context:\n{rag_context}")

        user_msg = "\n".join(context_parts) + f"\n\nUser question: {query}"
        response = self.chat(system_prompt, user_msg, temperature=0.5)
        if response and not response.startswith("LLM request failed"):
            return response
        return self._rule_based_context_answer(query, pipeline_context, rag_context)

    def _rule_based_context_answer(
        self, query: str, ctx: Dict[str, Any], rag_context: str = ""
    ) -> str:
        q = query.lower()
        if any(kw in q for kw in ["improv", "better", "scope", "enhance"]):
            return self._rule_based_improvements(ctx)
        if any(kw in q for kw in ["result", "achiev", "summary", "analyz"]):
            return self._rule_based_results_analysis(ctx)
        if rag_context:
            return f"Here's relevant information from the knowledge base:\n\n{rag_context[:2000]}"
        return self._rule_based_results_analysis(ctx)

    def _llm_what_if(self, query: str, current_state: Dict[str, Any]) -> str:
        system_prompt = (
            "You are an expert data scientist assistant. The user is working on "
            "an ML project and asking questions. Use the project state provided "
            "to give concise, actionable answers with specific references to their data."
        )
        state_summary = json.dumps(current_state, indent=2, default=str)
        user_message = f"Current project state:\n{state_summary}\n\nQuestion: {query}"
        return self.chat(system_prompt, user_message, temperature=0.5)

    def _rule_based_analysis(self, profile_dict: Dict[str, Any]) -> Dict[str, Any]:
        problem_type = profile_dict.get("problem_type", "eda_only")
        target_column = profile_dict.get("target_column")
        schema = profile_dict.get("schema", {})
        stats = profile_dict.get("stats", {})
        missing_pct = stats.get("missing_pct", {})
        quality_score = profile_dict.get("quality_score", 0)

        kpi_map = {
            "binary_classification": ["accuracy", "f1_score", "roc_auc", "precision", "recall"],
            "multiclass_classification": ["accuracy", "f1_macro", "f1_weighted"],
            "regression": ["rmse", "mae", "r2_score"],
            "time_series": ["rmse", "mae", "mape"],
            "clustering": ["silhouette_score", "calinski_harabasz"],
            "eda_only": ["descriptive_statistics"],
        }
        suggested_kpis = kpi_map.get(problem_type, ["accuracy"])

        recommendations: List[str] = []
        high_missing = [col for col, pct in missing_pct.items() if pct > 30]
        if high_missing:
            recommendations.append(f"Columns with >30% missing: {', '.join(high_missing[:5])}. Consider imputation or removal.")
        if quality_score < 50:
            recommendations.append(f"Data quality score is low ({quality_score:.1f}/100). Significant cleaning may be required.")

        cat_cols = [c for c, t in schema.items() if t == "categorical"]
        num_cols = [c for c, t in schema.items() if t == "numeric"]
        if cat_cols:
            recommendations.append(f"Found {len(cat_cols)} categorical column(s) - encoding will be needed.")
        if num_cols and len(num_cols) > 1:
            recommendations.append(f"Found {len(num_cols)} numeric column(s) - consider feature scaling.")

        id_cols = profile_dict.get("id_columns", [])
        if id_cols:
            recommendations.append(f"ID columns detected ({', '.join(id_cols)}). These should be excluded from modelling.")
        leakage_cols = profile_dict.get("leakage_columns", [])
        if leakage_cols:
            recommendations.append(f"Potential leakage columns: {', '.join(leakage_cols)}. Review before training.")
        if not recommendations:
            recommendations.append("Dataset looks clean. Ready for modelling.")

        return {
            "problem_type": problem_type,
            "target_column": target_column,
            "suggested_kpis": suggested_kpis,
            "recommendations": recommendations,
            "analysis_source": "rule_based",
        }

    def _rule_based_what_if(self, query: str, current_state: Dict[str, Any]) -> str:
        query_lower = query.lower()
        problem_type = current_state.get("problem_type", "unknown")
        target = current_state.get("target_column", "unknown")

        lines = [f"**Project:** problem_type={problem_type}, target={target}\n"]

        if "feature" in query_lower or "column" in query_lower:
            lines.append("**Suggestion:** Review feature importance after model training. Removing low-importance features can reduce overfitting.")
        elif "model" in query_lower or "algorithm" in query_lower:
            model_suggestions = {
                "binary_classification": "LogisticRegression, RandomForest, XGBoost, LightGBM",
                "multiclass_classification": "RandomForest, XGBoost, LightGBM, SVM",
                "regression": "LinearRegression, RandomForest, XGBoost, LightGBM",
            }
            suggestion = model_suggestions.get(problem_type, "Start with exploratory data analysis.")
            lines.append(f"**Recommended models for {problem_type}:** {suggestion}")
        elif "metric" in query_lower or "kpi" in query_lower:
            metric_map = {
                "binary_classification": "F1 score, ROC-AUC, precision, recall",
                "multiclass_classification": "Macro-F1, weighted-F1, accuracy",
                "regression": "RMSE, MAE, R2 score",
            }
            metrics = metric_map.get(problem_type, "Choose metrics based on your problem type.")
            lines.append(f"**Recommended metrics for {problem_type}:** {metrics}")
        else:
            summary = current_state.get("summary", "")
            if summary:
                lines.append(summary)
            else:
                lines.append("Ask about features, models, metrics, or improvements for specific guidance.")

        return "\n".join(lines)

    def _rule_based_results_analysis(self, ctx: Dict[str, Any]) -> str:
        parts = ["## Pipeline Results Summary\n"]
        profile = ctx.get("profile", {})
        model_data = ctx.get("model_results", {})
        eng_data = ctx.get("engineering_data", {})

        if profile:
            stats = profile.get("stats", {})
            parts.append(f"**Dataset:** {stats.get('row_count', '?')} rows x {stats.get('column_count', '?')} columns")
            parts.append(f"**Problem:** {profile.get('problem_type', 'unknown').replace('_', ' ').title()}")
            parts.append(f"**Quality:** {profile.get('quality_score', 0):.0f}/100\n")

        if model_data:
            parts.append(f"**Champion:** {model_data.get('champion_name', 'N/A')}")
            parts.append(f"**Best Score:** {model_data.get('champion_score', 'N/A')}")
            lb = model_data.get("leaderboard", [])
            if lb:
                parts.append("\n**Model Comparison:**")
                for entry in lb[:5]:
                    parts.append(f"- {entry.get('name', '?')}: {entry.get('score', 0):.4f}")

        return "\n".join(parts)

    def _rule_based_improvements(self, ctx: Dict[str, Any]) -> str:
        parts = ["## Improvement Suggestions\n"]
        model_data = ctx.get("model_results", {})
        profile = ctx.get("profile", {})
        problem_type = profile.get("problem_type", "unknown")

        score = model_data.get("champion_score", 0)
        if isinstance(score, (int, float)):
            if problem_type in ("binary_classification", "multiclass_classification"):
                if score < 0.7:
                    parts.append("- **Low accuracy** - Consider feature engineering, class balancing (SMOTE), or trying ensemble methods")
                elif score < 0.85:
                    parts.append("- **Moderate accuracy** - Try hyperparameter tuning with Optuna, or engineer domain-specific features")
                elif score < 0.95:
                    parts.append("- **Good accuracy** - Fine-tune with extended Optuna trials, or try stacking/blending models")
                else:
                    parts.append("- **Excellent accuracy** - Focus on model interpretability and deployment robustness")
            elif problem_type == "regression":
                parts.append("- Consider polynomial features or interaction terms for non-linear relationships")
                parts.append("- Try target variable transformations (log, Box-Cox) if residuals are skewed")

        parts.append("- **Feature Engineering:** Create interaction features between top predictors")
        parts.append("- **Cross-Validation:** Use stratified k-fold to ensure robust evaluation")
        parts.append("- **Data Augmentation:** If dataset is small, consider bootstrapping or synthetic data")

        return "\n".join(parts)


def _strip_thinking_tags(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    return text.strip()


def _build_results_context(ctx: Dict[str, Any]) -> str:
    parts = []

    profile = ctx.get("profile", {})
    if profile:
        stats = profile.get("stats", {})
        parts.append("## Dataset Profile")
        parts.append(f"- Rows: {stats.get('row_count', '?')}, Columns: {stats.get('column_count', '?')}")
        parts.append(f"- Problem type: {profile.get('problem_type', 'unknown')}")
        parts.append(f"- Target: {profile.get('target_column', 'unknown')}")
        parts.append(f"- Quality score: {profile.get('quality_score', 0):.0f}/100")
        schema = profile.get("schema", {})
        if schema:
            numeric = [k for k, v in schema.items() if v == "numeric"]
            categorical = [k for k, v in schema.items() if v == "categorical"]
            parts.append(f"- Numeric features: {', '.join(numeric[:10])}")
            parts.append(f"- Categorical features: {', '.join(categorical[:10])}")
        missing = stats.get("missing_pct", {})
        high_missing = {k: v for k, v in missing.items() if v > 5}
        if high_missing:
            parts.append(f"- Features with >5% missing: {json.dumps(high_missing, default=str)}")

    eng = ctx.get("engineering_data", {})
    if eng:
        report = eng.get("engineering_report", {})
        if report:
            parts.append("\n## Feature Engineering Results")
            transforms = report.get("transformations_summary", [])
            if transforms:
                parts.append(f"- {len(transforms)} transformations applied")
                for t in transforms[:8]:
                    parts.append(f"  - {t.get('step', '?')}: {t.get('description', '')}")
            skew = report.get("skewness_correction", {}).get("corrections", {})
            if skew:
                parts.append(f"- Skewness corrections: {len(skew)} features corrected")
            impute = report.get("imputation", {}).get("details", {})
            if impute:
                parts.append(f"- Imputation: {len(impute)} features imputed")

    exploration = ctx.get("exploration_data", {})
    if exploration:
        parts.append("\n## Exploration & Statistical Tests")
        normality = exploration.get("normality_tests", [])
        if normality:
            non_normal = [t for t in normality if "not normal" in t.get("conclusion", "").lower()]
            parts.append(f"- Normality tests: {len(normality)} features tested, {len(non_normal)} non-normal")
        hypothesis = exploration.get("hypothesis_tests", [])
        if hypothesis:
            significant = [t for t in hypothesis if t.get("p_value", 1) < 0.05]
            parts.append(f"- Hypothesis tests: {len(hypothesis)} run, {len(significant)} significant (p<0.05)")
        vif = exploration.get("vif_results", [])
        if vif:
            high_vif = [v for v in vif if v.get("VIF", 0) > 5]
            parts.append(f"- VIF analysis: {len(vif)} features, {len(high_vif)} with high multicollinearity (VIF>5)")

    model_data = ctx.get("model_results", {})
    if model_data:
        parts.append("\n## Model Results")
        parts.append(f"- Champion: {model_data.get('champion_name', 'N/A')}")
        parts.append(f"- Best score: {model_data.get('champion_score', 'N/A')}")
        lb = model_data.get("leaderboard", [])
        if lb:
            parts.append("- Leaderboard:")
            for entry in lb[:6]:
                name = entry.get("name", "?")
                score = entry.get("score", 0)
                time_s = entry.get("train_time", 0)
                parts.append(f"  - {name}: score={score:.4f}, train_time={time_s:.1f}s")
        params = model_data.get("champion_params", {})
        if params:
            parts.append(f"- Champion hyperparameters: {json.dumps(params, default=str)}")

    mlops = ctx.get("mlops_data", {})
    if mlops:
        parts.append("\n## Deployment Status")
        parts.append(f"- Deployment ready: {mlops.get('deployment_ready', False)}")
        parts.append(f"- Model path: {mlops.get('model_path', 'N/A')}")

    return "\n".join(parts) if parts else "No pipeline data available."


OllamaLLMClient = LLMClient
