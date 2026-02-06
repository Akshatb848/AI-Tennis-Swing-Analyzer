"""
Prompt Templates for LLM-powered Coordinator Agent
"""

from typing import Any, Dict, List, Optional


class PromptTemplates:
    """All prompt templates used by the coordinator agent."""

    @staticmethod
    def coordinator_system(agent_descriptions: str) -> str:
        return f"""You are an expert AI Data Scientist assistant coordinating a multi-agent data science platform.

You have access to these specialized agents:
{agent_descriptions}

Your responsibilities:
1. Understand the user's data science goals from natural language
2. Decide which agent(s) to invoke and in what order
3. Interpret results and explain findings in clear, actionable language
4. Suggest next steps based on the analysis results
5. Answer data science questions using your expertise

When the user asks you to do something, respond naturally and explain what you're doing.
When presenting results, focus on INSIGHTS, not just numbers — explain what the data means.

Always be concise but thorough. Use bullet points for key findings."""

    @staticmethod
    def intent_analysis(user_message: str, has_dataset: bool, has_target: bool, dataset_summary: str) -> str:
        return f"""Analyze the user's message and determine what action to take.

User message: "{user_message}"

Current state:
- Dataset loaded: {has_dataset}
- Target column set: {has_target}
- Dataset summary: {dataset_summary if has_dataset else 'No dataset loaded yet'}

Respond with a JSON object containing:
{{
    "intent": "<one of: run_agent, run_pipeline, upload_data, set_target, help, status, general>",
    "agent": "<agent name if intent is run_agent, e.g. EDAAgent, DataCleanerAgent, ModelTrainerAgent, FeatureEngineerAgent, AutoMLAgent, DataVisualizerAgent, DashboardBuilderAgent>",
    "action": "<specific action for the agent>",
    "explanation": "<brief explanation of what you'll do and why>"
}}

Intent guidelines:
- "run_agent": User wants a specific analysis (EDA, cleaning, training, etc.)
- "run_pipeline": User wants the full end-to-end analysis pipeline
- "upload_data": User wants to load or change data
- "set_target": User wants to set or change the target variable
- "help": User is asking what the platform can do
- "status": User wants to know current state
- "general": General data science question or conversation

Respond ONLY with the JSON object, no other text."""

    @staticmethod
    def interpret_results(
        agent_name: str,
        action: str,
        result_data: Dict[str, Any],
        user_context: str,
    ) -> str:
        # Truncate large result data to avoid token limits
        result_str = str(result_data)
        if len(result_str) > 3000:
            result_str = result_str[:3000] + "... (truncated)"

        return f"""You are an expert data scientist interpreting analysis results for a user.

The {agent_name} just completed the "{action}" task.

Results:
{result_str}

User's context/question: {user_context}

Provide a clear, insightful interpretation:
1. Summarize the KEY findings (not every metric — focus on what matters)
2. Highlight anything surprising or concerning
3. Give actionable recommendations for next steps
4. If there are model results, explain which model is best and why
5. If there are data quality issues, explain their impact

Be concise. Use bullet points. Speak like an expert data scientist advising a colleague."""

    @staticmethod
    def plan_workflow(
        dataset_summary: str,
        target_column: Optional[str],
        user_goal: str,
    ) -> str:
        return f"""Plan a data science workflow for the user's goal.

Dataset summary:
{dataset_summary}

Target column: {target_column or 'Not specified (unsupervised)'}
User's goal: {user_goal}

Available agents and their actions:
- DataCleanerAgent: clean_data (remove duplicates, handle missing values, clip outliers)
- EDAAgent: full_eda (statistical profiling, correlations, insights)
- FeatureEngineerAgent: engineer_features (datetime extraction, interactions, encoding, scaling)
- ModelTrainerAgent: train_models (train 6+ models with cross-validation)
- AutoMLAgent: auto_select_models (intelligent model selection and training)
- DataVisualizerAgent: generate_visualizations (distributions, correlations, scatter plots)
- DashboardBuilderAgent: build_dashboard (compile results into dashboard)

Respond with a JSON object:
{{
    "steps": [
        {{"agent": "<agent name>", "action": "<action>", "reason": "<why this step>"}},
        ...
    ],
    "explanation": "<brief overview of the plan>"
}}

Only include steps that are relevant to the user's goal and dataset.
If no target column is set, skip ModelTrainerAgent and AutoMLAgent.
Respond ONLY with the JSON object."""

    @staticmethod
    def dataset_summary(info: Dict[str, Any]) -> str:
        """Build a concise text summary of a dataset for use in prompts."""
        rows = info.get("shape", {}).get("rows", "?")
        cols = info.get("shape", {}).get("columns", "?")
        dtypes = info.get("dtypes", {})
        missing = info.get("missing_values", {})
        total_missing = sum(v for v in missing.values() if isinstance(v, (int, float)))

        numeric_cols = [c for c, t in dtypes.items() if "int" in t or "float" in t]
        categorical_cols = [c for c, t in dtypes.items() if t == "object" or t == "category"]

        return (
            f"{rows} rows x {cols} columns | "
            f"{len(numeric_cols)} numeric, {len(categorical_cols)} categorical | "
            f"{total_missing} total missing values | "
            f"Columns: {', '.join(list(dtypes.keys())[:15])}"
            + (" ..." if len(dtypes) > 15 else "")
        )
