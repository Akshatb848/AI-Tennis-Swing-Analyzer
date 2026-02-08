"""
Coordinator Agent - LLM-Powered Master Orchestrator

Manages the data science workflow by:
- Understanding user intent via LLM (or rule-based fallback)
- Planning dynamic workflows based on data characteristics
- Dispatching tasks to specialized agents
- Interpreting results and generating insights
- Maintaining conversational memory across turns
"""

import asyncio
import json
import logging
import re
from datetime import datetime
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field

from .base_agent import BaseAgent, AgentState, TaskResult, generate_uuid

logger = logging.getLogger(__name__)


@dataclass
class WorkflowStep:
    """Represents a single step in the data science workflow."""
    id: str
    name: str
    agent: str
    task: Dict[str, Any]
    dependencies: List[str] = field(default_factory=list)
    status: str = "pending"
    result: Optional[TaskResult] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "agent": self.agent,
            "task": self.task,
            "dependencies": self.dependencies,
            "status": self.status,
            "result": self.result.to_dict() if self.result else None,
        }


@dataclass
class Workflow:
    """Complete workflow definition."""
    id: str
    name: str
    project_id: str
    steps: List[WorkflowStep] = field(default_factory=list)
    status: str = "created"
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "project_id": self.project_id,
            "steps": [step.to_dict() for step in self.steps],
            "status": self.status,
            "created_at": self.created_at.isoformat()
            if isinstance(self.created_at, datetime)
            else self.created_at,
            "updated_at": self.updated_at.isoformat()
            if isinstance(self.updated_at, datetime)
            else self.updated_at,
            "metadata": self.metadata,
        }


class CoordinatorAgent(BaseAgent):
    """Master orchestrator that coordinates all specialized agents via LLM reasoning."""

    def __init__(self, llm_client=None, agent_registry: Optional[Dict[str, BaseAgent]] = None):
        super().__init__(
            name="Coordinator",
            description="Master orchestrator for data science workflow coordination",
            capabilities=[
                "workflow_planning",
                "task_distribution",
                "agent_coordination",
                "user_intent_understanding",
                "progress_tracking",
                "error_recovery",
                "result_interpretation",
            ],
        )
        self.llm_client = llm_client
        self.agent_registry: Dict[str, BaseAgent] = agent_registry or {}
        self.active_workflows: Dict[str, Workflow] = {}
        self.projects: Dict[str, Dict[str, Any]] = {}
        self.conversation_history: List[Dict[str, str]] = []
        self.analysis_history: List[Dict[str, Any]] = []

    def get_system_prompt(self) -> str:
        agent_desc = self._get_agent_descriptions()
        from llm.prompts import PromptTemplates
        return PromptTemplates.coordinator_system(agent_desc)

    def _get_agent_descriptions(self) -> str:
        """Build a text block describing all registered agents."""
        lines = []
        for name, agent in self.agent_registry.items():
            caps = ", ".join(agent.capabilities) if agent.capabilities else "general"
            lines.append(f"- {name}: {agent.description} (capabilities: {caps})")
        return "\n".join(lines) if lines else "No agents registered."

    # ------------------------------------------------------------------
    # Agent registration
    # ------------------------------------------------------------------

    def register_agent(self, agent: BaseAgent):
        self.agent_registry[agent.name] = agent
        logger.info(f"Registered agent: {agent.name}")

    # ------------------------------------------------------------------
    # Project management
    # ------------------------------------------------------------------

    def create_project(
        self, name: str, description: str = "", config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        project_id = generate_uuid()[:8]
        project = {
            "id": project_id,
            "name": name,
            "description": description,
            "config": config or {},
            "datasets": [],
            "workflows": [],
            "models": [],
            "artifacts": [],
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "status": "created",
        }
        self.projects[project_id] = project
        logger.info(f"Created project: {name} (ID: {project_id})")
        return project

    def add_dataset_to_project(self, project_id: str, dataset_info: Dict[str, Any]) -> bool:
        if project_id not in self.projects:
            return False
        self.projects[project_id]["datasets"].append(dataset_info)
        self.projects[project_id]["updated_at"] = datetime.now().isoformat()
        return True

    # ------------------------------------------------------------------
    # Conversational memory
    # ------------------------------------------------------------------

    def add_to_memory(self, role: str, content: str):
        """Append a message to conversation history, keeping last 50 turns."""
        self.conversation_history.append({
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat(),
        })
        # Keep memory bounded
        if len(self.conversation_history) > 50:
            self.conversation_history = self.conversation_history[-50:]

    def get_llm_messages(self) -> List[Dict[str, str]]:
        """Convert conversation history to LLM message format."""
        return [
            {"role": msg["role"], "content": msg["content"]}
            for msg in self.conversation_history
        ]

    def record_analysis(self, agent_name: str, action: str, result_summary: Dict[str, Any]):
        """Record an analysis result for context in future turns."""
        self.analysis_history.append({
            "agent": agent_name,
            "action": action,
            "summary": result_summary,
            "timestamp": datetime.now().isoformat(),
        })
        if len(self.analysis_history) > 20:
            self.analysis_history = self.analysis_history[-20:]

    # ------------------------------------------------------------------
    # Intent analysis (LLM-powered with fallback)
    # ------------------------------------------------------------------

    async def analyze_user_intent(
        self, user_message: str, context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Understand what the user wants using the LLM."""
        context = context or {}
        has_dataset = context.get("has_dataset", False)
        has_target = context.get("has_target", False)
        dataset_summary = context.get("dataset_summary", "")

        if self.llm_client is None:
            return {
                "intent": "llm_unavailable",
                "explanation": "LLM is not configured or validated. Please connect a valid provider.",
                "raw_message": user_message,
                "context": context,
                "blocked": True,
            }

        from llm.client import FallbackClient
        if isinstance(self.llm_client, FallbackClient):
            return {
                "intent": "llm_unavailable",
                "explanation": "Fallback LLM is disabled for intelligence features.",
                "raw_message": user_message,
                "context": context,
                "blocked": True,
            }

        from llm.prompts import PromptTemplates

        prompt = PromptTemplates.intent_analysis(
            user_message, has_dataset, has_target, dataset_summary
        )

        try:
            result = await self.llm_client.chat_json(
                messages=[{"role": "user", "content": prompt}],
                system=self.get_system_prompt(),
                temperature=0.1,
            )
            result["raw_message"] = user_message
            result["context"] = context
            return result
        except Exception as e:
            logger.error(f"LLM intent analysis failed: {e}")
            fallback = FallbackClient()
            result = await fallback.chat_json(
                messages=[{"role": "user", "content": user_message}]
            )
            result["raw_message"] = user_message
            result["context"] = context
            return result

    # ------------------------------------------------------------------
    # Result interpretation (LLM-powered)
    # ------------------------------------------------------------------

    async def interpret_results(
        self, agent_name: str, action: str, result_data: Dict[str, Any], user_context: str = ""
    ) -> str:
        """Have the LLM interpret agent results into human-readable insights."""
        from llm.client import FallbackClient
        if self.llm_client is None or isinstance(self.llm_client, FallbackClient):
            return "LLM unavailable. Result interpretation is paused until a validated LLM is connected."

        from llm.prompts import PromptTemplates

        prompt = PromptTemplates.interpret_results(
            agent_name, action, result_data, user_context
        )

        try:
            interpretation = await self.llm_client.chat(
                messages=[{"role": "user", "content": prompt}],
                system=self.get_system_prompt(),
                temperature=0.3,
            )
            return interpretation
        except Exception as e:
            logger.error(f"LLM interpretation failed: {e}")
            return self._fallback_interpret(agent_name, action, result_data)

    def _fallback_interpret(
        self, agent_name: str, action: str, result_data: Dict[str, Any]
    ) -> str:
        """Template-based result interpretation when LLM is unavailable."""
        lines = [f"**{agent_name}** completed `{action}` successfully.\n"]

        if agent_name == "EDAAgent":
            info = result_data.get("dataset_info", {})
            shape = info.get("shape", {})
            lines.append(f"- **Dataset**: {shape.get('rows', '?')} rows x {shape.get('columns', '?')} columns")
            missing = info.get("missing_values", {})
            total_missing = sum(v for v in missing.values() if isinstance(v, (int, float)))
            lines.append(f"- **Missing values**: {total_missing}")
            lines.append(f"- **Duplicates**: {info.get('duplicates', 0)}")
            for insight in result_data.get("insights", [])[:5]:
                lines.append(f"- {insight}")
            if result_data.get("recommendations"):
                lines.append("\n**Recommendations:**")
                for rec in result_data["recommendations"][:3]:
                    lines.append(f"- {rec}")

        elif agent_name == "ModelTrainerAgent":
            best = result_data.get("best_model", "N/A")
            best_metrics = result_data.get("best_metrics", {})
            task_type = result_data.get("task_type", "")
            lines.append(f"- **Best model**: {best}")
            if task_type == "classification":
                lines.append(f"- **Accuracy**: {best_metrics.get('accuracy', 0):.4f}")
                lines.append(f"- **F1 Score**: {best_metrics.get('f1', 0):.4f}")
            else:
                lines.append(f"- **R2 Score**: {best_metrics.get('r2', 0):.4f}")
                lines.append(f"- **RMSE**: {best_metrics.get('rmse', 0):.4f}")
            lines.append(f"- **CV Mean**: {best_metrics.get('cv_mean', 0):.4f} (+/- {best_metrics.get('cv_std', 0):.4f})")
            n_models = len(result_data.get("results", {}))
            lines.append(f"- Trained {n_models} models total")
            n_feats = result_data.get("n_features")
            if n_feats:
                lines.append(f"- Used {n_feats} features on {result_data.get('n_samples', '?')} samples")
            if result_data.get("target_encoded"):
                lines.append("- Target column was auto-encoded (categorical → numeric)")
            id_dropped = result_data.get("id_columns_dropped", [])
            text_dropped = result_data.get("text_columns_dropped", [])
            cat_encoded = result_data.get("categorical_features_encoded", [])
            if id_dropped:
                lines.append(f"- Dropped ID columns: {', '.join(id_dropped)}")
            if text_dropped:
                lines.append(f"- Dropped text columns: {', '.join(text_dropped)}")
            if cat_encoded:
                lines.append(f"- Auto-encoded {len(cat_encoded)} categorical features")

        elif agent_name == "DataCleanerAgent":
            report = result_data.get("cleaning_report", result_data)
            orig = report.get("original_shape", ())
            final = report.get("final_shape", ())
            if orig and final:
                lines.append(f"- **Before**: {orig[0] if isinstance(orig, (list, tuple)) else orig} rows")
                lines.append(f"- **After**: {final[0] if isinstance(final, (list, tuple)) else final} rows")
            orig_missing = report.get("original_missing_total", 0)
            if orig_missing:
                lines.append(f"- **Original missing values**: {orig_missing}")
            for step in report.get("steps", []):
                step_name = step.get("step", "")
                if step_name == "remove_duplicates":
                    lines.append(f"- Duplicates removed: {step.get('removed', 0)}")
                elif step_name == "handle_missing":
                    filled = step.get('values_filled', 0)
                    lines.append(f"- Missing values filled: {filled}")
                    details = step.get("details", [])
                    for d in details[:5]:
                        lines.append(f"  - `{d['column']}`: {d['filled']} values ({d['strategy']})")
                    if len(details) > 5:
                        lines.append(f"  - ... and {len(details) - 5} more columns")
                elif step_name == "handle_outliers":
                    clipped = step.get('clipped', 0)
                    lines.append(f"- Outliers clipped: {clipped}")
                    details = step.get("details", [])
                    for d in details[:5]:
                        lines.append(f"  - `{d['column']}`: {d['clipped']} outliers")
                else:
                    lines.append(f"- {step_name}: {step.get('removed', 0)} items")

        elif agent_name == "FeatureEngineerAgent":
            report = result_data.get("feature_report", result_data)
            created = report.get("created_features", [])
            encoded = report.get("encoded_features", [])
            dropped = report.get("dropped_columns", [])
            lines.append(f"- **Created** {len(created)} new features")
            lines.append(f"- **Encoded** {len(encoded)} categorical features")
            if dropped:
                lines.append(f"- **Dropped** {len(dropped)} non-feature columns (IDs, text)")
            orig = len(report.get("original_features", []))
            final = len(report.get("final_features", []))
            lines.append(f"- Feature count: {orig} -> {final}")

        elif agent_name == "AutoMLAgent":
            best = result_data.get("best_model", "N/A")
            lines.append(f"- **Best model**: {best}")
            recs = result_data.get("recommendations", [])
            if recs:
                lines.append("- **Top recommendations:**")
                for rec in recs[:3]:
                    lines.append(f"  - {rec.get('model', '?')}: score {rec.get('score', 0):.2f} ({', '.join(rec.get('reasons', []))})")

        elif agent_name == "ForecastAgent":
            forecast = result_data.get("forecast", {})
            if forecast:
                periods = forecast.get("periods", 0)
                metric = forecast.get("metric", "")
                lines.append(f"- **Forecast metric**: {metric}")
                lines.append(f"- **Periods ahead**: {periods}")
                values = forecast.get("forecast_values", [])
                if values:
                    lines.append(f"- **Last predicted value**: {values[-1]:.2f}" if isinstance(values[-1], (int, float)) else "")
            eligible = result_data.get("eligible_metrics", [])
            if eligible:
                lines.append(f"- **Forecast-eligible metrics**: {', '.join(eligible[:5])}")

        elif agent_name == "InsightsAgent":
            insights = result_data.get("insights", [])
            high_count = sum(1 for i in insights if i.get("priority") == "high")
            lines.append(f"- **Total insights**: {len(insights)} ({high_count} critical)")
            for ins in insights[:5]:
                prio = ins.get("priority", "")
                narr = ins.get("narrative", ins.get("title", ""))
                prefix = "[!]" if prio == "high" else "[-]"
                lines.append(f"  {prefix} {narr}")
            summary = result_data.get("executive_summary", "")
            if summary:
                lines.append(f"\n**Executive Summary:** {summary}")
            recs = result_data.get("recommendations", [])
            if recs:
                lines.append("\n**Recommendations:**")
                for r in recs[:4]:
                    lines.append(f"- **{r.get('action', '')}**: {r.get('detail', '')}")

        elif agent_name == "ReportGeneratorAgent":
            title = result_data.get("title", "Report")
            formats = []
            if "markdown" in result_data:
                formats.append("Markdown")
            if "html" in result_data:
                formats.append("HTML")
            if "csv_summary" in result_data:
                formats.append("CSV")
            lines.append(f"- **Report**: {title}")
            lines.append(f"- **Formats generated**: {', '.join(formats)}")
            lines.append("- Use the download buttons below to export your report.")

        else:
            # Generic summary
            for key, value in list(result_data.items())[:5]:
                if isinstance(value, (str, int, float)):
                    lines.append(f"- **{key}**: {value}")

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Conversational reply (GPT-like interactive responses)
    # ------------------------------------------------------------------

    async def generate_conversational_reply(
        self, user_message: str, context: Optional[Dict[str, Any]] = None
    ) -> str:
        """Generate an interactive, context-aware reply for general queries.

        When the user's message doesn't map to a specific agent action,
        this method produces a helpful, conversational response that
        references the current state and suggests next steps — similar
        to how a GPT-style assistant would respond.
        """
        context = context or {}

        from llm.client import FallbackClient
        if self.llm_client is None or isinstance(self.llm_client, FallbackClient):
            return self._fallback_conversational_reply(user_message, context)

        # LLM-powered conversational reply
        from llm.prompts import PromptTemplates

        prompt = PromptTemplates.conversational_reply(
            user_message, context, self.analysis_history
        )

        # Include recent conversation history for continuity
        messages = self.get_llm_messages()
        messages.append({"role": "user", "content": prompt})

        try:
            reply = await self.llm_client.chat(
                messages=messages,
                system=self.get_system_prompt(),
                temperature=0.5,
            )
            return reply
        except Exception as e:
            logger.error(f"LLM conversational reply failed: {e}")
            return self._fallback_conversational_reply(user_message, context)

    def _fallback_conversational_reply(
        self, user_message: str, context: Dict[str, Any]
    ) -> str:
        """Generate a rich, context-aware reply without an LLM."""
        has_dataset = context.get("has_dataset", False)
        has_target = context.get("has_target", False)
        dataset_summary = context.get("dataset_summary", "")
        completed_analyses = context.get("completed_analyses", {})
        completed = list(completed_analyses.keys()) if completed_analyses else []

        lines: List[str] = []
        msg_lower = user_message.lower().strip()

        # --- Acknowledge the user's message naturally ---
        if any(g in msg_lower for g in ["hi", "hello", "hey", "good morning", "good afternoon", "good evening"]):
            lines.append("Hello! I'm your AI Data Scientist assistant. Great to have you here!")
        elif any(g in msg_lower for g in ["thank", "thanks", "appreciate"]):
            lines.append("You're welcome! Happy to help with your data science tasks.")
        elif any(g in msg_lower for g in ["what can you", "capabilities", "what do you"]):
            return self.get_help_message()
        elif any(g in msg_lower for g in ["what is", "explain", "how does", "difference between", "when to use", "why"]):
            lines.append("That's a great data science question! I work best when analyzing your actual data, so I can demonstrate concepts hands-on.")
            lines.append("")
            lines.append("Upload a dataset and I can show you these concepts in action with real results!")
        else:
            lines.append("I'd be happy to help with your data science needs!")

        lines.append("")

        # --- Context-aware suggestions ---
        if not has_dataset:
            lines.append("**To get started**, upload a dataset using the sidebar — I support CSV, Excel, JSON, and Parquet formats. You can also try one of the sample datasets.")
        elif not completed:
            lines.append(f"**Your dataset is loaded** ({dataset_summary})." if dataset_summary else "**Your dataset is loaded.**")
            lines.append("")
            lines.append("Here's what I'd suggest next:")
            lines.append("- **\"Analyze my data\"** — Run exploratory data analysis to understand your dataset")
            lines.append("- **\"Clean the data\"** — Handle missing values, duplicates, and outliers")
            if has_target:
                lines.append("- **\"Train models\"** — Build and compare machine learning models")
                lines.append("- **\"Run full analysis\"** — Execute the complete pipeline from cleaning to modeling")
            else:
                lines.append("- **Set a target column** in the sidebar if you want to train predictive models")
        else:
            lines.append(f"**Your dataset is loaded** ({dataset_summary})." if dataset_summary else "**Your dataset is loaded.**")
            lines.append(f"**Completed so far**: {', '.join(completed)}")
            lines.append("")

            # Suggest next steps based on what's been done
            all_steps = ["cleaning", "eda", "feature_engineering", "modeling", "automl", "visualization", "dashboard", "insights", "forecast", "report"]
            remaining = [s for s in all_steps if s not in completed]
            if remaining:
                next_suggestions = {
                    "cleaning": "**\"Clean the data\"** — preprocess and handle data quality issues",
                    "eda": "**\"Analyze my data\"** — run exploratory data analysis",
                    "feature_engineering": "**\"Engineer features\"** — create and transform features",
                    "modeling": "**\"Train models\"** — build and compare ML models",
                    "automl": "**\"Run AutoML\"** — automatically find the best model",
                    "visualization": "**\"Show visualizations\"** — generate charts and plots",
                    "dashboard": "**\"Build dashboard\"** — compile a summary report",
                    "insights": "**\"Generate insights\"** — business intelligence analysis",
                    "forecast": "**\"Forecast trends\"** — time series forecasting",
                    "report": "**\"Generate report\"** — export HTML/Markdown/CSV report",
                }
                lines.append("**Suggested next steps:**")
                for step in remaining[:3]:
                    if step in next_suggestions:
                        lines.append(f"- {next_suggestions[step]}")
            else:
                lines.append("You've completed a thorough analysis! You can re-run any step or upload a new dataset to start fresh.")

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Dynamic workflow planning
    # ------------------------------------------------------------------

    async def plan_workflow(
        self,
        project_id: str,
        dataset_info: Dict[str, Any],
        user_requirements: Optional[Dict[str, Any]] = None,
    ) -> Workflow:
        target_column = dataset_info.get("target_column")
        user_goal = ""
        if user_requirements:
            user_goal = user_requirements.get("goal", "Full data science analysis")

        # Try LLM-based planning
        if self.llm_client is not None:
            try:
                plan = await self._llm_plan_workflow(dataset_info, target_column, user_goal)
                if plan and plan.get("steps"):
                    return self._build_workflow_from_plan(project_id, plan, target_column)
            except Exception as e:
                logger.error(f"LLM workflow planning failed: {e}")

        # Fallback to deterministic pipeline
        return self._default_workflow(project_id, dataset_info, target_column)

    async def _llm_plan_workflow(
        self, dataset_info: Dict[str, Any], target_column: Optional[str], user_goal: str
    ) -> Dict[str, Any]:
        from llm.prompts import PromptTemplates

        dataset_summary = dataset_info.get("summary", str(dataset_info))
        prompt = PromptTemplates.plan_workflow(dataset_summary, target_column, user_goal)

        return await self.llm_client.chat_json(
            messages=[{"role": "user", "content": prompt}],
            system=self.get_system_prompt(),
            temperature=0.1,
        )

    def _build_workflow_from_plan(
        self, project_id: str, plan: Dict[str, Any], target_column: Optional[str]
    ) -> Workflow:
        workflow_id = generate_uuid()[:8]
        steps = []
        for i, step_def in enumerate(plan.get("steps", []), 1):
            steps.append(
                WorkflowStep(
                    id=f"step_{i}",
                    name=step_def.get("reason", step_def.get("agent", f"Step {i}")),
                    agent=step_def["agent"],
                    task={"action": step_def["action"], "target_column": target_column},
                    dependencies=[f"step_{i - 1}"] if i > 1 else [],
                )
            )
        workflow = Workflow(
            id=workflow_id,
            name=f"workflow_{workflow_id}",
            project_id=project_id,
            steps=steps,
            metadata={"target_column": target_column, "plan": plan},
        )
        self.active_workflows[workflow_id] = workflow
        return workflow

    def _default_workflow(
        self, project_id: str, dataset_info: Dict[str, Any], target_column: Optional[str]
    ) -> Workflow:
        """Deterministic fallback workflow when LLM planning is unavailable."""
        workflow_id = generate_uuid()[:8]
        steps = []
        step_counter = 1

        # Always clean and explore
        steps.append(
            WorkflowStep(
                id=f"step_{step_counter}",
                name="Data Cleaning & Preprocessing",
                agent="DataCleanerAgent",
                task={"action": "clean_data", "dataset_id": dataset_info.get("id")},
                dependencies=[],
            )
        )
        step_counter += 1

        steps.append(
            WorkflowStep(
                id=f"step_{step_counter}",
                name="Exploratory Data Analysis",
                agent="EDAAgent",
                task={"action": "full_eda", "dataset_id": dataset_info.get("id")},
                dependencies=[f"step_{step_counter - 1}"],
            )
        )
        step_counter += 1

        steps.append(
            WorkflowStep(
                id=f"step_{step_counter}",
                name="Data Visualization",
                agent="DataVisualizerAgent",
                task={"action": "generate_visualizations", "dataset_id": dataset_info.get("id")},
                dependencies=[f"step_{step_counter - 1}"],
            )
        )
        step_counter += 1

        if target_column:
            steps.append(
                WorkflowStep(
                    id=f"step_{step_counter}",
                    name="Feature Engineering",
                    agent="FeatureEngineerAgent",
                    task={"action": "engineer_features", "target_column": target_column},
                    dependencies=[f"step_{step_counter - 1}"],
                )
            )
            step_counter += 1

            steps.append(
                WorkflowStep(
                    id=f"step_{step_counter}",
                    name="AutoML Model Selection",
                    agent="AutoMLAgent",
                    task={"action": "auto_select_models", "target_column": target_column},
                    dependencies=[f"step_{step_counter - 1}"],
                )
            )
            step_counter += 1

            steps.append(
                WorkflowStep(
                    id=f"step_{step_counter}",
                    name="Model Training & Evaluation",
                    agent="ModelTrainerAgent",
                    task={"action": "train_models", "target_column": target_column},
                    dependencies=[f"step_{step_counter - 1}"],
                )
            )
            step_counter += 1

        steps.append(
            WorkflowStep(
                id=f"step_{step_counter}",
                name="Dashboard Generation",
                agent="DashboardBuilderAgent",
                task={"action": "build_dashboard", "project_id": project_id},
                dependencies=[f"step_{step_counter - 1}"],
            )
        )

        workflow = Workflow(
            id=workflow_id,
            name=f"workflow_{workflow_id}",
            project_id=project_id,
            steps=steps,
            metadata={"target_column": target_column, "user_requirements": None},
        )
        self.active_workflows[workflow_id] = workflow
        return workflow

    # ------------------------------------------------------------------
    # Workflow execution
    # ------------------------------------------------------------------

    async def execute_workflow(
        self, workflow_id: str, progress_callback=None
    ) -> Dict[str, Any]:
        if workflow_id not in self.active_workflows:
            return {"success": False, "error": "Workflow not found"}

        workflow = self.active_workflows[workflow_id]
        workflow.status = "running"
        results = {}
        completed_steps = set()

        for step in workflow.steps:
            if not all(dep in completed_steps for dep in step.dependencies):
                step.status = "skipped"
                continue

            step.status = "running"
            if progress_callback:
                await progress_callback(
                    {
                        "workflow_id": workflow_id,
                        "step_id": step.id,
                        "step_name": step.name,
                        "status": "running",
                        "progress": len(completed_steps) / len(workflow.steps) * 100,
                    }
                )

            try:
                agent = self.agent_registry.get(step.agent)
                if agent:
                    result = await agent.run(step.task)
                    step.result = result
                    step.status = "completed" if result.success else "failed"
                    if result.success:
                        completed_steps.add(step.id)
                    results[step.id] = result.to_dict()
                else:
                    step.status = "completed"
                    completed_steps.add(step.id)
                    results[step.id] = {"success": True, "data": f"Simulated: {step.name}"}
            except Exception as e:
                step.status = "failed"
                results[step.id] = {"success": False, "error": str(e)}

            if progress_callback:
                await progress_callback(
                    {
                        "workflow_id": workflow_id,
                        "step_id": step.id,
                        "step_name": step.name,
                        "status": step.status,
                        "progress": len(completed_steps) / len(workflow.steps) * 100,
                    }
                )

        workflow.status = (
            "completed"
            if all(s.status in ["completed", "skipped"] for s in workflow.steps)
            else "failed"
        )
        workflow.updated_at = datetime.now()

        return {
            "success": workflow.status == "completed",
            "workflow_id": workflow_id,
            "status": workflow.status,
            "results": results,
            "completed_steps": len(completed_steps),
            "total_steps": len(workflow.steps),
        }

    # ------------------------------------------------------------------
    # Main execute (BaseAgent interface)
    # ------------------------------------------------------------------

    async def execute(self, task: Dict[str, Any]) -> TaskResult:
        action = task.get("action", "")

        try:
            if action == "create_project":
                project = self.create_project(
                    name=task.get("name", "Untitled"),
                    description=task.get("description", ""),
                    config=task.get("config"),
                )
                return TaskResult(success=True, data=project)

            elif action == "analyze_intent":
                intent = await self.analyze_user_intent(
                    user_message=task.get("message", ""),
                    context=task.get("context"),
                )
                return TaskResult(success=True, data=intent)

            elif action == "plan_workflow":
                workflow = await self.plan_workflow(
                    project_id=task.get("project_id"),
                    dataset_info=task.get("dataset_info", {}),
                    user_requirements=task.get("requirements"),
                )
                return TaskResult(success=True, data=workflow.to_dict())

            elif action == "execute_workflow":
                result = await self.execute_workflow(
                    workflow_id=task.get("workflow_id"),
                    progress_callback=task.get("progress_callback"),
                )
                return TaskResult(success=result["success"], data=result)

            elif action == "interpret":
                interpretation = await self.interpret_results(
                    agent_name=task.get("agent_name", ""),
                    action=task.get("agent_action", ""),
                    result_data=task.get("result_data", {}),
                    user_context=task.get("user_context", ""),
                )
                return TaskResult(success=True, data=interpretation)

            elif action == "get_status":
                return TaskResult(
                    success=True,
                    data={
                        "projects": list(self.projects.keys()),
                        "active_workflows": list(self.active_workflows.keys()),
                        "registered_agents": list(self.agent_registry.keys()),
                        "conversation_turns": len(self.conversation_history),
                        "analyses_completed": len(self.analysis_history),
                    },
                )

            return TaskResult(success=False, error=f"Unknown action: {action}")
        except Exception as e:
            return TaskResult(success=False, error=str(e))

    # ------------------------------------------------------------------
    # UI helpers
    # ------------------------------------------------------------------

    def get_welcome_message(self) -> str:
        return (
            "Welcome to the **AI Data Science Platform**! I'm your AI data scientist.\n\n"
            "Here's what I can do:\n"
            "- **Analyze** your data (EDA, statistics, correlations)\n"
            "- **Clean** your dataset (missing values, duplicates, outliers)\n"
            "- **Engineer** features automatically\n"
            "- **Train** machine learning models and compare them\n"
            "- **Visualize** distributions, correlations, and more\n"
            "- **Forecast** future trends with time series analysis\n"
            "- **Generate insights** with business intelligence narratives\n"
            "- **Export reports** in HTML, Markdown, or CSV formats\n"
            "- **Run a full pipeline** from cleaning to model training\n\n"
            "Upload a dataset using the sidebar, then ask me anything!"
        )

    def get_help_message(self) -> str:
        return (
            "**Available Commands:**\n"
            "- *\"Analyze my data\"* — Run exploratory data analysis\n"
            "- *\"Clean the data\"* — Handle missing values, duplicates, outliers\n"
            "- *\"Engineer features\"* — Create and transform features\n"
            "- *\"Train models\"* — Train and compare ML models\n"
            "- *\"Run full analysis\"* — Execute the complete pipeline\n"
            "- *\"Show visualizations\"* — Generate charts and plots\n"
            "- *\"Forecast trends\"* — Time series forecasting with Prophet\n"
            "- *\"Generate insights\"* — Business intelligence analysis\n"
            "- *\"Generate report\"* — Export HTML/Markdown/CSV report\n"
            "- *\"What's the status?\"* — Check current progress\n\n"
            "You can also ask data science questions in natural language!"
        )
