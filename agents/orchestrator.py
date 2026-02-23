"""Central orchestrator for the 5-pillar data science pipeline.

Coordinates: BusinessStrategy -> DataEngineering -> ExploratoryAnalysis ->
ModelingML -> MLOpsDeployment, with optional LLM and RAG support.
"""

import logging
import os
from typing import Any, Dict, List, Optional

from agents.base import AgentResult
from agents.business_strategy_agent import BusinessStrategyAgent
from agents.data_engineering_agent import DataEngineeringAgent
from agents.exploratory_analysis_agent import ExploratoryAnalysisAgent
from agents.modeling_ml_agent import ModelingMLAgent
from agents.mlops_deployment_agent import MLOpsDeploymentAgent

logger = logging.getLogger("Orchestrator")

PIPELINE_STEPS = ["strategy", "engineering", "exploration", "modeling", "mlops"]


class Orchestrator:
    """Central orchestrator that runs the full 5-pillar data-science pipeline."""

    def __init__(self, llm_client=None, rag_client=None) -> None:
        self.agents: Dict[str, Any] = {
            "strategy": BusinessStrategyAgent(),
            "engineering": DataEngineeringAgent(),
            "exploration": ExploratoryAnalysisAgent(),
            "modeling": ModelingMLAgent(),
            "mlops": MLOpsDeploymentAgent(),
        }
        self._results: Dict[str, AgentResult] = {}
        self._state: Dict[str, str] = {step: "pending" for step in PIPELINE_STEPS}
        self._context: Dict[str, Any] = {}
        self.llm_client = llm_client
        self.rag_client = rag_client

    def run_pipeline(
        self,
        file_path: str,
        target_col: Optional[str] = None,
        use_optuna: bool = True,
    ) -> Dict[str, Any]:
        """Execute the full 5-pillar pipeline end-to-end."""
        logger.info("Pipeline started – file=%s, target=%s", file_path, target_col)

        self._results.clear()
        self._state = {step: "pending" for step in PIPELINE_STEPS}
        self._context = {
            "file_path": file_path,
            "target_col": target_col,
            "use_optuna": use_optuna,
        }

        self._run_ingest_and_strategy(file_path, target_col)
        self._run_engineering(target_col)
        self._run_exploration()
        self._run_modeling(use_optuna)
        self._run_mlops()

        logger.info("Pipeline finished – state: %s", self._state)
        return {
            "pipeline_state": dict(self._state),
            "step_results": {k: self._summarise(v) for k, v in self._results.items()},
            "final_insights": self._results.get("mlops"),
        }

    def get_pipeline_state(self) -> Dict[str, str]:
        return dict(self._state)

    def get_step_result(self, step_name: str) -> Optional[AgentResult]:
        return self._results.get(step_name)

    def _run_ingest_and_strategy(self, file_path: str, target_col: Optional[str]) -> None:
        """Load data and run business strategy analysis."""
        self._state["strategy"] = "running"
        try:
            from utils.csv_loader import load_csv
            ext = os.path.splitext(file_path)[1].lstrip(".").lower()

            if ext in ("xlsx", "xls"):
                import pandas as pd
                df = pd.read_excel(file_path)
                from utils.csv_loader import _build_profile, _clean_columns
                warnings = []
                df = _clean_columns(df, warnings)
                profile = _build_profile(df, warnings, "n/a", "n/a")
            elif ext == "parquet":
                import pandas as pd
                df = pd.read_parquet(file_path)
                from utils.csv_loader import _build_profile, _clean_columns
                warnings = []
                df = _clean_columns(df, warnings)
                profile = _build_profile(df, warnings, "n/a", "n/a")
            elif ext == "json":
                import pandas as pd
                df = pd.read_json(file_path)
                from utils.csv_loader import _build_profile, _clean_columns
                warnings = []
                df = _clean_columns(df, warnings)
                profile = _build_profile(df, warnings, "n/a", "n/a")
            else:
                df, profile = load_csv(file_path)

            if target_col and target_col in df.columns:
                profile["target_column"] = target_col

            self._context["df"] = df
            self._context["profile"] = profile
            self._context["target_col"] = target_col or profile.get("target_column")
            self._context["problem_type"] = profile.get("problem_type", "regression")

            strategy_result = self.agents["strategy"].execute(
                dataset_profile=profile,
                llm_client=self.llm_client,
            )

            if strategy_result.success:
                objective = strategy_result.data.get("objective", {})
                if objective.get("problem_type"):
                    self._context["problem_type"] = objective["problem_type"]

            ingest_data = {
                "df": df,
                "profile": profile,
                "strategy": strategy_result.data if strategy_result.success else {},
            }
            self._results["strategy"] = AgentResult(
                success=True,
                data=ingest_data,
                metadata={"file_path": file_path},
            )
            self._state["strategy"] = "completed"

        except Exception as exc:
            logger.error("Strategy step failed: %s", exc, exc_info=True)
            self._results["strategy"] = AgentResult(success=False, errors=[str(exc)])
            self._state["strategy"] = "failed"

    def _run_engineering(self, target_col: Optional[str]) -> None:
        strategy = self._results.get("strategy")
        if not strategy or not strategy.success:
            self._state["engineering"] = "skipped"
            return

        self._state["engineering"] = "running"
        try:
            df = self._context["df"]
            tc = self._context.get("target_col") or target_col
            profile = self._context.get("profile", {})
            problem_type = self._context.get("problem_type", "regression")

            result = self.agents["engineering"].execute(
                df=df,
                target_col=tc,
                problem_type=problem_type,
                id_columns=profile.get("id_columns", []),
            )
            self._results["engineering"] = result
            self._state["engineering"] = "completed" if result.success else "failed"

            if result.success:
                self._context["X_train"] = result.data["X_train"]
                self._context["X_test"] = result.data["X_test"]
                self._context["y_train"] = result.data["y_train"]
                self._context["y_test"] = result.data["y_test"]
                self._context["feature_names"] = result.data.get("feature_names", [])

        except Exception as exc:
            logger.error("Engineering step failed: %s", exc, exc_info=True)
            self._results["engineering"] = AgentResult(success=False, errors=[str(exc)])
            self._state["engineering"] = "failed"

    def _run_exploration(self) -> None:
        strategy = self._results.get("strategy")
        if not strategy or not strategy.success:
            self._state["exploration"] = "skipped"
            return

        self._state["exploration"] = "running"
        try:
            df = self._context["df"]
            tc = self._context.get("target_col")
            problem_type = self._context.get("problem_type")

            result = self.agents["exploration"].execute(
                df=df,
                target_col=tc,
                problem_type=problem_type,
            )
            self._results["exploration"] = result
            self._state["exploration"] = "completed" if result.success else "failed"
        except Exception as exc:
            logger.error("Exploration step failed: %s", exc, exc_info=True)
            self._results["exploration"] = AgentResult(success=False, errors=[str(exc)])
            self._state["exploration"] = "failed"

    def _run_modeling(self, use_optuna: bool = True) -> None:
        eng = self._results.get("engineering")
        if not eng or not eng.success:
            self._state["modeling"] = "skipped"
            return

        self._state["modeling"] = "running"
        try:
            problem_type = self._context.get("problem_type", "regression")
            result = self.agents["modeling"].execute(
                X_train=self._context["X_train"],
                y_train=self._context["y_train"],
                X_test=self._context["X_test"],
                y_test=self._context["y_test"],
                problem_type=problem_type,
                feature_names=self._context.get("feature_names"),
                use_optuna=use_optuna,
            )
            self._results["modeling"] = result
            self._state["modeling"] = "completed" if result.success else "failed"
        except Exception as exc:
            logger.error("Modeling step failed: %s", exc, exc_info=True)
            self._results["modeling"] = AgentResult(success=False, errors=[str(exc)])
            self._state["modeling"] = "failed"

    def _run_mlops(self) -> None:
        model_res = self._results.get("modeling")
        if not model_res or not model_res.success:
            self._state["mlops"] = "skipped"
            return

        self._state["mlops"] = "running"
        try:
            problem_type = self._context.get("problem_type", "regression")
            eng = self._results.get("engineering")
            eval_metrics = {}
            if model_res.data.get("leaderboard"):
                eval_metrics = {
                    "champion_score": model_res.data.get("champion_score"),
                    "leaderboard": model_res.data.get("leaderboard"),
                }

            result = self.agents["mlops"].execute(
                champion_model=model_res.data["champion_model"],
                champion_name=model_res.data.get("champion_name", "unknown"),
                feature_names=self._context.get("feature_names", []),
                problem_type=problem_type,
                model_metrics=eval_metrics,
            )
            self._results["mlops"] = result
            self._state["mlops"] = "completed" if result.success else "failed"
        except Exception as exc:
            logger.error("MLOps step failed: %s", exc, exc_info=True)
            self._results["mlops"] = AgentResult(success=False, errors=[str(exc)])
            self._state["mlops"] = "failed"

    @staticmethod
    def _summarise(result: AgentResult) -> Dict[str, Any]:
        return {
            "success": result.success,
            "errors": result.errors,
            "data_keys": list(result.data.keys()) if result.data else [],
        }
