"""Business Strategy Agent – analyses dataset profiles to define ML objectives."""

from typing import Any, Dict, List, Optional

from agents.base import AgentResult, BaseAgent


_KPI_MAP: Dict[str, Dict[str, str]] = {
    "binary_classification": {"primary": "accuracy", "secondary": "f1_score", "description": "Maximise correct predictions of the binary outcome"},
    "multiclass_classification": {"primary": "accuracy", "secondary": "f1_weighted", "description": "Maximise correct predictions across all classes"},
    "regression": {"primary": "rmse", "secondary": "r2", "description": "Minimise prediction error for the continuous target"},
    "time_series": {"primary": "rmse", "secondary": "mae", "description": "Minimise forecast error over time"},
    "clustering": {"primary": "silhouette_score", "secondary": "calinski_harabasz", "description": "Maximise cluster separation and cohesion"},
    "eda_only": {"primary": "n/a", "secondary": "n/a", "description": "Exploratory analysis only – no predictive KPI"},
}


class BusinessStrategyAgent(BaseAgent):
    """Analyses dataset profiles to define ML objectives, KPIs, and constraints.

    When an LLM client is available and connected the agent delegates to the
    language model for richer, context-aware analysis.  Otherwise it falls
    back to deterministic rule-based logic derived from the dataset profile.
    """

    @property
    def name(self) -> str:
        return "BusinessStrategyAgent"

    def execute(
        self,
        dataset_profile: Dict[str, Any],
        user_prompt: Optional[str] = None,
        llm_client: Optional[Any] = None,
        **kwargs: Any,
    ) -> AgentResult:
        """Define ML objectives from a dataset profile.

        Args:
            dataset_profile: Profile dict produced by ``utils.csv_loader.load_csv``.
            user_prompt: Optional natural-language guidance from the user.
            llm_client: Optional LLM client with a ``generate`` / ``chat`` method.

        Returns:
            AgentResult with *objective* (dict) and *recommendations* (list[str]).
        """
        try:
            self.logger.info("BusinessStrategyAgent starting analysis")

            if llm_client is not None and self._llm_is_connected(llm_client):
                return self._llm_analysis(dataset_profile, user_prompt, llm_client)

            return self._rule_based_analysis(dataset_profile, user_prompt)

        except Exception as exc:
            self.logger.error("BusinessStrategyAgent failed: %s", exc, exc_info=True)
            return AgentResult(success=False, errors=[str(exc)])

    def _llm_is_connected(self, llm_client: Any) -> bool:
        """Return True when the LLM client looks usable."""
        if hasattr(llm_client, "is_connected"):
            try:
                return bool(llm_client.is_connected())
            except Exception:
                return False
        for attr in ("generate", "chat", "complete"):
            if callable(getattr(llm_client, attr, None)):
                return True
        return False

    def _llm_analysis(
        self,
        profile: Dict[str, Any],
        user_prompt: Optional[str],
        llm_client: Any,
    ) -> AgentResult:
        """Use the LLM to produce a richer strategy analysis."""
        problem_type = profile.get("problem_type", "eda_only")
        target = profile.get("target_column")
        schema = profile.get("schema", {})
        stats = profile.get("stats", {})
        quality = profile.get("quality_score", 0)

        system_msg = (
            "You are a senior data-science strategist. Given a dataset profile, "
            "define the ML objective, recommend KPIs, identify constraints, and "
            "list actionable recommendations. Respond in structured plain text."
        )

        user_msg_parts = [
            f"Problem type detected: {problem_type}",
            f"Target column: {target}",
            f"Number of columns: {stats.get('column_count', len(schema))}",
            f"Number of rows: {stats.get('row_count', 'unknown')}",
            f"Data quality score: {quality}/100",
            f"Schema: {schema}",
        ]
        if user_prompt:
            user_msg_parts.append(f"User guidance: {user_prompt}")

        prompt_text = "\n".join(user_msg_parts)

        try:
            if hasattr(llm_client, "chat"):
                response = llm_client.chat(
                    system_prompt=system_msg,
                    user_message=prompt_text,
                )
            elif hasattr(llm_client, "generate"):
                response = llm_client.generate(f"{system_msg}\n\n{prompt_text}")
            else:
                response = llm_client.complete(f"{system_msg}\n\n{prompt_text}")

            llm_text = str(response) if not isinstance(response, str) else response

            kpi_info = _KPI_MAP.get(problem_type, _KPI_MAP["eda_only"])
            objective = {
                "problem_type": problem_type,
                "target": target,
                "kpi": kpi_info["primary"],
                "kpi_secondary": kpi_info["secondary"],
                "kpi_description": kpi_info["description"],
                "constraints": self._detect_constraints(profile),
            }

            recommendations = [
                line.strip("- ").strip()
                for line in llm_text.split("\n")
                if line.strip() and len(line.strip()) > 10
            ][:10]

            if not recommendations:
                recommendations = self._generate_recommendations(profile, problem_type)

            self.logger.info("LLM-based strategy analysis complete")
            return AgentResult(
                success=True,
                data={"objective": objective, "recommendations": recommendations},
                metadata={"source": "llm", "problem_type": problem_type},
            )

        except Exception as exc:
            self.logger.warning("LLM analysis failed (%s), falling back to rules", exc)
            return self._rule_based_analysis(profile, user_prompt)

    def _rule_based_analysis(
        self,
        profile: Dict[str, Any],
        user_prompt: Optional[str],
    ) -> AgentResult:
        """Deterministic strategy based on the dataset profile."""
        problem_type = profile.get("problem_type", "eda_only")
        target = profile.get("target_column")
        quality = profile.get("quality_score", 0)
        stats = profile.get("stats", {})
        schema = profile.get("schema", {})

        kpi_info = _KPI_MAP.get(problem_type, _KPI_MAP["eda_only"])

        constraints = self._detect_constraints(profile)

        objective: Dict[str, Any] = {
            "problem_type": problem_type,
            "target": target,
            "kpi": kpi_info["primary"],
            "kpi_secondary": kpi_info["secondary"],
            "kpi_description": kpi_info["description"],
            "constraints": constraints,
        }

        recommendations = self._generate_recommendations(profile, problem_type)

        if user_prompt:
            recommendations.append(f"User guidance noted: '{user_prompt}' – incorporate into modelling strategy.")

        self.logger.info(
            "Rule-based strategy: problem=%s, target=%s, kpi=%s, constraints=%d",
            problem_type, target, kpi_info["primary"], len(constraints),
        )

        return AgentResult(
            success=True,
            data={"objective": objective, "recommendations": recommendations},
            metadata={"source": "rule_based", "problem_type": problem_type},
        )

    def _detect_constraints(self, profile: Dict[str, Any]) -> List[str]:
        """Identify practical constraints from the profile."""
        constraints: List[str] = []
        stats = profile.get("stats", {})
        row_count = stats.get("row_count", 0)
        col_count = stats.get("column_count", 0)
        quality = profile.get("quality_score", 0)
        missing = stats.get("missing_pct", {})
        id_cols = profile.get("id_columns", [])
        leakage_cols = profile.get("leakage_columns", [])

        if row_count < 500:
            constraints.append(f"Small dataset ({row_count} rows) – use cross-validation and avoid complex models")
        elif row_count < 2000:
            constraints.append(f"Moderate dataset size ({row_count} rows) – regularisation recommended")

        if col_count > 100:
            constraints.append(f"High dimensionality ({col_count} features) – dimensionality reduction advised")

        if quality < 50:
            constraints.append(f"Low data quality ({quality:.1f}/100) – aggressive cleaning required")

        high_missing = [c for c, pct in missing.items() if pct > 30]
        if high_missing:
            constraints.append(f"{len(high_missing)} column(s) have >30% missing values")

        if id_cols:
            constraints.append(f"ID columns detected ({', '.join(id_cols[:3])}) – must be excluded from features")

        if leakage_cols:
            constraints.append(f"Potential leakage columns ({', '.join(leakage_cols[:3])}) – must be excluded")

        return constraints

    def _generate_recommendations(self, profile: Dict[str, Any], problem_type: str) -> List[str]:
        """Generate actionable recommendations."""
        recs: List[str] = []
        stats = profile.get("stats", {})
        schema = profile.get("schema", {})
        quality = profile.get("quality_score", 0)
        row_count = stats.get("row_count", 0)
        warnings = profile.get("warnings", [])

        if problem_type in ("binary_classification", "multiclass_classification"):
            recs.append("Evaluate class balance; apply SMOTE or class weights if imbalanced")
            recs.append("Use stratified cross-validation to preserve class distribution")
            recs.append("Track precision and recall alongside accuracy for business impact assessment")
        elif problem_type == "regression":
            recs.append("Check target distribution for skewness; consider log-transform if needed")
            recs.append("Use RMSE as primary metric and R² for explanatory power")
            recs.append("Investigate non-linear models if linear baseline under-performs")
        elif problem_type == "time_series":
            recs.append("Use time-based train/test split (no random shuffling)")
            recs.append("Engineer lag and rolling-window features from the target")
            recs.append("Evaluate stationarity before modelling")
        elif problem_type == "clustering":
            recs.append("Standardise features before clustering")
            recs.append("Try multiple values of k and use the elbow method / silhouette score")
            recs.append("Validate clusters with domain knowledge")
        else:
            recs.append("Perform thorough exploratory analysis to understand data distributions")
            recs.append("Identify potential target variables for supervised learning")

        cat_cols = [c for c, t in schema.items() if t == "categorical"]
        num_cols = [c for c, t in schema.items() if t == "numeric"]

        if len(cat_cols) > 10:
            recs.append(f"High number of categorical features ({len(cat_cols)}) – consider target encoding or embedding")

        if quality < 70:
            recs.append("Prioritise data cleaning: handle missing values, outliers, and encoding issues before modelling")

        if row_count > 50000:
            recs.append("Dataset is large – consider sampling for rapid prototyping, then train on full data")

        datetime_cols = profile.get("datetime_candidates", [])
        if datetime_cols and problem_type != "time_series":
            recs.append(f"Datetime columns detected ({', '.join(datetime_cols[:2])}) – extract temporal features (day of week, month, etc.)")

        if warnings:
            recs.append(f"Address {len(warnings)} data-loading warning(s) before finalising the pipeline")

        return recs
