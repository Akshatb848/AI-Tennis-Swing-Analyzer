"""
AutoML Agent - Automated Model Selection

Works on ANY dataset by:
- Encoding categorical target columns automatically
- Auto-encoding categorical features internally
- Dropping ID-like and text columns
- Handling NaN values
- Graceful fallback when no models can be trained
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime

from .base_agent import BaseAgent, TaskResult

logger = logging.getLogger(__name__)

# Reuse the shared _prepare_features from model_trainer_agent
from .model_trainer_agent import _is_id_column, _is_text_column, _prepare_features


class AutoMLAgent(BaseAgent):
    """Agent for automated machine learning."""

    def __init__(self):
        super().__init__(
            name="AutoMLAgent",
            description="Automated machine learning and model selection",
            capabilities=["auto_model_selection", "data_analysis", "model_recommendation"]
        )
        self.recommendations: List[Dict[str, Any]] = []
        self.automl_report: Dict[str, Any] = {}

    def get_system_prompt(self) -> str:
        return "You are an expert AutoML Agent."

    async def execute(self, task: Dict[str, Any]) -> TaskResult:
        action = task.get("action", "auto_select_models")

        try:
            if action == "auto_select_models":
                return await self._auto_select_models(task)
            elif action == "recommend_models":
                return await self._recommend_models(task)
            elif action == "analyze_data_for_ml":
                return await self._analyze_data(task)
            return TaskResult(success=False, error=f"Unknown action: {action}")
        except Exception as e:
            return TaskResult(success=False, error=str(e))

    async def _auto_select_models(self, task: Dict[str, Any]) -> TaskResult:
        df = task.get("dataframe")
        if df is None:
            return TaskResult(success=False, error="No dataframe provided")

        target_column = task.get("target_column")
        if target_column is None or target_column not in df.columns:
            return TaskResult(success=False, error="Invalid target column")

        df = df.copy()
        # Convert pandas 3.x StringDtype to object for numpy compatibility
        for col in df.columns:
            if pd.api.types.is_string_dtype(df[col]) and df[col].dtype != "object":
                df[col] = df[col].astype(object)

        # ---- Prepare target ----
        y = df[target_column].copy()
        label_encoder = None
        target_encoded = False

        if y.dtype == "object" or y.dtype.name == "category" or pd.api.types.is_string_dtype(y):
            from sklearn.preprocessing import LabelEncoder
            label_encoder = LabelEncoder()
            y = y.fillna("_missing_")
            y = pd.Series(label_encoder.fit_transform(y.astype(str)), index=y.index)
            target_encoded = True

        # Drop rows where target is NaN
        valid_mask = y.notna()
        if valid_mask.sum() < 10:
            return TaskResult(
                success=False,
                error=f"Only {int(valid_mask.sum())} rows have valid target values. Need at least 10.",
            )

        # Analyze data characteristics
        data_analysis = self._analyze_data_characteristics(df, target_column)
        task_type = data_analysis["suggested_task_type"]

        # Get recommendations
        recommendations = self._get_model_recommendations(data_analysis, task_type)
        self.recommendations = recommendations

        # ---- Prepare features (auto-encode categoricals) ----
        X, id_cols, text_cols, encoded_cols = _prepare_features(df, target_column)

        if X.shape[1] == 0:
            return TaskResult(
                success=False,
                error=(
                    "No usable features available for training after dropping "
                    "ID/text columns and encoding categoricals."
                ),
            )

        X = X.loc[valid_mask]
        y = y.loc[valid_mask]

        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train recommended models
        results = {}
        training_errors = []
        for rec in recommendations[:5]:
            model_name = rec["model"]
            try:
                model, metrics = self._train_and_evaluate(model_name, X_train, X_test, y_train, y_test, task_type)
                results[model_name] = {
                    "metrics": metrics,
                    "recommendation_score": rec["score"],
                    "reasons": rec["reasons"]
                }
            except Exception as e:
                training_errors.append(f"{model_name}: {e}")
                logger.error(f"Error training {model_name}: {e}")

        if not results:
            error_detail = "; ".join(training_errors[:3]) if training_errors else "Unknown issue"
            return TaskResult(
                success=False,
                error=f"No models were successfully trained. Errors: {error_detail}",
            )

        metric_key = "accuracy" if task_type == "classification" else "r2"
        best_model = max(results, key=lambda k: results[k]["metrics"].get(metric_key, 0))

        self.automl_report = {
            "timestamp": datetime.now().isoformat(),
            "data_analysis": data_analysis,
            "task_type": task_type,
            "recommendations": recommendations,
            "results": results,
            "best_model": best_model,
            "n_features": X.shape[1],
            "n_samples": len(y),
            "target_encoded": target_encoded,
        }

        return TaskResult(success=True, data=self.automl_report, metrics={"models_evaluated": len(results)})

    def _analyze_data_characteristics(self, df: pd.DataFrame, target_column: str) -> Dict[str, Any]:
        target = df[target_column]
        features = df.drop(columns=[target_column])

        n_samples = len(df)
        n_features = len(features.columns)
        is_classification = (
            target.nunique() <= 10
            or target.dtype == 'object'
            or pd.api.types.is_string_dtype(target)
        )

        class_balance = None
        if is_classification:
            vc = target.value_counts()
            if len(vc) > 0:
                class_balance = float(vc.max() / vc.min()) if vc.min() > 0 else float('inf')
            else:
                class_balance = None

        return {
            "n_samples": n_samples,
            "n_features": n_features,
            "n_numeric": len(features.select_dtypes(include=[np.number]).columns),
            "n_categorical": len(features.select_dtypes(include=['object', 'category', 'string']).columns),
            "is_classification": is_classification,
            "suggested_task_type": "classification" if is_classification else "regression",
            "class_balance_ratio": class_balance,
            "is_imbalanced": class_balance > 3 if class_balance else False,
            "is_small_dataset": n_samples < 1000,
            "is_large_dataset": n_samples > 100000
        }

    def _get_model_recommendations(self, data_analysis: Dict, task_type: str) -> List[Dict]:
        recommendations = []

        if task_type == "classification":
            recommendations = [
                {"model": "RandomForestClassifier", "score": 0.9, "reasons": ["Robust", "Feature importance"]},
                {"model": "GradientBoostingClassifier", "score": 0.85, "reasons": ["High accuracy"]},
                {"model": "LogisticRegression", "score": 0.7, "reasons": ["Fast", "Interpretable"]},
                {"model": "DecisionTreeClassifier", "score": 0.65, "reasons": ["Interpretable"]},
                {"model": "KNeighborsClassifier", "score": 0.6, "reasons": ["Simple"]}
            ]
        else:
            recommendations = [
                {"model": "RandomForestRegressor", "score": 0.85, "reasons": ["Robust"]},
                {"model": "GradientBoostingRegressor", "score": 0.88, "reasons": ["High accuracy"]},
                {"model": "Ridge", "score": 0.7, "reasons": ["Regularized"]},
                {"model": "LinearRegression", "score": 0.6, "reasons": ["Fast", "Interpretable"]},
                {"model": "DecisionTreeRegressor", "score": 0.55, "reasons": ["Interpretable"]}
            ]

        return sorted(recommendations, key=lambda x: x["score"], reverse=True)

    def _train_and_evaluate(self, model_name: str, X_train, X_test, y_train, y_test, task_type: str) -> Tuple[Any, Dict]:
        from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge
        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
        from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
        from sklearn.neighbors import KNeighborsClassifier

        model_map = {
            "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42),
            "RandomForestClassifier": RandomForestClassifier(n_estimators=100, random_state=42),
            "GradientBoostingClassifier": GradientBoostingClassifier(n_estimators=100, random_state=42),
            "DecisionTreeClassifier": DecisionTreeClassifier(random_state=42),
            "KNeighborsClassifier": KNeighborsClassifier(),
            "LinearRegression": LinearRegression(),
            "Ridge": Ridge(random_state=42),
            "RandomForestRegressor": RandomForestRegressor(n_estimators=100, random_state=42),
            "GradientBoostingRegressor": GradientBoostingRegressor(n_estimators=100, random_state=42),
            "DecisionTreeRegressor": DecisionTreeRegressor(random_state=42)
        }

        model = model_map.get(model_name)
        if model is None:
            raise ValueError(f"Unknown model: {model_name}")

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        if task_type == "classification":
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
            metrics = {
                "accuracy": float(accuracy_score(y_test, y_pred)),
                "precision": float(precision_score(y_test, y_pred, average='weighted', zero_division=0)),
                "recall": float(recall_score(y_test, y_pred, average='weighted', zero_division=0)),
                "f1": float(f1_score(y_test, y_pred, average='weighted', zero_division=0))
            }
        else:
            from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
            metrics = {
                "mse": float(mean_squared_error(y_test, y_pred)),
                "rmse": float(np.sqrt(mean_squared_error(y_test, y_pred))),
                "mae": float(mean_absolute_error(y_test, y_pred)),
                "r2": float(r2_score(y_test, y_pred))
            }

        return model, metrics

    async def _recommend_models(self, task: Dict[str, Any]) -> TaskResult:
        df = task.get("dataframe")
        target = task.get("target_column")
        if df is None or target is None:
            return TaskResult(success=False, error="Missing data")

        analysis = self._analyze_data_characteristics(df, target)
        recs = self._get_model_recommendations(analysis, analysis["suggested_task_type"])
        return TaskResult(success=True, data={"analysis": analysis, "recommendations": recs})

    async def _analyze_data(self, task: Dict[str, Any]) -> TaskResult:
        df = task.get("dataframe")
        target = task.get("target_column")
        if df is None or target is None:
            return TaskResult(success=False, error="Missing data")
        return TaskResult(success=True, data=self._analyze_data_characteristics(df, target))
