"""
Model Trainer Agent - Robust model training and evaluation

Works on ANY dataset by:
- Encoding categorical target columns automatically (LabelEncoder)
- Auto-encoding categorical features internally (label encode + one-hot)
- Dropping ID-like and text columns
- Handling NaN in target/features
- Graceful fallback when no models can be trained
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List
from datetime import datetime
import time

from .base_agent import BaseAgent, TaskResult, generate_uuid

logger = logging.getLogger(__name__)


def _is_id_column(df: pd.DataFrame, col: str) -> bool:
    """Heuristic: detect ID-like columns that shouldn't be used as features."""
    col_lower = col.lower().strip()
    if col_lower in ("id", "index", "row_id", "row_number", "unnamed: 0"):
        return True
    if col_lower.endswith(("_id", " id")):
        return True
    if col_lower.startswith(("id_", "id ")):
        return True
    # Unique integer column with almost as many unique values as rows
    if pd.api.types.is_integer_dtype(df[col]) and len(df) > 20:
        if df[col].nunique() >= len(df) * 0.9:
            return True
    return False


def _is_text_column(df: pd.DataFrame, col: str) -> bool:
    """Detect free-text columns that shouldn't be used as features."""
    if not (df[col].dtype == "object" or pd.api.types.is_string_dtype(df[col])):
        return False
    sample = df[col].dropna()
    if len(sample) == 0:
        return False
    avg_len = sample.astype(str).str.len().mean()
    if avg_len > 50:
        return True
    if sample.nunique() > 20 and sample.nunique() / len(sample) > 0.8:
        return True
    col_lower = col.lower()
    if any(kw in col_lower for kw in ["email", "name", "description", "text", "comment",
                                       "address", "url", "subject", "body", "note"]):
        if sample.nunique() > 20:
            return True
    return False


def _prepare_features(df: pd.DataFrame, target_column: str) -> tuple:
    """Prepare features by encoding categoricals and dropping unusable columns.

    Returns (X, id_cols_dropped, text_cols_dropped, encoded_cols) where X is a fully numeric DataFrame.
    """
    feature_df = df.drop(columns=[target_column])

    # Convert pandas 3.x StringDtype to object for numpy compatibility
    str_dtype_cols = feature_df.select_dtypes(include=['string']).columns
    if len(str_dtype_cols) > 0:
        feature_df[str_dtype_cols] = feature_df[str_dtype_cols].astype(object)

    # Drop ID-like columns
    id_cols = [c for c in feature_df.columns if _is_id_column(feature_df, c)]
    if id_cols:
        feature_df = feature_df.drop(columns=id_cols)

    # Drop text columns
    text_cols = [c for c in feature_df.columns if _is_text_column(feature_df, c)]
    if text_cols:
        feature_df = feature_df.drop(columns=text_cols)

    # Encode remaining categorical columns
    encoded_cols = []
    cat_cols = [c for c in feature_df.select_dtypes(include=['object', 'category', 'str']).columns]
    for col in cat_cols:
        n_unique = feature_df[col].nunique()
        if n_unique <= 2:
            feature_df[col] = pd.factorize(feature_df[col])[0]
        elif n_unique <= 10:
            dummies = pd.get_dummies(feature_df[col], prefix=col, drop_first=True)
            # Ensure boolean dummies are int
            for d in dummies.columns:
                dummies[d] = dummies[d].astype(int)
            feature_df = pd.concat([feature_df.drop(columns=[col]), dummies], axis=1)
        else:
            # Frequency encoding for high-cardinality categoricals
            freq_map = feature_df[col].value_counts(normalize=True).to_dict()
            feature_df[f'{col}_freq'] = feature_df[col].map(freq_map).fillna(0)
            feature_df = feature_df.drop(columns=[col])
        encoded_cols.append(col)

    X = feature_df.select_dtypes(include=[np.number]).fillna(0)
    X = X.replace([np.inf, -np.inf], 0)

    return X, id_cols, text_cols, encoded_cols


class ModelTrainerAgent(BaseAgent):
    """Agent for model training and evaluation."""

    def __init__(self):
        super().__init__(
            name="ModelTrainerAgent",
            description="Model training, tuning, and evaluation",
            capabilities=["model_training", "hyperparameter_tuning", "cross_validation", "model_comparison"]
        )
        self.trained_models: Dict[str, Any] = {}
        self.training_history: List[Dict[str, Any]] = []
        self.best_model: Optional[Dict[str, Any]] = None

    def get_system_prompt(self) -> str:
        return "You are an expert Model Training Agent."

    async def execute(self, task: Dict[str, Any]) -> TaskResult:
        action = task.get("action", "train_models")

        try:
            if action == "train_models":
                return await self._train_models(task)
            elif action == "train_single_model":
                return await self._train_single_model(task)
            elif action == "get_best_model":
                return TaskResult(success=True, data=self.best_model)
            return TaskResult(success=False, error=f"Unknown action: {action}")
        except Exception as e:
            logger.error(f"Training error: {e}")
            return TaskResult(success=False, error=str(e))

    async def _train_models(self, task: Dict[str, Any]) -> TaskResult:
        df = task.get("dataframe")
        if df is None:
            return TaskResult(success=False, error="No dataframe provided")

        target_column = task.get("target_column")
        if target_column is None or target_column not in df.columns:
            return TaskResult(success=False, error="Invalid target column")

        df = df.copy()
        # Convert pandas 3.x StringDtype to object for numpy compatibility
        str_dtype_cols = df.select_dtypes(include=['string']).columns
        if len(str_dtype_cols) > 0:
            df[str_dtype_cols] = df[str_dtype_cols].astype(object)

        cv_folds = task.get("cv_folds", 5)

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

        # Drop rows where target is NaN (for numeric targets)
        valid_mask = y.notna()
        if valid_mask.sum() < 10:
            return TaskResult(
                success=False,
                error=f"Only {int(valid_mask.sum())} rows have valid target values. Need at least 10.",
            )

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

        # Align X and y on valid rows
        X = X.loc[valid_mask]
        y = y.loc[valid_mask]

        # Adjust CV folds for small datasets
        cv_folds = min(cv_folds, max(2, len(y) // 10))

        task_type = "classification" if y.nunique() <= 10 else "regression"

        from sklearn.model_selection import train_test_split, cross_val_score
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        results = {}
        training_errors = []

        if task_type == "classification":
            from sklearn.linear_model import LogisticRegression
            from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
            from sklearn.tree import DecisionTreeClassifier
            from sklearn.neighbors import KNeighborsClassifier
            from sklearn.naive_bayes import GaussianNB
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

            models = {
                "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42),
                "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
                "GradientBoosting": GradientBoostingClassifier(n_estimators=100, random_state=42),
                "DecisionTree": DecisionTreeClassifier(random_state=42),
                "KNN": KNeighborsClassifier(),
                "NaiveBayes": GaussianNB()
            }

            for name, model in models.items():
                try:
                    start = time.time()
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                    cv = cross_val_score(model, X_train, y_train, cv=cv_folds, scoring='accuracy')

                    results[name] = {
                        "metrics": {
                            "accuracy": float(accuracy_score(y_test, y_pred)),
                            "precision": float(precision_score(y_test, y_pred, average='weighted', zero_division=0)),
                            "recall": float(recall_score(y_test, y_pred, average='weighted', zero_division=0)),
                            "f1": float(f1_score(y_test, y_pred, average='weighted', zero_division=0)),
                            "cv_mean": float(cv.mean()), "cv_std": float(cv.std())
                        },
                        "training_time": time.time() - start,
                        "feature_importance": dict(zip(X.columns.tolist(), model.feature_importances_.tolist())) if hasattr(model, 'feature_importances_') else {}
                    }
                    self.trained_models[name] = model
                except Exception as e:
                    training_errors.append(f"{name}: {e}")
                    logger.error(f"Error training {name}: {e}")

        else:  # regression
            from sklearn.linear_model import LinearRegression, Ridge, Lasso
            from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
            from sklearn.tree import DecisionTreeRegressor
            from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

            models = {
                "LinearRegression": LinearRegression(),
                "Ridge": Ridge(random_state=42),
                "Lasso": Lasso(random_state=42),
                "RandomForest": RandomForestRegressor(n_estimators=100, random_state=42),
                "GradientBoosting": GradientBoostingRegressor(n_estimators=100, random_state=42),
                "DecisionTree": DecisionTreeRegressor(random_state=42)
            }

            for name, model in models.items():
                try:
                    start = time.time()
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                    cv = cross_val_score(model, X_train, y_train, cv=cv_folds, scoring='r2')

                    results[name] = {
                        "metrics": {
                            "mse": float(mean_squared_error(y_test, y_pred)),
                            "rmse": float(np.sqrt(mean_squared_error(y_test, y_pred))),
                            "mae": float(mean_absolute_error(y_test, y_pred)),
                            "r2": float(r2_score(y_test, y_pred)),
                            "cv_mean": float(cv.mean()), "cv_std": float(cv.std())
                        },
                        "training_time": time.time() - start,
                        "feature_importance": dict(zip(X.columns.tolist(), model.feature_importances_.tolist())) if hasattr(model, 'feature_importances_') else {}
                    }
                    self.trained_models[name] = model
                except Exception as e:
                    training_errors.append(f"{name}: {e}")
                    logger.error(f"Error training {name}: {e}")

        if not results:
            error_detail = "; ".join(training_errors[:3]) if training_errors else "Unknown issue"
            return TaskResult(
                success=False,
                error=f"No models were successfully trained. Errors: {error_detail}",
            )

        metric_key = "accuracy" if task_type == "classification" else "r2"
        best_name = max(results, key=lambda k: results[k]["metrics"].get(metric_key, 0))

        self.best_model = {
            "name": best_name,
            "metrics": results[best_name]["metrics"],
            "task_type": task_type
        }

        self.training_history.append({
            "timestamp": datetime.now().isoformat(),
            "task_type": task_type,
            "best_model": best_name
        })

        return TaskResult(
            success=True,
            data={
                "results": results,
                "best_model": best_name,
                "best_metrics": results[best_name]["metrics"],
                "task_type": task_type,
                "n_features": X.shape[1],
                "n_samples": len(y),
                "target_encoded": target_encoded,
                "id_columns_dropped": id_cols,
                "text_columns_dropped": text_cols,
                "categorical_features_encoded": encoded_cols,
            },
            metrics={"models_trained": len(results), "best_model": best_name}
        )

    async def _train_single_model(self, task: Dict[str, Any]) -> TaskResult:
        return TaskResult(success=False, error="Not implemented")
