"""Modeling ML Agent – trains, tunes, and selects champion models."""

import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import cross_val_score

from agents.base import AgentResult, BaseAgent

_HAS_XGB = False
XGBClassifier = None
XGBRegressor = None
try:
    from xgboost import XGBClassifier, XGBRegressor
    _HAS_XGB = True
except Exception:
    pass

_HAS_LGB = False
LGBMClassifier = None
LGBMRegressor = None
try:
    from lightgbm import LGBMClassifier, LGBMRegressor
    _HAS_LGB = True
except Exception:
    pass

_HAS_OPTUNA = False
optuna = None
try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    _HAS_OPTUNA = True
except Exception:
    pass


class ModelingMLAgent(BaseAgent):
    """Trains multiple algorithms, optionally tunes with Optuna, and selects a champion.

    Supports classification (binary / multiclass) and regression.  Optional
    XGBoost, LightGBM, and Optuna integrations are used when available.
    """

    @property
    def name(self) -> str:
        return "ModelingMLAgent"

    def execute(
        self,
        X_train: Any,
        y_train: Any,
        X_test: Any,
        y_test: Any,
        problem_type: str = "regression",
        feature_names: Optional[List[str]] = None,
        use_optuna: bool = True,
        **kwargs: Any,
    ) -> AgentResult:
        """Train, evaluate, and rank models.

        Args:
            X_train: Training features (array-like).
            y_train: Training target (array-like).
            X_test: Test features (array-like).
            y_test: Test target (array-like).
            problem_type: ML problem type string.
            feature_names: Optional column names.
            use_optuna: Whether to attempt Optuna hyper-parameter tuning.

        Returns:
            AgentResult with champion_model, champion_name, champion_params,
            champion_score, leaderboard, and all_models.
        """
        try:
            self.logger.info(
                "ModelingMLAgent started – problem=%s, train=%d, test=%d, optuna=%s",
                problem_type, len(X_train), len(X_test), use_optuna and _HAS_OPTUNA,
            )

            X_tr = np.asarray(X_train, dtype=float)
            X_te = np.asarray(X_test, dtype=float)
            y_tr = np.asarray(y_train)
            y_te = np.asarray(y_test)

            is_classification = "classification" in problem_type

            candidates = self._get_candidates(is_classification)

            leaderboard: List[Dict[str, Any]] = []
            all_models: Dict[str, Any] = {}

            for name, model in candidates:
                t0 = time.time()
                try:
                    model.fit(X_tr, y_tr)
                    score = self._evaluate(model, X_te, y_te, is_classification)
                    train_time = round(time.time() - t0, 3)
                    params = self._safe_params(model)
                    leaderboard.append({
                        "name": name,
                        "score": round(score, 6),
                        "params": params,
                        "train_time": train_time,
                    })
                    all_models[name] = model
                    self.logger.info("  %s – score=%.4f (%.2fs)", name, score, train_time)
                except Exception as exc:
                    self.logger.warning("  %s failed: %s", name, exc)

            if not leaderboard:
                return AgentResult(success=False, errors=["All candidate models failed to train"])

            leaderboard.sort(key=lambda x: x["score"], reverse=True)

            if use_optuna and _HAS_OPTUNA and len(leaderboard) >= 2:
                top_names = [e["name"] for e in leaderboard[:2]]
                for tname in top_names:
                    tuned_model, tuned_score, tuned_params, tune_time = self._optuna_tune(
                        tname, X_tr, y_tr, X_te, y_te, is_classification,
                    )
                    if tuned_model is not None:
                        entry_name = f"{tname}_tuned"
                        leaderboard.append({
                            "name": entry_name,
                            "score": round(tuned_score, 6),
                            "params": tuned_params,
                            "train_time": tune_time,
                        })
                        all_models[entry_name] = tuned_model
                        self.logger.info("  %s (tuned) – score=%.4f", tname, tuned_score)

                leaderboard.sort(key=lambda x: x["score"], reverse=True)

            champion_entry = leaderboard[0]
            champion_name = champion_entry["name"]
            champion_model = all_models[champion_name]

            cv_score = self._cross_validate(champion_model, X_tr, y_tr, is_classification)
            champion_entry["cv_score"] = round(cv_score, 6)

            self.logger.info(
                "Champion: %s – score=%.4f, cv=%.4f",
                champion_name, champion_entry["score"], cv_score,
            )

            return AgentResult(
                success=True,
                data={
                    "champion_model": champion_model,
                    "champion_name": champion_name,
                    "champion_params": champion_entry["params"],
                    "champion_score": champion_entry["score"],
                    "leaderboard": leaderboard,
                    "all_models": all_models,
                },
                metadata={
                    "problem_type": problem_type,
                    "feature_names": feature_names,
                    "optuna_available": _HAS_OPTUNA,
                    "xgboost_available": _HAS_XGB,
                    "lightgbm_available": _HAS_LGB,
                },
            )

        except Exception as exc:
            self.logger.error("ModelingMLAgent failed: %s", exc, exc_info=True)
            return AgentResult(success=False, errors=[str(exc)])

    def _get_candidates(self, is_classification: bool) -> List[Tuple[str, Any]]:
        """Return list of (name, model) pairs for the problem type."""
        if is_classification:
            candidates: List[Tuple[str, Any]] = [
                ("LogisticRegression", LogisticRegression(max_iter=1000, random_state=42)),
                ("RandomForest", RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42)),
                ("GradientBoosting", GradientBoostingClassifier(n_estimators=100, max_depth=5, random_state=42)),
            ]
            if _HAS_XGB:
                candidates.append(("XGBoost", XGBClassifier(
                    n_estimators=100, max_depth=6, learning_rate=0.1,
                    use_label_encoder=False, eval_metric="logloss", random_state=42, verbosity=0,
                )))
            if _HAS_LGB:
                candidates.append(("LightGBM", LGBMClassifier(
                    n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42, verbose=-1,
                )))
        else:
            candidates = [
                ("LinearRegression", LinearRegression()),
                ("RandomForest", RandomForestRegressor(n_estimators=100, max_depth=8, random_state=42)),
                ("GradientBoosting", GradientBoostingRegressor(n_estimators=100, max_depth=5, random_state=42)),
            ]
            if _HAS_XGB:
                candidates.append(("XGBoost", XGBRegressor(
                    n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42, verbosity=0,
                )))
            if _HAS_LGB:
                candidates.append(("LightGBM", LGBMRegressor(
                    n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42, verbose=-1,
                )))
        return candidates

    def _evaluate(self, model: Any, X_test: np.ndarray, y_test: np.ndarray, is_classification: bool) -> float:
        """Return a single scalar score (higher is better)."""
        preds = model.predict(X_test)
        if is_classification:
            return float(f1_score(y_test, preds, average="weighted", zero_division=0))
        else:
            rmse = float(np.sqrt(mean_squared_error(y_test, preds)))
            r2 = float(r2_score(y_test, preds)) if len(np.unique(y_test)) > 1 else 0.0
            return r2

    def _cross_validate(self, model: Any, X: np.ndarray, y: np.ndarray, is_classification: bool) -> float:
        """5-fold cross-validation score."""
        try:
            scoring = "f1_weighted" if is_classification else "r2"
            n_folds = min(5, len(X))
            if is_classification:
                unique_counts = np.bincount(y.astype(int)) if np.issubdtype(y.dtype, np.integer) else None
                if unique_counts is not None:
                    n_folds = min(n_folds, int(unique_counts.min()))
                n_folds = max(2, n_folds)
            scores = cross_val_score(model, X, y, cv=n_folds, scoring=scoring, error_score="raise")
            return float(scores.mean())
        except Exception as exc:
            self.logger.warning("Cross-validation failed: %s", exc)
            return 0.0

    def _optuna_tune(
        self,
        model_name: str,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        is_classification: bool,
    ) -> Tuple[Optional[Any], float, Dict[str, Any], float]:
        """Tune a model using Optuna (max 30 trials).

        Returns (model, score, params, train_time) or (None, 0, {}, 0) on failure.
        """
        t0 = time.time()
        try:
            def objective(trial: "optuna.Trial") -> float:
                params = self._suggest_params(trial, model_name, is_classification)
                model = self._build_model(model_name, params, is_classification)
                model.fit(X_train, y_train)
                return self._evaluate(model, X_test, y_test, is_classification)

            study = optuna.create_study(direction="maximize")
            study.optimize(objective, n_trials=30, timeout=120, show_progress_bar=False)

            best_params = study.best_params
            best_model = self._build_model(model_name, best_params, is_classification)
            best_model.fit(X_train, y_train)
            best_score = self._evaluate(best_model, X_test, y_test, is_classification)
            train_time = round(time.time() - t0, 3)
            return best_model, best_score, best_params, train_time

        except Exception as exc:
            self.logger.warning("Optuna tuning failed for %s: %s", model_name, exc)
            return None, 0.0, {}, round(time.time() - t0, 3)

    def _suggest_params(self, trial: "optuna.Trial", model_name: str, is_classification: bool) -> Dict[str, Any]:
        """Suggest hyper-parameters for an Optuna trial."""
        if model_name in ("RandomForest",):
            return {
                "n_estimators": trial.suggest_int("n_estimators", 50, 300),
                "max_depth": trial.suggest_int("max_depth", 3, 15),
                "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
            }
        elif model_name in ("GradientBoosting",):
            return {
                "n_estimators": trial.suggest_int("n_estimators", 50, 300),
                "max_depth": trial.suggest_int("max_depth", 3, 10),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            }
        elif model_name in ("XGBoost",):
            return {
                "n_estimators": trial.suggest_int("n_estimators", 50, 300),
                "max_depth": trial.suggest_int("max_depth", 3, 10),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            }
        elif model_name in ("LightGBM",):
            return {
                "n_estimators": trial.suggest_int("n_estimators", 50, 300),
                "max_depth": trial.suggest_int("max_depth", 3, 12),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "num_leaves": trial.suggest_int("num_leaves", 15, 127),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            }
        elif model_name in ("LogisticRegression",):
            return {
                "C": trial.suggest_float("C", 1e-4, 100.0, log=True),
                "max_iter": 1000,
            }
        else:
            return {}

    def _build_model(self, model_name: str, params: Dict[str, Any], is_classification: bool) -> Any:
        """Instantiate a model from name + params."""
        if model_name == "RandomForest":
            cls = RandomForestClassifier if is_classification else RandomForestRegressor
            return cls(random_state=42, **params)
        elif model_name == "GradientBoosting":
            cls = GradientBoostingClassifier if is_classification else GradientBoostingRegressor
            return cls(random_state=42, **params)
        elif model_name == "XGBoost" and _HAS_XGB:
            if is_classification:
                return XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=42, verbosity=0, **params)
            return XGBRegressor(random_state=42, verbosity=0, **params)
        elif model_name == "LightGBM" and _HAS_LGB:
            if is_classification:
                return LGBMClassifier(random_state=42, verbose=-1, **params)
            return LGBMRegressor(random_state=42, verbose=-1, **params)
        elif model_name == "LogisticRegression":
            return LogisticRegression(random_state=42, **params)
        elif model_name == "LinearRegression":
            return LinearRegression()
        else:
            raise ValueError(f"Unknown model: {model_name}")

    @staticmethod
    def _safe_params(model: Any) -> Dict[str, Any]:
        """Extract serialisable parameters from a model."""
        try:
            params = model.get_params()
            return {k: v for k, v in params.items() if isinstance(v, (int, float, str, bool, type(None)))}
        except Exception:
            return {}
