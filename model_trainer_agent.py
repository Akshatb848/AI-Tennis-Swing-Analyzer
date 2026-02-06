"""
Model Trainer Agent - Model training and evaluation
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List
from datetime import datetime
import time

from .base_agent import BaseAgent, TaskResult, generate_uuid

logger = logging.getLogger(__name__)


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
        
        cv_folds = task.get("cv_folds", 5)
        
        X = df.drop(columns=[target_column]).select_dtypes(include=[np.number]).fillna(0)
        y = df[target_column]
        
        task_type = "classification" if y.nunique() <= 10 else "regression"
        
        from sklearn.model_selection import train_test_split, cross_val_score
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        results = {}
        
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
                        "feature_importance": dict(zip(X.columns, model.feature_importances_.tolist())) if hasattr(model, 'feature_importances_') else {}
                    }
                    self.trained_models[name] = model
                except Exception as e:
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
                        "feature_importance": dict(zip(X.columns, model.feature_importances_.tolist())) if hasattr(model, 'feature_importances_') else {}
                    }
                    self.trained_models[name] = model
                except Exception as e:
                    logger.error(f"Error training {name}: {e}")
        
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
                "task_type": task_type
            },
            metrics={"models_trained": len(results), "best_model": best_name}
        )
    
    async def _train_single_model(self, task: Dict[str, Any]) -> TaskResult:
        return TaskResult(success=False, error="Not implemented")
