"""MLOps Deployment Agent – serialises models and generates deployment artefacts."""

import os
import datetime
from typing import Any, Dict, List, Optional

from agents.base import AgentResult, BaseAgent


class MLOpsDeploymentAgent(BaseAgent):
    """Handles model persistence, inference script generation, monitoring
    configuration, and deployment recommendations.
    """

    MODELS_DIR = "models"

    @property
    def name(self) -> str:
        return "MLOpsDeploymentAgent"

    def execute(
        self,
        champion_model: Any,
        champion_name: str,
        feature_names: List[str],
        problem_type: str,
        model_metrics: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> AgentResult:
        """Package the champion model for deployment.

        Args:
            champion_model: Fitted scikit-learn-compatible estimator.
            champion_name: Human-readable model name.
            feature_names: Ordered list of feature names expected at inference.
            problem_type: ML problem type string.
            model_metrics: Optional dict of evaluation metrics.

        Returns:
            AgentResult with model_path, inference_script, monitoring_config,
            deployment_recommendations, and deployment_ready.
        """
        try:
            self.logger.info("MLOpsDeploymentAgent started – model=%s", champion_name)

            model_path = self._save_model(champion_model, champion_name)

            inference_script = self._generate_inference_script(
                champion_name, feature_names, problem_type, model_path,
            )

            monitoring_config = self._generate_monitoring_config(
                feature_names, problem_type, model_metrics,
            )

            deployment_recs = self._generate_deployment_recommendations(
                champion_name, problem_type, model_metrics,
            )

            deployment_ready = model_path is not None and os.path.exists(model_path)

            self.logger.info(
                "MLOpsDeploymentAgent complete – model_path=%s, ready=%s",
                model_path, deployment_ready,
            )

            return AgentResult(
                success=True,
                data={
                    "model_path": model_path,
                    "inference_script": inference_script,
                    "monitoring_config": monitoring_config,
                    "deployment_recommendations": deployment_recs,
                    "deployment_ready": deployment_ready,
                },
                metadata={
                    "champion_name": champion_name,
                    "problem_type": problem_type,
                    "timestamp": datetime.datetime.utcnow().isoformat(),
                },
            )

        except Exception as exc:
            self.logger.error("MLOpsDeploymentAgent failed: %s", exc, exc_info=True)
            return AgentResult(success=False, errors=[str(exc)])

    def _save_model(self, model: Any, model_name: str) -> str:
        """Serialise the model to disk using joblib."""
        import joblib

        os.makedirs(self.MODELS_DIR, exist_ok=True)
        ts = datetime.datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        safe_name = model_name.replace(" ", "_").replace("/", "_")
        filename = f"{safe_name}_{ts}.joblib"
        model_path = os.path.join(self.MODELS_DIR, filename)

        joblib.dump(model, model_path)
        self.logger.info("Model saved to %s", model_path)
        return model_path

    def _generate_inference_script(
        self,
        model_name: str,
        feature_names: List[str],
        problem_type: str,
        model_path: str,
    ) -> str:
        """Generate a standalone predict.py script as a string."""
        features_repr = repr(feature_names)
        is_classification = "classification" in problem_type

        script = f'''"""Auto-generated inference script for {model_name}.

Usage:
    python predict.py                       # interactive mode
    python predict.py --input data.csv      # batch mode
"""

import sys
import argparse
import json

import joblib
import numpy as np
import pandas as pd

MODEL_PATH = "{model_path}"
FEATURE_NAMES = {features_repr}
PROBLEM_TYPE = "{problem_type}"

model = joblib.load(MODEL_PATH)


def predict(input_data: dict) -> dict:
    """Run inference on a single record.

    Args:
        input_data: dict mapping feature name -> value.

    Returns:
        dict with 'prediction' key (and 'probability' for classification).
    """
    row = pd.DataFrame([input_data], columns=FEATURE_NAMES)
    row = row.fillna(0)
    prediction = model.predict(row)[0]

    result = {{"prediction": float(prediction) if not isinstance(prediction, (str, np.integer)) else int(prediction)}}

    {"if hasattr(model, 'predict_proba'):" if is_classification else "if False:"}
        {"probas = model.predict_proba(row)[0]" if is_classification else "pass"}
        {"result['probability'] = {int(c): round(float(p), 4) for c, p in enumerate(probas)}" if is_classification else ""}

    return result


def predict_batch(df: pd.DataFrame) -> pd.DataFrame:
    """Run inference on a DataFrame.

    Args:
        df: DataFrame with columns matching FEATURE_NAMES.

    Returns:
        DataFrame with a 'prediction' column appended.
    """
    for col in FEATURE_NAMES:
        if col not in df.columns:
            df[col] = 0
    X = df[FEATURE_NAMES].fillna(0)
    df["prediction"] = model.predict(X)
    return df


def main():
    parser = argparse.ArgumentParser(description="Model inference")
    parser.add_argument("--input", type=str, help="Path to input CSV for batch prediction")
    parser.add_argument("--output", type=str, default="predictions.csv", help="Output CSV path")
    args = parser.parse_args()

    if args.input:
        df = pd.read_csv(args.input)
        result_df = predict_batch(df)
        result_df.to_csv(args.output, index=False)
        print(f"Predictions saved to {{args.output}}")
    else:
        print("Interactive mode. Enter feature values (or 'quit' to exit):")
        print(f"Features: {{FEATURE_NAMES}}")
        while True:
            try:
                raw = input("\\nJSON input> ")
                if raw.strip().lower() in ("quit", "exit", "q"):
                    break
                data = json.loads(raw)
                result = predict(data)
                print(f"Result: {{json.dumps(result, indent=2)}}")
            except (json.JSONDecodeError, KeyboardInterrupt):
                break


if __name__ == "__main__":
    main()
'''
        return script

    def _generate_monitoring_config(
        self,
        feature_names: List[str],
        problem_type: str,
        model_metrics: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Generate a monitoring configuration dict."""
        drift_thresholds: Dict[str, Any] = {
            "psi_threshold": 0.2,
            "kl_divergence_threshold": 0.1,
            "feature_drift_check_interval_hours": 24,
            "monitored_features": feature_names[:20],
        }

        performance_thresholds: Dict[str, Any] = {}
        if model_metrics:
            for metric_name, value in model_metrics.items():
                if isinstance(value, (int, float)):
                    performance_thresholds[metric_name] = {
                        "baseline": round(float(value), 4),
                        "alert_threshold": round(float(value) * 0.9, 4),
                        "critical_threshold": round(float(value) * 0.8, 4),
                    }

        logging_config = {
            "log_predictions": True,
            "log_input_features": True,
            "log_latency": True,
            "sample_rate": 1.0 if len(feature_names) < 50 else 0.1,
            "log_format": "json",
            "log_destination": "stdout",
            "retention_days": 90,
        }

        alerting = {
            "channels": ["email", "slack"],
            "alert_on_drift": True,
            "alert_on_performance_degradation": True,
            "alert_on_high_latency_ms": 500,
            "alert_on_error_rate_pct": 5.0,
            "cooldown_minutes": 60,
        }

        return {
            "drift_thresholds": drift_thresholds,
            "performance_thresholds": performance_thresholds,
            "logging": logging_config,
            "alerting": alerting,
            "retraining": {
                "trigger": "drift_detected_or_scheduled",
                "schedule_cron": "0 2 * * 0",
                "min_samples_for_retrain": 1000,
            },
        }

    def _generate_deployment_recommendations(
        self,
        model_name: str,
        problem_type: str,
        model_metrics: Optional[Dict[str, Any]],
    ) -> List[str]:
        """Generate deployment platform recommendations."""
        recs: List[str] = []

        recs.append(
            "Streamlit Cloud: Wrap the inference script in a Streamlit app for "
            "interactive predictions with a simple UI. Free tier available for "
            "public apps."
        )

        recs.append(
            "Hugging Face Spaces: Deploy as a Gradio or Streamlit app on HF Spaces "
            "for easy sharing and community visibility. Supports free GPU for "
            "small models."
        )

        recs.append(
            "FastAPI + Docker: Containerise the model with FastAPI for a production-grade "
            "REST API. Deploy to any cloud provider (AWS ECS, GCP Cloud Run, Azure "
            "Container Apps)."
        )

        recs.append(
            "Replit Deployments: Use Replit's built-in deployment to publish the "
            "prediction API directly from this workspace with zero configuration."
        )

        recs.append(
            "MLflow / BentoML: For enterprise MLOps, register the model in MLflow "
            "or package with BentoML for versioning, A/B testing, and canary "
            "deployments."
        )

        if model_metrics:
            primary = list(model_metrics.values())[0] if model_metrics else None
            if isinstance(primary, (int, float)) and primary < 0.5:
                recs.append(
                    "CAUTION: Model performance is below typical production thresholds. "
                    "Consider additional feature engineering or data collection before "
                    "deploying to production."
                )

        recs.append(
            "CI/CD: Set up automated retraining and model validation in a CI/CD "
            "pipeline (GitHub Actions, GitLab CI) to keep the model fresh."
        )

        recs.append(
            "Monitoring: Integrate drift detection and performance monitoring "
            "(see monitoring_config) to catch data distribution shifts early."
        )

        return recs
