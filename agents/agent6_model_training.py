from autogluon.tabular import TabularPredictor
import shap
from utils.db_manager import load_df, save_df_to_table

class ModelTrainingAgent:
    def __init__(self, project_id: str):
        self.project_id = project_id

    def run(self) -> str:
        df = load_df(self.project_id, "feature_store")  # Assume features ready
        label = 'target'  # Assume or detect
        predictor = TabularPredictor(label=label).fit(df)  # Auto ML/DL

        # Best model + metrics
        leaderboard = predictor.leaderboard()
        best_model = leaderboard.iloc[0]['model']
        metrics = {"rmse": leaderboard.iloc[0]['score_val'], "best_model": best_model}

        # SHAP explainability
        explainer = shap.Explainer(predictor)
        shap_values = explainer(df)
        shap.summary_plot(shap_values, df, show=False, file_name=f"artifacts/{self.project_id}/shap.png")

        save_df_to_table(pd.DataFrame([metrics]), "model_results", self.project_id)
        return "model_results/best"
