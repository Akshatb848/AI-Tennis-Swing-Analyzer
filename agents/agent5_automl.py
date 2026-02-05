from utils.llm_client import query_llm
from utils.db_manager import save_df_to_table

class AutoMLAgent:
    def __init__(self, project_id: str, proposal: str = ""):
        self.project_id = project_id
        self.proposal = proposal

    def run(self) -> str:
        # Use provided LLM proposal or generate
        if not self.proposal:
            df = self._load_df()
            prompt = f"Propose ML/DL/RL model for data: {df.head().to_string()}. Rationale:"
            self.proposal = query_llm(prompt)

        # Recommendations (e.g., AutoGluon handles ML/DL)
        recs = {"proposal": self.proposal, "framework": "AutoGluon for auto-implementation"}
        save_df_to_table(pd.DataFrame([recs]), "model_recs", self.project_id)
        return "model_recs/latest"
