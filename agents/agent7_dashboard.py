import pandas as pd
from utils.db_manager import load_table

class DashboardAgent:
    def __init__(self, project_id: str):
        self.project_id = project_id

    def run(self):
        return {
            "project": load_table("project_metadata", self.project_id),
            "agents": load_table("agent_status", self.project_id),
            "features": load_table("feature_store", self.project_id),
            "models": load_table("model_results", self.project_id),
        }
