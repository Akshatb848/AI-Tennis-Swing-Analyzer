import streamlit as st
from streamlit_option_menu import option_menu
from streamlit_aggrid import AgGrid
import plotly.express as px
from utils.db_manager import load_df

class DashboardAgent:
    def __init__(self, project_id: str):
        self.project_id = project_id

    def run(self) -> dict:
        # Load all data
        project = load_df(self.project_id, "project_metadata")
        agents = load_df(self.project_id, "agent_status")
        features = load_df(self.project_id, "feature_store")
        models = load_df(self.project_id, "model_results")

        # Interactive components (better than PowerBI: code-dynamic, real-time)
        data = {
            "project": project,
            "agents": agents,
            "features": features,
            "models": models
        }
        return data  # app.py renders

    # Note: Rendering happens in app.py for interactivity
