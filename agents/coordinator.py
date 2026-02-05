from utils.db_manager import (
    log_agent_start,
    log_agent_success,
    log_agent_failure,
)

class CoordinatorAgent:
    def __init__(self, project_id: str):
        self.project_id = project_id

    def run_agent(self, name, fn):
        log_agent_start(self.project_id, name)
        try:
            fn()
            log_agent_success(self.project_id, name)
        except Exception as e:
            log_agent_failure(self.project_id, name, str(e))
            raise

    def run_pipeline(self):
        print(f"[Coordinator] Running pipeline for project: {self.project_id}")

        from agents.agent3_warehouse import WarehouseAgent
        from agents.agent1_cleaner import DataCleanerAgent
        from agents.agent4_features import FeatureEngineeringAgent
        from agents.agent2_visualizer import VisualizerAgent
        from agents.agent6_model_training import ModelTrainingAgent

        self.run_agent("warehouse", WarehouseAgent(self.project_id).run)
        self.run_agent("data_cleaner", DataCleanerAgent(self.project_id).run)
        self.run_agent("feature_engineering", FeatureEngineeringAgent(self.project_id).run)
        self.run_agent("visualizer", VisualizerAgent(self.project_id).run)
        self.run_agent("model_training", ModelTrainingAgent(self.project_id).run)

        print("[Coordinator] Pipeline completed")
