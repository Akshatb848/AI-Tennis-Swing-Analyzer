"""
Coordinator Agent - LLM-Powered Master Orchestrator
"""

import asyncio
import json
import logging
import re
from datetime import datetime
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field

from .base_agent import BaseAgent, AgentState, TaskResult, generate_uuid

logger = logging.getLogger(__name__)


@dataclass
class WorkflowStep:
    """Represents a single step in the data science workflow"""
    id: str
    name: str
    agent: str
    task: Dict[str, Any]
    dependencies: List[str] = field(default_factory=list)
    status: str = "pending"
    result: Optional[TaskResult] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "agent": self.agent,
            "task": self.task,
            "dependencies": self.dependencies,
            "status": self.status,
            "result": self.result.to_dict() if self.result else None
        }


@dataclass
class Workflow:
    """Complete workflow definition"""
    id: str
    name: str
    project_id: str
    steps: List[WorkflowStep] = field(default_factory=list)
    status: str = "created"
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "project_id": self.project_id,
            "steps": [step.to_dict() for step in self.steps],
            "status": self.status,
            "created_at": self.created_at.isoformat() if isinstance(self.created_at, datetime) else self.created_at,
            "updated_at": self.updated_at.isoformat() if isinstance(self.updated_at, datetime) else self.updated_at,
            "metadata": self.metadata
        }


class CoordinatorAgent(BaseAgent):
    """Master orchestrator that coordinates all specialized agents."""
    
    def __init__(self, llm_client=None, agent_registry: Optional[Dict[str, BaseAgent]] = None):
        super().__init__(
            name="Coordinator",
            description="Master orchestrator for data science workflow coordination",
            capabilities=[
                "workflow_planning", "task_distribution", "agent_coordination",
                "user_intent_understanding", "progress_tracking", "error_recovery"
            ]
        )
        self.llm_client = llm_client
        self.agent_registry: Dict[str, BaseAgent] = agent_registry or {}
        self.active_workflows: Dict[str, Workflow] = {}
        self.projects: Dict[str, Dict[str, Any]] = {}
        self.conversation_history: List[Dict[str, str]] = []
        
        self.command_patterns = {
            "new_project": [r"(?:create|make|new|start)\s+(?:a\s+)?(?:new\s+)?project", r"new\s+project"],
            "load_project": [r"(?:load|open|use|select)\s+project"],
            "upload_dataset": [r"(?:upload|add|import|load)\s+(?:a\s+)?(?:data|dataset|file)", r"(?:enter|create)\s+dataset"],
            "proceed": [r"proceed", r"continue", r"go\s+ahead", r"start\s+(?:analysis|processing)", r"run", r"execute"],
            "status": [r"(?:show|get|what\'?s?\s+(?:the)?)\s+status", r"status"],
            "help": [r"help", r"what\s+can\s+you\s+do", r"commands"]
        }
    
    def get_system_prompt(self) -> str:
        return """You are the Coordinator Agent - an expert AI data scientist orchestrator."""

    def register_agent(self, agent: BaseAgent):
        self.agent_registry[agent.name] = agent
        logger.info(f"Registered agent: {agent.name}")
    
    def create_project(self, name: str, description: str = "", config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        project_id = generate_uuid()[:8]
        project = {
            "id": project_id,
            "name": name,
            "description": description,
            "config": config or {},
            "datasets": [],
            "workflows": [],
            "models": [],
            "artifacts": [],
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "status": "created"
        }
        self.projects[project_id] = project
        logger.info(f"Created project: {name} (ID: {project_id})")
        return project
    
    def add_dataset_to_project(self, project_id: str, dataset_info: Dict[str, Any]) -> bool:
        if project_id not in self.projects:
            return False
        self.projects[project_id]["datasets"].append(dataset_info)
        self.projects[project_id]["updated_at"] = datetime.now().isoformat()
        return True
    
    async def analyze_user_intent(self, user_message: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        user_message_lower = user_message.lower().strip()
        for command, patterns in self.command_patterns.items():
            for pattern in patterns:
                if re.search(pattern, user_message_lower):
                    return {"intent": command, "confidence": 0.95, "raw_message": user_message, "context": context}
        return {"intent": "general_query", "confidence": 0.5, "raw_message": user_message, "context": context}
    
    async def plan_workflow(self, project_id: str, dataset_info: Dict[str, Any], 
                           user_requirements: Optional[Dict[str, Any]] = None) -> Workflow:
        workflow_id = generate_uuid()[:8]
        target_column = dataset_info.get("target_column")
        
        steps = []
        step_counter = 1
        
        # Data Cleaning
        steps.append(WorkflowStep(
            id=f"step_{step_counter}",
            name="Data Cleaning & Preprocessing",
            agent="DataCleanerAgent",
            task={"action": "clean_data", "dataset_id": dataset_info.get("id")},
            dependencies=[]
        ))
        step_counter += 1
        
        # EDA
        steps.append(WorkflowStep(
            id=f"step_{step_counter}",
            name="Exploratory Data Analysis",
            agent="EDAAgent",
            task={"action": "full_eda", "dataset_id": dataset_info.get("id")},
            dependencies=[f"step_{step_counter - 1}"]
        ))
        step_counter += 1
        
        # Visualization
        steps.append(WorkflowStep(
            id=f"step_{step_counter}",
            name="Data Visualization",
            agent="DataVisualizerAgent",
            task={"action": "generate_visualizations", "dataset_id": dataset_info.get("id")},
            dependencies=[f"step_{step_counter - 1}"]
        ))
        step_counter += 1
        
        if target_column:
            # Feature Engineering
            steps.append(WorkflowStep(
                id=f"step_{step_counter}",
                name="Feature Engineering",
                agent="FeatureEngineerAgent",
                task={"action": "engineer_features", "target_column": target_column},
                dependencies=[f"step_{step_counter - 1}"]
            ))
            step_counter += 1
            
            # AutoML
            steps.append(WorkflowStep(
                id=f"step_{step_counter}",
                name="AutoML Model Selection",
                agent="AutoMLAgent",
                task={"action": "auto_select_models", "target_column": target_column},
                dependencies=[f"step_{step_counter - 1}"]
            ))
            step_counter += 1
            
            # Model Training
            steps.append(WorkflowStep(
                id=f"step_{step_counter}",
                name="Model Training & Evaluation",
                agent="ModelTrainerAgent",
                task={"action": "train_models", "target_column": target_column},
                dependencies=[f"step_{step_counter - 1}"]
            ))
            step_counter += 1
        
        # Dashboard
        steps.append(WorkflowStep(
            id=f"step_{step_counter}",
            name="Dashboard Generation",
            agent="DashboardBuilderAgent",
            task={"action": "build_dashboard", "project_id": project_id},
            dependencies=[f"step_{step_counter - 1}"]
        ))
        
        workflow = Workflow(
            id=workflow_id,
            name=f"workflow_{workflow_id}",
            project_id=project_id,
            steps=steps,
            metadata={"target_column": target_column, "user_requirements": user_requirements}
        )
        self.active_workflows[workflow_id] = workflow
        return workflow
    
    async def execute_workflow(self, workflow_id: str, progress_callback=None) -> Dict[str, Any]:
        if workflow_id not in self.active_workflows:
            return {"success": False, "error": "Workflow not found"}
        
        workflow = self.active_workflows[workflow_id]
        workflow.status = "running"
        results = {}
        completed_steps = set()
        
        for step in workflow.steps:
            if not all(dep in completed_steps for dep in step.dependencies):
                step.status = "skipped"
                continue
            
            step.status = "running"
            if progress_callback:
                await progress_callback({
                    "workflow_id": workflow_id, "step_id": step.id,
                    "step_name": step.name, "status": "running",
                    "progress": len(completed_steps) / len(workflow.steps) * 100
                })
            
            try:
                agent = self.agent_registry.get(step.agent)
                if agent:
                    result = await agent.run(step.task)
                    step.result = result
                    step.status = "completed" if result.success else "failed"
                    if result.success:
                        completed_steps.add(step.id)
                    results[step.id] = result.to_dict()
                else:
                    step.status = "completed"
                    completed_steps.add(step.id)
                    results[step.id] = {"success": True, "data": f"Simulated: {step.name}"}
            except Exception as e:
                step.status = "failed"
                results[step.id] = {"success": False, "error": str(e)}
            
            if progress_callback:
                await progress_callback({
                    "workflow_id": workflow_id, "step_id": step.id,
                    "step_name": step.name, "status": step.status,
                    "progress": len(completed_steps) / len(workflow.steps) * 100
                })
        
        workflow.status = "completed" if all(s.status in ["completed", "skipped"] for s in workflow.steps) else "failed"
        workflow.updated_at = datetime.now()
        
        return {
            "success": workflow.status == "completed",
            "workflow_id": workflow_id,
            "status": workflow.status,
            "results": results,
            "completed_steps": len(completed_steps),
            "total_steps": len(workflow.steps)
        }
    
    async def execute(self, task: Dict[str, Any]) -> TaskResult:
        action = task.get("action", "")
        
        try:
            if action == "create_project":
                project = self.create_project(
                    name=task.get("name", "Untitled"),
                    description=task.get("description", ""),
                    config=task.get("config")
                )
                return TaskResult(success=True, data=project)
            
            elif action == "plan_workflow":
                workflow = await self.plan_workflow(
                    project_id=task.get("project_id"),
                    dataset_info=task.get("dataset_info", {}),
                    user_requirements=task.get("requirements")
                )
                return TaskResult(success=True, data=workflow.to_dict())
            
            elif action == "execute_workflow":
                result = await self.execute_workflow(
                    workflow_id=task.get("workflow_id"),
                    progress_callback=task.get("progress_callback")
                )
                return TaskResult(success=result["success"], data=result)
            
            elif action == "get_status":
                return TaskResult(success=True, data={
                    "projects": list(self.projects.keys()),
                    "active_workflows": list(self.active_workflows.keys()),
                    "registered_agents": list(self.agent_registry.keys())
                })
            
            return TaskResult(success=False, error=f"Unknown action: {action}")
        except Exception as e:
            return TaskResult(success=False, error=str(e))
    
    def get_welcome_message(self) -> str:
        return """
╔══════════════════════════════════════════════════════════════════╗
║           🔬 Data Science Agent Platform 🔬                       ║
╠══════════════════════════════════════════════════════════════════╣
║  Step 1: "create project" or "make new project"                  ║
║  Step 2: "upload dataset" or "enter dataset"                     ║
║  Step 3: "proceed" or "start analysis"                           ║
║                                                                  ║
║  Type "help" for more commands!                                  ║
╚══════════════════════════════════════════════════════════════════╝
"""
    
    def get_help_message(self) -> str:
        return """
📚 Commands:
• "create project [name]" - Create new project
• "upload dataset" - Upload data
• "proceed" - Start analysis
• "status" - Show status
• "help" - Show help
"""
