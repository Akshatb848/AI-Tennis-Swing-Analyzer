"""
Base Agent Class - Foundation for all specialized agents
Implements common functionality, state management, and communication protocols
"""

import asyncio
import logging
import json
import uuid
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass, field
from enum import Enum
import traceback

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AgentState(Enum):
    """Agent execution states"""
    IDLE = "idle"
    RUNNING = "running"
    WAITING = "waiting"
    COMPLETED = "completed"
    FAILED = "failed"
    PAUSED = "paused"


class MessageType(Enum):
    """Types of messages exchanged between agents"""
    TASK = "task"
    RESULT = "result"
    ERROR = "error"
    STATUS = "status"
    DATA = "data"
    QUERY = "query"
    RESPONSE = "response"
    CHECKPOINT = "checkpoint"


def generate_uuid() -> str:
    """Generate a valid UUID string"""
    return str(uuid.uuid4())


def is_valid_uuid(val: Any) -> bool:
    """Check if a value is a valid UUID"""
    if val is None:
        return False
    try:
        uuid.UUID(str(val))
        return True
    except (ValueError, AttributeError, TypeError):
        return False


@dataclass
class AgentMessage:
    """Message structure for inter-agent communication"""
    id: str = field(default_factory=generate_uuid)
    sender: str = ""
    receiver: str = ""
    message_type: MessageType = MessageType.TASK
    content: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    priority: int = 1
    requires_response: bool = False
    parent_message_id: Optional[str] = None  # String, not UUID type
    
    def __post_init__(self):
        """Validate and convert fields after initialization"""
        # Ensure id is a valid UUID string
        if not is_valid_uuid(self.id):
            self.id = generate_uuid()
        
        # Ensure parent_message_id is valid if provided
        if self.parent_message_id is not None and not is_valid_uuid(self.parent_message_id):
            self.parent_message_id = None
        
        # Ensure timestamp is a string
        if isinstance(self.timestamp, datetime):
            self.timestamp = self.timestamp.isoformat()
        
        # Ensure message_type is correct type
        if isinstance(self.message_type, str):
            try:
                self.message_type = MessageType(self.message_type)
            except ValueError:
                self.message_type = MessageType.TASK
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "sender": self.sender,
            "receiver": self.receiver,
            "message_type": self.message_type.value if isinstance(self.message_type, MessageType) else str(self.message_type),
            "content": self.content,
            "timestamp": self.timestamp,
            "priority": self.priority,
            "requires_response": self.requires_response,
            "parent_message_id": self.parent_message_id
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AgentMessage":
        message_type = data.get("message_type", "task")
        if isinstance(message_type, str):
            try:
                message_type = MessageType(message_type)
            except ValueError:
                message_type = MessageType.TASK
        
        return cls(
            id=data.get("id") or generate_uuid(),
            sender=data.get("sender", ""),
            receiver=data.get("receiver", ""),
            message_type=message_type,
            content=data.get("content") or {},
            timestamp=data.get("timestamp") or datetime.now().isoformat(),
            priority=data.get("priority", 1),
            requires_response=data.get("requires_response", False),
            parent_message_id=data.get("parent_message_id")
        )


@dataclass
class TaskResult:
    """Result structure for agent task execution"""
    success: bool
    data: Any = None
    error: Optional[str] = None
    metrics: Dict[str, Any] = field(default_factory=dict)
    artifacts: List[str] = field(default_factory=list)
    execution_time: float = 0.0
    agent_name: str = ""
    task_id: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        serializable_data = self.data
        if self.data is not None:
            if not isinstance(self.data, (dict, list, str, int, float, bool, type(None))):
                try:
                    serializable_data = str(self.data)
                except Exception:
                    serializable_data = "Non-serializable data"
        
        return {
            "success": self.success,
            "data": serializable_data,
            "error": self.error,
            "metrics": self.metrics,
            "artifacts": self.artifacts,
            "execution_time": self.execution_time,
            "agent_name": self.agent_name,
            "task_id": self.task_id
        }


class BaseAgent(ABC):
    """
    Abstract base class for all agents in the system.
    Provides common functionality for state management, logging, and communication.
    """
    
    def __init__(
        self,
        name: str,
        description: str = "",
        capabilities: Optional[List[str]] = None,
        max_retries: int = 3,
        timeout: int = 300
    ):
        self.id = generate_uuid()
        self.name = name
        self.description = description
        self.capabilities = capabilities if capabilities is not None else []
        self.max_retries = max_retries
        self.timeout = timeout
        
        self.state = AgentState.IDLE
        self.current_task: Optional[Dict[str, Any]] = None
        self.message_queue: asyncio.Queue = asyncio.Queue()
        self.history: List[Dict[str, Any]] = []
        self.checkpoints: List[Dict[str, Any]] = []
        self.created_at = datetime.now()
        self.last_active = datetime.now()
        
        self._on_state_change: Optional[Callable] = None
        self._on_task_complete: Optional[Callable] = None
        self._on_error: Optional[Callable] = None
        
        logger.info(f"Agent '{self.name}' initialized with ID: {self.id}")
    
    @abstractmethod
    async def execute(self, task: Dict[str, Any]) -> TaskResult:
        """Execute the main task of the agent."""
        pass
    
    @abstractmethod
    def get_system_prompt(self) -> str:
        """Return the system prompt for this agent's LLM interactions."""
        pass
    
    async def run(self, task: Dict[str, Any]) -> TaskResult:
        """Run the agent with retry logic and error handling."""
        task_id = task.get("id") or generate_uuid()
        start_time = datetime.now()
        
        self._set_state(AgentState.RUNNING)
        self.current_task = task
        self.last_active = datetime.now()
        
        for attempt in range(self.max_retries):
            try:
                logger.info(f"Agent '{self.name}' executing task {task_id} (attempt {attempt + 1}/{self.max_retries})")
                
                result = await asyncio.wait_for(
                    self.execute(task),
                    timeout=self.timeout
                )
                
                result.execution_time = (datetime.now() - start_time).total_seconds()
                result.agent_name = self.name
                result.task_id = task_id
                
                self._log_execution(task, result)
                self._set_state(AgentState.COMPLETED)
                self.current_task = None
                
                if self._on_task_complete:
                    try:
                        await self._on_task_complete(result)
                    except Exception as e:
                        logger.warning(f"Callback error: {e}")
                
                return result
                
            except asyncio.TimeoutError:
                logger.warning(f"Agent '{self.name}' timeout on attempt {attempt + 1}")
                if attempt == self.max_retries - 1:
                    self._set_state(AgentState.FAILED)
                    return TaskResult(
                        success=False,
                        error=f"Task timed out after {self.timeout} seconds",
                        execution_time=(datetime.now() - start_time).total_seconds(),
                        agent_name=self.name,
                        task_id=task_id
                    )
                    
            except Exception as e:
                logger.error(f"Agent '{self.name}' error: {str(e)}")
                logger.error(traceback.format_exc())
                
                if attempt == self.max_retries - 1:
                    self._set_state(AgentState.FAILED)
                    if self._on_error:
                        try:
                            await self._on_error(e, task)
                        except Exception:
                            pass
                    return TaskResult(
                        success=False,
                        error=str(e),
                        execution_time=(datetime.now() - start_time).total_seconds(),
                        agent_name=self.name,
                        task_id=task_id
                    )
                
                await asyncio.sleep(2 ** attempt)
        
        return TaskResult(success=False, error="Unknown error", agent_name=self.name, task_id=task_id)
    
    def _set_state(self, new_state: AgentState):
        old_state = self.state
        self.state = new_state
        logger.debug(f"Agent '{self.name}' state: {old_state.value} -> {new_state.value}")
    
    def _log_execution(self, task: Dict[str, Any], result: TaskResult):
        self.history.append({
            "timestamp": datetime.now().isoformat(),
            "task": task,
            "result": result.to_dict(),
            "state": self.state.value
        })
    
    def save_checkpoint(self, checkpoint_data: Optional[Dict[str, Any]] = None) -> str:
        checkpoint_id = generate_uuid()
        self.checkpoints.append({
            "id": checkpoint_id,
            "timestamp": datetime.now().isoformat(),
            "agent_name": self.name,
            "state": self.state.value,
            "current_task": self.current_task,
            "custom_data": checkpoint_data or {}
        })
        return checkpoint_id
    
    def restore_checkpoint(self, checkpoint_id: str) -> bool:
        for cp in self.checkpoints:
            if cp["id"] == checkpoint_id:
                self.current_task = cp.get("current_task")
                return True
        return False
    
    async def send_message(self, receiver: str, content: Dict[str, Any], 
                          message_type: MessageType = MessageType.DATA) -> AgentMessage:
        return AgentMessage(sender=self.name, receiver=receiver, 
                          message_type=message_type, content=content)
    
    async def receive_message(self) -> Optional[AgentMessage]:
        try:
            return await asyncio.wait_for(self.message_queue.get(), timeout=1.0)
        except asyncio.TimeoutError:
            return None
    
    def get_status(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "state": self.state.value,
            "current_task": self.current_task,
            "last_active": self.last_active.isoformat(),
            "history_count": len(self.history),
            "checkpoint_count": len(self.checkpoints)
        }
    
    def register_callbacks(self, on_state_change: Optional[Callable] = None,
                          on_task_complete: Optional[Callable] = None,
                          on_error: Optional[Callable] = None):
        self._on_state_change = on_state_change
        self._on_task_complete = on_task_complete
        self._on_error = on_error
    
    def __repr__(self) -> str:
        return f"Agent(name='{self.name}', state={self.state.value})"
