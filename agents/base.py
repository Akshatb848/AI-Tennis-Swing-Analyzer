import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List

from config.settings import LOG_FORMAT


@dataclass
class AgentResult:
    """Standardised result returned by every agent execution."""

    success: bool
    data: Dict[str, Any] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class BaseAgent(ABC):
    """Abstract base class that every agent must implement."""

    def __init__(self) -> None:
        self.logger = logging.getLogger(self.name)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter(LOG_FORMAT))
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique identifier for this agent."""
        ...

    @abstractmethod
    def execute(self, **kwargs: Any) -> AgentResult:
        """Run the agent's main logic.

        Args:
            **kwargs: Arbitrary keyword arguments specific to each agent.

        Returns:
            AgentResult: The outcome of the execution.
        """
        ...

    def validate_input(self, required_keys: List[str], data: Dict[str, Any]) -> List[str]:
        """Check that *data* contains every key listed in *required_keys*.

        Args:
            required_keys: Keys that must be present.
            data: The input dictionary to validate.

        Returns:
            A list of missing-key error messages (empty when valid).
        """
        missing = [k for k in required_keys if k not in data]
        errors = [f"Missing required key: {k}" for k in missing]
        if errors:
            self.logger.warning("Input validation failed: %s", errors)
        return errors
