from abc import ABC, abstractmethod
from typing import Any

class BaseAgent(ABC):
    """
    Core abstraction for all specialized agents within the Multi-Agent pipeline.
    Agents interact via a standardized `run()` signature, sharing state through
    a centralized dictionary.
    """

    def __init__(self, name: str):
        self.name = name
        self.input_queue: list[Any] = []
        self.output_queue: list[Any] = []

    @abstractmethod
    async def run(self, state: dict[str, Any]) -> dict[str, Any]:
        """
        Executes the agent's primary domain logic.
        
        Args:
            state (dict): The active orchestration state containing queries, traces, and subtasks.
            
        Returns:
            dict: The modified state dictionary updated with the agent's payload.
        """
        raise NotImplementedError(f"Agent '{self.name}' must implement `run`")
