from typing import Type
from app.agents.base_agent import BaseAgent

class AgentRegistry:
    """
    Centralized registry tracking active agents in the MCP pipeline topology.
    """
    _agents: dict[str, BaseAgent] = {}

    @classmethod
    def register(cls, agent: BaseAgent) -> None:
        """Registers an instantiated agent instance."""
        cls._agents[agent.name] = agent

    @classmethod
    def get(cls, name: str) -> BaseAgent:
        """Resolves an agent by name ensuring correct orchestration bounds."""
        if name not in cls._agents:
            raise KeyError(f"Agent '{name}' is not registered in the pipeline.")
        return cls._agents[name]

    @classmethod
    def clear(cls) -> None:
        """Wipes registry bindings (useful during testing lifecycle)."""
        cls._agents.clear()
