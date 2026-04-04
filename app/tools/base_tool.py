import time
from typing import Any, Dict, Optional
from pydantic import BaseModel, Field

class ToolResult(BaseModel):
    """
    Standardized payload formatting responses crossing across executable tool bounds.
    Replaces brittle string logging ensuring downstream pipelines validate formats natively.
    """
    tool: str = Field(..., description="The name of the tool executed (e.g. 'file', 'git', 'log')")
    action: str = Field(..., description="The exact action triggered (e.g. 'read_file', 'git_blame')")
    success: bool = Field(..., description="Whether the tool safely executed the requested action.")
    data: Dict[str, Any] = Field(default_factory=dict, description="Standard structured data retrieved by the tool.")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Metrics surrounding execution (size restrictions, latencies, etc).")
    error: Optional[str] = Field(None, description="Explicit error payloads if the execution hit exceptions over boundaries.")
    timestamp: float = Field(default_factory=time.time, description="Epoch timestamp routing recency integrations dynamically.")

class Tool:
    """
    Root Abstract Base Interface wrapping executable channels safely bypassing
    RAG chunk distances utilizing exact system lookups.
    """
    name: str = "BaseTool"
    description: str = "Root representation of an executable data channel."

    async def execute(self, **kwargs) -> ToolResult:
        """
        Subclasses map specific domain actions overriding this wrapper.
        Always returns a standardized Pydantic ToolResult validating boundaries!
        """
        raise NotImplementedError(f"Tool {self.name} has not implemented an 'execute' method.")
