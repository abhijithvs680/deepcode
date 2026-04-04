from typing import Dict, Optional
from app.tools.base_tool import Tool
from app.tools.file_tool import FileTool
from app.tools.git_tool import GitTool
from app.tools.log_tool import LogTool
import os

class ToolRegistry:
    """
    Centralized mapping holding safe execution endpoints securely restricted over initialized arguments.
    """
    def __init__(self, workspace_root: str = os.getcwd()):
        self._tools: Dict[str, Tool] = {}
        
        # Auto-bind standard channels
        ft = FileTool(workspace_root)
        gt = GitTool(workspace_root)
        lt = LogTool(workspace_root)
        
        self._tools[ft.name] = ft
        self._tools[gt.name] = gt
        self._tools[lt.name] = lt

    def get_tool(self, name: str) -> Optional[Tool]:
        return self._tools.get(name)

    async def execute_tool(self, name: str, action: str, **kwargs):
        t = self.get_tool(name)
        if not t:
            from app.tools.base_tool import ToolResult
            return ToolResult(tool=name, action=action, success=False, error=f"Tool '{name}' not found.")
        return await t.execute(action, **kwargs)
