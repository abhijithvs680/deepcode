from __future__ import annotations

import os
from typing import TYPE_CHECKING, Dict, Optional

from app.tools.file_tool import FileTool
from app.tools.file_edit_tool import FileEditTool
from app.tools.git_tool import GitTool
from app.tools.log_tool import LogTool
from app.tools.base_tool import Tool

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession


class ToolRegistry:
    """
    Centralized mapping holding safe execution endpoints securely restricted over initialized arguments.
    """

    def __init__(
        self,
        workspace_root: str = os.getcwd(),
        db: Optional[AsyncSession] = None,
        repo_name: Optional[str] = None,
    ):
        self._tools: Dict[str, Tool] = {}
        self.workspace_root = workspace_root

        ft = FileTool(workspace_root, db=db, repo_name=repo_name)
        gt = GitTool(workspace_root)
        lt = LogTool(workspace_root)
        ed = FileEditTool(workspace_root)

        self._tools[ft.name] = ft
        self._tools[gt.name] = gt
        self._tools[lt.name] = lt
        self._tools[ed.name] = ed

    def get_tool(self, name: str) -> Optional[Tool]:
        """Return registered tool instance (e.g. ``file_edit`` for session finalize)."""
        return self._tools.get(name)

    async def execute_tool(self, name: str, action: str, **kwargs):
        t = self.get_tool(name)
        if not t:
            from app.tools.base_tool import ToolResult

            return ToolResult(tool=name, action=action, success=False, error=f"Tool '{name}' not found.")
        return await t.execute(action, **kwargs)
