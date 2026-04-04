import os
import re
from typing import Optional

from app.tools.base_tool import Tool, ToolResult

class LogTool(Tool):
    """
    Ingests local workspace standard output structures securely mapping error schemas
    and avoiding token crashes handling standard unstructured debugging strings.
    """
    name = "log"
    description = "Searches system trace log outputs locally parsing exceptions via structured formats."

    def __init__(self, workspace_root: str):
        self.workspace_root = workspace_root

    def _resolve_path(self, path: str) -> str:
        fp = os.path.abspath(os.path.join(self.workspace_root, path))
        if not fp.startswith(os.path.abspath(self.workspace_root)):
            raise PermissionError("Illegal path traversal bounds exception.")
        return fp

    async def execute(self, action: str, **kwargs) -> ToolResult:
        try:
            if action == "search_logs":
                return await self.search_logs(**kwargs)
            elif action == "get_recent_failures":
                return await self.get_recent_failures(**kwargs)
            else:
                return ToolResult(tool=self.name, action=action, success=False, error=f"Unknown Tool Action: '{action}'")
        except Exception as e:
            return ToolResult(tool=self.name, action=action, success=False, error=str(e))

    def _get_log_files(self) -> list[str]:
        """Heuristically retrieve local workspace logic dumps mapping `.log` files securely."""
        files = []
        for root, dirs, f in os.walk(self.workspace_root):
            # Skip hidden and cache dependencies
            dirs[:] = [d for d in dirs if not d.startswith('.') and d not in ["__pycache__", "node_modules", "venv"]]
            for file in f:
                if file.endswith(".log"):
                    files.append(os.path.join(root, file))
        return files

    async def search_logs(self, query: str) -> ToolResult:
        """Simple chronological filter evaluating matches natively."""
        pattern = re.compile(re.escape(query), re.IGNORECASE)
        matches = []
        try:
            for path in self._get_log_files():
                with open(path, "r", encoding="utf-8", errors="replace") as f:
                    for i, line in enumerate(f):
                        if pattern.search(line):
                            matches.append({
                                "file": os.path.relpath(path, self.workspace_root), 
                                "line_no": i + 1, 
                                "content": line.strip()
                            })
            return ToolResult(
                tool=self.name, 
                action="search_logs", 
                success=True, 
                data={"matches": matches[-50:]}, # Return latest tail matches
                metadata={"found_count": len(matches)}
            )
        except Exception as e:
            return ToolResult(tool=self.name, action="search_logs", success=False, error=str(e))

    async def get_recent_failures(self, limit: int = 20) -> ToolResult:
        """
        Deep structural grep targeting explicit stack trace behaviors aggregating error exceptions.
        """
        errors = []
        # common unstructured triggers
        error_pattern = re.compile(r"error|exception|traceback|fail", re.IGNORECASE)
        
        try:
            for path in self._get_log_files():
                with open(path, "r", encoding="utf-8", errors="replace") as f:
                    for i, line in enumerate(f):
                        if error_pattern.search(line):
                            errors.append({
                                "file": os.path.relpath(path, self.workspace_root),
                                "line_no": i + 1,
                                "message": line.strip()
                            })
            
            tail = errors[-limit:]
            return ToolResult(
                tool=self.name, 
                action="get_recent_failures", 
                success=True, 
                data={"failures": tail}, 
                metadata={"total_errors": len(errors)}
            )
        except Exception as e:
            return ToolResult(tool=self.name, action="get_recent_failures", success=False, error=str(e))
