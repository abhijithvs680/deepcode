import subprocess
import logging
from typing import Optional

from app.tools.base_tool import Tool, ToolResult

logger = logging.getLogger(__name__)

class GitTool(Tool):
    """
    Direct hook into the Workspace git bindings pulling raw accountability metrics
    (Blame, History, Churn) explicitly formatted tracking recent state limits.
    """
    name = "git"
    description = "Pulls codebase history natively targeting blames, commits, and diffs restricting ANN distances."

    def __init__(self, workspace_root: str):
        self.workspace_root = workspace_root

    def _exec(self, cmd: list[str]) -> str:
        """Helper evaluating safe shell contexts running natively over the Git backend."""
        res = subprocess.run(cmd, cwd=self.workspace_root, capture_output=True, text=True, check=True)
        return res.stdout.strip()
        
    async def execute(self, action: str, **kwargs) -> ToolResult:
        try:
            if action == "git_log":
                return await self.git_log(**kwargs)
            elif action == "git_blame":
                return await self.git_blame(**kwargs)
            elif action == "git_diff":
                return await self.git_diff(**kwargs)
            elif action == "get_recent_changes":
                return await self.get_recent_changes(**kwargs)
            else:
                return ToolResult(tool=self.name, action=action, success=False, error=f"Unknown Tool Action: '{action}'")
        except subprocess.CalledProcessError as e:
            return ToolResult(tool=self.name, action=action, success=False, error=f"Git Error: {e.stderr.strip()}")
        except Exception as e:
            return ToolResult(tool=self.name, action=action, success=False, error=str(e))

    async def git_log(self, file_path: Optional[str] = None, n: int = 10) -> ToolResult:
        cmd = ["git", "log", f"-n {n}", "--pretty=format:%H|%an|%ad|%s"]
        if file_path:
            cmd.extend(["--", file_path])
            
        out = self._exec(cmd)
        commits = []
        for line in out.split("\n"):
            if line:
                parts = line.split("|", 3)
                if len(parts) == 4:
                    commits.append({"hash": parts[0], "author": parts[1], "date": parts[2], "message": parts[3]})
        
        return ToolResult(tool=self.name, action="git_log", success=True, data={"commits": commits}, metadata={"count": len(commits)})

    async def git_blame(self, file_path: str, line: Optional[int] = None) -> ToolResult:
        cmd = ["git", "blame"]
        if line:
            cmd.extend(["-L", f"{line},{line}"])
        cmd.append(file_path)
            
        out = self._exec(cmd)
        return ToolResult(tool=self.name, action="git_blame", success=True, data={"blame": out.split("\n")})

    async def git_diff(self, commit_a: str, commit_b: str) -> ToolResult:
        cmd = ["git", "diff", f"{commit_a}..{commit_b}"]
        out = self._exec(cmd)
        return ToolResult(tool=self.name, action="git_diff", success=True, data={"diff": out[:5000]}, metadata={"truncated": len(out) > 5000})

    async def get_recent_changes(self, file_path: str, n: int = 5) -> ToolResult:
        cmd = ["git", "log", f"-n {n}", "-p", "--", file_path]
        out = self._exec(cmd)
        return ToolResult(tool=self.name, action="get_recent_changes", success=True, data={"recent_diffs": out[:5000]}, metadata={"file": file_path, "commits": n})
