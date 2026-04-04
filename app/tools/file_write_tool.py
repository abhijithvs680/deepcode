import difflib
import logging
import os
import subprocess
from pathlib import Path

from app.tools.base_tool import Tool, ToolResult

logger = logging.getLogger(__name__)

class FileWriteTool(Tool):
    """
    Safely handles modifications to the filesystem natively generating Diffs
    and persisting Git backup limits seamlessly protecting memory scopes!
    """
    name = "FileWriteTool"
    description = "Modifies a file after calculating and validating Diff boundaries natively."

    def __init__(self, repo_path: str):
        self.repo_path = repo_path

    def _git_safety_hook(self) -> str | None:
        import uuid
        branch_name = f"deepcode/patch-{uuid.uuid4().hex[:8]}"
        try:
            # Phase 3: Create isolated patch branches elegantly 
            subprocess.run(["git", "checkout", "-b", branch_name], cwd=self.repo_path, check=True, capture_output=True)
            logger.info(f"Git Backup target evaluated accurately: {branch_name}")
            return branch_name
        except Exception as e:
            logger.warning(f"Git checkout execution bounded failed (maybe not a git repo?): {e}")
            return None

    def rollback(self) -> None:
        try:
            subprocess.run(["git", "reset", "--hard", "HEAD"], cwd=self.repo_path, check=True, capture_output=True)
            logger.info("Atomic apply crashed; Rolled back executed mutations!")
        except Exception as e:
            logger.error(f"Atomic Rollback bounded error unexpectedly: {e}")

    def generate_diff(self, file_path: str, new_content: str) -> str:
        abs_path = os.path.join(self.repo_path, file_path)
        if not os.path.exists(abs_path):
            original_lines = []
        else:
            with open(abs_path, 'r', encoding='utf-8') as f:
                original_lines = f.readlines()
        
        new_lines = [line + '\n' for line in new_content.splitlines()]
        if new_content and not new_content.endswith('\n'):
            new_lines[-1] = new_lines[-1].rstrip('\n') # Fix splitlines append trace

        diff = difflib.unified_diff(
            original_lines, 
            new_lines, 
            fromfile=f"a/{file_path}", 
            tofile=f"b/{file_path}"
        )
        return "".join(diff)

    def apply_patches(self, patches: list[dict], allowed_files: list[str]) -> ToolResult:
        """
        Phase 3: Atomic Multi-file Patch deployments!
        """
        # Scope validation all or nothing limits
        for p in patches:
            fp = p["file_path"]
            if fp not in allowed_files and not any(fp.endswith(f) for f in allowed_files):
                return ToolResult(
                    tool=self.name,
                    action="apply_patches",
                    success=False,
                    error=f"Atomic Refused modify: {fp} wasn't explicitly evaluated statically by Retriever."
                )

        branch = self._git_safety_hook()
        
        try:
            for p in patches:
                fp = p["file_path"]
                nc = p["new_content"]
                abs_path = os.path.join(self.repo_path, fp)
                # Ensure directory bounds!
                os.makedirs(os.path.dirname(abs_path), exist_ok=True)
                with open(abs_path, 'w', encoding='utf-8') as f:
                    f.write(nc)
                    
            return ToolResult(
                tool=self.name,
                action="apply_patches",
                success=True,
                data={
                    "message": "Successfully applied Phase 3 Atomic multi-file mappings!",
                    "branch": branch
                }
            )
        except Exception as e:
            self.rollback()
            return ToolResult(
                tool=self.name,
                action="apply_patches",
                success=False,
                error=f"Atomic Patch deployment failed natively tracking explicit rollback limits: {e}"
            )
