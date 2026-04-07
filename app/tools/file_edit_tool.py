"""
Transactional search/replace edits with snapshot rollback and optional Python syntax check.
"""
from __future__ import annotations

import ast
import logging
import os
from difflib import unified_diff
from typing import Any

from app.core.config import get_settings
from app.tools.base_tool import Tool, ToolResult

logger = logging.getLogger(__name__)


def validate_python_syntax(source: str, path: str) -> tuple[bool, str]:
    if not path.endswith(".py"):
        return True, ""
    try:
        ast.parse(source)
        return True, ""
    except SyntaxError as e:
        return False, f"{e.msg} at line {e.lineno}"


class FileEditTool(Tool):
    name = "file_edit"
    description = "Search/replace patch application with session rollback."

    def __init__(self, repo_path: str):
        self.repo_path = os.path.abspath(repo_path)
        self._original: dict[str, str] = {}
        self._working: dict[str, str] = {}
        self._active = False
        self._dry_run = False

    def set_dry_run(self, enabled: bool) -> None:
        self._dry_run = bool(enabled)

    def has_active_session(self) -> bool:
        return self._active

    def has_pending_changes(self) -> bool:
        if not self._active:
            return False
        for rel, newc in self._working.items():
            if newc != self._original.get(rel, ""):
                return True
        return False

    def touched_files(self) -> list[str]:
        return sorted(self._working.keys())

    def unified_diff_preview(self) -> str:
        """Unified diff for all files touched in the active session (empty if unchanged)."""
        chunks: list[str] = []
        for rel in sorted(self._working.keys()):
            old = self._original.get(rel, "")
            new = self._working.get(rel, "")
            if old == new:
                continue
            a = old.splitlines(keepends=True)
            b = new.splitlines(keepends=True)
            chunks.extend(
                unified_diff(
                    a,
                    b,
                    fromfile=f"a/{rel}",
                    tofile=f"b/{rel}",
                )
            )
        return "".join(chunks)

    def begin_session(self) -> None:
        self._original.clear()
        self._working.clear()
        self._active = True

    def _read(self, rel: str) -> str:
        abs_p = os.path.abspath(os.path.join(self.repo_path, rel))
        if not abs_p.startswith(self.repo_path):
            raise PermissionError("path outside repo")
        if not os.path.isfile(abs_p):
            return ""
        with open(abs_p, "r", encoding="utf-8", errors="replace") as f:
            return f.read()

    def _ensure_working(self, rel: str) -> str:
        rel = rel.replace("\\", "/")
        if rel not in self._working:
            content = self._read(rel)
            self._original[rel] = content
            self._working[rel] = content
        return self._working[rel]

    def apply_search_replace(
        self,
        file_path: str,
        search: str,
        replace: str,
        *,
        expected_occurrences: int = 1,
    ) -> ToolResult:
        if not self._active:
            if getattr(get_settings(), "REACT_SUBGOAL_EDIT_TRANSACTION", True):
                return ToolResult(
                    tool=self.name,
                    action="apply_search_replace",
                    success=False,
                    error="No active edit session — begin_session must run first.",
                )
            self.begin_session()
        rel = file_path.replace("\\", "/")
        try:
            w = self._ensure_working(rel)
        except Exception as e:
            return ToolResult(
                tool=self.name,
                action="apply_search_replace",
                success=False,
                error=str(e),
            )
        n = w.count(search)
        if n != expected_occurrences:
            return ToolResult(
                tool=self.name,
                action="apply_search_replace",
                success=False,
                error=f"search block occurrences={n}, expected={expected_occurrences}",
            )
        self._working[rel] = w.replace(search, replace, expected_occurrences)
        ok, err = validate_python_syntax(self._working[rel], rel)
        if not ok:
            self._working[rel] = w
            return ToolResult(
                tool=self.name,
                action="apply_search_replace",
                success=False,
                error=f"syntax check failed: {err}",
            )
        return ToolResult(
            tool=self.name,
            action="apply_search_replace",
            success=True,
            data={"file": rel, "bytes": len(self._working[rel])},
        )

    def commit_session(self, allowed_files: set[str]) -> ToolResult:
        if not self._active:
            return ToolResult(
                tool=self.name,
                action="commit_session",
                success=False,
                error="No active session.",
            )
        try:
            from app.agents.light_syntax import validate_text_for_path

            to_write: list[tuple[str, str]] = []
            for rel, content in self._working.items():
                if rel not in allowed_files and not any(
                    rel.startswith(a + "/") for a in allowed_files
                ):
                    continue
                abs_p = os.path.abspath(os.path.join(self.repo_path, rel))
                if not abs_p.startswith(self.repo_path):
                    continue
                ok, err = validate_text_for_path(rel, content)
                if not ok:
                    return ToolResult(
                        tool=self.name,
                        action="commit_session",
                        success=False,
                        error=f"{rel}: {err}",
                    )
                to_write.append((rel, content))

            if self._dry_run:
                written = [r for r, _ in to_write]
                self._working.clear()
                self._original.clear()
                self._active = False
                return ToolResult(
                    tool=self.name,
                    action="commit_session",
                    success=True,
                    data={"dry_run": True, "files_skipped": written},
                )

            for rel, content in to_write:
                abs_p = os.path.abspath(os.path.join(self.repo_path, rel))
                os.makedirs(os.path.dirname(abs_p), exist_ok=True)
                with open(abs_p, "w", encoding="utf-8", newline="") as f:
                    f.write(content)
            written = [r for r, _ in to_write]
            self._working.clear()
            self._original.clear()
            self._active = False
            return ToolResult(
                tool=self.name,
                action="commit_session",
                success=True,
                data={"files_written": written},
            )
        except Exception as e:
            return ToolResult(
                tool=self.name,
                action="commit_session",
                success=False,
                error=str(e),
            )

    def rollback_session(self) -> ToolResult:
        if not self._active and not self._original and not self._working:
            return ToolResult(
                tool=self.name,
                action="rollback_session",
                success=True,
                data={"noop": True},
            )
        try:
            for rel, content in self._original.items():
                abs_p = os.path.abspath(os.path.join(self.repo_path, rel))
                if not abs_p.startswith(self.repo_path):
                    continue
                if os.path.isfile(abs_p):
                    with open(abs_p, "w", encoding="utf-8", newline="") as f:
                        f.write(content)
            self._working.clear()
            self._original.clear()
            self._active = False
            return ToolResult(
                tool=self.name,
                action="rollback_session",
                success=True,
                data={},
            )
        except Exception as e:
            return ToolResult(
                tool=self.name,
                action="rollback_session",
                success=False,
                error=str(e),
            )

    async def execute(self, action: str, **kwargs: Any) -> ToolResult:
        if action == "begin_session":
            self.begin_session()
            return ToolResult(tool=self.name, action=action, success=True, data={})
        if action == "apply_search_replace":
            return self.apply_search_replace(
                kwargs.get("file_path", kwargs.get("path", "")),
                kwargs.get("search", ""),
                kwargs.get("replace", ""),
                expected_occurrences=int(kwargs.get("expected_occurrences", 1)),
            )
        if action == "commit_session":
            af = kwargs.get("allowed_files") or []
            return self.commit_session({str(x).replace("\\", "/") for x in af})
        if action == "rollback_session":
            return self.rollback_session()
        if action == "configure":
            if "dry_run" in kwargs:
                self.set_dry_run(bool(kwargs.get("dry_run")))
            return ToolResult(
                tool=self.name,
                action=action,
                success=True,
                data={"dry_run": self._dry_run},
            )
        return ToolResult(
            tool=self.name,
            action=action,
            success=False,
            error=f"unknown action {action}",
        )
