"""
Transactional search/replace edits with snapshot rollback, checkpoints, and optional pre-commit snapshots.
"""
from __future__ import annotations

import ast
import copy
import logging
import os
import shutil
from datetime import datetime
from difflib import unified_diff
from typing import Any

from app.core.config import get_settings
from app.core.event_bus import EventBus
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
        self._checkpoints: list[tuple[dict[str, str], dict[str, str]]] = []

    def set_dry_run(self, enabled: bool) -> None:
        self._dry_run = bool(enabled)

    def has_active_session(self) -> bool:
        return self._active

    def has_checkpoints(self) -> bool:
        return bool(self._checkpoints)

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

    def session_diff_for_file(self, rel: str) -> str:
        rel = rel.replace("\\", "/")
        old = self._original.get(rel, "")
        new = self._working.get(rel, "")
        if old == new:
            return ""
        return "".join(
            unified_diff(
                old.splitlines(keepends=True),
                new.splitlines(keepends=True),
                fromfile=f"a/{rel}",
                tofile=f"b/{rel}",
            )
        )

    def begin_session(self) -> None:
        self._original.clear()
        self._working.clear()
        self._checkpoints.clear()
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

    def checkpoint_session(self) -> ToolResult:
        if not self._active:
            return ToolResult(
                tool=self.name,
                action="checkpoint_session",
                success=False,
                error="No active session.",
            )
        self._checkpoints.append(
            (copy.deepcopy(self._working), copy.deepcopy(self._original))
        )
        cap = int(getattr(get_settings(), "REACT_EDIT_CHECKPOINT_MAX", 5) or 5)
        cap = max(1, min(cap, 12))
        while len(self._checkpoints) > cap:
            self._checkpoints.pop(0)
        return ToolResult(
            tool=self.name,
            action="checkpoint_session",
            success=True,
            data={"checkpoints": len(self._checkpoints)},
        )

    def rollback_to_last_checkpoint(self) -> ToolResult:
        if not self._checkpoints:
            return self.rollback_session()
        w, o = self._checkpoints.pop()
        self._working = copy.deepcopy(w)
        self._original = copy.deepcopy(o)
        self._active = True
        return ToolResult(
            tool=self.name,
            action="rollback_to_last_checkpoint",
            success=True,
            data={"checkpoints_remaining": len(self._checkpoints)},
        )

    def apply_search_replace(
        self,
        file_path: str,
        search: str,
        replace: str,
        *,
        expected_occurrences: int = 1,
        dry_run: bool = False,
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
        new_content = w.replace(search, replace, expected_occurrences)
        ok, err = validate_python_syntax(new_content, rel)
        if not ok:
            return ToolResult(
                tool=self.name,
                action="apply_search_replace",
                success=False,
                error=f"syntax check failed: {err}",
            )
        if dry_run:
            diff = "".join(
                unified_diff(
                    w.splitlines(keepends=True),
                    new_content.splitlines(keepends=True),
                    fromfile=f"a/{rel}",
                    tofile=f"b/{rel}",
                )
            )
            return ToolResult(
                tool=self.name,
                action="apply_search_replace",
                success=True,
                data={
                    "file": rel,
                    "dry_run": True,
                    "bytes": len(new_content),
                    "diff_preview": diff[:4000],
                },
            )
        self._working[rel] = new_content
        return ToolResult(
            tool=self.name,
            action="apply_search_replace",
            success=True,
            data={"file": rel, "bytes": len(self._working[rel])},
        )

    def _snapshot_files_before_write(
        self, pairs: list[tuple[str, str]]
    ) -> str | None:
        stg = get_settings()
        if not getattr(stg, "PRE_COMMIT_SNAPSHOT_ENABLED", True):
            return None
        rel_snap = str(getattr(stg, "PRE_COMMIT_SNAPSHOT_DIR", ".deepcode/snapshots") or ".deepcode/snapshots").strip("/\\")
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        snap_root = os.path.abspath(
            os.path.join(self.repo_path, rel_snap.replace("\\", "/"), ts)
        )
        if not snap_root.startswith(self.repo_path):
            return None
        try:
            os.makedirs(snap_root, exist_ok=True)
            for rel, _ in pairs:
                abs_p = os.path.abspath(os.path.join(self.repo_path, rel))
                if not abs_p.startswith(self.repo_path) or not os.path.isfile(abs_p):
                    continue
                dst = os.path.join(snap_root, rel)
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                shutil.copy2(abs_p, dst)
            meta = os.path.join(snap_root, "_snapshot_meta.txt")
            with open(meta, "w", encoding="utf-8") as f:
                f.write(f"timestamp={ts}\nrepo={self.repo_path}\n")
            return snap_root
        except OSError as exc:
            logger.warning("pre-commit snapshot failed: %s", exc)
            return None

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
                self._checkpoints.clear()
                self._active = False
                return ToolResult(
                    tool=self.name,
                    action="commit_session",
                    success=True,
                    data={"dry_run": True, "files_skipped": written},
                )

            snap = self._snapshot_files_before_write(to_write)

            for rel, content in to_write:
                abs_p = os.path.abspath(os.path.join(self.repo_path, rel))
                os.makedirs(os.path.dirname(abs_p), exist_ok=True)
                with open(abs_p, "w", encoding="utf-8", newline="") as f:
                    f.write(content)
            written = [r for r, _ in to_write]
            self._working.clear()
            self._original.clear()
            self._checkpoints.clear()
            self._active = False
            return ToolResult(
                tool=self.name,
                action="commit_session",
                success=True,
                data={
                    "files_written": written,
                    "snapshot_dir": snap,
                },
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
                    with open(abs_p, "w", encoding="utf-8") as f:
                        f.write(content)
            self._working.clear()
            self._original.clear()
            self._checkpoints.clear()
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
        if action == "checkpoint_session":
            return self.checkpoint_session()
        if action == "rollback_to_last_checkpoint":
            return self.rollback_to_last_checkpoint()
        if action == "apply_search_replace":
            r = self.apply_search_replace(
                kwargs.get("file_path", kwargs.get("path", "")),
                kwargs.get("search", ""),
                kwargs.get("replace", ""),
                expected_occurrences=int(kwargs.get("expected_occurrences", 1)),
                dry_run=bool(kwargs.get("dry_run")),
            )
            if r.success and r.data:
                rel = str((r.data or {}).get("file") or "").replace("\\", "/")
                if rel:
                    diff_txt = str((r.data or {}).get("diff_preview") or "")
                    if not diff_txt and not bool(kwargs.get("dry_run")):
                        diff_txt = self.session_diff_for_file(rel)
                    if diff_txt:
                        await EventBus.emit(
                            {
                                "type": "diff_ready",
                                "file": rel,
                                "diff": diff_txt[:16000],
                                "workspace": self.repo_path,
                                "dry_run": bool(kwargs.get("dry_run")),
                            }
                        )
            return r
        if action == "commit_session":
            af = kwargs.get("allowed_files") or []
            r = self.commit_session({str(x).replace("\\", "/") for x in af})
            if r.success and r.data and not (r.data or {}).get("dry_run"):
                for fp in (r.data or {}).get("files_written") or []:
                    p = str(fp).replace("\\", "/")
                    await EventBus.emit(
                        {
                            "type": "file_updated",
                            "file": p,
                            "workspace": self.repo_path,
                        }
                    )
            return r
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
