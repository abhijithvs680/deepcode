"""Mini terminal file browser: filterable list, preview, TextArea edit + save."""
from __future__ import annotations

import os
from pathlib import Path

from textual import on
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import Screen
from textual.widgets import DataTable, Footer, Header, Input, Static, TextArea

_SKIP_DIR_NAMES = frozenset(
    {
        ".git",
        "__pycache__",
        ".venv",
        "venv",
        "node_modules",
        ".mypy_cache",
        ".pytest_cache",
        ".tox",
        ".eggs",
        ".ruff_cache",
        ".idea",
        ".cursor",
    }
)


def _lexer_for(path: str) -> str | None:
    ext = Path(path).suffix.lower()
    return {
        ".py": "python",
        ".md": "markdown",
        ".toml": "toml",
        ".json": "json",
        ".yaml": "yaml",
        ".yml": "yaml",
        ".rs": "rust",
        ".go": "go",
        ".js": "javascript",
        ".ts": "typescript",
        ".tsx": "tsx",
        ".jsx": "jsx",
        ".css": "css",
        ".html": "html",
        ".sql": "sql",
        ".sh": "bash",
    }.get(ext)


def _collect_files(root: Path, *, max_files: int = 8000) -> list[str]:
    root = root.resolve()
    out: list[str] = []
    if not root.is_dir():
        return out
    root_s = str(root)
    for dirpath, dirnames, filenames in os.walk(root_s, topdown=True):
        dirnames[:] = [d for d in dirnames if d not in _SKIP_DIR_NAMES]
        for name in filenames:
            p = Path(dirpath) / name
            try:
                rel = p.relative_to(root)
            except ValueError:
                continue
            s = str(rel).replace("\\", "/")
            out.append(s)
            if len(out) >= max_files:
                return sorted(out)
    return sorted(out)


async def _record_manual_edit(repo_id: str, rel_path: str, ok: bool) -> None:
    if not repo_id:
        return
    try:
        from app.core.database import async_session_factory, create_tables, dispose_engine

        await create_tables()
        async with async_session_factory() as db:
            from app.memory.intel_store import record_edit_learning

            await record_edit_learning(
                db,
                repo_id,
                query="tui_manual_edit",
                files=[rel_path],
                success=ok,
                detail="manual_edit",
            )
            await db.commit()
        await dispose_engine()
    except Exception:
        pass


def run_file_explorer(root: str, repo_id: str | None = None) -> None:
    r = Path(root).resolve()
    if not r.is_dir():
        raise SystemExit(f"Not a directory: {r}")

    class _Host(App[None]):
        def on_mount(self) -> None:
            self.push_screen(FileExplorerScreen(str(r), repo_id=repo_id))

    _Host().run()


class FileExplorerScreen(Screen[None]):
    """Filter with the search box; arrows to move; highlight loads preview; e edits; Ctrl+S saves."""

    BINDINGS = [
        Binding("escape", "back_or_quit", "Back / Close", show=True),
        Binding("q", "quit", "Close", show=True),
        Binding("e", "toggle_edit", "Edit", show=True),
        Binding("ctrl+s", "save_buffer", "Save", show=True),
    ]

    CSS = """
    FileExplorerScreen {
        width: 100%;
        height: 100%;
        layout: vertical;
    }
    #main-row {
        height: 1fr;
    }
    #left-col {
        width: 40%;
        min-width: 24;
        border-right: solid $primary;
    }
    #path-filter {
        dock: top;
        margin: 0 1;
    }
    #ftable {
        height: 1fr;
    }
    #right-col {
        width: 1fr;
    }
    #path-label {
        dock: top;
        padding: 0 1;
        background: $surface;
    }
    #editor-pane {
        height: 1fr;
        margin: 0 1 1 1;
    }
    TextArea {
        height: 1fr;
    }
    """

    def __init__(self, root: str, repo_id: str | None = None):
        super().__init__()
        self._root = Path(root).resolve()
        self._repo_id = (repo_id or "").strip() or None
        self._all_paths = _collect_files(self._root)
        self._current_rel: str | None = None

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Horizontal(id="main-row"):
            with Vertical(id="left-col"):
                yield Input(placeholder="Filter paths…", id="path-filter")
                yield DataTable(id="ftable", cursor_type="row", zebra_stripes=True)
            with VerticalScroll(id="right-col"):
                yield Static("", id="path-label")
                yield TextArea(
                    "",
                    id="editor-pane",
                    language=None,
                    read_only=True,
                    show_line_numbers=True,
                    soft_wrap=True,
                )
        yield Footer()

    def on_mount(self) -> None:
        table = self.query_one("#ftable", DataTable)
        table.add_column("path")
        self._apply_filter("")

    def _apply_filter(self, q: str) -> None:
        table = self.query_one("#ftable", DataTable)
        table.clear(columns=True)
        table.add_column("path")
        qn = (q or "").strip().lower()
        shown = [p for p in self._all_paths if not qn or qn in p.lower()]
        for p in shown[:5000]:
            table.add_row(p, key=p)
        if not shown:
            self.query_one("#path-label", Static).update("(no matches)")
            self.query_one("#editor-pane", TextArea).load_text("")

    @on(Input.Changed, "#path-filter")
    def _on_filter_changed(self, event: Input.Changed) -> None:
        self._apply_filter(event.value)

    @on(DataTable.RowHighlighted, "#ftable")
    def _on_row_highlight(self, event: DataTable.RowHighlighted) -> None:
        key = event.row_key
        if key is None or key.value is None:
            return
        rel = str(key.value)
        self._load_file(rel)

    def _load_file(self, rel: str) -> None:
        self._current_rel = rel
        path = self._root / rel
        self.query_one("#path-label", Static).update(str(path))
        ta = self.query_one("#editor-pane", TextArea)
        ta.read_only = True
        if not path.is_file():
            ta.load_text("")
            return
        try:
            raw = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            ta.load_text("(could not read file)")
            return
        if len(raw) > 400_000:
            raw = raw[:400_000] + "\n\n… truncated …"
        ta.language = _lexer_for(rel)
        ta.load_text(raw)

    def _pop_or_exit(self) -> None:
        try:
            self.app.pop_screen()
        except Exception:
            self.app.exit()

    def action_toggle_edit(self) -> None:
        if not self._current_rel:
            return
        ta = self.query_one("#editor-pane", TextArea)
        ta.read_only = not ta.read_only
        if ta.read_only:
            self.notify("Read-only", severity="information")
        else:
            self.notify("Editable — Ctrl+S to save, Esc to leave read-only", severity="information")

    async def action_save_buffer(self) -> None:
        if not self._current_rel:
            return
        path = self._root / self._current_rel
        ta = self.query_one("#editor-pane", TextArea)
        if ta.read_only:
            self.notify("Press e to enable editing first", severity="warning")
            return
        try:
            path.write_text(ta.text, encoding="utf-8")
        except OSError as exc:
            self.notify(f"Save failed: {exc}", severity="error")
            if self._repo_id:
                await _record_manual_edit(self._repo_id, self._current_rel, False)
            return
        self.notify("Saved", severity="information")
        if self._repo_id:
            await _record_manual_edit(self._repo_id, self._current_rel, True)

    def action_back_or_quit(self) -> None:
        ta = self.query_one("#editor-pane", TextArea)
        if not ta.read_only:
            ta.read_only = True
            if self._current_rel:
                self._load_file(self._current_rel)
            self.notify("Reverted to read-only (reload from disk)", severity="information")
            return
        self._pop_or_exit()

    def action_quit(self) -> None:
        self._pop_or_exit()
