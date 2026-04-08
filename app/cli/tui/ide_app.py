"""
DeepCode V2 — agent IDE runtime (Textual): VS Code–style workspace — chat sidebar |
editor (center) | explorer + subgoals (right); integrated terminal docked along the bottom.

Run: deepcode ui --ide [REPO]
"""
from __future__ import annotations

import json
import os
import socket
from pathlib import Path
from typing import Any

from rich.markup import escape
from textual import on, work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal, Vertical, VerticalScroll
from textual.widgets import DirectoryTree, Footer, Header, RichLog, Static

from app.cli.tui.widgets.ide_composer import IdeComposer

from app.cli.commands.ask import run_ask_async, run_long_goal_async
from app.cli.tui.app import ConfirmDiffScreen, ConfirmShellScreen, IngestionLoaderScreen
from app.cli.tui.clipboard_util import copy_plain_text, strip_rich_markup
from app.cli.tui.ide_event_buffer import IdeEventDebouncer
from app.cli.tui.ide_shell import IdeSubprocessShell, ShellPromptLine, format_shell_line
from app.cli.tui.ide_state import IDEState
from app.cli.tui.widgets import chat as chatfmt
from app.cli.tui.widgets.command_palette import CommandPaletteScreen
from app.core.config import get_settings

settings = get_settings()


def _normalize_unified_diff_header_path(token: str) -> str | None:
    t = token.strip().split("\t", 1)[0].strip()
    if not t or t in ("/dev/null", "-", "null", "NUL"):
        return None
    for prefix in ("a/", "b/", "i/", "w/", "c/"):
        if t.startswith(prefix):
            return t[len(prefix) :].replace("\\", "/")
    return t.replace("\\", "/")


def _paths_from_unified_diff(diff: str) -> set[str]:
    out: set[str] = set()
    for line in (diff or "").splitlines():
        s = line.strip("\r")
        if s.startswith("--- ") or s.startswith("+++ "):
            p = _normalize_unified_diff_header_path(s[4:])
            if p:
                out.add(p)
    return out


def _diff_covers_active_path(diff: str, active_rel: str) -> bool:
    """True if unified diff headers reference the same repo-relative path as active_rel."""
    if not diff or not active_rel or active_rel == "_session":
        return False
    want = active_rel.replace("\\", "/").strip().lstrip("./")
    if not want:
        return False
    for p in _paths_from_unified_diff(diff):
        pn = p.strip().lstrip("./")
        if pn == want or pn.endswith("/" + want) or want.endswith("/" + pn):
            return True
    return False


class DeepCodeIdeApp(App):
    """Agent IDE: editor-first workspace, docked terminal, sidebars — same orchestrator as the TUI."""

    TITLE = "DeepCode IDE"
    SUB_TITLE = f"{settings.LLM_PROVIDER} / {settings.LLM_MODEL}"

    BINDINGS = [
        Binding("ctrl+p", "command_palette", "Commands", show=True),
        Binding("ctrl+l", "clear_chat", "Clear chat", show=True),
        Binding("ctrl+b", "toggle_bottom_visibility", "Terminal", show=True),
        Binding(
            "ctrl+shift+t",
            "focus_shell_prompt",
            "Shell",
            show=True,
        ),
        Binding("ctrl+i", "next_tab", "Next tab", show=True),
        Binding("ctrl+w", "close_tab", "Close tab", show=True),
        Binding("ctrl+shift+e", "copy_explorer", "Copy explorer", show=False),
        Binding("ctrl+shift+d", "copy_editor", "Copy editor", show=False),
        Binding(
            "ctrl+c",
            "cancel_agent",
            "Cancel run",
            show=True,
            priority=True,
        ),
    ]

    CSS = """
    Screen { layout: vertical; background: #1e1e1e; }
    Header { background: #252526; color: #cccccc; }
    Footer { background: #252526; color: #888888; }
    #ide-workspace {
        height: 1fr;
        width: 100%;
        layout: vertical;
        min-height: 8;
        background: #1e1e1e;
    }
    #ide-top {
        height: 1fr;
        layout: horizontal;
        min-height: 8;
        background: #1e1e1e;
    }
    .ide-sidebar {
        height: 1fr;
        layout: vertical;
        background: #181818;
        border: none;
    }
    #ide-chat {
        width: 38;
        min-width: 32;
        max-width: 46;
        padding: 0 1;
        border-right: solid #2a2a2a;
        background: #181818;
        color: #bbbbbb;
    }
    #ide-right {
        width: 38;
        min-width: 30;
        max-width: 48;
        padding: 0 1;
        border-left: solid #2a2a2a;
        background: #181818;
        color: #bbbbbb;
    }
    #ide-chat-scroll {
        height: 1fr;
        min-height: 8;
        padding: 0;
        background: transparent;
    }
    #ide-chat-scroll Static.chat-line {
        color: #aaaaaa;
    }
    #ide-center {
        width: 1fr;
        height: 1fr;
        min-width: 24;
        layout: vertical;
        background: #1e1e1e;
        border: none;
    }
    #ide-editor-stack {
        height: 1fr;
        min-height: 6;
        layout: vertical;
        background: transparent;
        border: none;
    }
    #ide-terminal-stack {
        height: 12;
        min-height: 6;
        max-height: 22;
        layout: vertical;
        border-top: solid #333333;
        padding: 0 1;
        background: #202020;
    }
    #ide-terminal-stack.hidden {
        display: none;
        height: 0;
        min-height: 0;
        max-height: 0;
        border: none;
    }
    #ide-explorer-col {
        height: 1fr;
        min-height: 5;
        layout: vertical;
        border: none;
        background: transparent;
    }
    #ide-explorer-body {
        height: 1fr;
        min-height: 5;
        layout: vertical;
        background: transparent;
    }
    #ide-explorer-tree {
        height: 2fr;
        min-height: 8;
        padding: 0;
        background: transparent;
        color: #bbbbbb;
    }
    #ide-explorer-tree > .tree--highlight-line {
        background: #2a2a2a;
    }
    #ide-subgoals-scroll {
        height: 1fr;
        min-height: 4;
        padding: 0;
        background: transparent;
    }
    .ide-ribbon-label {
        height: 1;
        padding: 0 1 0 1;
        background: transparent;
        color: #999999;
        text-style: none;
        border: none;
    }
    #ide-editor-tabs {
        height: 1;
        padding: 0 2 0 2;
        background: #2a2a2a;
        color: #e6e6e6;
        border-bottom: solid #333333;
    }
    #ide-editor-path {
        padding: 0 2 0 2;
        background: #1e1e1e;
        height: 1;
        color: #c0c0c0;
        border-bottom: solid #333333;
    }
    #ide-editor-body {
        height: 1fr;
        padding: 1 2;
        background: #1e1e1e;
        color: #e6e6e6;
    }
    #ide-editor-body Static.chat-line {
        color: #e6e6e6;
    }
    #ide-explorer-scroll {
        height: 1fr;
        min-height: 4;
        padding: 0;
        background: transparent;
    }
    Static.ide-chat-events {
        padding: 0 1 0 1;
        margin: 0;
        background: #252526 35%;
        border-left: solid #007acc 55%;
        color: #dddddd;
        height: auto;
        min-height: 1;
    }
    #ide-composer-stack {
        layout: vertical;
        height: auto;
        min-height: 3;
        margin: 0;
        padding: 0;
        border: none;
        background: transparent;
    }
    #input {
        height: 4;
        min-height: 3;
        max-height: 10;
        background: #2a2a2a;
        border: solid #3a3a3a;
        color: #d4d4d4;
        padding: 0 1 0 1;
    }
    #input:focus {
        border: solid #505050;
    }
    #ide-composer-hint {
        height: auto;
        min-height: 1;
        padding: 0 1 0 1;
        background: transparent;
        color: #999999;
        border: none;
    }
    #ide-debug-trace {
        height: 6;
        min-height: 3;
        border-bottom: solid #333333;
        background: #202020;
    }
    #ide-shell-out {
        height: 1fr;
        min-height: 3;
        background: transparent;
        padding: 0 1 0 1;
        color: #cccccc;
    }
    #ide-shell-prompt {
        min-height: 1;
        height: auto;
        padding: 0 1;
        margin: 0;
        background: #252526;
        border: solid #3a3a3a;
        border-top: solid #3a3a3a;
    }
    """

    def __init__(
        self, *, raw_repo_input: str | None = None, debug: bool = False
    ) -> None:
        super().__init__()
        self.raw_repo_input = raw_repo_input
        self._ide_debug = bool(debug)
        self._repo_id = None
        self._repo_path = None
        self.ide_state = IDEState()
        self._bottom_visible = True
        self._ide_shell: IdeSubprocessShell | None = None
        self._shell_cwd: str | None = None
        self._ev_debounce = IdeEventDebouncer(
            interval_s=0.1, on_flush=self._on_debounced_events
        )
        self._shell_intro_written = False
        self._active_run_events_static: Static | None = None

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Vertical(id="ide-workspace"):
            with Horizontal(id="ide-top"):
                with Vertical(id="ide-chat", classes="ide-sidebar"):
                    yield Static("Chat", classes="ide-ribbon-label")
                    yield VerticalScroll(id="ide-chat-scroll")
                    with Vertical(id="ide-composer-stack"):
                        yield IdeComposer(
                            "",
                            id="input",
                            placeholder="Message…",
                            show_line_numbers=False,
                            soft_wrap=True,
                        )
                        yield Static("", id="ide-composer-hint")
                with Vertical(id="ide-center"):
                    with Vertical(id="ide-editor-stack"):
                        yield Static("", id="ide-editor-tabs")
                        yield Static("", id="ide-editor-path")
                        yield VerticalScroll(id="ide-editor-body")
                with Vertical(id="ide-right", classes="ide-sidebar"):
                    with Vertical(id="ide-explorer-col"):
                        yield Static("Explorer", classes="ide-ribbon-label")
                        with Container(id="ide-explorer-body"):
                            yield VerticalScroll(id="ide-explorer-scroll")
                        yield Static("Task", classes="ide-ribbon-label")
                        yield VerticalScroll(id="ide-subgoals-scroll")
            with Vertical(id="ide-terminal-stack"):
                if self._ide_debug:
                    yield Static("Debug", classes="ide-ribbon-label")
                    yield RichLog(
                        id="ide-debug-trace",
                        auto_scroll=True,
                        wrap=True,
                        markup=True,
                    )
                yield RichLog(
                    id="ide-shell-out",
                    auto_scroll=True,
                    wrap=True,
                    markup=True,
                )
                yield ShellPromptLine(
                    tip_fn=lambda: self._ide_shell_prompt_tip(),
                    id="ide-shell-prompt",
                )
        yield Footer()

    def on_mount(self) -> None:
        self._ev_debounce.bind_app(self)
        try:
            self.query_one("#input", IdeComposer).focus()
        except Exception:
            pass
        try:
            self.query_one("#ide-composer-hint", Static).update(
                self.ide_state.composer_chrome_hint()
            )
        except Exception:
            pass
        try:
            self.query_one("#ide-shell-out", RichLog).can_focus = True
            if self._ide_debug:
                self.query_one("#ide-debug-trace", RichLog).can_focus = False
        except Exception:
            pass
        try:
            self.query_one("#ide-shell-prompt", ShellPromptLine).refresh_tip()
        except Exception:
            pass
        self._shell_intro_once()
        self.push_screen(IngestionLoaderScreen(self.raw_repo_input))

    def on_unmount(self) -> None:
        sh = getattr(self, "_ide_shell", None)
        if sh is not None:
            import asyncio

            try:
                loop = asyncio.get_running_loop()
                loop.create_task(sh.stop())
            except RuntimeError:
                pass

    def apply_repo_context(self, repo_id: str, repo_path: str) -> None:
        self._repo_id = repo_id
        self._repo_path = repo_path
        if self._repo_path:
            self.sub_title = f"{settings.LLM_MODEL} | {Path(self._repo_path).name}"
        self._dispose_shell_worker()
        self._shell_intro_written = False
        self._setup_explorer_workspace_tree()
        self._mount_chat(chatfmt.ingest_ready_banner())
        self._shell_intro_once()
        self._refresh_all_panels()
        try:
            self.query_one("#ide-shell-prompt", ShellPromptLine).refresh_tip()
        except Exception:
            pass

    def _setup_explorer_workspace_tree(self) -> None:
        """Mount a directory tree above the ranked-context panel when a repo root is known."""
        try:
            body = self.query_one("#ide-explorer-body", Container)
        except Exception:
            return
        body.remove_children()
        rp = (self._repo_path or "").strip()
        if not rp:
            body.mount(VerticalScroll(id="ide-explorer-scroll"))
            return
        try:
            root = Path(rp).expanduser().resolve()
        except OSError:
            body.mount(VerticalScroll(id="ide-explorer-scroll"))
            return
        try:
            body.mount(DirectoryTree(root, id="ide-explorer-tree"))
        except OSError:
            pass
        body.mount(VerticalScroll(id="ide-explorer-scroll"))

    @on(DirectoryTree.FileSelected)
    def on_directory_tree_file_selected(self, event: DirectoryTree.FileSelected) -> None:
        if getattr(event.control, "id", None) != "ide-explorer-tree":
            return
        if not self._repo_path:
            return
        path = event.path
        try:
            path = path.resolve()
        except OSError:
            return
        if not path.is_file():
            return
        try:
            root = Path(self._repo_path).expanduser().resolve()
            rel = path.relative_to(root).as_posix()
        except ValueError:
            return
        self._flush_ide_events()
        self.ide_state.apply_event({"type": "agent_file_open", "path": rel})
        self._ide_focus_editor_after_refresh = True
        self._refresh_all_panels()

    @work(exclusive=True, group="shell_reset")
    async def _dispose_shell_worker(self) -> None:
        if self._ide_shell is not None:
            await self._ide_shell.stop()
        self._ide_shell = None
        self._shell_cwd = None

    def action_toggle_bottom_visibility(self) -> None:
        self._flush_ide_events()
        term = self.query_one("#ide-terminal-stack", Vertical)
        self._bottom_visible = not self._bottom_visible
        term.set_class(not self._bottom_visible, "hidden")

    def action_focus_shell_prompt(self) -> None:
        self._flush_ide_events()
        if not self._bottom_visible:
            self.action_toggle_bottom_visibility()
        try:
            sp = self.query_one("#ide-shell-prompt", ShellPromptLine)
            sp.refresh_tip()
            sp.focus()
        except Exception:
            pass

    def _ide_shell_prompt_tip(self) -> str:
        cwd = self._repo_path or str(Path.cwd().resolve())
        u = os.environ.get("USERNAME") or os.environ.get("USER", "user")
        h = (socket.gethostname() or "host").split(".")[0]
        leaf = Path(cwd).name or cwd
        return f"{u}@{h}:{leaf}$ "

    def _flush_ide_events(self) -> None:
        try:
            self._ev_debounce.flush_now()
        except Exception:
            pass

    def action_next_tab(self) -> None:
        self._flush_ide_events()
        self.ide_state.next_tab()
        self._refresh_all_panels()

    def action_close_tab(self) -> None:
        self._flush_ide_events()
        self.ide_state.close_active_tab()
        self._refresh_all_panels()

    def action_copy_explorer(self) -> None:
        block = (
            self.ide_state.subgoals_tasks_panel_text()
            + "\n\n"
            + self.ide_state.explorer_panel_text()
        )
        if copy_plain_text(strip_rich_markup(block)):
            self._mount_chat(
                chatfmt.system_block("Subgoals / task + explorer copied (plain text).")
            )
        else:
            self._mount_chat(chatfmt.error_block("Clipboard unavailable on this system."))

    def action_copy_editor(self) -> None:
        diff = self._resolve_diff_raw_for_active()
        if diff:
            ok = copy_plain_text(diff[:500_000])
        else:
            p = self._active_file_path()
            ok = False
            if p is not None and p.is_file():
                try:
                    ok = copy_plain_text(
                        p.read_text(encoding="utf-8", errors="replace")[:500_000]
                    )
                except OSError:
                    ok = False
        if ok:
            self._mount_chat(chatfmt.system_block("Editor content copied (plain text)."))
        else:
            self._mount_chat(chatfmt.error_block("Nothing to copy or clipboard failed."))

    def _on_debounced_events(self, batch: list[dict]) -> None:
        self._ide_focus_editor_after_refresh = any(
            e.get("type") == "agent_file_open" for e in batch
        )
        for ev in batch:
            self.ide_state.apply_event(ev)
        self.call_next(self._refresh_all_panels)

    def _chat_scroll(self) -> VerticalScroll:
        return self.query_one("#ide-chat-scroll", VerticalScroll)

    def _refresh_all_panels(self) -> None:
        def _apply() -> None:
            try:
                ex = self.query_one("#ide-explorer-scroll", VerticalScroll)
                ex.remove_children()
                ex.mount(
                    Static(
                        self.ide_state.explorer_panel_text(),
                        classes="chat-line",
                    )
                )
                sg = self.query_one("#ide-subgoals-scroll", VerticalScroll)
                sg.remove_children()
                sg.mount(
                    Static(
                        self.ide_state.subgoals_tasks_panel_text(),
                        classes="chat-line",
                    )
                )
                ar = getattr(self, "_active_run_events_static", None)
                if ar is not None:
                    try:
                        ar.update(self.ide_state.run_events_panel_text())
                    except Exception:
                        pass
                try:
                    hint = self.query_one("#ide-composer-hint", Static)
                    hint.update(self.ide_state.composer_chrome_hint())
                except Exception:
                    pass
                if self._ide_debug:
                    tr = self.query_one("#ide-debug-trace", RichLog)
                    tr.clear()
                    tr.write(self.ide_state.trace_panel_text())
                tabs_w = self.query_one("#ide-editor-tabs", Static)
                tabs_w.update(self._editor_tab_strip_markup())
                path_w = self.query_one("#ide-editor-path", Static)
                af = self.ide_state.active_file or "(no file)"
                diff_raw = self._resolve_diff_raw_for_active()
                path_w.update(
                    self._editor_path_bar_markup(
                        af,
                        diff_mode=bool(
                            self.ide_state.diff_mode and diff_raw and af != "(no file)"
                        ),
                        diff_raw=diff_raw,
                    )
                )
                ed = self.query_one("#ide-editor-body", VerticalScroll)
                ed.remove_children()
                body = self._editor_markup()
                ed.mount(Static(body, classes="chat-line"))
                if getattr(self, "_ide_focus_editor_after_refresh", False):
                    self._ide_focus_editor_after_refresh = False
                    try:
                        ed.focus()
                    except Exception:
                        pass
            except Exception:
                pass

        self.call_next(_apply)

    @staticmethod
    def _session_diff_references_path(session_diff: str, af: str) -> bool:
        if not session_diff or not af or af == "_session":
            return False
        a = af.replace("\\", "/").strip()
        if _diff_covers_active_path(session_diff, a):
            return True
        blob = session_diff.replace("\\", "/")
        return a in blob

    def _resolve_diff_raw_for_active(self) -> str:
        s = self.ide_state
        af = (s.active_file or "").strip()
        pd = s.pending_diff
        if not pd:
            return ""
        if af and af != "_session":
            d = pd.get(af)
            if d:
                return str(d)
        sess = pd.get("_session")
        if sess and af and af != "_session":
            ss = str(sess)
            if self._session_diff_references_path(ss, af):
                return ss
            return ""
        if sess and (not af or af == "_session"):
            return str(sess)
        if af:
            return str(pd.get(af) or "")
        return ""

    def _active_file_path(self) -> Path | None:
        root = self._repo_path
        af = (self.ide_state.active_file or "").strip()
        if not root or not af or af == "_session":
            return None
        rel = af.replace("\\", "/").lstrip("/")
        try:
            root_p = Path(root).expanduser().resolve()
            cand = (root_p / rel).resolve()
            if root_p not in cand.parents and cand != root_p:
                return None
            return cand if cand.is_file() else None
        except (OSError, ValueError):
            return None

    @staticmethod
    def _diff_line_markup(line: str) -> str:
        if line.startswith("+") and not line.startswith("+++"):
            return f"[green]{escape(line)}[/green]"
        if line.startswith("-") and not line.startswith("---"):
            return f"[red]{escape(line)}[/red]"
        if line.startswith("@@"):
            return f"[cyan]{escape(line)}[/cyan]"
        if line.startswith("--- ") or line.startswith("+++ "):
            return f"[yellow]{escape(line)}[/yellow]"
        return f"[dim]{escape(line)}[/dim]"

    @staticmethod
    def _diff_plus_minus(diff: str) -> tuple[int, int]:
        if not diff:
            return 0, 0
        adds = dels = 0
        for line in diff.splitlines():
            if line.startswith(("+++ ", "@@")):
                continue
            if line.startswith("+") and not line.startswith("+++"):
                adds += 1
            elif line.startswith("-") and not line.startswith("---"):
                dels += 1
        return adds, dels

    @staticmethod
    def _filename_from_tab_path(tab_path: str) -> str:
        p = (tab_path or "").replace("\\", "/").strip()
        if p == "_session":
            return "_session"
        return Path(p).name or p or "?"

    @staticmethod
    def _breadcrumb_markup(rel: str) -> str:
        if not rel or rel == "(no file)":
            return "[dim]—[/dim]"
        r = rel.replace("\\", "/").strip()
        if r == "_session":
            return "[dim]_session[/dim]"
        parts = [x for x in r.split("/") if x]
        if not parts:
            return escape(rel)
        if len(parts) == 1:
            return f"[bold]{escape(parts[0])}[/bold]"
        head = " › ".join(escape(x) for x in parts[:-1])
        tail = parts[-1]
        return f"[dim]{head} ›[/dim] [bold]{escape(tail)}[/bold]"

    def _editor_tab_strip_markup(self) -> str:
        st = self.ide_state
        tabs = st.tabs
        hint = "  [dim]· Ctrl+I · W[/dim]"
        if not tabs:
            return f"[dim]—[/dim]{hint}"
        parts: list[str] = []
        for i, t in enumerate(tabs):
            label = self._filename_from_tab_path(t)
            if i == st.active_tab_index:
                parts.append(f"[reverse bold] {escape(label)} [/reverse bold]")
            else:
                parts.append(f"[dim]{escape(label)}[/dim]")
        return "  ".join(parts) + hint

    def _editor_path_bar_markup(
        self, active_display: str, *, diff_mode: bool, diff_raw: str
    ) -> str:
        af = active_display
        if diff_mode and diff_raw and af not in ("(no file)", "") and af != "_session":
            plus, minus = self._diff_plus_minus(diff_raw)
            crumb = self._breadcrumb_markup(af)
            return f"[yellow]DIFF[/yellow] [dim]+{plus} −{minus}[/dim]  {crumb}"
        if not af or af == "(no file)":
            return "[dim]—[/dim]"
        return self._breadcrumb_markup(af)

    def _editor_markup(self) -> str:
        self.ide_state.ensure_active_file_fallback()
        af = self.ide_state.active_file
        if not af:
            return (
                "[dim]No file open — use Explorer (tree), agent read_file, or run a task.[/dim]"
            )
        if self.ide_state.diff_mode:
            diff = self._resolve_diff_raw_for_active()
            if diff:
                lines = diff[:14000].splitlines()
                return "\n".join(self._diff_line_markup(L) for L in lines[:500])
        fp = self._active_file_path()
        if fp is None:
            root = self._repo_path or ""
            if not root:
                return escape(af)
            return f"[dim]{escape(af)} — not on disk or outside repo root[/dim]"
        try:
            text = fp.read_text(encoding="utf-8", errors="replace")
        except OSError as e:
            return f"[red]{escape(str(e))}[/red]"
        cap = text[:24000]
        if len(text) > 24000:
            cap += "\n\n[dim]… truncated …[/dim]"
        return escape(cap)

    def _mount_chat(self, markup: str) -> Static:
        w = Static(markup, classes="chat-line")
        self._chat_scroll().mount(w)
        self._chat_scroll().scroll_end(animate=False)
        return w

    def _clear_chat_only(self) -> None:
        self._chat_scroll().remove_children()
        self._active_run_events_static = None

    def _shell_intro_once(self) -> None:
        if getattr(self, "_shell_intro_written", False):
            return
        try:
            lg = self.query_one("#ide-shell-out", RichLog)
            lg.write(
                "[dim]Interactive shell — click the prompt row below or Ctrl+Shift+T, "
                "then type and press Enter (one line per run).[/dim]"
            )
            self._shell_intro_written = True
        except Exception:
            pass

    def action_clear_chat(self) -> None:
        self._clear_chat_only()

    def _is_terminal_focused(self) -> bool:
        """True when focus is in the bottom terminal (prompt, log, or terminal stack)."""
        try:
            w = self.screen.focused
            if w is None:
                return False
            wid = getattr(w, "id", None) or ""
            if wid in ("ide-shell-prompt", "ide-shell-out"):
                return True
            node: Any = w
            for _ in range(16):
                pid = getattr(node, "id", None) or ""
                if pid == "ide-terminal-stack":
                    return True
                node = getattr(node, "parent", None)
                if node is None:
                    break
            return False
        except Exception:
            return False

    @work(exclusive=False, group="ide_shell")
    async def _shell_interrupt_worker(self) -> None:
        sh = getattr(self, "_ide_shell", None)
        if sh is None:
            return
        await sh.start()
        await sh.send_interrupt()

        def _log() -> None:
            try:
                self.query_one("#ide-shell-out", RichLog).write(
                    "[dim]^C  [interrupt sent to shell][/dim]"
                )
            except Exception:
                pass

        self.call_next(_log)

    def action_cancel_agent(self) -> None:
        if self._is_terminal_focused():
            self._shell_interrupt_worker()
            return
        if getattr(self, "_orchestrator_busy", False):
            ev = getattr(self, "_react_cancel_event", None)
            if ev is not None and hasattr(ev, "set"):
                ev.set()
                self._mount_chat(chatfmt.system_block("Cancelling current run…"))
        else:
            self.exit()

    @work
    async def action_command_palette(self) -> None:
        result = await self.push_screen_wait(CommandPaletteScreen())
        if result is not None:
            if result.endswith(" "):
                inp = self.query_one("#input", IdeComposer)
                inp.text = result
                inp.focus()
            else:
                self._handle_slash(result)

    @on(IdeComposer.Changed)
    def on_ide_composer_changed(self, event: IdeComposer.Changed) -> None:
        ta = event.text_area
        if getattr(ta, "id", None) != "input":
            return
        if ta.text == "/":
            ta.text = ""
            self.action_command_palette()

    @work(exclusive=False, group="ide_shell")
    async def _run_shell_command(self, line: str) -> None:
        def append_out(msg: str) -> None:
            try:
                self.query_one("#ide-shell-out", RichLog).write(format_shell_line(msg))
            except Exception:
                pass

        cwd = self._repo_path or str(Path.cwd().resolve())
        if self._ide_shell is None or self._shell_cwd != cwd:
            if self._ide_shell is not None:
                await self._ide_shell.stop()
            self._ide_shell = IdeSubprocessShell(
                cwd=cwd,
                on_line=lambda t: self.call_next(lambda: append_out(t)),
            )
            self._shell_cwd = cwd
            u = os.environ.get("USERNAME") or os.environ.get("USER", "user")
            h = (socket.gethostname() or "host").split(".")[0]
            leaf = Path(cwd).name or cwd
            tip = f"{u}@{h}:{leaf}$ "
            self.call_next(lambda: append_out(tip))

        self.call_next(lambda: append_out(f"> {line}"))
        await self._ide_shell.send_line(line)
        self.call_next(self._refresh_shell_prompt_tip)

    def _refresh_shell_prompt_tip(self) -> None:
        try:
            self.query_one("#ide-shell-prompt", ShellPromptLine).refresh_tip()
        except Exception:
            pass

    @on(ShellPromptLine.CommandSubmitted)
    def _on_shell_prompt_command(self, event: ShellPromptLine.CommandSubmitted) -> None:
        raw = (event.line or "").strip()
        if raw:
            self._run_shell_command(raw)

    @on(IdeComposer.Submitted)
    def on_ide_composer_submitted(self, event: IdeComposer.Submitted) -> None:
        if getattr(self, "_orchestrator_busy", False):
            return
        raw = (event.text or "").strip()
        try:
            self.query_one("#input", IdeComposer).text = ""
        except Exception:
            pass
        if not raw:
            return
        mode = "ask"
        auto_apply = False
        dry_run = False
        long_run = False

        if raw.startswith("/"):
            parts = raw.split(maxsplit=1)
            cmd = parts[0].lower()
            arg = parts[1].strip() if len(parts) > 1 else ""
            if cmd in ("/plan", "/edit", "/ask", "/build", "/run"):
                if cmd == "/run":
                    long_run = True
                    if "--auto-apply" in arg:
                        auto_apply = True
                        arg = arg.replace("--auto-apply", "").strip()
                    if "--dry-run" in arg:
                        dry_run = True
                        arg = arg.replace("--dry-run", "").strip()
                    if not arg:
                        self._mount_chat(
                            chatfmt.error_block('Usage: /run "your macro goal"')
                        )
                        return
                    self.run_orchestrator(
                        arg,
                        mode="edit",
                        auto_apply=auto_apply,
                        dry_run=dry_run,
                        long_run=True,
                    )
                    return
                mode = cmd[1:]
                if mode == "build":
                    mode = "edit"
                if "--auto-apply" in arg:
                    auto_apply = True
                    arg = arg.replace("--auto-apply", "").strip()
                if "--dry-run" in arg:
                    dry_run = True
                    arg = arg.replace("--dry-run", "").strip()
                if not arg:
                    self._mount_chat(chatfmt.error_block(f"Usage: {cmd} <query>"))
                    return
                self.run_orchestrator(arg, mode=mode, auto_apply=auto_apply, dry_run=dry_run)
                return
            self._handle_slash(raw)
            return
        self.run_orchestrator(raw, mode=mode, auto_apply=auto_apply, dry_run=dry_run)

    def _handle_slash(self, line: str) -> None:
        parts = line.split(maxsplit=1)
        cmd = parts[0].lower()
        arg = parts[1].strip() if len(parts) > 1 else ""

        if cmd in ("/help", "/?"):
            self._mount_chat(
                chatfmt.system_block(
                    "IDE: /help /metrics /clear /browse /repo /run /edit /plan …\n"
                    "Ctrl+P palette  |  Ctrl+B terminal panel  |  Ctrl+Shift+T shell prompt  |  "
                    "Ctrl+I tab  |  Ctrl+W close tab  |  Ctrl+L clear chat  |  "
                    "Ctrl+Shift+E/D copy explorer / editor"
                )
            )
            return
        if cmd == "/clear":
            self._clear_chat_only()
            self._mount_chat(chatfmt.system_block("Chat cleared."))
            return
        if cmd == "/metrics":
            from app.telemetry.metrics_engine import metrics_engine

            metrics_engine.load()
            snap = metrics_engine.snapshot()
            blob = json.dumps(snap, indent=2, default=str)[:12000]
            self._mount_chat(
                chatfmt.system_block(f"[metrics]\n{escape(blob)}")
            )
            return
        if cmd == "/browse":
            self._push_file_explorer()
            return
        if cmd in ("/repos", "/open", "/find", "/explain"):
            if cmd == "/open" and not arg:
                self._push_file_explorer()
                return
            if cmd != "/repos" and arg:
                self.run_orchestrator(f"{cmd[1:]} {arg}", mode="ask")
                return
            if cmd == "/repos":
                from app.core.repo_registry import load_registry

                registry = load_registry()
                if not registry:
                    self._mount_chat(chatfmt.system_block("No repos registered."))
                    return
                lines = [f"{rid[:8]} —> {entry.get('path', 'unknown')}" for rid, entry in registry.items()]
                self._mount_chat(
                    chatfmt.system_block("[repos]\n" + escape("\n".join(lines)))
                )
                return
        if cmd == "/repo":
            if not arg:
                self._mount_chat(
                    chatfmt.error_block("Usage: /repo <repo_name> (matches ingest name)")
                )
                return
            self._repo_id = arg
            self._mount_chat(chatfmt.system_block(f"Repo set to: {arg}"))
            return
        self._mount_chat(chatfmt.error_block(f"Unknown command: {cmd}"))

    @work(exclusive=False, group="file_explorer")
    async def _push_file_explorer(self) -> None:
        self._flush_ide_events()
        from app.cli.tui.file_explorer import FileExplorerScreen

        root = self._repo_path or str(Path.cwd().resolve())
        screen = FileExplorerScreen(root, repo_id=self._repo_id)
        await self.push_screen_wait(screen)

    @work(exclusive=True, group="orchestrator", name="ide_ask")
    async def run_orchestrator(
        self,
        question: str,
        mode: str = "ask",
        auto_apply: bool = False,
        dry_run: bool = False,
        long_run: bool = False,
    ) -> None:
        self._mount_chat(chatfmt.user_block(question))
        answer_body = Static(chatfmt.assistant_label() + "\n", classes="chat-line")
        events_w = Static(
            self.ide_state.run_events_panel_text(),
            classes="chat-line ide-chat-events",
        )
        self._active_run_events_static = events_w
        scroll = self._chat_scroll()
        scroll.mount(answer_body)
        scroll.mount(events_w)
        scroll.scroll_end(animate=False)
        self._final_answer_via_bus = False

        stream_buffer = ""
        import time

        last_update_t = time.perf_counter()

        def _apply_answer_markup(markup: str) -> None:
            answer_body.update(markup)
            self._chat_scroll().scroll_end(animate=False)

        async def _stream_callback(chunk: str) -> None:
            nonlocal stream_buffer, last_update_t
            stream_buffer += chunk
            if time.perf_counter() - last_update_t > 0.05:
                buf = stream_buffer

                def _push_partial() -> None:
                    _apply_answer_markup(
                        chatfmt.assistant_label() + "\n" + escape(buf)
                    )

                self.call_next(_push_partial)
                last_update_t = time.perf_counter()

        async def _confirm_diff_callback(
            combined_diff: str, *, confirm_reasons: list[str] | None = None
        ) -> bool:
            meta = getattr(self, "ide_state", None)
            if meta and combined_diff:
                self.ide_state.pending_diff["_session"] = combined_diff
                self.ide_state.diff_mode = True
                self.call_next(self._refresh_all_panels)
            self.ide_state.last_confirm_reasons = list(confirm_reasons or [])
            self.ide_state.ui_phase = "awaiting_confirm"
            self.call_next(self._refresh_all_panels)
            try:
                return await self.push_screen_wait(
                    ConfirmDiffScreen(
                        diff_text=combined_diff, confirm_reasons=confirm_reasons
                    )
                )
            finally:
                self.ide_state.ui_phase = "running"
                self.call_next(self._refresh_all_panels)

        async def _confirm_shell_callback(
            command: str,
            *,
            title: str | None = None,
            cwd: str | None = None,
            tag_labels: list[str] | None = None,
            warning: str | None = None,
            risk: str | None = None,
        ) -> str:
            self.ide_state.ui_phase = "awaiting_shell"
            self.call_next(self._refresh_all_panels)
            try:
                return await self.push_screen_wait(
                    ConfirmShellScreen(
                        command=command,
                        title=title,
                        tag_labels=tag_labels,
                        cwd_hint=cwd,
                        warning=warning,
                        risk=risk,
                    )
                )
            finally:
                self.ide_state.ui_phase = "running"
                self.call_next(self._refresh_all_panels)

        async def _status_callback(msg: str) -> None:
            self.ide_state.orchestrator_status_line = str(msg)[:280]
            self.call_next(self._refresh_all_panels)

        from app.core.event_bus import EventBus

        async def _on_ide_event(event: dict) -> None:
            t = event.get("type")
            if t == "final_answer":
                content = str(event.get("content") or "")
                self._final_answer_via_bus = True
                try:
                    self._ev_debounce.flush_now()
                except Exception:
                    pass

                def _apply_final() -> None:
                    answer_body.update(
                        chatfmt.assistant_label() + "\n" + escape(content)
                    )
                    self._chat_scroll().scroll_end(animate=False)

                self.call_next(_apply_final)
                return
            self._ev_debounce.push(event)

        import asyncio

        self._react_cancel_event = asyncio.Event()
        self._orchestrator_busy = True
        self.ide_state.ui_phase = "running"
        self.ide_state.terminal_log.clear()
        self.ide_state.orchestrator_status_line = ""
        try:
            self.query_one("#input", IdeComposer).disabled = True
        except Exception:
            pass

        EventBus.subscribe(_on_ide_event)
        try:
            if long_run:
                result = await run_long_goal_async(
                    question,
                    debug=True,
                    context=None,
                    repo_id=self._repo_id,
                    auto_apply=auto_apply,
                    dry_run=dry_run,
                    stream_callback=_stream_callback,
                    confirm_diff_callback=_confirm_diff_callback,
                    confirm_shell_callback=_confirm_shell_callback,
                    status_callback=_status_callback,
                    react_cancel_event=self._react_cancel_event,
                )
            else:
                result = await run_ask_async(
                    question,
                    debug=True,
                    context=None,
                    repo_id=self._repo_id,
                    mode=mode,
                    auto_apply=auto_apply,
                    dry_run=dry_run,
                    stream_callback=_stream_callback,
                    confirm_diff_callback=_confirm_diff_callback,
                    confirm_shell_callback=_confirm_shell_callback,
                    status_callback=_status_callback,
                    react_cancel_event=self._react_cancel_event,
                )
        except Exception as exc:
            answer_body.update(chatfmt.error_block(str(exc)))
            self._chat_scroll().scroll_end(animate=False)
            return
        finally:
            try:
                self._ev_debounce.flush_now()
            except Exception:
                pass
            EventBus.unsubscribe(_on_ide_event)
            self._orchestrator_busy = False
            self.ide_state.ui_phase = "idle"
            self.ide_state.orchestrator_status_line = ""
            self.call_next(self._refresh_all_panels)

            def _unlock() -> None:
                try:
                    w = self.query_one("#input", IdeComposer)
                    w.disabled = False
                    w.focus()
                except Exception:
                    pass

            self.call_next(_unlock)

        answer = result.get("answer") or ""
        if not getattr(self, "_final_answer_via_bus", False):
            _apply_answer_markup(chatfmt.assistant_label() + "\n" + escape(answer))
        meta = result.get("meta")
        if meta:
            self.ide_state.apply_meta(meta)
            ac = meta.get("answer_confidence")
            if ac is not None:
                self.ide_state.confidence = float(ac)
            if meta.get("require_user_confirm"):
                self._mount_chat(
                    chatfmt.system_block(
                        "Edits may require confirmation when confidence is low "
                        "(orchestrator enforces modal vs auto-apply)."
                    )
                )
            esr = meta.get("early_stop_reason") or meta.get("react_early_stop_reason")
            if esr and esr in (
                "high_confidence",
                "planner_done",
                "enough_context",
                "max_steps_exhausted",
            ):
                nfr = len(meta.get("files_touched") or [])
                nsy = len(meta.get("symbols_used") or [])
                if nfr < 2 and nsy == 0 and float(meta.get("answer_confidence") or 0) < 0.55:
                    self._mount_chat(
                        chatfmt.system_block(
                            f"Note: early stop ({esr}) with weak file/symbol grounding."
                        )
                    )
        self.call_next(self._refresh_all_panels)

        if meta and meta.get("long_run"):
            lr = meta.get("goal_plan") or []
            st = meta.get("goal_status") or "?"
            lines = [f"Long-run status={st}\n"]
            for t in lr[:14]:
                if isinstance(t, dict):
                    lines.append(
                        f"  {t.get('id', '?')}: {str(t.get('description', ''))[:48]} "
                        f"[{t.get('status', '')}]\n"
                    )
            self._mount_chat(chatfmt.system_block(escape("".join(lines))))

        self._chat_scroll().scroll_end(animate=False)


def run_ide_tui(
    *, raw_repo_input: str | None = None, debug: bool = False
) -> None:
    from app.core.logger import quiet_app_loggers_for_tui

    quiet_app_loggers_for_tui()
    DeepCodeIdeApp(raw_repo_input=raw_repo_input, debug=debug).run()
