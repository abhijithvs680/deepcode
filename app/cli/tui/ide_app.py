"""
DeepCode V2 — agent IDE runtime (Textual): chat | editor | explorer + terminal.

Run: deepcode ui --ide [REPO]
"""
from __future__ import annotations

import json
from pathlib import Path

from rich.markup import escape
from textual import work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.widgets import Footer, Header, Input, RichLog, Static

from app.cli.commands.ask import run_ask_async, run_long_goal_async
from app.cli.tui.app import ConfirmDiffScreen, ConfirmShellScreen, IngestionLoaderScreen
from app.cli.tui.clipboard_util import copy_plain_text, strip_rich_markup
from app.cli.tui.ide_event_buffer import IdeEventDebouncer
from app.cli.tui.ide_shell import IdeSubprocessShell, format_shell_line
from app.cli.tui.ide_state import IDEState
from app.cli.tui.widgets import chat as chatfmt
from app.cli.tui.widgets.command_palette import CommandPaletteScreen
from app.core.config import get_settings

settings = get_settings()


class DeepCodeIdeApp(App):
    """Grid-style agent IDE: same orchestrator as classic TUI, IDE-centric state + panels."""

    TITLE = "DeepCode IDE"
    SUB_TITLE = f"{settings.LLM_PROVIDER} / {settings.LLM_MODEL}"

    BINDINGS = [
        Binding("ctrl+p", "command_palette", "Commands", show=True),
        Binding("ctrl+l", "clear_chat", "Clear chat", show=True),
        Binding("ctrl+b", "toggle_bottom_visibility", "Terminal", show=True),
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
    Screen { layout: vertical; }
    Header { background: $primary 10%; }
    Footer { background: $panel; color: $text-muted; }
    #ide-top { height: 1fr; layout: horizontal; min-height: 8; }
    #ide-chat { width: 1fr; height: 1fr; border-right: outer $primary 25%; }
    #ide-editor-col { width: 2fr; height: 1fr; border-right: outer $primary 25%; }
    #ide-explorer-col { width: 1fr; height: 1fr; }
    .ide-panel-title {
        background: $primary 14%;
        text-style: bold;
        padding: 0 1;
        height: 1;
    }
    #ide-chat-scroll { height: 1fr; padding: 0 1; }
    #ide-editor-path { padding: 0 1; background: $panel; height: 1; }
    #ide-editor-body { height: 1fr; padding: 0 1; }
    #ide-explorer-scroll { height: 1fr; padding: 0 1; }
    #ide-orchestrator-bar {
        height: auto;
        min-height: 1;
        padding: 0 1;
        background: $warning 8%;
        border-bottom: solid $primary 20%;
    }
    #ide-debug-trace {
        height: 8;
        min-height: 4;
        border-top: solid $primary 15%;
        border-bottom: solid $primary 15%;
    }
    #ide-bottom-dock {
        height: 10;
        min-height: 4;
        border-top: outer $primary 30%;
        layout: vertical;
        padding: 0 1;
    }
    #ide-bottom-dock.hidden { display: none; height: 0; min-height: 0; border: none; }
    #ide-shell-out { height: 1fr; min-height: 3; }
    #ide-shell-in { height: 1; margin-top: 1; }
    #ide-editor-tabs { height: 1; padding: 0 1; background: $panel 80%; }
    #input { border: tall $primary 30%; margin: 0 1 1 1; dock: bottom; }
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

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Horizontal(id="ide-top"):
            with Vertical(id="ide-chat"):
                yield Static("Chat", classes="ide-panel-title")
                yield VerticalScroll(id="ide-chat-scroll")
            with Vertical(id="ide-editor-col"):
                yield Static("Editor / Diff", classes="ide-panel-title")
                yield Static("", id="ide-editor-tabs")
                yield Static("", id="ide-editor-path")
                yield VerticalScroll(id="ide-editor-body")
            with Vertical(id="ide-explorer-col"):
                yield Static("Explorer", classes="ide-panel-title")
                yield VerticalScroll(id="ide-explorer-scroll")
        with Vertical(id="ide-bottom-dock"):
            yield Static("", id="ide-orchestrator-bar")
            if self._ide_debug:
                yield Static("Debug trace", classes="ide-panel-title")
                yield RichLog(
                    id="ide-debug-trace",
                    auto_scroll=True,
                    wrap=True,
                    markup=True,
                )
            yield Static("Terminal", classes="ide-panel-title")
            yield RichLog(id="ide-shell-out", auto_scroll=True, wrap=True, markup=True)
            yield Input(placeholder="Shell command… Enter", id="ide-shell-in")
        yield Input(
            placeholder="Message…  Ctrl+P  /ask /edit /run /plan /metrics",
            id="input",
        )
        yield Footer()

    def on_mount(self) -> None:
        self._ev_debounce.bind_app(self)
        self.query_one("#input", Input).focus()
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
        self._mount_chat(chatfmt.ingest_ready_banner())
        self._refresh_all_panels()

    @work(exclusive=True, group="shell_reset")
    async def _dispose_shell_worker(self) -> None:
        if self._ide_shell is not None:
            await self._ide_shell.stop()
        self._ide_shell = None
        self._shell_cwd = None

    def action_toggle_bottom_visibility(self) -> None:
        self._flush_ide_events()
        dock = self.query_one("#ide-bottom-dock", Vertical)
        self._bottom_visible = not self._bottom_visible
        dock.set_class(not self._bottom_visible, "hidden")

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
            self.ide_state.timeline_text()
            + "\n\n"
            + self.ide_state.explorer_text()
        )
        if copy_plain_text(strip_rich_markup(block)):
            self._mount_chat(chatfmt.system_block("Explorer + timeline copied (plain text)."))
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
                        "[bold]Subgoals[/bold]\n"
                        + self.ide_state.timeline_text()
                        + "\n\n"
                        + self.ide_state.explorer_text(),
                        classes="chat-line",
                    )
                )
                obar = self.query_one("#ide-orchestrator-bar", Static)
                obar.update(self.ide_state.status_bar_text())
                if self._ide_debug:
                    tr = self.query_one("#ide-debug-trace", RichLog)
                    tr.clear()
                    tr.write(self.ide_state.trace_panel_text())
                tabs_w = self.query_one("#ide-editor-tabs", Static)
                nt = len(self.ide_state.tabs)
                cur = self.ide_state.active_tab_index + 1 if nt else 0
                tab_hint = (
                    " | ".join(
                        (
                            f"[bold]{escape(t)}[/bold]"
                            if i == self.ide_state.active_tab_index
                            else escape(t)
                        )
                        for i, t in enumerate(self.ide_state.tabs)
                    )
                    if self.ide_state.tabs
                    else "[dim](no tabs)[/dim]"
                )
                tabs_w.update(
                    f"Tabs {cur}/{nt}  [dim]Ctrl+I[/dim] [dim]Ctrl+W[/dim] close  {tab_hint}"
                )
                path_w = self.query_one("#ide-editor-path", Static)
                af = self.ide_state.active_file or "(no file)"
                diff_raw = self._resolve_diff_raw_for_active()
                if self.ide_state.diff_mode and diff_raw and af != "(no file)":
                    plus, minus = self._diff_plus_minus(diff_raw)
                    path_w.update(
                        f"[DIFF +{plus} -{minus}] {escape(af)}"
                    )
                else:
                    mode = "view"
                    path_w.update(f"[{mode}] {escape(af)}")
                ed = self.query_one("#ide-editor-body", VerticalScroll)
                ed.remove_children()
                body = self._editor_markup()
                ed.mount(Static(body, classes="chat-line"))
            except Exception:
                pass

        self.call_next(_apply)

    @staticmethod
    def _session_diff_references_path(session_diff: str, af: str) -> bool:
        if not session_diff or not af or af == "_session":
            return False
        a = af.replace("\\", "/")
        blob = session_diff.replace("\\", "/")
        if a in blob:
            return True
        base = Path(a).name
        return bool(base) and base in blob

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

    def _editor_markup(self) -> str:
        af = self.ide_state.active_file
        if not af:
            return "[dim]Open a file via agent read_file or explorer.[/dim]"
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

    def action_clear_chat(self) -> None:
        self._clear_chat_only()

    def action_cancel_agent(self) -> None:
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
                inp = self.query_one("#input", Input)
                inp.value = result
                inp.focus()
                try:
                    inp.cursor_position = len(result)
                except Exception:
                    pass
            else:
                self._handle_slash(result)

    async def on_input_changed(self, event: Input.Changed) -> None:
        if event.input.id == "input" and event.value == "/":
            event.input.value = ""
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
            self.call_next(lambda: append_out(f"[cwd {cwd}]"))

        self.call_next(lambda: append_out(f"> {line}"))
        await self._ide_shell.send_line(line)

    def on_input_submitted(self, event: Input.Submitted) -> None:
        if event.input.id == "ide-shell-in":
            raw = event.value.strip()
            event.input.value = ""
            if raw:
                self._run_shell_command(raw)
            return
        if getattr(self, "_orchestrator_busy", False):
            return
        if event.input.id != "input":
            return
        raw = event.value.strip()
        event.input.value = ""
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
                    "Ctrl+P palette  |  Ctrl+B hide/show terminal  |  Ctrl+I tab  |  "
                    "Ctrl+W close tab  |  Ctrl+L clear chat  |  Ctrl+Shift+E/D copy explorer / editor"
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
        answer_widget = Static(chatfmt.assistant_label() + "\n", classes="chat-line")
        self._chat_scroll().mount(answer_widget)
        self._chat_scroll().scroll_end(animate=False)
        self._final_answer_via_bus = False

        stream_buffer = ""
        import time

        last_update_t = time.perf_counter()

        def _apply_answer_markup(markup: str) -> None:
            answer_widget.update(markup)
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
                    answer_widget.update(
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
            self.query_one("#input", Input).disabled = True
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
            answer_widget.update(chatfmt.error_block(str(exc)))
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

            def _unlock() -> None:
                try:
                    w = self.query_one("#input", Input)
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
