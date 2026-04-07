"""
DeepCode interactive TUI (Textual): chat + intelligence panel + command palette.

Run: deepcode ui
"""
from __future__ import annotations

import json
from pathlib import Path

from rich.markup import escape
from textual import work
from textual.app import App, ComposeResult
from textual.binding import Binding

from textual.containers import Container, Horizontal, Vertical, VerticalScroll
from textual.widgets import Footer, Header, Input, Static, Button, RichLog
from textual.screen import ModalScreen

class ConfirmDiffScreen(ModalScreen[bool]):
    def __init__(self, diff_text: str):
        super().__init__()
        self.diff_text = diff_text

    def compose(self) -> ComposeResult:
        with Vertical(id="diff-dialog"):
            yield Static("⚠ [bold]Proposed File Modifications[/bold]")
            yield RichLog(id="diff-view")
            with Horizontal(id="diff-dialog-buttons"):
                yield Button("Apply Patch (Y)", id="btn_apply", variant="success")
                yield Button("Discard (N)", id="btn_discard", variant="error")

    def on_mount(self) -> None:
        log = self.query_one("#diff-view", RichLog)
        from rich.syntax import Syntax
        log.write(Syntax(self.diff_text, "diff", theme="monokai", word_wrap=True))
        
    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "btn_apply":
            self.dismiss(True)
        else:
            self.dismiss(False)

from app.cli.commands.ask import run_ask_async
from app.cli.tui.widgets import chat as chatfmt
from app.cli.tui.widgets.command_palette import CommandPaletteScreen
from app.core.config import get_settings

class IngestionLoaderScreen(ModalScreen):
    CSS = """
    IngestionLoaderScreen {
        align: center middle;
    }
    #ingest-backdrop {
        width: 100%;
        height: 100%;
        align: center middle;
        background: $background 70%;
    }
    #ingestion-dialog {
        width: 72;
        max-width: 92%;
        height: auto;
        max-height: 88%;
        background: $surface;
        border: thick $primary;
        padding: 1 2;
        layout: vertical;
        align: center middle;
    }
    #ingest-title {
        text-align: center;
        width: 100%;
        margin-bottom: 1;
    }
    #ingest-current {
        text-align: center;
        width: 100%;
        height: auto;
        min-height: 3;
        margin: 1 0;
        padding: 0 1;
        background: $primary 12%;
        border: solid $primary 35%;
    }
    #ingest-log {
        width: 100%;
        height: 16;
        min-height: 8;
        border: solid $primary 40%;
        background: $surface;
        margin-top: 1;
        padding: 0 1;
    }
    """

    def __init__(self, raw_repo_input: str | None):
        super().__init__()
        self.raw_repo_input = raw_repo_input

    def compose(self) -> ComposeResult:
        with Container(id="ingest-backdrop"):
            with Vertical(id="ingestion-dialog"):
                yield Static(
                    "🚀 [bold magenta]Initializing DeepCode[/bold magenta]\n"
                    "[dim]Repository scan, chunking, embeddings, and intel[/dim]",
                    id="ingest-title",
                )
                yield Static(
                    "[dim]Starting…[/dim]",
                    id="ingest-current",
                )
                yield RichLog(id="ingest-log", auto_scroll=True, wrap=True, markup=True)

    def _schedule_status(self, msg: str) -> None:
        """Marshal UI updates onto the Textual message pump (safe from workers)."""
        safe = escape(msg)

        def _apply() -> None:
            try:
                self.query_one("#ingest-current", Static).update(
                    f"[bold cyan]{safe}[/bold cyan]"
                )
                self.query_one("#ingest-log", RichLog).write(f"[dim]•[/dim] {safe}")
            except Exception:
                pass

        try:
            self.app.call_next(_apply)
        except Exception:
            _apply()

    async def on_mount(self) -> None:
        self.run_ingestion()

    @work
    async def run_ingestion(self) -> None:
        from app.core.repo_manager import resolve_and_prepare_repo_async

        def _echo(msg: str) -> None:
            self._schedule_status(msg)

        repo_path, repo_id = await resolve_and_prepare_repo_async(
            self.raw_repo_input, echo=_echo
        )
        self.app.apply_repo_context(repo_id, repo_path)
        self.dismiss()

settings = get_settings()
TRACE_DIR = Path(".ai_assistant/logs")


class DeepCodeTuiApp(App):
    """Split chat / trace-meta panel with slash commands and Ctrl+P palette."""

    TITLE = "DeepCode Agent"
    SUB_TITLE = f"{settings.LLM_PROVIDER} / {settings.LLM_MODEL}"

    BINDINGS = [
        Binding("ctrl+p", "command_palette", "Commands", show=True),
        Binding("ctrl+l", "clear_chat", "Clear chat", show=True),
    ]

    CSS = """
    Screen {
        layout: vertical;
    }
    #main-row {
        height: 1fr;
        layout: horizontal;
    }
    #chat-col {
        width: 3fr;
        height: 1fr;
        layout: vertical;
    }
    #trace-col {
        width: 1fr;
        height: 1fr;
        layout: vertical;
        border-left: outer $primary;
        background: $surface;
    }
    #chat-scroll {
        height: 1fr;
        border: tall $success 30%;
        padding: 0 1;
    }
    #trace-scroll {
        height: 1fr;
        padding: 0 1;
    }
    .panel-title {
        text-style: bold;
        padding: 0 1;
        background: $primary 15%;
    }
    #input {
        dock: bottom;
        margin: 0 1 1 1;
    }
    Static.chat-line {
        margin-bottom: 1;
        height: auto;
    }
    #diff-dialog {
        background: $surface;
        border: thick $primary;
        width: 80%;
        height: 80%;
        align: center middle;
    }
    #diff-view {
        height: 1fr;
        border: solid $accent;
    }
    #diff-dialog-buttons {
        height: 3;
        align: center middle;
    }
    """

    def __init__(
        self,
        *,
        raw_repo_input: str | None = None,
    ) -> None:
        super().__init__()
        self.raw_repo_input = raw_repo_input
        self._repo_id = None
        self._repo_path = None

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Horizontal(id="main-row"):
            with Vertical(id="chat-col"):
                yield Static("Chat", classes="panel-title")
                yield VerticalScroll(id="chat-scroll")
            with Vertical(id="trace-col"):
                yield Static("Trace / Meta", classes="panel-title")
                yield VerticalScroll(id="trace-scroll")
        yield Input(
            placeholder="Message…  Ctrl+P palette  /help for slash commands",
            id="input",
        )
        yield Footer()

    def on_mount(self) -> None:
        self.query_one("#input", Input).focus()
        # Immediately invoke blocking overlay preventing UI bypasses
        self.push_screen(IngestionLoaderScreen(self.raw_repo_input))
        
    def apply_repo_context(self, repo_id: str, repo_path: str) -> None:
        self._repo_id = repo_id
        self._repo_path = repo_path
        if self._repo_path:
            label = Path(self._repo_path).name
            self.sub_title = f"{settings.LLM_MODEL} | {label}"
            
        rid = escape(self._repo_id or "(none)")
        rp = escape(self._repo_path or "")
        
        self._set_trace_text(
            f"[dim]repo_id:[/dim] {rid}\n[dim]path:[/dim] {rp}\n\n"
            "[dim]Pipeline meta and trace appear here after each reply. "
            "Try /metrics or /trace.[/dim]"
        )
        self._mount_chat(chatfmt.ingest_ready_banner())
        
    def _chat_scroll(self) -> VerticalScroll:
        return self.query_one("#chat-scroll", VerticalScroll)

    def _trace_scroll(self) -> VerticalScroll:
        return self.query_one("#trace-scroll", VerticalScroll)

    def _set_trace_text(self, markup: str) -> None:
        ts = self._trace_scroll()
        ts.remove_children()
        ts.mount(Static(markup, classes="chat-line"))

    def _append_trace(self, markup: str) -> None:
        self._trace_scroll().mount(Static(markup, classes="chat-line"))

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

    def action_clear_chat(self) -> None:
        self._clear_chat_only()

    def _clear_chat_only(self) -> None:
        self._chat_scroll().remove_children()

    def _mount_chat(self, markup: str) -> Static:
        w = Static(markup, classes="chat-line")
        self._chat_scroll().mount(w)
        self._chat_scroll().scroll_end(animate=False)
        return w

    def on_input_submitted(self, event: Input.Submitted) -> None:
        if event.input.id != "input":
            return
        raw = event.value.strip()
        event.input.value = ""
        if not raw:
            return
            
        mode = "ask"
        auto_apply = False
        dry_run = False

        if raw.startswith("/"):
            parts = raw.split(maxsplit=1)
            cmd = parts[0].lower()
            arg = parts[1].strip() if len(parts) > 1 else ""
            if cmd in ("/plan", "/build", "/ask"):
                mode = cmd[1:]
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
        else:
            self.run_orchestrator(raw, mode=mode, auto_apply=auto_apply, dry_run=dry_run)

    def _handle_slash(self, line: str) -> None:
        parts = line.split(maxsplit=1)
        cmd = parts[0].lower()
        arg = parts[1].strip() if len(parts) > 1 else ""

        if cmd in ("/help", "/?"):
            self._mount_chat(
                chatfmt.system_block(
                    "Slash: /help /metrics /trace /clear /repo <name>\n"
                    "Ctrl+P — command palette  |  Ctrl+L — clear chat"
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
            self._set_trace_text(escape(json.dumps(snap, indent=2, default=str)))
            self._mount_chat(chatfmt.system_block("Metrics loaded into side panel."))
            return

        if cmd == "/trace":
            if not TRACE_DIR.is_dir():
                self._mount_chat(chatfmt.error_block("No .ai_assistant/logs directory."))
                return
            files = sorted(
                TRACE_DIR.glob("trace_*.json"),
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            )
            if not files:
                self._mount_chat(chatfmt.error_block("No trace_*.json files."))
                return
            data = json.loads(files[0].read_text(encoding="utf-8"))
            self._set_trace_text(escape(json.dumps(data, indent=2, default=str)))
            self._mount_chat(
                chatfmt.system_block(f"Latest trace: {files[0].name} (side panel)")
            )
            return

        if cmd in ("/repos", "/open", "/find", "/explain"):
            # If standard inputs hit here with text attached, bounce back to orchestrator directly!
            if cmd != "/repos" and arg:
                # Remove slash so orchestrator just treats it as raw query alongside hint text!
                self.run_orchestrator(f"{cmd[1:]} {arg}", mode="ask")
                return
                
            if cmd == "/repos":
                from app.core.repo_registry import load_registry
                registry = load_registry()
                if not registry:
                    self._mount_chat(chatfmt.system_block("No repos registered."))
                    return
                lines = []
                for rid, entry in registry.items():
                    p = entry.get('path', 'unknown')
                    lines.append(f"{rid[:8]} —> {p}")
                self._set_trace_text("[bold]Registered Repos[/bold]\n" + escape("\n".join(lines)))
                self._mount_chat(chatfmt.system_block("Loaded repos into side panel."))
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

    @work(exclusive=True, group="orchestrator", name="ask")
    async def run_orchestrator(
        self, 
        question: str, 
        mode: str = "ask",
        auto_apply: bool = False,
        dry_run: bool = False,
    ) -> None:
        self._mount_chat(chatfmt.user_block(question))

        answer_widget = Static(chatfmt.assistant_label() + "\n", classes="chat-line")
        self._chat_scroll().mount(answer_widget)
        self._chat_scroll().scroll_end(animate=False)

        stream_buffer = ""
        import time
        last_update_t = time.perf_counter()

        def _refocus_input() -> None:
            try:
                self.query_one("#input", Input).focus()
            except Exception:
                pass

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

        async def _confirm_diff_callback(combined_diff: str) -> bool:
            return await self.push_screen_wait(ConfirmDiffScreen(diff_text=combined_diff))

        async def _status_callback(msg: str) -> None:
            if not stream_buffer:

                def _push_status() -> None:
                    _apply_answer_markup(
                        chatfmt.assistant_label()
                        + f"\n[dim italic]{escape(msg)}[/dim italic]"
                    )

                self.call_next(_push_status)

        try:
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
                status_callback=_status_callback,
            )
        except Exception as exc:
            answer_widget.update(chatfmt.error_block(str(exc)))
            self._chat_scroll().scroll_end(animate=False)
            self.call_next(_refocus_input)
            return

        answer = result.get("answer") or ""
        _apply_answer_markup(chatfmt.assistant_label() + "\n" + escape(answer))

        meta = result.get("meta")
        if meta is not None:
            classification = meta.get("classification") or {}
            q_type = classification.get("query_type", "factual")
            q_conf = classification.get("confidence", 1.0)

            tool_name = meta.get('selected_tool') or 'Pure RAG'
            tool_conf = float(meta.get('router_confidence') or 0.0)
            
            cancel_str = ""
            if meta.get('early_cancel_rag'): cancel_str = " (Early Cancel: RAG)"
            if meta.get('early_cancel_tool'): cancel_str = " (Early Cancel: Tool)"
            
            adap_cfg = meta.get('adaptive_config') or {}
            adap_k = adap_cfg.get('rag_top_k', 'N/A')
            adap_before = adap_cfg.get('rag_top_k_before', adap_k)
            adap_retries = adap_cfg.get('max_retries', 'N/A')
            git_bias_str = "enabled" if adap_cfg.get("enable_git_bias") else "disabled"

            markup = (
                f"🧠 [bold]Query Type:[/bold] {q_type} ({q_conf:.2f})\n"
                f"⚙️ [bold]Strategy:[/bold]\n"
                f"  - rag_top_k: {adap_before} -> {adap_k}\n"
                f"  - git_bias: {git_bias_str}\n"
                f"🔧 [bold]Tool:[/bold] {escape(tool_name)} ({tool_conf:.2f}){cancel_str}\n"
                f"🧠 [bold]Adaptive:[/bold]\n"
                f"  - retries: {adap_retries}\n"
            )
            self._set_trace_text(markup)

        tr = result.get("trace")
        if tr is not None:
            self._append_trace(
                "[bold]Debug trace[/bold]\n" + escape(json.dumps(tr, indent=2, default=str))
            )

        branch = (meta or {}).get("patch_branch")
        if branch:
            self._mount_chat(
                chatfmt.system_block(
                    f"Branch: {escape(str(branch))} (patch workflow)"
                )
            )

        self._chat_scroll().scroll_end(animate=False)
        self.call_next(_refocus_input)


def run_tui(
    *,
    raw_repo_input: str | None = None,
) -> None:
    DeepCodeTuiApp(raw_repo_input=raw_repo_input).run()
