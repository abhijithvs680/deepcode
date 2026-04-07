"""Modal command palette (Ctrl+P)."""
from __future__ import annotations

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Vertical
from textual.screen import ModalScreen
from textual.widgets import Label, OptionList

# (value inserted into input or executed prefix, description)
PALETTE_ENTRIES: list[tuple[str, str]] = [
    ("/run ", "Long-run: macro goal → multiple edit-mode cycles"),
    ("/edit ", "Edit mode: explore → plan → patches (replaces /build)"),
    ("/plan ", "Create an implementation plan without writing"),
    ("/ask ", "Ask a standard question (Default)"),
    ("/browse", "Mini IDE: browse files under the workspace"),
    ("/open ", "Open/View a specific file (with query)"),
    ("/find ", "Search for a symbol or text"),
    ("/explain ", "Explain a function or component"),
    ("/metrics", "Show metrics snapshot in chat"),
    ("/repos", "Manage known repositories"),
    ("/clear", "Clear chat history"),
    ("/help", "Show slash commands"),
]


class CommandPaletteScreen(ModalScreen[str | None]):
    """Returns selected command prefix (or None if dismissed)."""

    BINDINGS = [
        Binding("escape", "dismiss_empty", "Close", show=True),
    ]

    DEFAULT_CSS = """
    CommandPaletteScreen {
        align: center middle;
    }
    #palette-box {
        width: 64;
        height: auto;
        max-height: 90%;
        border: thick $primary;
        background: $surface;
        padding: 1 2;
    }
    OptionList {
        width: 100%;
        height: auto;
        margin-top: 1;
    }
    """

    def compose(self) -> ComposeResult:
        with Vertical(id="palette-box"):
            yield Label("Command palette — pick one (Esc to close)")
            yield OptionList(
                *[f"{val} — {desc}" for val, desc in PALETTE_ENTRIES],
                id="cmd_options"
            )

    def on_mount(self) -> None:
        self.query_one(OptionList).focus()

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        value, _ = PALETTE_ENTRIES[event.option_index]
        self.dismiss(value)

    def action_dismiss_empty(self) -> None:
        self.dismiss(None)
