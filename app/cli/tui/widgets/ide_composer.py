"""Multiline chat composer: Enter sends, Shift+Enter newline (IDE layout)."""
from __future__ import annotations

from textual import events
from textual.message import Message
from textual.widgets import TextArea


class IdeComposer(TextArea):
    """Rich-style follow-up field; mirrors common chat UX."""

    class Submitted(Message):
        bubble = True

        def __init__(self, text: str) -> None:
            self.text = text
            super().__init__()

    def on_key(self, event: events.Key) -> None:
        if event.key == "shift+enter":
            return
        if event.key == "enter":
            event.prevent_default()
            self.post_message(self.Submitted(self.text))
            return
