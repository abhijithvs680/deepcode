"""Smoke tests for Textual TUI (headless)."""
import pytest
from textual.widgets import Input

from app.cli.tui.app import DeepCodeTuiApp


@pytest.mark.asyncio
async def test_tui_mounts_input_and_panels():
    app = DeepCodeTuiApp()
    async with app.run_test():
        assert app.query_one("#input", Input) is not None
        assert app.query_one("#chat-scroll") is not None
        assert app.query_one("#trace-scroll") is not None


def test_run_tui_is_callable():
    from app.cli.tui.app import run_tui

    assert callable(run_tui)
