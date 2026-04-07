"""Best-effort clipboard copy (no hard dependency)."""
from __future__ import annotations


def copy_plain_text(text: str) -> bool:
    if not text:
        return False
    try:
        import pyperclip  # type: ignore

        pyperclip.copy(text)
        return True
    except Exception:
        pass
    try:
        import tkinter as tk

        r = tk.Tk()
        r.withdraw()
        r.clipboard_clear()
        r.clipboard_append(text)
        r.update()
        r.destroy()
        return True
    except Exception:
        pass
    return False


def strip_rich_markup(markup: str) -> str:
    try:
        from rich.text import Text

        return Text.from_markup(markup or "").plain
    except Exception:
        return markup or ""
