"""Inspect exported JSON traces under .ai_assistant/logs."""
from __future__ import annotations

import json
from pathlib import Path

import typer

TRACE_DIR = Path(".ai_assistant/logs")


def show_latest_trace() -> None:
    if not TRACE_DIR.is_dir():
        typer.echo("No trace directory (.ai_assistant/logs). Run with --debug first.", err=True)
        raise typer.Exit(1)
    files = sorted(
        TRACE_DIR.glob("trace_*.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not files:
        typer.echo("No trace_*.json files found.")
        raise typer.Exit(1)
    latest = files[0]
    data = json.loads(latest.read_text(encoding="utf-8"))
    typer.echo(json.dumps(data, indent=2, default=str))
    typer.echo(f"\n# file: {latest}", err=True)


def list_traces(limit: int = 15) -> None:
    if not TRACE_DIR.is_dir():
        typer.echo("No trace directory.")
        raise typer.Exit(1)
    files = sorted(
        TRACE_DIR.glob("trace_*.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )[:limit]
    if not files:
        typer.echo("No traces.")
        return
    for p in files:
        typer.echo(f"{p.stat().st_mtime:.0f}\t{p.name}")
