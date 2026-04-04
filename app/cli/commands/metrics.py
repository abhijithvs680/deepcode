"""Print aggregated metrics snapshot (reloads from disk)."""
from __future__ import annotations

import json

import typer

from app.telemetry.metrics_engine import metrics_engine


def show_metrics() -> None:
    metrics_engine.load()
    typer.echo(json.dumps(metrics_engine.snapshot(), indent=2, default=str))
