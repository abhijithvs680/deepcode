"""Run eval benchmark JSON."""
from __future__ import annotations

import json

import typer

from app.cli.util import run_async
from app.evals.evaluator import run_benchmark


def cmd_eval(benchmark_path: str) -> None:
    try:
        report = run_async(run_benchmark(benchmark_path, db=None))
        typer.echo(json.dumps(report, indent=2, default=str))
    except Exception as exc:
        typer.echo(f"Eval failed: {exc}", err=True)
        raise typer.Exit(1) from exc
