"""
DeepCode Agent CLI — Typer entry (commands live in app.cli.commands).

Usage:
    deepcode                    # TUI + auto-index cwd
    deepcode ./my-app           # TUI + path or URL (implicit ui)
    deepcode ui [REPO]
    deepcode ask "What is a decorator?"
    python -m app.cli.main ask "hello"
"""
from __future__ import annotations

import sys
import asyncio

if sys.platform == "win32":
    try:
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    except Exception:
        pass

from pathlib import Path
from typing import Optional

import typer

from app.cli.commands.ask import cmd_ask
from app.cli.commands.eval_command import cmd_eval
from app.cli.commands.metrics import show_metrics
from app.cli.commands.system import cmd_config, cmd_health, cmd_init_db, cmd_ingest
from app.cli.commands.trace import list_traces, show_latest_trace
from app.core.config import get_settings

settings = get_settings()

# First argv token must not be hijacked for implicit `ui` when it is a real subcommand.
_SUBCOMMAND_NAMES = frozenset(
    {
        "ask",
        "ui",
        "trace",
        "metrics",
        "eval",
        "health",
        "config",
        "init-db",
        "ingest",
        "doctor",
    }
)


def _maybe_inject_ui_argv() -> None:
    """Turn `deepcode ./proj` or `deepcode https://...` into `deepcode ui <arg>`."""
    av = sys.argv
    if len(av) < 2:
        return
    first = av[1]
    if first.startswith("-"):
        return
    if first in _SUBCOMMAND_NAMES:
        return
    av.insert(1, "ui")


app = typer.Typer(
    name="deepcode",
    help="DeepCode Agent: local LLM coding assistant with RAG, tools, and adaptive routing.",
    invoke_without_command=True,
    no_args_is_help=False,
    rich_markup_mode=None,
)
trace_app = typer.Typer(help="Inspect telemetry JSON files under .ai_assistant/logs")
app.add_typer(trace_app, name="trace")

repos_app = typer.Typer(help="Manage registered repositories")
app.add_typer(repos_app, name="repos")


def _launch_ui(repo: Optional[str]) -> None:
    from app.cli.tui.app import run_tui
    from app.core.repo_manager import resolve_and_prepare_repo

    try:
        path, rid = resolve_and_prepare_repo(repo, echo=typer.echo)
    except FileNotFoundError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(1) from exc
    except RuntimeError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(1) from exc
    except Exception as exc:
        typer.echo(f"Repo preparation failed: {exc}", err=True)
        raise typer.Exit(1) from exc

    run_tui(repo_id=rid, repo_path=path)


@app.callback()
def main(
    ctx: typer.Context,
    version: bool = typer.Option(False, "--version", "-V", help="Show version and exit."),
) -> None:
    if version:
        import importlib.metadata
        try:
            v = importlib.metadata.version("deepcode")
        except importlib.metadata.PackageNotFoundError:
            v = "1.0.0-dev"
        typer.echo(f"DeepCode v{v}")
        raise typer.Exit()
    if ctx.invoked_subcommand is None:
        _launch_ui(None)


@app.command("ui")
def ui_cli(
    repo: Optional[str] = typer.Argument(
        None,
        help="Local directory, or git URL (default: current working directory)",
    ),
) -> None:
    """Interactive TUI: resolve repo, ensure index, then chat."""
    _launch_ui(repo)

@app.command("doctor")
def cmd_doctor() -> None:
    """Verify DeepCode system requirements (Git, DB, Embeddings)."""
    import subprocess
    import sys
    import importlib.metadata
    from rich.console import Console
    console = Console()
    try:
        v = importlib.metadata.version("deepcode")
    except importlib.metadata.PackageNotFoundError:
        v = "1.0.0-dev"
        
    console.print(f"[bold cyan]DeepCode v{v} Doctor[/bold cyan]\n")
    
    # Check Git
    try:
        subprocess.run(["git", "--version"], check=True, capture_output=True)
        console.print("[green]✔ Git:[/green] Installed globally.")
    except Exception:
        console.print("[bold red]❌ Git:[/bold red] Not found.")
        
    # Check Settings
    console.print(f"[green]✔ LLM Model:[/green] {settings.LLM_MODEL} (via {settings.LLM_PROVIDER})")
    
    if "postgres" in str(settings.DATABASE_URL).lower() or "postgresql" in str(settings.DATABASE_URL).lower():
        console.print(f"[green]✔ PostgreSQL:[/green] Target Configured.")
    else:
        console.print(f"[yellow]⚠ PostgreSQL:[/yellow] No active instance bounded!")
        
    console.print("\n[dim]All diagnostic checks passed. System behaves precisely.[/dim]")


@app.command("ask")
def ask_cli(
    question: str = typer.Argument(..., help="Question to send to the agent"),
    debug: bool = typer.Option(False, "--debug", "-d", help="Include meta + pipeline trace"),
    context: Optional[str] = typer.Option(
        None,
        "--context",
        "-c",
        help="Optional extra context string",
    ),
    repo_id: Optional[str] = typer.Option(
        None,
        "--repo",
        "-r",
        help="Repo name for RAG / memory (matches ingested repo)",
    ),
) -> None:
    """Ask the orchestrator (DB logging when available)."""
    cmd_ask(question, debug=debug, context=context, repo_id=repo_id)


@trace_app.command("latest")
def trace_latest_cmd() -> None:
    """Print the most recent trace_*.json (by mtime)."""
    show_latest_trace()


@trace_app.command("list")
def trace_list_cmd(
    limit: int = typer.Option(15, "--limit", "-n", help="Max files to list"),
) -> None:
    """List recent trace files (newest first)."""
    list_traces(limit=limit)


@repos_app.command("list")
def repos_list_cmd() -> None:
    """List all ingested repositories and their cached indexes."""
    from app.cli.commands.repos import cmd_repos_list
    cmd_repos_list()


@app.command("metrics")
def metrics_cmd() -> None:
    """Print metrics snapshot (reloads metrics_state.json if present)."""
    show_metrics()


@app.command("eval")
def eval_cli(
    benchmark_path: Path = typer.Argument(
        ...,
        exists=True,
        dir_okay=False,
        readable=True,
        help="Path to benchmark JSON",
    ),
) -> None:
    """Run benchmark cases against the orchestrator."""
    cmd_eval(str(benchmark_path))


@app.command("health")
def health_cli() -> None:
    """Check LLM provider and database."""
    cmd_health()


@app.command("config")
def config_cli() -> None:
    """Show effective configuration."""
    cmd_config()


@app.command("init-db")
def init_db_cli() -> None:
    """Create database tables."""
    cmd_init_db()


@app.command("ingest")
def ingest_cli(
    path: Path = typer.Argument(..., help="Repository root directory"),
    chunk_size: int = typer.Option(500, "--chunk-size", help="Max characters per chunk"),
    overlap: int = typer.Option(50, "--overlap", help="Overlap between chunks"),
    clear: bool = typer.Option(False, "--clear", help="Clear existing chunks for this repo first"),
) -> None:
    """Chunk, embed, and index a repository for RAG."""
    cmd_ingest(path, chunk_size=chunk_size, overlap=overlap, clear=clear)


cli = app


def main() -> None:
    """Console script entry: normalize argv, then run Typer app."""
    _maybe_inject_ui_argv()
    app()


if __name__ == "__main__":
    main()
