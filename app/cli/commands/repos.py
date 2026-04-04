from __future__ import annotations

import typer
from rich.console import Console
from rich.table import Table

from app.core.repo_registry import load_registry

def cmd_repos_list() -> None:
    """Print the tracked repositories from the JSON registry."""
    console = Console()
    registry = load_registry()
    if not registry:
        console.print("[yellow]No repositories tracked yet. Run `deepcode ingest` or `deepcode ui <path>`.[/yellow]")
        return

    table = Table(title="Tracked Repositories")
    table.add_column("Repo ID", style="cyan", overflow="fold")
    table.add_column("Path / URL", style="magenta")
    table.add_column("Git HEAD", style="green")
    table.add_column("Last Indexed", style="dim")
    table.add_column("Ver", style="dim", justify="right")

    for rid, entry in registry.items():
        table.add_row(
            rid[:12] + "..." if len(rid) > 12 else rid,
            entry.get("path", "unknown"),
            (entry.get("head_hash") or "")[:8],
            entry.get("last_indexed", ""),
            str(entry.get("index_version", 1)),
        )

    console.print(table)
