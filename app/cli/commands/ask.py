"""Ask — run the orchestrator with optional debug trace and meta."""
from __future__ import annotations

import json
from typing import Any

import typer

from app.cli.util import run_async
from app.core.config import get_settings
from app.core.logger import get_logger

logger = get_logger(__name__)
settings = get_settings()


async def run_ask_async(
    question: str,
    *,
    debug: bool = False,
    context: str | None = None,
    repo_id: str | None = None,
    mode: str = "ask",
    auto_apply: bool = False,
    dry_run: bool = False,
    **kwargs: Any,
) -> dict[str, Any]:
    try:
        import app.core.models  # noqa: F401
        from app.core.database import async_session_factory, create_tables, dispose_engine

        await create_tables()

        async with async_session_factory() as db:
            from app.agents.orchestrator import AgentOrchestrator

            orchestrator = AgentOrchestrator(db=db)
            result = await orchestrator.run(
                question=question,
                context=context,
                debug=debug,
                source="cli",
                repo_id=repo_id,
                mode=mode,
                auto_apply=auto_apply,
                dry_run=dry_run,
                **kwargs,
            )
            await db.commit()

        await dispose_engine()
        return result
    except Exception as db_err:
        logger.warning("DB unavailable (%s), running without logging", db_err)
        from app.agents.orchestrator import AgentOrchestrator

        orchestrator = AgentOrchestrator()
        return await orchestrator.run(
            question=question,
            context=context,
            debug=debug,
            source="cli",
            repo_id=repo_id,
            mode=mode,
            auto_apply=auto_apply,
            dry_run=dry_run,
            **kwargs,
        )


def run_ask(
    question: str,
    *,
    debug: bool = False,
    context: str | None = None,
    repo_id: str | None = None,
    mode: str = "ask",
    auto_apply: bool = False,
    dry_run: bool = False,
) -> dict[str, Any]:
    return run_async(
        run_ask_async(
            question, 
            debug=debug, 
            context=context, 
            repo_id=repo_id, 
            mode=mode, 
            auto_apply=auto_apply, 
            dry_run=dry_run
        )
    )


def cmd_ask(
    question: str,
    debug: bool = False,
    context: str | None = None,
    repo_id: str | None = None,
) -> None:
    typer.echo(f"\nAsking: {question}")
    typer.echo(f"Provider: {settings.LLM_PROVIDER} | Model: {settings.LLM_MODEL}\n")
    try:
        result = run_ask(question, debug=debug, context=context, repo_id=repo_id)
        typer.echo("─" * 60)
        typer.echo(result.get("answer", ""))
        typer.echo("─" * 60)
        if debug:
            meta = result.get("meta")
            if meta is not None:
                typer.echo("\n=== Meta ===\n")
                typer.echo(json.dumps(meta, indent=2, default=str))
            tr = result.get("trace")
            if tr is not None:
                typer.echo("\n=== Trace ===\n")
                typer.echo(json.dumps(tr, indent=2, default=str))
    except ConnectionError as exc:
        typer.echo(f"\nConnection failed: {exc}", err=True)
        raise typer.Exit(1) from exc
    except Exception as exc:
        typer.echo(f"\nError: {exc}", err=True)
        raise typer.Exit(1) from exc
