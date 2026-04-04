"""Health, config, database init, and repo ingest."""
from __future__ import annotations

from pathlib import Path

import typer

from app.cli.util import run_async
from app.core.config import get_settings
from app.core.logger import get_logger
from app.providers.factory import get_provider

logger = get_logger(__name__)
settings = get_settings()


def cmd_health() -> None:
    typer.echo(f"Checking LLM provider ({settings.LLM_PROVIDER})...")

    async def _check_provider():
        provider = get_provider()
        return await provider.health_check()

    ok_llm = run_async(_check_provider())
    if ok_llm:
        typer.echo("  OK LLM provider is healthy.")
    else:
        typer.echo("  LLM provider is unreachable.", err=True)

    typer.echo("\nChecking database...")

    async def _check_db():
        try:
            from app.core.database import engine
            from sqlalchemy import text

            async with engine.connect() as conn:
                await conn.execute(text("SELECT 1"))
            return True
        except Exception as exc:
            logger.error("DB check failed: %s", exc)
            return False

    ok_db = run_async(_check_db())
    if ok_db:
        typer.echo("  OK Database is connected.")
    else:
        typer.echo("  Database is unreachable.", err=True)

    if not (ok_llm and ok_db):
        raise typer.Exit(1)
    typer.echo("\nAll systems operational.")


def cmd_config() -> None:
    typer.echo("\nCurrent configuration")
    typer.echo("─" * 50)
    typer.echo(f"  Provider      : {settings.LLM_PROVIDER}")
    typer.echo(f"  Model         : {settings.LLM_MODEL}")
    typer.echo(f"  Embed Model   : {settings.EMBEDDING_MODEL}")
    typer.echo(f"  Ollama URL    : {settings.OLLAMA_BASE_URL}")
    typer.echo(f"  Temperature   : {settings.LLM_TEMPERATURE}")
    typer.echo(f"  Max Tokens    : {settings.LLM_MAX_TOKENS}")
    typer.echo(f"  Database      : {settings.DATABASE_URL}")
    typer.echo(f"  Log Level     : {settings.LOG_LEVEL}")
    typer.echo("─" * 50)


def cmd_init_db() -> None:
    typer.echo("Creating database tables...")

    async def _init():
        import app.core.models  # noqa: F401
        import app.models.code  # noqa: F401
        import app.models.repo_intel  # noqa: F401
        from app.core.database import create_tables, dispose_engine

        await create_tables()
        await dispose_engine()

    try:
        run_async(_init())
        typer.echo("Tables created successfully.")
        typer.echo("   • sessions, interaction_logs, code_chunks")
        typer.echo("   • repo_symbols, repo_symbol_meta, repo_import_edges, repo_file_git")
    except Exception as exc:
        typer.echo(f"Failed: {exc}", err=True)
        raise typer.Exit(1) from exc


def cmd_ingest(
    path: Path,
    chunk_size: int = 500,
    overlap: int = 50,
    clear: bool = False,
) -> None:
    repo_path = path.resolve()
    if not repo_path.is_dir():
        typer.echo(f"Not a directory: {repo_path}", err=True)
        raise typer.Exit(1)

    typer.echo(f"\nIngesting repository: {repo_path.name}")
    typer.echo(f"   Path: {repo_path}")
    typer.echo(f"   Chunk size: {chunk_size} | Overlap: {overlap}\n")

    async def _ingest():
        from app.core.config import get_settings as ingest_settings
        from app.ingestion.full_ingest import ingest_repository

        st = ingest_settings()
        inserted = await ingest_repository(
            str(repo_path),
            repo_path.name,
            clear=clear,
            chunk_size=chunk_size,
            overlap=overlap,
            max_source_files=st.REPO_INDEX_MAX_FILES,
            dispose_after=True,
        )

        if not inserted:
            typer.echo("No source files found to ingest.")
        return inserted

    try:
        count = run_async(_ingest())
        if count:
            typer.echo(f"\nIngested {count} chunks into code_chunks table")
    except ConnectionError as exc:
        typer.echo(f"Cannot reach LLM provider: {exc}", err=True)
        raise typer.Exit(1) from exc
    except Exception as exc:
        typer.echo(f"Ingestion failed: {exc}", err=True)
        raise typer.Exit(1) from exc
