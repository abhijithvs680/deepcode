"""
Full ingest: chunks + embeddings + repo intel (single entry for CLI and repo_manager).
"""
from __future__ import annotations

from pathlib import Path

from app.core.config import get_settings
from app.core.database import async_session_factory, create_tables, dispose_engine
from app.core.logger import get_logger
from app.ingestion.ast_analysis import build_repo_intel_from_files
from app.ingestion.git_metadata import build_repo_file_git_rows
from app.ingestion.loader import load_repo
from app.ingestion.processor import process_repo
from app.memory.intel_store import replace_repo_file_git, replace_repo_intel
from app.memory.vector_store import clear_repo, insert_chunks
from app.providers.factory import get_provider

logger = get_logger(__name__)


async def ingest_repository(
    repo_path: str,
    repo_name: str,
    *,
    clear: bool = False,
    chunk_size: int = 500,
    overlap: int = 50,
    max_source_files: int | None = None,
    dispose_after: bool = False,
) -> int:
    """
    Embed chunks and write intel rows under ``repo_name``.

    Returns number of chunks inserted (0 if none).
    """
    import app.models.code  # noqa: F401
    import app.models.repo_intel  # noqa: F401

    root = Path(repo_path).resolve()
    await create_tables()

    if clear:
        deleted = await clear_repo(repo_name)
        logger.info("Cleared %s prior rows for '%s'", deleted, repo_name)

    chunks = process_repo(
        str(root),
        chunk_size=chunk_size,
        overlap=overlap,
        repo_name=repo_name,
        max_source_files=max_source_files,
    )

    if not chunks:
        return 0

    provider = get_provider()
    inserted = await insert_chunks(chunks, provider)

    st = get_settings()
    files = load_repo(str(root), max_files=max_source_files)
    sym_rows, meta_rows, edge_rows = build_repo_intel_from_files(
        files, repo_name, st.CODE_INTEL_PACKAGE_ROOTS
    )
    git_rows = build_repo_file_git_rows(files, root, repo_name)
    async with async_session_factory() as db:
        await replace_repo_intel(db, repo_name, sym_rows, edge_rows, meta_rows)
        await replace_repo_file_git(db, repo_name, git_rows)
        await db.commit()

    if dispose_after:
        await dispose_engine()
    return inserted
