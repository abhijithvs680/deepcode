"""
Resolve a local path or git URL, ensure vectors exist, return (path, repo_id).

``repo_id`` is the ``code_chunks.repo_name`` key: MD5 of resolved path (local)
or normalized URL (remote). Clones live under ``~/.deepcode/repos/<repo_id>``.
"""
from __future__ import annotations

import asyncio
import hashlib
import os
import subprocess
from collections.abc import Callable
from pathlib import Path

from sqlalchemy import func, select

from datetime import datetime, timezone

from app.core.config import get_settings
from app.core.database import async_session_factory
from app.core.logger import get_logger
from app.ingestion.full_ingest import ingest_repository
from app.models.code import CodeChunk

logger = get_logger(__name__)

INGEST_STATUS: dict[str, dict] = {}

def get_ingest_state(repo_id: str) -> dict:
    return INGEST_STATUS.get(repo_id, {"status": "idle"})


def normalize_git_url(url: str) -> str:
    url = url.strip().lower()
    if url.endswith(".git"):
        url = url[:-4]
    url = url.replace("git@github.com:", "https://github.com/")
    return url


def get_git_head_hash(path: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=path,
            stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return None


def stable_repo_id_from_path(resolved: Path) -> str:
    # First attempt to normalize remote ID
    try:
        remote_url = subprocess.check_output(
            ["git", "remote", "get-url", "origin"],
            cwd=resolved,
            stderr=subprocess.DEVNULL
        ).decode().strip()
        if remote_url:
            return hashlib.md5(normalize_git_url(remote_url).encode("utf-8")).hexdigest()
    except Exception:
        pass
    
    # Fallback absolute path identity safely 
    return hashlib.md5(str(resolved.resolve()).encode("utf-8")).hexdigest()


def stable_repo_id_from_url(url: str) -> str:
    return hashlib.md5(normalize_git_url(url).encode("utf-8")).hexdigest()


def is_remote_ref(ref: str) -> bool:
    s = ref.strip()
    return s.startswith("http://") or s.startswith("https://") or s.startswith("git@")


def repo_cache_root() -> Path:
    p = Path.home() / ".deepcode" / "repos"
    p.mkdir(parents=True, exist_ok=True)
    return p


async def is_repo_indexed(repo_id: str) -> bool:
    async with async_session_factory() as db:
        q = select(func.count()).select_from(CodeChunk).where(CodeChunk.repo_name == repo_id)
        n = (await db.execute(q)).scalar_one()
        return int(n or 0) > 0


def _clone_repo(url: str, dest: Path) -> None:
    import shutil

    from git import Repo

    if dest.is_dir():
        if (dest / ".git").is_dir():
            return
        shutil.rmtree(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    Repo.clone_from(url, str(dest), depth=1)


async def execute_ingest(
    repo_path: str,
    repo_id: str,
    chunk_size: int,
    overlap: int,
    cap: int,
    echo: Callable[[str], None],
    head_hash: str | None,
    clear_first: bool = False
) -> None:
    try:
        INGEST_STATUS[repo_id] = {
            "status": "running",
            "started_at": datetime.now(timezone.utc).isoformat()
        }
        inserted = await ingest_repository(
            repo_path,
            repo_id,
            clear=clear_first,
            chunk_size=chunk_size,
            overlap=overlap,
            max_source_files=cap,
            status_callback=echo,
        )
        echo(f"   Ingest complete — {inserted} chunks indexed.")
        from app.core.repo_registry import update_repo_registry
        update_repo_registry(repo_id, repo_path, head_hash, index_version=1)
        INGEST_STATUS[repo_id] = {"status": "done", "finished_at": datetime.now(timezone.utc).isoformat()}
    except Exception as e:
        logger.error(f"Ingest failed: {e}")
        INGEST_STATUS[repo_id] = {"status": "failed", "error": str(e)}


async def prepare_remote_repo_async(
    url: str,
    *,
    chunk_size: int = 500,
    overlap: int = 50,
    echo: Callable[[str], None] = print,
) -> tuple[str, str]:
    repo_id = stable_repo_id_from_url(url)
    dest = repo_cache_root() / repo_id
    echo(f"Remote repository")
    echo(f"   URL fingerprint (repo id): {repo_id}")
    if dest.is_dir() and (dest / ".git").is_dir():
        echo(f"   Using cached clone: {dest}")
    else:
        echo("   Cloning (shallow)…")
        try:
            await asyncio.to_thread(_clone_repo, url.strip(), dest)
        except Exception as exc:
            logger.exception("git clone failed")
            raise RuntimeError(f"Clone failed: {exc}") from exc
        echo(f"   Cloned to: {dest}")

    resolved = str(dest.resolve())
    st = get_settings()
    cap = st.REPO_INDEX_MAX_FILES

    try:
        indexed = await is_repo_indexed(repo_id)
    except Exception as exc:
        echo(f"Warning: could not query index ({exc}); launch without auto-ingest.")
        return resolved, repo_id

    from app.core.repo_registry import get_repo_entry
    entry = get_repo_entry(repo_id)
    stored_hash = entry.get("head_hash") if entry else None
    current_hash = get_git_head_hash(dest)

    if not indexed:
        echo("   Not indexed — executing synchronous TUI ingest…")
        await execute_ingest(resolved, repo_id, chunk_size, overlap, cap, echo, current_hash, clear_first=False)
    elif current_hash and current_hash != stored_hash:
        echo("   Stale index detected (git HEAD drift) — executing synchronous TUI re-index…")
        await execute_ingest(resolved, repo_id, chunk_size, overlap, cap, echo, current_hash, clear_first=True)
    else:
        echo("   Already indexed.")

    return resolved, repo_id


async def prepare_local_repo_async(
    path_str: str,
    *,
    chunk_size: int = 500,
    overlap: int = 50,
    echo: Callable[[str], None] = print,
) -> tuple[str, str]:
    root = Path(path_str).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Not a directory: {root}")

    repo_id = stable_repo_id_from_path(root)
    echo(f"Local repository: {root}")
    echo(f"Repo ID: {repo_id}")

    st = get_settings()
    cap = st.REPO_INDEX_MAX_FILES

    try:
        indexed = await is_repo_indexed(repo_id)
    except Exception as exc:
        echo(f"Warning: could not query index ({exc}); launch without auto-ingest.")
        return str(root), repo_id

    from app.core.repo_registry import get_repo_entry
    entry = get_repo_entry(repo_id)
    stored_hash = entry.get("head_hash") if entry else None
    current_hash = get_git_head_hash(root)

    if not indexed:
        echo("   Not indexed — executing synchronous TUI ingest…")
        await execute_ingest(str(root), repo_id, chunk_size, overlap, cap, echo, current_hash, clear_first=False)
    elif current_hash and current_hash != stored_hash:
        echo("   Stale index detected (git HEAD drift) — executing synchronous TUI re-index…")
        await execute_ingest(str(root), repo_id, chunk_size, overlap, cap, echo, current_hash, clear_first=True)
    else:
        echo("   Already indexed.")

    return str(root), repo_id


async def resolve_and_prepare_repo_async(
    repo: str | None,
    *,
    chunk_size: int = 500,
    overlap: int = 50,
    echo: Callable[[str], None] = print,
) -> tuple[str, str]:
    """
    ``repo`` is None or "." → cwd; http(s)/git@ → clone cache; else → local path string.
    """
    if repo is None or not str(repo).strip() or str(repo).strip() == ".":
        return await prepare_local_repo_async(
            os.getcwd(),
            chunk_size=chunk_size,
            overlap=overlap,
            echo=echo,
        )
    r = str(repo).strip()
    if is_remote_ref(r):
        return await prepare_remote_repo_async(
            r, chunk_size=chunk_size, overlap=overlap, echo=echo
        )
    return await prepare_local_repo_async(
        r, chunk_size=chunk_size, overlap=overlap, echo=echo
    )


def resolve_and_prepare_repo(
    repo: str | None,
    *,
    chunk_size: int = 500,
    overlap: int = 50,
    echo: Callable[[str], None] = print,
) -> tuple[str, str]:
    from app.cli.util import run_async

    return run_async(
        resolve_and_prepare_repo_async(
            repo, chunk_size=chunk_size, overlap=overlap, echo=echo
        )
    )
