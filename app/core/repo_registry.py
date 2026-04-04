"""
Registry for tracking ingested repositories and their metadata.
Backing file: ~/.deepcode/repos.json
"""

import json
from pathlib import Path
from typing import TypedDict, Optional
from datetime import datetime, timezone

from app.core.logger import get_logger

logger = get_logger(__name__)

def _registry_path() -> Path:
    p = Path.home() / ".deepcode" / "repos.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    return p

class RepoRegistryEntry(TypedDict):
    repo_id: str
    path: str
    head_hash: Optional[str]
    last_indexed: str
    index_version: int

def load_registry() -> dict[str, RepoRegistryEntry]:
    p = _registry_path()
    if not p.is_file():
        return {}
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            return {}
        return data  # type: ignore
    except Exception as e:
        logger.warning(f"Failed to parse repo registry: {e}")
        return {}

def save_registry(registry: dict[str, RepoRegistryEntry]) -> None:
    p = _registry_path()
    try:
        p.write_text(json.dumps(registry, indent=2), encoding="utf-8")
    except Exception as e:
        logger.error(f"Failed to save repo registry: {e}")

def update_repo_registry(
    repo_id: str,
    path: str,
    head_hash: Optional[str] = None,
    index_version: int = 1
) -> None:
    reg = load_registry()
    entry: RepoRegistryEntry = {
        "repo_id": repo_id,
        "path": str(Path(path).resolve()),
        "head_hash": head_hash,
        "last_indexed": datetime.now(timezone.utc).isoformat(),
        "index_version": index_version,
    }
    reg[repo_id] = entry
    save_registry(reg)

def get_repo_entry(repo_id: str) -> Optional[RepoRegistryEntry]:
    return load_registry().get(repo_id)
