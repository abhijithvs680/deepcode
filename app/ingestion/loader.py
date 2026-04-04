"""
Repository file loader.

Walks a directory tree and returns a list of file paths,
skipping common non-source directories.
If ``<root>/.gitignore`` exists, paths are filtered with Git ignore rules (via pathspec).
"""
from __future__ import annotations

import os
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING

from app.core.logger import get_logger

if TYPE_CHECKING:
    import pathspec

logger = get_logger(__name__)

# Directories to skip during ingestion
IGNORE_DIRS = {
    ".git", "node_modules", "venv", ".venv", "__pycache__",
    ".pytest_cache", ".mypy_cache", ".tox", "dist", "build",
    ".egg-info", "htmlcov", ".idea", ".vscode",
}

# File extensions to include (source code + docs)
SOURCE_EXTENSIONS = {
    ".py", ".js", ".ts", ".jsx", ".tsx",
    ".java", ".go", ".rs", ".rb", ".php",
    ".c", ".cpp", ".h", ".hpp", ".cs",
    ".html", ".css", ".scss",
    ".sql", ".sh", ".bash",
    ".yaml", ".yml", ".toml", ".json",
    ".md", ".txt", ".rst",
    ".dockerfile",
}


def _load_root_gitignore_spec(root: Path) -> pathspec.PathSpec | None:
    """Parse repo-root ``.gitignore`` only; return None if missing or unusable."""
    gi = root / ".gitignore"
    if not gi.is_file():
        return None
    try:
        import pathspec
    except ImportError:
        logger.warning(
            "Found .gitignore at %s but pathspec is not installed; ignoring it. "
            "Install pathspec to honor .gitignore.",
            gi,
        )
        return None
    try:
        text = gi.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        logger.warning("Could not read .gitignore: %s", exc)
        return None
    lines = text.splitlines()
    if not any(ln.strip() and not ln.lstrip().startswith("#") for ln in lines):
        return None
    try:
        return pathspec.PathSpec.from_lines("gitignore", lines)
    except Exception as exc:
        logger.warning("Could not parse .gitignore: %s", exc)
        return None


def _gitignore_ignores(spec: pathspec.PathSpec, rel_posix: str, *, is_dir: bool) -> bool:
    if spec.match_file(rel_posix):
        return True
    if is_dir and spec.match_file(rel_posix + "/"):
        return True
    return False


def load_repo(path: str, max_files: int | None = None) -> list[dict[str, str]]:
    """
    Walk a repository and return loadable source files.

    Args:
        path: Root directory of the repository.
        max_files: Stop after this many source files (huge-repo guard).

    Returns:
        List of dicts with 'file_path' (relative) and 'abs_path'.
    """
    root = Path(path).resolve()
    if not root.is_dir():
        raise ValueError(f"Not a directory: {root}")

    gitignore = _load_root_gitignore_spec(root)
    if gitignore is not None:
        logger.info("Using root .gitignore for %s", root)

    files = []

    for dirpath, dirnames, filenames in os.walk(root):
        # Prune ignored directories in-place
        dirnames[:] = [d for d in dirnames if d not in IGNORE_DIRS]

        if gitignore is not None:
            rel_parent = os.path.relpath(dirpath, root)
            if rel_parent in (".", ""):
                rel_parent_posix = ""
            else:
                rel_parent_posix = str(PurePosixPath(rel_parent))

            def _dir_rel(name: str) -> str:
                if not rel_parent_posix:
                    return name.replace("\\", "/")
                return f"{rel_parent_posix}/{name}"

            dirnames[:] = [
                d
                for d in dirnames
                if not _gitignore_ignores(gitignore, _dir_rel(d), is_dir=True)
            ]

        for filename in filenames:
            ext = Path(filename).suffix.lower()
            # Include Dockerfile (no extension match)
            if ext not in SOURCE_EXTENSIONS and filename.lower() != "dockerfile":
                continue

            abs_path = os.path.join(dirpath, filename)
            rel_path = os.path.relpath(abs_path, root)
            rel_posix = str(PurePosixPath(rel_path))

            if gitignore is not None and _gitignore_ignores(
                gitignore, rel_posix, is_dir=False
            ):
                continue

            files.append({
                "file_path": rel_posix,
                "abs_path": abs_path,
                "language": ext.lstrip(".") if ext else "text",
            })
            if max_files is not None and len(files) >= max_files:
                logger.warning(
                    "Stopped at max_files=%s while walking %s (more files may exist)",
                    max_files,
                    root,
                )
                break
        if max_files is not None and len(files) >= max_files:
            break

    logger.info("Loaded %d files from %s", len(files), root)
    return files
