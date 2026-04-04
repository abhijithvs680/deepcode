"""
Repository processor — orchestrates loading + chunking.

Takes a repo path, walks the files, reads each one, chunks it,
and returns a flat list of chunk dicts ready for embedding.
"""
from pathlib import Path

from tqdm import tqdm

from app.core.logger import get_logger
from app.ingestion.loader import load_repo
from app.ingestion.chunker import chunk_file

logger = get_logger(__name__)


def process_repo(
    path: str,
    chunk_size: int = 500,
    overlap: int = 50,
    *,
    repo_name: str | None = None,
    max_source_files: int | None = None,
) -> list[dict]:
    """
    Load all source files from a repo and chunk them.

    Args:
        path: Root directory of the repository.
        chunk_size: Max characters per chunk.
        overlap: Overlap between consecutive chunks.

    Returns:
        List of dicts: {file_path, content, language, chunk_index, repo_name}

    Complexity:
        Time: O(F * C) where F = files, C = avg file size
        Space: O(total_chars) for all chunks in memory
    """
    root = Path(path).resolve()
    resolved_name = repo_name or root.name

    files = load_repo(path, max_files=max_source_files)
    results = []
    skipped = 0

    for file_info in tqdm(files, desc="Processing files", unit="file"):
        try:
            with open(file_info["abs_path"], "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()

            if not content.strip():
                skipped += 1
                continue

            chunks = chunk_file(content, chunk_size=chunk_size, overlap=overlap)

            for idx, chunk in enumerate(chunks):
                results.append({
                    "file_path": file_info["file_path"],
                    "content": chunk,
                    "language": file_info["language"],
                    "chunk_index": idx,
                    "repo_name": resolved_name,
                })

        except Exception as exc:
            logger.warning("Skipping %s: %s", file_info["file_path"], exc)
            skipped += 1
            continue

    logger.info(
        "Processed repo '%s': %d files → %d chunks (%d skipped)",
        resolved_name,
        len(files),
        len(results),
        skipped,
    )
    return results
