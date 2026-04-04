"""
Code chunker — splits file content into manageable pieces.

Uses a sliding-window approach with overlap to preserve context
at chunk boundaries. This is a simple character-based chunker;
Week 3 upgrades to AST-based chunking for better semantic boundaries.
"""
from app.core.logger import get_logger

logger = get_logger(__name__)


def chunk_file(
    content: str,
    chunk_size: int = 500,
    overlap: int = 50,
) -> list[str]:
    """
    Split file content into overlapping chunks.

    Args:
        content: The full text content of a file.
        chunk_size: Maximum characters per chunk.
        overlap: Characters of overlap between consecutive chunks.

    Returns:
        List of text chunks. Empty list if content is empty/whitespace.

    Complexity:
        Time: O(n) where n = len(content)
        Space: O(n) for the output list
    """
    content = content.strip()
    if not content:
        return []

    # Small files → single chunk
    if len(content) <= chunk_size:
        return [content]

    chunks = []
    start = 0
    step = max(chunk_size - overlap, 1)  # Guard against zero/negative step

    while start < len(content):
        end = start + chunk_size
        chunk = content[start:end]
        if chunk.strip():  # Skip whitespace-only chunks
            chunks.append(chunk)
        start += step

    logger.debug("Chunked %d chars → %d chunks (size=%d, overlap=%d)",
                 len(content), len(chunks), chunk_size, overlap)
    return chunks
