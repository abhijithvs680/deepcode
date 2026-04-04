"""
Dependency Resolver — Follows imports in retrieved code chunks
to gather necessary cross-file context.
"""
import re
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.logger import get_logger
from app.models.code import CodeChunk

logger = get_logger(__name__)

# Matches "from X import Y, Z" and "import X, Y"
# This is a basic heuristic for Python
FROM_IMPORT_REGEX = r'^from\s+([a-zA-Z0-9_\.]+)\s+import\s+([a-zA-Z0-9_,\s]+)'
IMPORT_REGEX = r'^import\s+([a-zA-Z0-9_,\s]+)'


def extract_imports(text: str) -> list[tuple[str, str]]:
    """
    Extract (module, symbol) tuples from python code.
    If 'import module', symbol is '*'.
    """
    dependencies = []
    
    for line in text.split('\n'):
        line = line.strip()
        if not line:
            continue
            
        from_match = re.match(FROM_IMPORT_REGEX, line)
        if from_match:
            module = from_match.group(1)
            symbols = [s.strip() for s in from_match.group(2).split(',')]
            for symbol in symbols:
                dependencies.append((module, symbol))
            continue
            
        import_match = re.match(IMPORT_REGEX, line)
        if import_match:
            symbols = [s.strip() for s in import_match.group(1).split(',')]
            for symbol in symbols:
                dependencies.append((symbol, '*'))
                
    return dependencies


async def resolve_dependencies(
    chunks: list[dict],
    db: AsyncSession,
    repo_name: str | None = None
) -> tuple[list[dict], list[dict]]:
    """
    Analyze chunks for imports and retrieve the chunks defining those symbols.
    
    Returns:
        (resolved_chunks, trace_entries)
    """
    resolved_chunks = []
    trace_entries = []
    
    for chunk in chunks:
        imports = extract_imports(chunk['content'])
        
        for module, symbol in imports:
            # Only resolve local app imports (heuristic)
            if not module.startswith("app."):
                continue
                
            # Convert app.core.database to app/core/database.py
            expected_file_part = module.replace('.', '/')
            
            stmt = select(CodeChunk)
            if repo_name:
                stmt = stmt.where(CodeChunk.repo_name == repo_name)
                
            stmt = stmt.where(CodeChunk.file_path.ilike(f"%{expected_file_part}%"))
            
            # If a specific symbol was imported, find its definition
            if symbol != '*':
                # Look for "def symbol(" or "class symbol(" or "symbol ="
                # Using ilike for rough matching
                stmt = stmt.where(CodeChunk.content.ilike(f"%{symbol}%"))
                
            result = await db.execute(stmt)
            # Take the first chunk that matches the definition
            row = result.scalars().first()
            
            if row:
                resolved = {
                    "file_path": row.file_path,
                    "content": row.content,
                    "language": row.language,
                    "chunk_index": row.chunk_index,
                    "repo_name": row.repo_name,
                    "score": 0.5,
                    "score_breakdown": {
                        "semantic": 0.0,
                        "keyword": 0.0,
                        "symbol": 0.5,
                        "path": 0.75,
                        "git": 0.0,
                        "git_recency": 0.0,
                        "git_churn": 0.0,
                    },
                    "symbol_tier": "medium",
                    "symbol_reason": "dependency_resolver",
                }
                resolved_chunks.append(resolved)
                
                trace_entries.append({
                    "source_file": chunk["file_path"],
                    "import_statement": f"from {module} import {symbol}",
                    "resolved_file": row.file_path
                })
                logger.info("Resolved dependency: %s -> %s", symbol, row.file_path)
                
    return resolved_chunks, trace_entries
