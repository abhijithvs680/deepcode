"""
Vector store — inserts code chunks with embeddings into PostgreSQL.
"""
import json
import uuid

from tqdm import tqdm

from app.core.config import get_settings
from app.core.database import async_session_factory
from app.core.logger import get_logger
from app.models.code import CodeChunk
from app.providers.base import LLMProvider

logger = get_logger(__name__)
_settings = get_settings()


async def insert_chunks(
    chunks: list[dict],
    provider: LLMProvider,
    batch_size: int = 20,
) -> int:
    inserted = 0

    async with async_session_factory() as db:
        for i, chunk in enumerate(tqdm(chunks, desc="Embedding & storing", unit="chunk")):
            try:
                embedding = await provider.embed(chunk["content"])
                clean_content = chunk["content"].replace("\x00", "")
                ev = None
                if embedding and len(embedding) == _settings.EMBEDDING_DIMENSION:
                    ev = embedding

                row = CodeChunk(
                    id=uuid.uuid4(),
                    repo_name=chunk["repo_name"],
                    file_path=chunk["file_path"],
                    chunk_index=chunk["chunk_index"],
                    content=clean_content,
                    language=chunk.get("language"),
                    embedding=json.dumps(embedding),
                    embedding_vec=ev,
                )
                db.add(row)
                inserted += 1

                if (i + 1) % batch_size == 0:
                    await db.commit()
                    logger.info("Committed batch %d (%d chunks)", (i + 1) // batch_size, inserted)

            except Exception as exc:
                logger.warning(
                    "Failed to embed chunk %s:%d — %s",
                    chunk["file_path"],
                    chunk["chunk_index"],
                    exc,
                )
                continue

        await db.commit()

    logger.info("Inserted %d/%d chunks into code_chunks", inserted, len(chunks))
    return inserted


async def clear_repo(repo_name: str) -> int:
    from sqlalchemy import delete

    from app.models.repo_intel import RepoFileGit, RepoImportEdge, RepoSymbol, RepoSymbolMeta

    async with async_session_factory() as db:
        await db.execute(delete(RepoSymbol).where(RepoSymbol.repo_name == repo_name))
        await db.execute(delete(RepoSymbolMeta).where(RepoSymbolMeta.repo_name == repo_name))
        await db.execute(
            delete(RepoImportEdge).where(RepoImportEdge.repo_name == repo_name)
        )
        await db.execute(delete(RepoFileGit).where(RepoFileGit.repo_name == repo_name))
        stmt = delete(CodeChunk).where(CodeChunk.repo_name == repo_name)
        result = await db.execute(stmt)
        await db.commit()
        count = result.rowcount
        logger.info("Cleared %d chunks for repo '%s'", count, repo_name)
        return count
