"""
Retriever — code-aware composite scoring with optional pgvector ANN.

score = 0.45*semantic + 0.2*keyword + 0.2*symbol + 0.1*path + 0.05*git
"""
from __future__ import annotations

import json

import numpy as np
from sqlalchemy import func, select

from app.core.config import get_settings
from app.core.database import async_session_factory
from app.core.logger import get_logger
from app.memory.scoring import (
    aggregate_file_score,
    composite_score,
    keyword_score,
    normalize_semantic,
    path_priority,
    symbol_tier_for_chunk,
)
from app.memory.symbol_extract import extract_query_identifiers
from app.models.code import CodeChunk
from app.providers.base import LLMProvider

logger = get_logger(__name__)
settings = get_settings()

GitScoreMap = dict[str, tuple[float, dict[str, float]]]


def cosine_similarity(a: list[float], b: list[float]) -> float:
    a_arr = np.array(a, dtype=np.float32)
    b_arr = np.array(b, dtype=np.float32)
    dot = np.dot(a_arr, b_arr)
    norm_a = np.linalg.norm(a_arr)
    norm_b = np.linalg.norm(b_arr)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(dot / (norm_a * norm_b))


async def get_query_embedding_cached(
    query: str,
    provider: LLMProvider,
    embed_cache: dict[str, list[float]],
) -> list[float]:
    key = query.strip().lower()
    if key not in embed_cache:
        emb = await provider.embed(query)
        embed_cache[key] = emb or []
    return embed_cache[key]


def _row_to_chunk_dict(
    row: CodeChunk,
    *,
    score: float,
    score_breakdown: dict,
    symbol_tier: str,
    symbol_reason: str,
    graph_hop: int | None = None,
    graph_decay: float | None = None,
) -> dict:
    d = {
        "file_path": row.file_path,
        "content": row.content,
        "language": row.language,
        "chunk_index": row.chunk_index,
        "repo_name": row.repo_name,
        "score": round(score, 4),
        "score_breakdown": score_breakdown,
        "symbol_tier": symbol_tier,
        "symbol_reason": symbol_reason,
    }
    if graph_hop is not None:
        d["graph_hop"] = graph_hop
    if graph_decay is not None:
        d["graph_decay"] = graph_decay
    return d


def _git_for_path(git_by_path: GitScoreMap | None, file_path: str) -> tuple[float, dict[str, float]]:
    if not git_by_path:
        return 0.0, {}
    fp = file_path.replace("\\", "/")
    return git_by_path.get(fp, (0.0, {}))


def _score_row(
    row: CodeChunk,
    query_for_keywords: str,
    full_question: str,
    query_embedding: list[float],
    indexed_hits: frozenset[str],
    git_by_path: GitScoreMap | None = None,
    focus_entities: list[str] | None = None,
) -> dict | None:
    try:
        chunk_embedding = json.loads(row.embedding) if row.embedding else None
    except (json.JSONDecodeError, TypeError):
        chunk_embedding = None
    if not chunk_embedding:
        return None

    semantic_raw = cosine_similarity(query_embedding, chunk_embedding)
    sem_norm = normalize_semantic(semantic_raw)
    kw = keyword_score(query_for_keywords, row.content)
    ids = extract_query_identifiers(full_question)
    sym_ch, sym_tier, sym_reason = symbol_tier_for_chunk(
        ids, row.file_path, row.content, indexed_hits
    )
    path_p = path_priority(row.file_path, full_question.lower())
    git_ch, git_subs = _git_for_path(git_by_path, row.file_path)
    total = composite_score(sem_norm, kw, sym_ch, path_p, git_ch)

    # Gap 4: Entity boosting natively mappings bridging structural searches explicitly avoiding complex limits 
    if focus_entities:
        c_content = row.content.lower()
        fp = row.file_path.lower()
        for ent in focus_entities:
            ent = ent.lower()
            if ent in fp or ent in c_content:
                total += 0.2
                break # Only offset once across contexts preventing massive explosions!

    breakdown = {
        "semantic": round(sem_norm, 4),
        "keyword": round(kw, 4),
        "symbol": round(sym_ch, 4),
        "path": round(path_p, 4),
        "git": round(git_ch, 4),
        "git_recency": round(git_subs.get("git_recency", 0.0), 4),
        "git_churn": round(git_subs.get("git_churn", 0.0), 4),
    }
    return {
        "total": total,
        "breakdown": breakdown,
        "symbol_tier": sym_tier,
        "symbol_reason": sym_reason,
    }


def keyword_only_scores(
    rows: list[CodeChunk],
    query: str,
    full_question: str,
    indexed_hits: frozenset[str],
    git_by_path: GitScoreMap | None = None,
) -> list[dict]:
    ids = extract_query_identifiers(full_question)
    out: list[dict] = []
    for row in rows:
        kw = keyword_score(query, row.content)
        sym_ch, sym_tier, sym_reason = symbol_tier_for_chunk(
            ids, row.file_path, row.content, indexed_hits
        )
        path_p = path_priority(row.file_path, full_question.lower())
        git_ch, git_subs = _git_for_path(git_by_path, row.file_path)
        total = composite_score(0.5, kw, sym_ch, path_p, git_ch)
        out.append(
            _row_to_chunk_dict(
                row,
                score=total,
                score_breakdown={
                    "semantic": 0.5,
                    "keyword": round(kw, 4),
                    "symbol": round(sym_ch, 4),
                    "path": round(path_p, 4),
                    "git": round(git_ch, 4),
                    "git_recency": round(git_subs.get("git_recency", 0.0), 4),
                    "git_churn": round(git_subs.get("git_churn", 0.0), 4),
                },
                symbol_tier=sym_tier,
                symbol_reason=sym_reason + "|fallback_keyword_only",
            )
        )
    out.sort(key=lambda x: x["score"], reverse=True)
    return out


async def _load_chunks(repo_name: str | None) -> list[CodeChunk]:
    async with async_session_factory() as db:
        stmt = select(CodeChunk).where(CodeChunk.embedding.isnot(None))
        if repo_name:
            stmt = stmt.where(CodeChunk.repo_name == repo_name)
        result = await db.execute(stmt)
        return list(result.scalars().all())


async def _ann_candidates(
    repo_name: str | None,
    query_embedding: list[float],
    limit: int,
) -> list[CodeChunk]:
    async with async_session_factory() as db:
        if db.bind.dialect.name != "postgresql":
            return []
        stmt = select(CodeChunk).where(CodeChunk.embedding_vec.isnot(None))
        if repo_name:
            stmt = stmt.where(CodeChunk.repo_name == repo_name)
        stmt = stmt.order_by(
            CodeChunk.embedding_vec.op("<=>")(query_embedding)
        ).limit(limit)
        result = await db.execute(stmt)
        return list(result.scalars().all())


async def _any_pgvector(repo_name: str | None) -> bool:
    async with async_session_factory() as db:
        if db.bind.dialect.name != "postgresql":
            return False
        stmt = select(func.count()).select_from(CodeChunk).where(
            CodeChunk.embedding_vec.isnot(None)
        )
        if repo_name:
            stmt = stmt.where(CodeChunk.repo_name == repo_name)
        result = await db.execute(stmt)
        n = result.scalar() or 0
        return n > 0


async def search(
    query: str,
    provider: LLMProvider,
    repo_name: str | None = None,
    top_k: int = 5,
    min_score: float = 0.25,
    *,
    full_question: str | None = None,
    embed_cache: dict[str, list[float]] | None = None,
    indexed_symbol_hits: frozenset[str] | None = None,
    git_by_path: GitScoreMap | None = None,
    focus_entities: list[str] | None = None,
    file_path_allowlist: frozenset[str] | None = None,
) -> list[dict]:
    fq = full_question if full_question is not None else query
    cache = embed_cache if embed_cache is not None else {}
    indexed = indexed_symbol_hits or frozenset()

    qemb = await get_query_embedding_cached(query, provider, cache)
    if not qemb:
        rows = await _load_chunks(repo_name)
        return keyword_only_scores(rows, query, fq, indexed, git_by_path)[:top_k]

    use_ann = await _any_pgvector(repo_name)
    cap = settings.RAG_CANDIDATE_CAP

    if use_ann:
        try:
            rows = await _ann_candidates(repo_name, qemb, cap)
        except Exception as exc:
            # e.g. query vector dim ≠ column dim after changing EMBEDDING_MODEL / DIMENSION
            logger.warning(
                "pgvector ANN query failed (%s); falling back to full chunk load + JSON embeddings.",
                exc,
            )
            rows = await _load_chunks(repo_name)
    else:
        rows = await _load_chunks(repo_name)

    scored: list[dict] = []
    for row in rows:
        s = _score_row(row, query, fq, qemb, indexed, git_by_path, focus_entities)
        if s is None:
            continue
        if s["total"] >= min_score:
            scored.append(
                _row_to_chunk_dict(
                    row,
                    score=s["total"],
                    score_breakdown=s["breakdown"],
                    symbol_tier=s["symbol_tier"],
                    symbol_reason=s["symbol_reason"],
                )
            )

    if not scored and rows:
        for row in rows:
            s = _score_row(row, query, fq, qemb, indexed, git_by_path)
            if s is None:
                continue
            scored.append(
                _row_to_chunk_dict(
                    row,
                    score=s["total"],
                    score_breakdown=s["breakdown"],
                    symbol_tier=s["symbol_tier"],
                    symbol_reason=s["symbol_reason"] + "|fallback_relaxed_threshold",
                )
            )

    scored_pre_allowlist = list(scored)
    if file_path_allowlist:
        norm = {x.replace("\\", "/").strip() for x in file_path_allowlist if x}
        if norm:

            def _path_in_focus(fp: str) -> bool:
                p = str(fp).replace("\\", "/")
                if p in norm:
                    return True
                return any(p.startswith(pref + "/") for pref in norm)

            narrowed = [c for c in scored if _path_in_focus(c.get("file_path", ""))]
            if narrowed:
                scored = narrowed
            else:
                scored_pre_allowlist.sort(key=lambda x: x["score"], reverse=True)
                scored = scored_pre_allowlist[:2]

    scored.sort(key=lambda x: x["score"], reverse=True)
    results = scored[:top_k]

    if not results and rows:
        results = keyword_only_scores(rows, query, fq, indexed, git_by_path)[:top_k]

    logger.info(
        "Retrieved %d chunks (repo=%s, top_k=%d, ann=%s, scanned=%d)",
        len(results),
        repo_name or "ALL",
        top_k,
        use_ann,
        len(rows),
    )
    return results


def orm_chunk_to_dict(
    row: CodeChunk,
    *,
    full_question: str,
    query_embedding: list[float],
    query_for_keywords: str,
    indexed_hits: frozenset[str],
    score_override: float | None = None,
    extra_reason: str = "",
    graph_hop: int | None = None,
    graph_decay: float | None = None,
    git_by_path: GitScoreMap | None = None,
) -> dict | None:
    s = _score_row(
        row, query_for_keywords, full_question, query_embedding, indexed_hits, git_by_path
    )
    if s is None:
        return None
    total = score_override if score_override is not None else max(s["total"], 0.45)
    reason = s["symbol_reason"] + (extra_reason or "")
    return _row_to_chunk_dict(
        row,
        score=total,
        score_breakdown=s["breakdown"],
        symbol_tier=s["symbol_tier"],
        symbol_reason=reason,
        graph_hop=graph_hop,
        graph_decay=graph_decay,
    )


def merge_chunk_results(
    prior: dict[tuple[str, int], dict],
    new_chunks: list[dict],
    prefer_higher_score: bool = True,
) -> None:
    for c in new_chunks:
        k = (c["file_path"], c["chunk_index"])
        if k not in prior:
            prior[k] = c
        elif prefer_higher_score and c["score"] > prior[k]["score"]:
            prior[k] = c


def rank_files_then_chunks(
    chunks: list[dict],
    top_files: int = 12,
    top_chunks: int = 15,
) -> list[dict]:
    by_file: dict[str, list[dict]] = {}
    for c in chunks:
        by_file.setdefault(c["file_path"], []).append(c)

    file_rows: list[tuple[str, float, float]] = []
    for fp, clist in by_file.items():
        scores = [float(x["score"]) for x in clist]
        sym_hit = max(
            (float(x["score_breakdown"].get("symbol", 0)) for x in clist),
            default=0.0,
        )
        fs = aggregate_file_score(scores, sym_hit)
        file_rows.append((fp, fs, max(scores)))

    file_rows.sort(key=lambda x: (x[1], x[2]), reverse=True)
    keep_files = {fp for fp, _, _ in file_rows[:top_files]}

    pool = [c for c in chunks if c["file_path"] in keep_files]
    pool.sort(key=lambda x: (x["score"], x["file_path"]), reverse=True)
    return pool[:top_chunks]
