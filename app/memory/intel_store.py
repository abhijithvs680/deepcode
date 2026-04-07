"""Persist repo symbol index, rich symbol meta, import graph, Git per file (replace per ingest)."""
from __future__ import annotations

import json
from datetime import datetime

from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.logger import get_logger
from app.memory.scoring import git_channel_score, memory_score
from app.models.memory import AgentDecision, AgentBugFix, AgentQueryHistory
from app.models.repo_intel import RepoFileGit, RepoImportEdge, RepoSymbol, RepoSymbolMeta

logger = get_logger(__name__)


async def replace_repo_intel(
    db: AsyncSession,
    repo_name: str,
    symbol_rows: list[dict],
    edge_rows: list[dict],
    meta_rows: list[dict] | None = None,
) -> tuple[int, int, int]:
    await db.execute(delete(RepoSymbol).where(RepoSymbol.repo_name == repo_name))
    await db.execute(delete(RepoSymbolMeta).where(RepoSymbolMeta.repo_name == repo_name))
    await db.execute(delete(RepoImportEdge).where(RepoImportEdge.repo_name == repo_name))

    sym_n = 0
    for row in symbol_rows:
        db.add(
            RepoSymbol(
                repo_name=row["repo_name"],
                symbol=row["symbol"],
                file_path=row["file_path"],
                kind=row.get("kind", "name"),
            )
        )
        sym_n += 1

    meta_n = 0
    if meta_rows:
        for row in meta_rows:
            db.add(
                RepoSymbolMeta(
                    repo_name=row["repo_name"],
                    file_path=row["file_path"],
                    qualified_name=row["qualified_name"],
                    kind=row["kind"],
                    parent_symbol=row.get("parent_symbol"),
                    line_start=int(row["line_start"]),
                    line_end=int(row["line_end"]),
                    bases_json=row.get("bases_json"),
                    signature=row.get("signature"),
                    return_annotation=row.get("return_annotation"),
                    docstring=row.get("docstring"),
                    is_async=bool(row.get("is_async", False)),
                    is_classmethod=bool(row.get("is_classmethod", False)),
                    is_staticmethod=bool(row.get("is_staticmethod", False)),
                )
            )
            meta_n += 1

    for row in edge_rows:
        db.add(
            RepoImportEdge(
                repo_name=row["repo_name"],
                source_file=row["source_file"],
                target_file=row["target_file"],
            )
        )

    await db.flush()
    logger.info(
        "Repo intel for '%s': %d symbols, %d meta, %d import edges",
        repo_name,
        sym_n,
        meta_n,
        len(edge_rows),
    )
    return sym_n, meta_n, len(edge_rows)


async def replace_repo_file_git(
    db: AsyncSession,
    repo_name: str,
    git_rows: list[dict],
) -> int:
    await db.execute(delete(RepoFileGit).where(RepoFileGit.repo_name == repo_name))
    n = 0
    for row in git_rows:
        db.add(
            RepoFileGit(
                repo_name=row["repo_name"],
                file_path=row["file_path"],
                last_commit_hash=row.get("last_commit_hash"),
                last_commit_time=row.get("last_commit_time"),
                last_author=row.get("last_author"),
                last_message=row.get("last_message"),
                recent_commits_json=row.get("recent_commits_json"),
                blame_summary_json=row.get("blame_summary_json"),
            )
        )
        n += 1
    await db.flush()
    logger.info("Repo file Git for '%s': %d rows", repo_name, n)
    return n


async def fetch_symbols_for_tokens(
    db: AsyncSession,
    repo_name: str,
    tokens: list[str],
) -> tuple[frozenset[str], list[dict]]:
    if not tokens or not repo_name:
        return frozenset(), []
    lows = {t.lower() for t in tokens if t}
    stmt = select(RepoSymbol).where(RepoSymbol.repo_name == repo_name)
    result = await db.execute(stmt)
    rows = result.scalars().all()
    hits: set[str] = set()
    trace: list[dict] = []
    seen: set[tuple[str, str]] = set()
    for r in rows:
        sym_l = r.symbol.lower()
        base = sym_l.split(".")[-1]
        fp = r.file_path.replace("\\", "/")
        stem = fp.rsplit("/", 1)[-1].replace(".py", "").lower()
        matched = False
        reason = "symbol_index_hit"
        if sym_l in lows or base in lows:
            matched = True
        elif stem in lows:
            matched = True
            reason = "symbol_index_stem"
        if matched:
            hits.add(sym_l)
            hits.add(base)
            key = (r.symbol, fp)
            if key not in seen:
                seen.add(key)
                trace.append(
                    {
                        "symbol": r.symbol,
                        "file_path": fp,
                        "kind": r.kind,
                        "reason": reason,
                    }
                )

    stmt_m = select(RepoSymbolMeta).where(RepoSymbolMeta.repo_name == repo_name)
    result_m = await db.execute(stmt_m)
    for m in result_m.scalars().all():
        qn_l = m.qualified_name.lower()
        bases = [p for p in qn_l.split(".") if p]
        matched = False
        reason = "symbol_meta_qualified"
        for t in lows:
            if t in qn_l or qn_l == t or qn_l.endswith("." + t) or t in bases:
                matched = True
                break
        if matched:
            hits.add(qn_l)
            if bases:
                hits.add(bases[-1])
            key = (m.qualified_name, m.file_path.replace("\\", "/"))
            if key not in seen:
                seen.add(key)
                trace.append(
                    {
                        "symbol": m.qualified_name,
                        "file_path": m.file_path.replace("\\", "/"),
                        "kind": m.kind,
                        "reason": reason,
                    }
                )

    return frozenset(hits), trace


async def load_import_edges(
    db: AsyncSession,
    repo_name: str,
) -> list[tuple[str, str]]:
    if not repo_name:
        return []
    stmt = select(RepoImportEdge).where(RepoImportEdge.repo_name == repo_name)
    result = await db.execute(stmt)
    rows = result.scalars().all()
    return [(r.source_file, r.target_file) for r in rows]


async def fetch_chunks_for_files(
    db: AsyncSession,
    repo_name: str,
    file_paths: set[str],
) -> list:
    if not file_paths or not repo_name:
        return []
    from app.models.code import CodeChunk

    stmt = select(CodeChunk).where(
        CodeChunk.repo_name == repo_name,
        CodeChunk.file_path.in_(sorted(file_paths)),
    )
    result = await db.execute(stmt)
    return list(result.scalars().all())


async def fetch_git_scores_for_repo(
    db: AsyncSession,
    repo_name: str,
    *,
    now: datetime | None = None,
) -> dict[str, tuple[float, dict[str, float]]]:
    """file_path -> (git scalar, subscores) for composite retrieval."""
    if not repo_name:
        return {}
    stmt = select(RepoFileGit).where(RepoFileGit.repo_name == repo_name)
    result = await db.execute(stmt)
    out: dict[str, tuple[float, dict[str, float]]] = {}
    for r in result.scalars().all():
        fp = r.file_path.replace("\\", "/")
        g, subs = git_channel_score(
            r.last_commit_time,
            r.recent_commits_json,
            now=now,
        )
        out[fp] = (g, subs)
    return out


async def fetch_recent_commit_hints(
    db: AsyncSession,
    repo_name: str,
    limit: int = 12,
) -> list[str]:
    """Distinct recent commit message snippets for query expansion."""
    if not repo_name or limit <= 0:
        return []
    stmt = select(RepoFileGit).where(RepoFileGit.repo_name == repo_name)
    result = await db.execute(stmt)
    hints: list[str] = []
    seen: set[str] = set()
    for r in result.scalars().all():
        if r.last_message and r.last_message.strip():
            msg = r.last_message.strip()[:160]
            if msg.lower() not in seen:
                seen.add(msg.lower())
                hints.append(msg)
        if r.recent_commits_json:
            try:
                arr = json.loads(r.recent_commits_json)
                for item in arr:
                    m = (item.get("message") or "").strip()[:160]
                    if m and m.lower() not in seen:
                        seen.add(m.lower())
                        hints.append(m)
            except (json.JSONDecodeError, TypeError):
                pass
        if len(hints) >= limit:
            break
    return hints[:limit]


async def fetch_symbol_anchor_text(
    db: AsyncSession,
    repo_name: str,
    file_paths: set[str],
    identifiers: list[str],
    max_lines: int = 80,
) -> str:
    """Formatted symbol metadata for context (qualified names, signatures, docstrings)."""
    if not file_paths or not repo_name:
        return ""
    fps = {p.replace("\\", "/") for p in file_paths}
    lows = {i.lower() for i in identifiers if i}
    stmt = select(RepoSymbolMeta).where(
        RepoSymbolMeta.repo_name == repo_name,
        RepoSymbolMeta.file_path.in_(sorted(fps)),
    )
    result = await db.execute(stmt)
    rows = list(result.scalars().all())
    if lows:
        filtered = []
        for m in rows:
            qn = m.qualified_name.lower()
            if any(t in qn or qn.endswith("." + t) for t in lows):
                filtered.append(m)
        if filtered:
            rows = filtered
    lines: list[str] = []
    for m in rows[:200]:
        bits = [
            f"- `{m.qualified_name}` ({m.kind})",
            f"  {m.file_path}:{m.line_start}-{m.line_end}",
        ]
        if m.signature:
            bits.append(f"  sig: {m.signature[:300]}")
        if m.docstring:
            bits.append(f"  doc: {m.docstring[:400]}")
        if m.is_staticmethod:
            bits.append("  @staticmethod")
        if m.is_classmethod:
            bits.append("  @classmethod")
        if m.is_async:
            bits.append("  async")
        lines.extend(bits)
        if len(lines) >= max_lines:
            break
    return "\n".join(lines)


async def fetch_git_context_text(
    db: AsyncSession,
    repo_name: str,
    file_paths: set[str],
    max_files: int = 12,
) -> str:
    """Recent change summaries for selected files."""
    if not file_paths or not repo_name:
        return ""
    fps_set = {p.replace("\\", "/") for p in file_paths}
    fps = sorted(fps_set)[:max_files]
    if not fps:
        return ""
    stmt = select(RepoFileGit).where(
        RepoFileGit.repo_name == repo_name,
        RepoFileGit.file_path.in_(fps),
    )
    result = await db.execute(stmt)
    parts: list[str] = []
    for r in result.scalars().all():
        fp = r.file_path.replace("\\", "/")
        if fp not in fps_set:
            continue
        block = [f"File `{fp}`:"]
        if r.last_commit_time:
            block.append(f"  last commit: {r.last_commit_time} ({r.last_commit_hash or '?'})")
        if r.last_author:
            block.append(f"  author: {r.last_author}")
        if r.last_message:
            block.append(f"  message: {r.last_message[:300]}")
        if r.blame_summary_json:
            try:
                segs = json.loads(r.blame_summary_json)
                for s in segs[:4]:
                    block.append(
                        f"  lines {s.get('line_start')}-{s.get('line_end')}: "
                        f"{s.get('author', '')} ({s.get('hash', '')})"
                    )
            except (json.JSONDecodeError, TypeError):
                pass
        parts.append("\n".join(block))
    return "\n\n".join(parts)


async def lookup_repo_symbols(
    db: AsyncSession,
    repo_name: str,
    name_substring: str,
    *,
    limit: int = 25,
) -> list[dict]:
    """
    SQL lookup of ingested symbols (no embeddings). Used by FileTool.find_symbol / ReAct loop.
    """
    if not repo_name or not (name_substring or "").strip():
        return []
    from app.models.repo_intel import RepoSymbol

    pat = f"%{name_substring.strip()}%"
    stmt = (
        select(RepoSymbol)
        .where(RepoSymbol.repo_name == repo_name)
        .where(RepoSymbol.symbol.ilike(pat))
        .limit(limit)
    )
    rows = (await db.execute(stmt)).scalars().all()
    return [
        {"symbol": r.symbol, "file": r.file_path, "kind": r.kind}
        for r in rows
    ]


# ── Long-Term Cognition (Week 5) ─────────────────────────────────────────────

async def store_decision(db: AsyncSession, repo_name: str, symbol: str, context: str) -> None:
    db.add(AgentDecision(repo_name=repo_name, symbol=symbol, decision_context=context))
    await db.commit()

async def store_bug(db: AsyncSession, repo_name: str, symbol: str, description: str) -> None:
    db.add(AgentBugFix(repo_name=repo_name, symbol=symbol, bug_description=description))
    await db.commit()

async def store_query(
    db: AsyncSession,
    repo_name: str,
    query_text: str,
    response_text: str,
    query_embedding: list[float] | None = None,
) -> None:
    from app.core.config import get_settings

    st = get_settings()
    dim = int(getattr(st, "EMBEDDING_DIMENSION", 0) or 0)
    vec: list[float] | None = None
    if query_embedding and dim > 0 and len(query_embedding) == dim:
        vec = query_embedding
    elif query_embedding:
        logger.warning(
            "Skipping query_embedding on store_query: len=%d != EMBEDDING_DIMENSION=%d",
            len(query_embedding),
            dim,
        )

    row = AgentQueryHistory(
        repo_name=repo_name,
        query_text=query_text,
        response_text=response_text,
        query_embedding=vec,
    )
    db.add(row)
    try:
        await db.commit()
    except Exception as exc:
        err = str(exc).lower()
        if "dimension" in err or "dataerror" in err:
            await db.rollback()
            db.add(
                AgentQueryHistory(
                    repo_name=repo_name,
                    query_text=query_text,
                    response_text=response_text,
                    query_embedding=None,
                )
            )
            await db.commit()
            logger.warning(
                "agent_query_history: DB pgvector size mismatch — stored without embedding. "
                "Run `deepcode init-db` on a fresh DB or ALTER columns to match EMBEDDING_DIMENSION=%s.",
                dim,
            )
        else:
            raise


async def fetch_memory_context(
    db: AsyncSession,
    repo_name: str,
    query_text: str,
    query_embedding: list[float] | None = None,
    k: int = 3,
) -> list[dict]:
    """Retrieve scored decisions, bug fixes, and prior query history."""
    if not repo_name:
        return []

    results: list[dict] = []
    lows = {x.lower() for x in query_text.split() if len(x) > 2}

    # 1. Fetch Decisions via string matching on symbols
    st_dec = select(AgentDecision).where(AgentDecision.repo_name == repo_name).order_by(AgentDecision.timestamp.desc()).limit(20)
    for row in (await db.execute(st_dec)).scalars().all():
        if row.symbol and row.symbol.lower() in lows:
            score = memory_score(1.0, row.timestamp)
            results.append({
                "type": "decision",
                "symbol": row.symbol,
                "context": row.decision_context,
                "score": score,
                "timestamp": row.timestamp.isoformat()
            })

    # 2. Fetch Bugs via string matching on symbols
    st_bug = select(AgentBugFix).where(AgentBugFix.repo_name == repo_name).order_by(AgentBugFix.resolved_at.desc()).limit(20)
    for row in (await db.execute(st_bug)).scalars().all():
        if row.symbol and row.symbol.lower() in lows:
            score = memory_score(1.0, row.resolved_at)
            results.append({
                "type": "bug_fix",
                "symbol": row.symbol,
                "context": row.bug_description,
                "score": score,
                "timestamp": row.resolved_at.isoformat()
            })

    # 3. Fetch past queries (with semantic similarity if vector exists)
    if query_embedding and db.bind.dialect.name == "postgresql":
        try:
            st_qh = (
                select(AgentQueryHistory)
                .where(
                    AgentQueryHistory.repo_name == repo_name,
                    AgentQueryHistory.query_embedding.isnot(None),
                )
                .order_by(AgentQueryHistory.query_embedding.op("<=>")(query_embedding))
                .limit(10)
            )
            qh_rows = (await db.execute(st_qh)).scalars().all()
        except Exception as exc:
            logger.warning("Semantic query history skipped: %s", exc)
            qh_rows = []
        for row in qh_rows:
            try:
                # cosine matching mapping over vector pg array
                # pg op('<=>') is distance, so cosine_sim = 1.0 - distance
                from app.memory.retriever import cosine_similarity
                import json
                row_vec = json.loads(row.query_embedding) if isinstance(row.query_embedding, str) else row.query_embedding
                sim = cosine_similarity(query_embedding, row_vec)
                # scale to [0,1]
                sim = (sim + 1.0) / 2.0
            except:
                sim = 0.5

            score = memory_score(sim, row.timestamp)
            if score > 0.4:  # Relevance threshold for past queries
                results.append({
                    "type": "past_query",
                    "query": row.query_text,
                    "response": row.response_text,
                    "score": score,
                    "timestamp": row.timestamp.isoformat()
                })

    # Sort and take top k
    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:k]
