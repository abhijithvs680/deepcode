"""
Query Expander — code-aware expansion plus light NLP rules for vague queries.
"""
import re

from app.memory.symbol_extract import expansion_variants_for_identifiers, extract_query_identifiers

EXPANSION_RULES = {
    "auth": ["authentication", "login", "jwt", "token", "password", "oauth"],
    "db": ["database", "sql", "postgres", "connection", "query", "session", "model", "sqlalchemy"],
    "database": ["db", "sql", "postgres", "connection", "query", "session", "model", "sqlalchemy"],
    "api": ["route", "endpoint", "controller", "fastapi", "router", "request", "response"],
    "retrieval": ["search", "rag", "embedding", "chunk", "vector", "similarity"],
    "state": ["store", "context", "redux", "provider", "memory"],
    "error": ["exception", "traceback", "log", "warning", "fail", "crash"],
}


def expand_query(query: str) -> list[str]:
    """Expand a query for retrieval: original + code-shaped variants + optional rules."""
    expansions: list[str] = [query]

    identifiers = extract_query_identifiers(query)
    for v in expansion_variants_for_identifiers(identifiers):
        expansions.append(v)
    for composed in (f"{query} {ident}" for ident in identifiers[:5]):
        expansions.append(composed.strip())

    tokens = re.findall(r"[a-z]+", query.lower())
    for token in tokens:
        if token in EXPANSION_RULES:
            for related_term in EXPANSION_RULES[token]:
                expansions.append(f"{query} {related_term}".strip())
                expansions.append(related_term)

    seen: set[str] = set()
    unique: list[str] = []
    for exp in expansions:
        e = exp.strip()
        if e and e not in seen:
            seen.add(e)
            unique.append(e)
    return unique


def merge_queries_with_repo_hints(
    base_queries: list[str],
    commit_hints: list[str],
    max_total: int = 40,
) -> list[str]:
    """Append bounded DB-backed commit message hints for retrieval (deduped)."""
    if not commit_hints:
        return base_queries[:max_total]
    seen: set[str] = {q.strip().lower() for q in base_queries if q.strip()}
    out = list(base_queries)
    for h in commit_hints:
        h = h.strip()
        if not h or h.lower() in seen:
            continue
        seen.add(h.lower())
        out.append(h)
        if len(out) >= max_total:
            break
    return out[:max_total]
