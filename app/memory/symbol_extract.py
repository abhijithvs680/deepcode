"""Extract code-like identifiers from natural-language queries."""
from __future__ import annotations

import re

_SYMBOL_STOPWORDS = frozenset({
    "logger", "config", "settings", "request", "response", "context", "database",
    "session", "model", "object", "string", "import", "return", "class", "function",
    "def", "what", "does", "how", "where", "when", "which", "this", "that", "with",
    "from", "into", "about", "file", "code", "test", "tests", "the", "and", "for",
})


def extract_query_identifiers(query: str) -> list[str]:
    """Pull snake_case, PascalCase, and dotted names from a user question."""
    seen: set[str] = set()
    out: list[str] = []

    for pattern in (
        r"\b([A-Z][a-zA-Z0-9]+(?:\.[A-Za-z_][a-zA-Z0-9_]*)*)\b",
        r"\b([a-z][a-z0-9_]*(?:\.[a-z][a-z0-9_]*)*)\b",
    ):
        for m in re.finditer(pattern, query):
            s = m.group(1).strip()
            if len(s) < 2:
                continue
            key = s.lower()
            if key not in seen:
                seen.add(key)
                out.append(s)
    return out


def snake_to_pascal(name: str) -> str:
    parts = name.replace(".", "_").split("_")
    return "".join(p[:1].upper() + p[1:] if p else "" for p in parts if p)


def expansion_variants_for_identifiers(identifiers: list[str]) -> list[str]:
    """Code-aware search strings (original + common variants)."""
    variants: list[str] = []
    seen: set[str] = set()

    def add(s: str) -> None:
        s = s.strip()
        if not s or s in seen:
            return
        seen.add(s)
        variants.append(s)

    for raw in identifiers:
        add(raw)
        if "_" in raw or raw.islower():
            add(snake_to_pascal(raw))
        spaced = raw.replace("_", " ").strip()
        if spaced != raw:
            add(spaced)
        stem = raw.split(".")[-1]
        if stem.endswith(".py"):
            stem = stem[:-3]
        add(f"file:{stem}.py")
        add(f"class {snake_to_pascal(stem) if '_' in stem or stem.islower() else stem}")
        add(f"def {stem}")
    return variants


def allow_partial_symbol_token(token: str) -> bool:
    t = token.lower()
    if len(t) < 6:
        return False
    if t in _SYMBOL_STOPWORDS:
        return False
    return True
