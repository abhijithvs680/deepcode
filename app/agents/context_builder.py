"""
Context Builder — assembles the final prompt from multiple sources.

This is a CRITICAL abstraction. All prompt engineering lives here.
When you add memory, RAG, tool results, or conversation history,
they get wired in through this builder — not scattered across routes.

Architecture:
    User Question
         │
    ContextBuilder.build()
         ├── system prompt
         ├── conversation history (from session)
         ├── user-provided context
         ├── (future) RAG results
         ├── (future) tool outputs
         │
         └──▶ Final Prompt → LLM
"""
from __future__ import annotations

from dataclasses import dataclass, field
import re

from app.core.config import get_settings

settings = get_settings()

# ── System Prompts ─────────────────────────────────────────────────────────────
SYSTEM_PROMPT = (
    "You are DeepCode Agent, a highly capable AI coding assistant. "
    "You provide clear, concise, and accurate answers. "
    "When relevant, include well-formatted code examples with explanations. "
    "If you're unsure about something, say so honestly."
)


@dataclass
class ContextPiece:
    """A single piece of context with a label and priority."""
    label: str
    content: str
    priority: int = 0  # Higher = included first


@dataclass
class BuiltContext:
    """The fully assembled prompt, ready to send to the LLM."""
    system_prompt: str
    full_prompt: str
    pieces_used: list[str] = field(default_factory=list)


def compress_context(content: str, query: str) -> str:
    """
    Compress a chunk's content to reduce noise by only extracting lines 
    that match query keywords.
    """
    lines = content.split("\n")
    query_words = set(query.lower().split())
    
    # If no words or content small, return as is
    if not query_words or len(lines) <= 15:
        return content
        
    relevant = []
    last_idx = -2
    
    for idx, line in enumerate(lines):
        line_lower = line.lower()
        if any(w in line_lower for w in query_words):
            # Optional: add visual break if jumping lines
            if idx > last_idx + 1 and last_idx != -2:
                relevant.append("    ...")
            relevant.append(line)
            last_idx = idx
            
    if not relevant:
        return "\n".join(lines[:15])

    return "\n".join(relevant[:15])


def estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4)


def truncate_to_token_budget(text: str, max_tokens: int) -> str:
    """Hard cap for tool / auxiliary blocks (char≈4×tokens)."""
    if max_tokens <= 0 or not text:
        return text
    max_chars = max_tokens * 4
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 24] + "\n… [truncated]"


def assemble_rag_context(
    chunks: list[dict],
    question: str,
    max_tokens: int | None = None,
) -> tuple[str, list[str], list[str]]:
    """
    Build Relevant Code text with hard token budget; priority via context_priority_tuple.
    Returns (rag_text, excluded_file_paths, included_file_paths in order).
    """
    from app.memory.scoring import context_priority_tuple

    cap = max_tokens if max_tokens is not None else settings.MAX_CONTEXT_TOKENS
    ordered = sorted(chunks, key=context_priority_tuple, reverse=True)
    parts: list[str] = []
    excluded: list[str] = []
    included_files: list[str] = []
    used = 0
    for c in ordered:
        body = compress_context(c["content"], question)
        block = (
            f"File: {c['file_path']} (score: {c.get('score', 0)})\n"
            f"```{c.get('language', '')}\n{body}\n```"
        )
        t = estimate_tokens(block)
        if used + t <= cap:
            parts.append(block)
            used += t
            fp = c["file_path"]
            if fp not in included_files:
                included_files.append(fp)
        else:
            excluded.append(c["file_path"])
    return "\n\n".join(parts), excluded, included_files


def assemble_memory_context(memories: list[dict], max_tokens: int = 1500) -> str:
    """
    Format Agent memory entries into a structured block.
    Limits injection token size to avoid blowing up the context window.
    """
    if not memories:
        return ""
    
    parts = []
    used = 0
    from app.memory.scoring import estimate_tokens
    
    for m in memories:
        if m.get("type") == "decision":
            block = f"[PAST DECISION | {m.get('symbol', 'Global')}]\n({m.get('timestamp', 'Unknown')}): {m.get('context', '')}"
        elif m.get("type") == "bug_fix":
            block = f"[PAST BUG FIX | {m.get('symbol', 'Global')}]\n({m.get('timestamp', 'Unknown')}): {m.get('context', '')}"
        else:
            block = f"[PAST INTERACTION]\nUser: {m.get('query', '')}\nAction: {m.get('response', '')}"
            
        t = estimate_tokens(block)
        if used + t <= max_tokens:
            parts.append(block)
            used += t
            
    return "\n\n".join(parts)


def assemble_symbol_git_rag_context(
    symbol_section: str,
    git_section: str,
    chunks: list[dict],
    question: str,
    max_tokens: int | None = None,
    *,
    symbol_budget: int | None = None,
    git_budget: int | None = None,
    rag_budget: int | None = None,
) -> tuple[str, list[str], list[str]]:
    """
    Shared token budget: symbol anchors first, Git summary second, RAG chunks last.
    Optional per-layer budgets override soft fractions when provided (Week 10).
    """
    if symbol_budget is not None and git_budget is not None and rag_budget is not None:
        cap = symbol_budget + git_budget + rag_budget
    else:
        cap = max_tokens if max_tokens is not None else settings.MAX_CONTEXT_TOKENS
    used = 0
    parts: list[str] = []

    def _append_block(title: str, body: str, soft_frac: float, hard_max: int | None) -> None:
        nonlocal used
        text = body.strip()
        if not text:
            return
        block = f"### {title}\n{text}"
        t = estimate_tokens(block)
        if hard_max is not None:
            soft_budget = hard_max
        else:
            soft_budget = max(120, int(cap * soft_frac))
        if t > soft_budget:
            block = block[: max(1, soft_budget * 4)]
            t = estimate_tokens(block)
        if used + t <= cap:
            parts.append(block)
            used += t
            return
        remain = cap - used
        if remain > 80:
            trimmed = block[: max(0, remain * 4 - 24)]
            parts.append(trimmed)
            used += estimate_tokens(trimmed)

    sym_b = symbol_budget if symbol_budget is not None else None
    git_b = git_budget if git_budget is not None else None
    _append_block("Symbol anchors", symbol_section, 0.35, sym_b)
    _append_block("Recent changes (Git)", git_section, 0.28, git_b)

    if rag_budget is not None:
        remaining = rag_budget
    else:
        remaining = max(100, cap - used)
    rag_text, excluded, included_files = assemble_rag_context(
        chunks, question, max_tokens=remaining
    )
    if rag_text.strip():
        parts.append("### Relevant code\n" + rag_text)
    else:
        excluded = [c["file_path"] for c in chunks]

    return "\n\n".join(parts), excluded, included_files


class ContextBuilder:
    """
    Assembles the final LLM prompt from multiple context sources.

    Usage:
        builder = ContextBuilder(question="What is Python?")
        builder.add_context("User hint", "Focus on async features")
        builder.add_conversation_history([...])
        result = builder.build()
        # result.full_prompt → send to LLM
    """

    def __init__(self, question: str):
        self.question = question
        self._pieces: list[ContextPiece] = []
        self._conversation_history: list[dict[str, str]] = []

    def add_context(self, label: str, content: str, priority: int = 0) -> "ContextBuilder":
        """
        Add a context piece (user-provided, RAG results, tool output, etc.)

        Args:
            label: Human-readable label (e.g., "User Context", "Code Search Results")
            content: The actual context text.
            priority: Higher priority pieces appear first.

        Returns:
            self — for method chaining.
        """
        if content and content.strip():
            self._pieces.append(ContextPiece(label=label, content=content.strip(), priority=priority))
        return self

    def add_conversation_history(self, history: list[dict[str, str]]) -> "ContextBuilder":
        """
        Add previous conversation turns for multi-turn context.

        Args:
            history: List of {"role": "user"|"assistant", "content": "..."} dicts.

        Returns:
            self — for method chaining.
        """
        self._conversation_history = history
        return self

    def build(self) -> BuiltContext:
        """
        Assemble the final prompt from all registered context pieces.

        Returns:
            BuiltContext with the full_prompt ready for the LLM.
        """
        parts: list[str] = [SYSTEM_PROMPT]
        pieces_used: list[str] = ["system_prompt"]

        # Conversation history (if multi-turn)
        if self._conversation_history:
            history_text = self._format_history()
            parts.append(f"\n--- Conversation History ---\n{history_text}")
            pieces_used.append("conversation_history")

        # Context pieces, sorted by priority (high first)
        sorted_pieces = sorted(self._pieces, key=lambda p: p.priority, reverse=True)
        for piece in sorted_pieces:
            parts.append(f"\n--- {piece.label} ---\n{piece.content}")
            pieces_used.append(piece.label)

        # The actual question
        parts.append(f"\n--- Question ---\n{self.question}")

        full_prompt = "\n".join(parts)

        return BuiltContext(
            system_prompt=SYSTEM_PROMPT,
            full_prompt=full_prompt,
            pieces_used=pieces_used,
        )

    def _format_history(self) -> str:
        """Format conversation history into a readable string."""
        lines = []
        for turn in self._conversation_history[-10:]:  # Last 10 turns max
            role = turn.get("role", "unknown").capitalize()
            content = turn.get("content", "")
            lines.append(f"{role}: {content}")
        return "\n".join(lines)
