"""Week 4 — extended AST / RepoSymbolMeta-style extraction."""
import json

import pytest

from app.agents.context_builder import assemble_symbol_git_rag_context
from app.agents.query_expander import merge_queries_with_repo_hints
from app.ingestion.ast_analysis import analyze_python_source, build_repo_intel_from_files
from app.memory.scoring import (
    W_GIT,
    W_KEYWORD,
    W_PATH,
    W_SEMANTIC,
    W_SYMBOL,
    composite_score,
    git_channel_score,
)


def test_weight_constants_sum_to_one():
    assert W_SEMANTIC + W_KEYWORD + W_SYMBOL + W_PATH + W_GIT == pytest.approx(1.0)


def test_composite_includes_git():
    a = composite_score(1.0, 0.0, 0.0, 0.0, 0.0)
    b = composite_score(1.0, 0.0, 0.0, 0.0, 1.0)
    assert b - a == pytest.approx(0.05, abs=0.001)


def test_git_channel_score_recency_and_churn():
    g0, subs0 = git_channel_score(None, "[]")
    assert g0 == 0.0
    assert subs0["git_recency"] == 0.0

    recent = json.dumps([{"hash": "a"}, {"hash": "b"}])
    g1, subs1 = git_channel_score("2099-01-01T00:00:00+00:00", recent)
    assert subs1["git_recency"] == pytest.approx(1.0, abs=0.01)
    assert subs1["git_churn"] == pytest.approx(0.1, abs=0.01)
    assert g1 == pytest.approx(0.65 * 1.0 + 0.35 * 0.1, abs=0.02)


def test_analyze_python_methods_decorators_constants():
    src = '''
"""Module doc."""
CONST = 42

class Foo:
    """Class doc."""
    @classmethod
    def cm(cls, x: int) -> None:
        """method doc"""
        pass
    @staticmethod
    def st():
        pass
    async def ar(self):
        pass

def top():
    pass
'''
    syms, metas, _edges = analyze_python_source("app/x.py", src, {"app"})
    by_q = {m["qualified_name"]: m for m in metas}
    assert by_q["Foo"]["kind"] == "class"
    assert json.loads(by_q["Foo"].get("bases_json") or "[]") == []
    assert by_q["Foo.cm"]["is_classmethod"] is True
    assert by_q["Foo.st"]["is_staticmethod"] is True
    assert by_q["Foo.ar"]["is_async"] is True
    assert "def cm" in (by_q["Foo.cm"].get("signature") or "")
    assert by_q["CONST"]["kind"] == "constant"


def test_build_repo_intel_from_files_empty_meta(tmp_path):
    p = tmp_path / "app" / "m.py"
    p.parent.mkdir(parents=True)
    p.write_text("def f(): pass\n", encoding="utf-8")
    files = [
        {
            "file_path": "app/m.py",
            "abs_path": str(p),
            "language": "py",
        }
    ]
    syms, meta, edges = build_repo_intel_from_files(files, "r1", "app")
    assert any(s["symbol"] == "f" for s in syms)
    assert any(m["qualified_name"] == "f" and m["kind"] == "function" for m in meta)


def test_merge_queries_with_repo_hints_cap():
    base = ["q1", "q2"]
    hints = ["h1", "h2", "h3"]
    out = merge_queries_with_repo_hints(base, hints, max_total=4)
    assert len(out) <= 4
    assert "h1" in out


def test_assemble_symbol_git_rag_order_small_budget():
    chunks = [
        {
            "file_path": "a.py",
            "content": "print(1)",
            "language": "python",
            "chunk_index": 0,
            "score": 0.9,
            "score_breakdown": {},
            "symbol_tier": "high",
            "symbol_reason": "t",
        }
    ]
    text, _ex, _inc = assemble_symbol_git_rag_context(
        "SYMBOL_BLOCK",
        "GIT_BLOCK",
        chunks,
        "print",
        max_tokens=5000,
    )
    assert text.index("Symbol anchors") < text.index("Relevant code")
    assert text.index("Recent changes (Git)") < text.index("Relevant code")
    assert "SYMBOL_BLOCK" in text
    assert "GIT_BLOCK" in text
