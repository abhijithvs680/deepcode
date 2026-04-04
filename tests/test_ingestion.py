"""
Tests for the ingestion pipeline (loader, chunker, processor).
"""
import os
import tempfile
from pathlib import Path

import pytest

from app.ingestion.loader import load_repo, IGNORE_DIRS
from app.ingestion.chunker import chunk_file
from app.ingestion.processor import process_repo


class TestLoader:
    """Tests for the repository file loader."""

    def test_loads_python_files(self, tmp_path):
        """Should find .py files."""
        (tmp_path / "main.py").write_text("print('hello')")
        (tmp_path / "utils.py").write_text("def foo(): pass")

        files = load_repo(str(tmp_path))
        paths = [f["file_path"] for f in files]

        assert "main.py" in paths
        assert "utils.py" in paths
        assert len(files) == 2

    def test_skips_ignored_dirs(self, tmp_path):
        """Should skip __pycache__, .git, node_modules, etc."""
        cache_dir = tmp_path / "__pycache__"
        cache_dir.mkdir()
        (cache_dir / "cached.pyc").write_text("bytecode")

        git_dir = tmp_path / ".git"
        git_dir.mkdir()
        (git_dir / "config").write_text("git stuff")

        (tmp_path / "app.py").write_text("print('keep')")

        files = load_repo(str(tmp_path))
        paths = [f["file_path"] for f in files]

        assert "app.py" in paths
        assert len(files) == 1  # Only app.py, not cached or git files

    def test_skips_non_source_files(self, tmp_path):
        """Should skip binary/non-source extensions."""
        (tmp_path / "photo.jpg").write_bytes(b"\xff\xd8\xff")
        (tmp_path / "data.bin").write_bytes(b"\x00\x01")
        (tmp_path / "code.py").write_text("x = 1")

        files = load_repo(str(tmp_path))
        assert len(files) == 1
        assert files[0]["file_path"] == "code.py"

    def test_returns_relative_paths(self, tmp_path):
        """Paths should be relative to the repo root."""
        sub = tmp_path / "src" / "core"
        sub.mkdir(parents=True)
        (sub / "main.py").write_text("pass")

        files = load_repo(str(tmp_path))
        assert files[0]["file_path"].replace("\\", "/") == "src/core/main.py"

    def test_includes_language(self, tmp_path):
        """Should detect language from extension."""
        (tmp_path / "app.py").write_text("pass")
        (tmp_path / "style.css").write_text("body {}")

        files = load_repo(str(tmp_path))
        langs = {f["file_path"]: f["language"] for f in files}

        assert langs["app.py"] == "py"
        assert langs["style.css"] == "css"

    def test_raises_on_invalid_path(self):
        """Should raise if path doesn't exist."""
        with pytest.raises(ValueError, match="Not a directory"):
            load_repo("/nonexistent/path/abc123")


class TestChunker:
    """Tests for the code chunker."""

    def test_basic_chunking(self):
        """Should split content into chunks of specified size."""
        content = "a" * 1000
        chunks = chunk_file(content, chunk_size=500, overlap=0)
        assert len(chunks) == 2
        assert all(len(c) == 500 for c in chunks)

    def test_small_file_single_chunk(self):
        """Files smaller than chunk_size → single chunk."""
        content = "def foo(): pass"
        chunks = chunk_file(content, chunk_size=500)
        assert len(chunks) == 1
        assert chunks[0] == content

    def test_empty_content_returns_empty(self):
        """Empty/whitespace content → empty list."""
        assert chunk_file("") == []
        assert chunk_file("   ") == []

    def test_overlap_creates_more_chunks(self):
        """Overlap should create more chunks than without."""
        content = "a" * 1000
        no_overlap = chunk_file(content, chunk_size=500, overlap=0)
        with_overlap = chunk_file(content, chunk_size=500, overlap=100)
        assert len(with_overlap) > len(no_overlap)

    def test_overlap_preserves_context(self):
        """Adjacent chunks should share overlap characters."""
        content = "0123456789" * 10  # 100 chars
        chunks = chunk_file(content, chunk_size=40, overlap=10)

        # End of first chunk should match start of second chunk
        overlap_region = chunks[0][-10:]
        assert chunks[1].startswith(overlap_region)


class TestProcessor:
    """Tests for the repository processor."""

    def test_processes_repo(self, tmp_path):
        """Should load, read, and chunk all source files."""
        (tmp_path / "main.py").write_text("x = 1\n" * 100)
        (tmp_path / "README.md").write_text("# Hello\n" * 10)

        results = process_repo(str(tmp_path))

        assert len(results) > 0
        assert all("file_path" in r for r in results)
        assert all("content" in r for r in results)
        assert all("repo_name" in r for r in results)
        assert all("chunk_index" in r for r in results)
        assert results[0]["repo_name"] == tmp_path.name

    def test_skips_unreadable_files(self, tmp_path):
        """Should skip files that can't be read without crashing."""
        (tmp_path / "good.py").write_text("x = 1")
        # Binary file that can't be chunked meaningfully
        (tmp_path / "bad.py").write_bytes(b"\x80\x81\x82")

        results = process_repo(str(tmp_path))
        # Should have at least the good file's chunks
        assert any(r["file_path"] == "good.py" for r in results)

    def test_empty_repo(self, tmp_path):
        """Empty repo → empty results."""
        results = process_repo(str(tmp_path))
        assert results == []
