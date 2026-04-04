"""Tests for root .gitignore filtering in load_repo."""

import pytest

pytest.importorskip("pathspec")

from app.ingestion.loader import load_repo


def test_gitignore_excludes_matching_file(tmp_path):
    (tmp_path / ".gitignore").write_text("secret.py\n", encoding="utf-8")
    (tmp_path / "secret.py").write_text("x = 1\n", encoding="utf-8")
    (tmp_path / "app.py").write_text("y = 2\n", encoding="utf-8")

    files = load_repo(str(tmp_path))
    paths = {f["file_path"] for f in files}

    assert "app.py" in paths
    assert "secret.py" not in paths


def test_gitignore_prunes_ignored_directory(tmp_path):
    (tmp_path / ".gitignore").write_text("build/\n", encoding="utf-8")
    build = tmp_path / "build"
    build.mkdir()
    (build / "nested.py").write_text("pass\n", encoding="utf-8")
    (tmp_path / "ok.py").write_text("pass\n", encoding="utf-8")

    files = load_repo(str(tmp_path))
    paths = {f["file_path"] for f in files}

    assert paths == {"ok.py"}


def test_no_gitignore_loads_all_source(tmp_path):
    (tmp_path / "a.py").write_text("1\n", encoding="utf-8")
    (tmp_path / "b.py").write_text("2\n", encoding="utf-8")

    files = load_repo(str(tmp_path))
    assert len(files) == 2
