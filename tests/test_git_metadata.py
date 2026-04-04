"""Week 4 — Git metadata ingest and scoring."""
import json
from datetime import datetime, timezone

import pytest

git_mod = pytest.importorskip("git")

from app.ingestion.git_metadata import build_repo_file_git_rows
from app.memory.scoring import git_channel_score


def test_build_repo_file_git_rows_ephemeral_repo(tmp_path):
    src = tmp_path / "lib.py"
    src.write_text("a = 1\n", encoding="utf-8")
    repo = git_mod.Repo.init(tmp_path)
    repo.index.add(["lib.py"])
    repo.index.commit("init lib")
    src.write_text("a = 2\n", encoding="utf-8")
    repo.index.add(["lib.py"])
    repo.index.commit("bump a")

    files = [
        {
            "file_path": "lib.py",
            "abs_path": str(src),
            "language": "py",
        }
    ]
    rows = build_repo_file_git_rows(files, tmp_path, "demo")
    assert len(rows) == 1
    row = rows[0]
    assert row["repo_name"] == "demo"
    assert row["file_path"] == "lib.py"
    assert row["last_commit_hash"]
    assert row["last_message"]
    recent = json.loads(row["recent_commits_json"])
    assert len(recent) >= 2


def test_build_repo_file_git_rows_skips_without_git(tmp_path):
    (tmp_path / "x.txt").write_text("hi")
    files = [{"file_path": "x.txt", "abs_path": str(tmp_path / "x.txt"), "language": "text"}]
    assert build_repo_file_git_rows(files, tmp_path, "n") == []


def test_git_channel_score_uses_fixed_now():
    now = datetime(2020, 6, 15, tzinfo=timezone.utc)
    past = datetime(2020, 6, 1, tzinfo=timezone.utc)
    g, subs = git_channel_score(
        past.isoformat(),
        json.dumps([{"hash": "1"}]),
        now=now,
    )
    assert 0.0 < g <= 1.0
    assert subs["git_churn"] == pytest.approx(0.05, abs=0.01)
