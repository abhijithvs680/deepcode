"""repo_manager helpers (no DB)."""
from pathlib import Path

from app.core.repo_manager import (
    is_remote_ref,
    stable_repo_id_from_path,
    stable_repo_id_from_url,
)


def test_stable_path_id_deterministic():
    p = Path("C:/tmp/foo") if Path.cwd().drive else Path("/tmp/foo")
    a = stable_repo_id_from_path(p.resolve())
    b = stable_repo_id_from_path(p.resolve())
    assert a == b
    assert len(a) == 32


def test_stable_url_id_strips_git_suffix():
    u1 = stable_repo_id_from_url("https://github.com/a/b.git")
    u2 = stable_repo_id_from_url("https://github.com/a/b")
    assert u1 == u2


def test_is_remote_ref():
    assert is_remote_ref("https://x.com/y")
    assert is_remote_ref("git@github.com:a/b.git")
    assert not is_remote_ref("./local")
