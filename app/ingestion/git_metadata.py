"""
Git log + blame summaries per file for ingest (GitPython).

Skips gracefully when path is not a git repo or GitPython is unavailable.
"""
from __future__ import annotations

import json
from pathlib import Path, PurePosixPath


def build_repo_file_git_rows(
    files: list[dict],
    repo_root: Path,
    repo_name: str,
    *,
    max_commits: int = 15,
    blame_segments_max: int = 8,
) -> list[dict]:
    """
    Build rows for RepoFileGit aligned with loader `file_path` (POSIX relative).

    Only includes files with at least one commit in history.
    """
    try:
        import git
    except ImportError:
        return []

    root = Path(repo_root).resolve()
    if not (root / ".git").exists():
        return []
    try:
        repo = git.Repo(root)
    except Exception:
        return []
    if repo.bare:
        return []

    rows: list[dict] = []
    for fi in files:
        rel = str(PurePosixPath(fi["file_path"]))
        abs_f = Path(fi["abs_path"])
        if not abs_f.is_file():
            continue
        try:
            commits = list(repo.iter_commits(paths=rel, max_count=max_commits))
        except Exception:
            commits = []
        if not commits:
            continue
        last = commits[0]
        recent: list[dict] = []
        for c in commits:
            msg = (c.message or "").split("\n")[0]
            recent.append(
                {
                    "hash": c.hexsha[:12],
                    "message": msg[:200],
                    "author": c.author.name if c.author else "",
                    "time": c.committed_datetime.isoformat()
                    if c.committed_datetime
                    else "",
                }
            )

        blame_summary: list[dict] = []
        try:
            blamed = repo.blame("HEAD", rel)
            segs: list[dict] = []
            line_start = 1
            for commit, line_list in blamed:
                n = len(line_list)
                segs.append(
                    {
                        "line_start": line_start,
                        "line_end": line_start + n - 1,
                        "author": (commit.author.name or "")[:120]
                        if commit.author
                        else "",
                        "hash": commit.hexsha[:8],
                        "lines": n,
                    }
                )
                line_start += n
            segs.sort(key=lambda x: int(x.get("lines", 0)), reverse=True)
            blame_summary = segs[:blame_segments_max]
            for s in blame_summary:
                s.pop("lines", None)
        except Exception:
            pass

        rows.append(
            {
                "repo_name": repo_name,
                "file_path": rel,
                "last_commit_hash": last.hexsha[:40],
                "last_commit_time": last.committed_datetime.isoformat()
                if last.committed_datetime
                else "",
                "last_author": (last.author.name or "")[:256] if last.author else "",
                "last_message": ((last.message or "").split("\n")[0][:500]),
                "recent_commits_json": json.dumps(recent[:max_commits]),
                "blame_summary_json": json.dumps(blame_summary),
            }
        )
    return rows
