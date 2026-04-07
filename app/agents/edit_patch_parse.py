"""Parse <edit> search/replace blocks from model output."""
from __future__ import annotations

import re
from typing import Any


_EDIT_BLOCK = re.compile(
    r"<edit\s+file\s*=\s*['\"](?P<fp>[^'\"]+)['\"]\s*>(?P<body>.*?)</edit>",
    re.DOTALL | re.IGNORECASE,
)
_SEARCH = re.compile(r"<search>(.*?)</search>", re.DOTALL | re.IGNORECASE)
_REPLACE = re.compile(r"<replace>(.*?)</replace>", re.DOTALL | re.IGNORECASE)


def parse_edit_blocks(answer: str) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for m in _EDIT_BLOCK.finditer(answer or ""):
        body = m.group("body") or ""
        sm = _SEARCH.search(body)
        rm = _REPLACE.search(body)
        if not sm or not rm:
            continue
        fp = (m.group("fp") or "").strip().replace("\\", "/")
        if not fp:
            continue
        out.append(
            {
                "file_path": fp,
                "search": sm.group(1),
                "replace": rm.group(1),
            }
        )
    return out
