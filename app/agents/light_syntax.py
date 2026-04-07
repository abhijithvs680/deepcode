"""Cheap syntax checks before persisting edited files (always-on safety net)."""
from __future__ import annotations

import ast
import json
from typing import Tuple


def validate_text_for_path(rel_path: str, content: str) -> Tuple[bool, str]:
    """
    Return (ok, error_message). Unknown extensions pass.
    """
    p = (rel_path or "").lower()
    if p.endswith(".py"):
        try:
            ast.parse(content)
            return True, ""
        except SyntaxError as e:
            return False, f"Python: {e.msg} at line {e.lineno}"
    if p.endswith(".json"):
        try:
            json.loads(content)
            return True, ""
        except json.JSONDecodeError as e:
            return False, f"JSON: {e.msg} at line {e.lineno} col {e.colno}"
    if p.endswith((".yaml", ".yml")):
        try:
            import yaml  # type: ignore[import-not-found]

            yaml.safe_load(content)
            return True, ""
        except ImportError:
            return True, ""
        except Exception as e:
            return False, f"YAML: {e}"
    return True, ""
