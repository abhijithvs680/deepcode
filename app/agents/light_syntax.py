"""Cheap syntax checks before persisting edited files (always-on safety net)."""
from __future__ import annotations

import ast
import json
import py_compile
import tempfile
from typing import Tuple


def validate_python_py_compile(content: str) -> Tuple[bool, str]:
    """Byte-compile check (catches some issues ast.parse alone may miss)."""
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".py",
            encoding="utf-8",
            delete=True,
        ) as tmp:
            tmp.write(content)
            tmp.flush()
            py_compile.compile(tmp.name, doraise=True)
        return True, ""
    except py_compile.PyCompileError as e:
        return False, f"py_compile: {e.msg}"
    except OSError as e:
        return False, str(e)


def validate_text_for_path(rel_path: str, content: str) -> Tuple[bool, str]:
    """
    Return (ok, error_message). Unknown extensions pass.
    """
    p = (rel_path or "").lower()
    if p.endswith(".py"):
        try:
            ast.parse(content)
        except SyntaxError as e:
            return False, f"Python: {e.msg} at line {e.lineno}"
        ok, err = validate_python_py_compile(content)
        if not ok:
            return False, err
        return True, ""
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
