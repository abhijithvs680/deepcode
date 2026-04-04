"""
Static analysis: symbols, RepoSymbolMeta rows, and import edges.
"""
from __future__ import annotations

import ast
import json
from pathlib import Path, PurePosixPath

DOCSTRING_MAX = 500


def _norm_rel(path: str) -> str:
    return str(PurePosixPath(path))


def _module_to_file(module: str, package_roots: set[str]) -> str | None:
    parts = module.split(".")
    if not parts or parts[0] not in package_roots:
        return None
    return _norm_rel(Path(*parts).with_suffix(".py"))


def _safe_unparse(node: ast.AST | None) -> str:
    if node is None:
        return ""
    try:
        return ast.unparse(node)
    except AttributeError:
        return ""


def _docstring(node: ast.AST) -> str:
    raw = ast.get_docstring(node) or ""
    return raw[:DOCSTRING_MAX]


def _decorator_flags(decs: list[ast.expr]) -> tuple[bool, bool]:
    static, cmethod = False, False
    for d in decs:
        name = None
        if isinstance(d, ast.Name):
            name = d.id
        elif isinstance(d, ast.Call):
            f = d.func
            if isinstance(f, ast.Name):
                name = f.id
            elif isinstance(f, ast.Attribute):
                name = f.attr
        if name == "staticmethod":
            static = True
        elif name == "classmethod":
            cmethod = True
    return static, cmethod


def _record_function_meta(
    node: ast.FunctionDef | ast.AsyncFunctionDef,
    qualified_name: str,
    parent: str | None,
    kind: str,
    meta_out: list[dict],
) -> None:
    is_async = isinstance(node, ast.AsyncFunctionDef)
    st, cm = _decorator_flags(node.decorator_list)
    args = _safe_unparse(node.args)
    ret = _safe_unparse(node.returns) if node.returns else ""
    prefix = "async def " if is_async else "def "
    sig_line = f"{prefix}{node.name}({args})"
    if ret:
        sig_line += f" -> {ret}"

    meta_out.append(
        {
            "qualified_name": qualified_name,
            "kind": kind,
            "parent_symbol": parent,
            "line_start": node.lineno,
            "line_end": getattr(node, "end_lineno", node.lineno) or node.lineno,
            "bases_json": None,
            "signature": sig_line[:2000],
            "return_annotation": ret[:500],
            "docstring": _docstring(node),
            "is_async": is_async,
            "is_classmethod": cm,
            "is_staticmethod": st,
        }
    )


def analyze_python_source(
    rel_file: str,
    source: str,
    package_roots: set[str],
) -> tuple[list[dict], list[dict], list[tuple[str, str]]]:
    """
    Returns:
        symbol_rows: simple {symbol, kind, file_path} for RepoSymbol
        meta_rows: rich dicts for RepoSymbolMeta
        edges: (source_file, target_file)
    """
    rel_file = _norm_rel(rel_file)
    symbol_rows: list[dict] = []
    meta_rows: list[dict] = []
    edges: list[tuple[str, str]] = []
    seen_sym: set[tuple[str, str, str]] = set()

    def add_symbol(sym: str, kind: str) -> None:
        key = (sym, kind, rel_file)
        if key not in seen_sym:
            seen_sym.add(key)
            symbol_rows.append({"symbol": sym, "kind": kind, "file_path": rel_file})

    try:
        tree = ast.parse(source)
    except SyntaxError:
        return symbol_rows, meta_rows, edges

    stem = PurePosixPath(rel_file).stem
    add_symbol(stem, "module")
    meta_rows.append(
        {
            "qualified_name": stem,
            "kind": "module",
            "parent_symbol": None,
            "line_start": 1,
            "line_end": len(source.splitlines()) or 1,
            "bases_json": None,
            "signature": "",
            "return_annotation": "",
            "docstring": _docstring(tree) if hasattr(tree, "body") else "",
            "is_async": False,
            "is_classmethod": False,
            "is_staticmethod": False,
        }
    )

    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            add_symbol(node.name, "class")
            bases = [_safe_unparse(b) for b in node.bases]
            meta_rows.append(
                {
                    "qualified_name": node.name,
                    "kind": "class",
                    "parent_symbol": None,
                    "line_start": node.lineno,
                    "line_end": getattr(node, "end_lineno", node.lineno) or node.lineno,
                    "bases_json": json.dumps(bases[:20]),
                    "signature": f"class {node.name}",
                    "return_annotation": "",
                    "docstring": _docstring(node),
                    "is_async": False,
                    "is_classmethod": False,
                    "is_staticmethod": False,
                }
            )
            for item in node.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    qn = f"{node.name}.{item.name}"
                    add_symbol(qn, "method")
                    _record_function_meta(item, qn, node.name, "method", meta_rows)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            add_symbol(node.name, "function")
            _record_function_meta(node, node.name, None, "function", meta_rows)
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            name = node.target.id
            add_symbol(name, "variable")
            meta_rows.append(
                {
                    "qualified_name": name,
                    "kind": "variable",
                    "parent_symbol": None,
                    "line_start": node.lineno,
                    "line_end": getattr(node, "end_lineno", node.lineno) or node.lineno,
                    "bases_json": None,
                    "signature": f"{name}: {_safe_unparse(node.annotation)}",
                    "return_annotation": "",
                    "docstring": "",
                    "is_async": False,
                    "is_classmethod": False,
                    "is_staticmethod": False,
                }
            )
        elif isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name):
                    add_symbol(t.id, "constant")
                    meta_rows.append(
                        {
                            "qualified_name": t.id,
                            "kind": "constant",
                            "parent_symbol": None,
                            "line_start": node.lineno,
                            "line_end": getattr(node, "end_lineno", node.lineno) or node.lineno,
                            "bases_json": None,
                            "signature": f"{t.id} = ...",
                            "return_annotation": "",
                            "docstring": "",
                            "is_async": False,
                            "is_classmethod": False,
                            "is_staticmethod": False,
                        }
                    )

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                mod = alias.name
                tgt = _module_to_file(mod, package_roots)
                if tgt:
                    edges.append((rel_file, tgt))
        elif isinstance(node, ast.ImportFrom):
            level = int(node.level or 0)
            mod = node.module
            if level == 0 and mod:
                tgt = _module_to_file(mod, package_roots)
                if tgt:
                    edges.append((rel_file, tgt))
            elif level > 0:
                pkg_dir = PurePosixPath(rel_file).parent
                for _ in range(level - 1):
                    pkg_dir = pkg_dir.parent
                if mod:
                    sub = Path(*mod.split(".")).with_suffix(".py")
                    candidate = _norm_rel((pkg_dir / sub).as_posix())
                    parts = PurePosixPath(candidate).parts
                    if parts and parts[0] in package_roots:
                        edges.append((rel_file, candidate))
                else:
                    for alias in node.names:
                        if alias.name == "*":
                            continue
                        candidate = _norm_rel((pkg_dir / Path(alias.name + ".py")).as_posix())
                        parts = PurePosixPath(candidate).parts
                        if parts and parts[0] in package_roots:
                            edges.append((rel_file, candidate))

    return symbol_rows, meta_rows, edges


def build_repo_intel_from_files(
    files: list[dict],
    repo_name: str,
    package_roots_csv: str,
) -> tuple[list[dict], list[dict], list[dict], list[dict]]:
    roots = {r.strip() for r in package_roots_csv.split(",") if r.strip()}
    sym_map: dict[tuple[str, str, str], dict] = {}
    meta_list: list[dict] = []
    edge_set: set[tuple[str, str, str]] = set()

    for fi in files:
        if fi.get("language") != "py":
            continue
        rel = fi["file_path"]
        try:
            with open(fi["abs_path"], "r", encoding="utf-8", errors="ignore") as fh:
                src = fh.read()
        except OSError:
            continue
        syms, metas, edgs = analyze_python_source(rel, src, roots)
        for s in syms:
            key = (repo_name, s["symbol"], _norm_rel(s["file_path"]))
            sym_map[key] = {
                "repo_name": repo_name,
                "symbol": s["symbol"],
                "file_path": _norm_rel(s["file_path"]),
                "kind": s["kind"],
            }
        for m in metas:
            row = {"repo_name": repo_name, "file_path": _norm_rel(rel)}
            row.update(m)
            meta_list.append(row)
        for src_f, tgt_f in edgs:
            edge_set.add((repo_name, _norm_rel(src_f), _norm_rel(tgt_f)))

    return (
        list(sym_map.values()),
        meta_list,
        [
            {"repo_name": a, "source_file": b, "target_file": c}
            for (a, b, c) in sorted(edge_set)
        ],
    )
