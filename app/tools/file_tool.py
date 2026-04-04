import os
from typing import Optional, Any
import logging

from app.tools.base_tool import Tool, ToolResult

logger = logging.getLogger(__name__)

class FileTool(Tool):
    """
    Precision file execution channel directly binding line limits and AST validations
    bypassing RAG fuzziness securely targeting token conservation bounds.
    """
    name = "file"
    description = "Retrieves live programmatic code state securely restricted over token caps."
    
    def __init__(self, workspace_root: str):
        self.workspace_root = workspace_root

    async def execute(self, action: str, **kwargs) -> ToolResult:
        """Main routing execution layer safely dispatching boundaries."""
        try:
            if action == "read_file":
                return await self.read_file(**kwargs)
            elif action == "search":
                return await self.search(**kwargs)
            elif action == "get_symbols":
                return await self.get_symbols(**kwargs)
            else:
                return ToolResult(tool=self.name, action=action, success=False, error=f"Unknown Tool Action: '{action}'")
        except Exception as e:
            return ToolResult(tool=self.name, action=action, success=False, error=str(e))

    def _resolve_path(self, path: str) -> str:
        """Ensures secure path resolutions blocking directory traversal."""
        full_path = os.path.abspath(os.path.join(self.workspace_root, path))
        if not full_path.startswith(os.path.abspath(self.workspace_root)):
            raise PermissionError("Directory traversal restricted natively outside workspace constraints.")
        return full_path

    async def read_file(self, path: str, start_line: Optional[int] = None, end_line: Optional[int] = None) -> ToolResult:
        """
        Retrieves line-delimited boundaries avoiding token explosion across bloated schemas.
        """
        try:
            fp = self._resolve_path(path)
            if not os.path.exists(fp):
                return ToolResult(tool=self.name, action="read_file", success=False, error="File does not exist.")
            
            with open(fp, "r", encoding="utf-8", errors="replace") as f:
                lines = f.readlines()
                
            total_lines = len(lines)
            s_idx = max(0, start_line - 1) if start_line else 0
            e_idx = min(total_lines, end_line) if end_line else total_lines
            
            content = "".join(lines[s_idx:e_idx])
            
            return ToolResult(
                tool=self.name,
                action="read_file",
                success=True,
                data={"content": content, "lines": f"{s_idx + 1}-{e_idx}"},
                metadata={"file_size": os.path.getsize(fp), "total_lines": total_lines, "truncated": e_idx - s_idx < total_lines}
            )
        except Exception as e:
            return ToolResult(tool=self.name, action="read_file", success=False, error=str(e))

    async def search(
        self,
        query: str = "",
        queries: Optional[list[str]] = None,
        file_path: Optional[str] = None,
    ) -> ToolResult:
        """
        Basic grep-like search. Pass `queries` for multiple terms (OR, deduped by file+line);
        otherwise a single `query` string is used (legacy).
        """
        import re

        terms: list[str] = []
        if queries:
            for q in queries:
                t = (q or "").strip()
                if len(t) >= 2:
                    terms.append(t)
        raw = (query or "").strip()
        if raw and raw not in terms:
            # avoid duplicating if same as a cleaned term
            if not any(raw.lower() == x.lower() for x in terms):
                terms.append(raw)

        if not terms:
            return ToolResult(
                tool=self.name,
                action="search",
                success=True,
                data={"matches": []},
                metadata={"total_matches_evaluated": 0, "search_terms": []},
            )

        matches: list[dict] = []
        seen_lines: set[tuple[str, int]] = set()
        try:
            target_files: list[str] = []
            if file_path:
                fp = self._resolve_path(file_path)
                if os.path.exists(fp):
                    target_files.append(fp)
            else:
                for root, _, files in os.walk(self.workspace_root):
                    for f in files:
                        if f.endswith((".py", ".md", ".json", ".yml", ".yaml")):
                            target_files.append(os.path.join(root, f))

            patterns = [re.compile(re.escape(t), re.IGNORECASE) for t in terms]

            for path in target_files:
                try:
                    with open(path, "r", encoding="utf-8", errors="replace") as f:
                        lines = f.readlines()
                    for i, line in enumerate(lines):
                        if any(p.search(line) for p in patterns):
                            rel_path = os.path.relpath(path, self.workspace_root)
                            key = (rel_path, i + 1)
                            if key in seen_lines:
                                continue
                            seen_lines.add(key)
                            matches.append(
                                {"file": rel_path, "line_no": i + 1, "content": line.strip()}
                            )
                            if len(matches) >= 50:
                                break
                except OSError:
                    continue
                if len(matches) >= 50:
                    break

            return ToolResult(
                tool=self.name,
                action="search",
                success=True,
                data={"matches": matches},
                metadata={
                    "total_matches_evaluated": len(matches),
                    "search_terms": terms[:12],
                },
            )
        except Exception as e:
            return ToolResult(tool=self.name, action="search", success=False, error=str(e))

    async def get_symbols(self, file_path: str) -> ToolResult:
        """
        Extracts AST classes/methods natively mapping symbol ranges directly inside tool scopes.
        """
        try:
            fp = self._resolve_path(file_path)
            if not os.path.exists(fp) or not fp.endswith(".py"):
                return ToolResult(tool=self.name, action="get_symbols", success=False, error="Invalid filepath or non-Python target constraints.")
                
            with open(fp, "r", encoding="utf-8", errors="replace") as f:
                code = f.read()
                
            from app.ingestion.ast_analysis import extract_python_symbols
            symbols = extract_python_symbols(code)
            
            structured = [{"name": s.name, "type": s.kind, "line_start": s.signature_line, "line_end": s.body_end_line} for s in symbols]
            
            return ToolResult(
                tool=self.name,
                action="get_symbols",
                success=True,
                data={"symbols": structured},
                metadata={"symbol_count": len(structured)}
            )
        except Exception as e:
            return ToolResult(tool=self.name, action="get_symbols", success=False, error=str(e))
