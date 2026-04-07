"""
Mutable IDE runtime state for DeepCode V2 (agent IDE). Updated from EventBus + orchestrator meta.

Event types: agent_file_open, diff_ready, file_updated, agent_status, final_answer,
  subgoal_start, subgoal_complete, state, rag_*, cycle_*, task_*, goal_complete
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from rich.markup import escape as rich_escape

from app.cli.tui.ide_event_buffer import normalize_paths

MAX_OPEN_FILES = 24
MAX_WORKING_SET = 16
MAX_TOUCHED = 32
MAX_TABS = 14


def _stars(score: float, *, max_stars: int = 4) -> str:
    if score <= 0:
        return "—"
    s = float(score)
    if s <= 1.0:
        n = max(1, min(max_stars, int(round(s * max_stars))))
    else:
        n = min(max_stars, max(1, int(round(s))))
    return "★" * n


@dataclass
class IDEState:
    """Agent + UI shared model (terminal IDE)."""

    active_file: str = ""
    open_files: list[str] = field(default_factory=list)
    tabs: list[str] = field(default_factory=list)
    active_tab_index: int = 0

    working_set: list[str] = field(default_factory=list)
    touched_files: list[str] = field(default_factory=list)
    file_scores: dict[str, float] = field(default_factory=dict)

    pending_diff: dict[str, str] = field(default_factory=dict)
    diff_mode: bool = False

    subgoal_timeline: list[dict[str, Any]] = field(default_factory=list)
    agent_status: dict[str, Any] = field(default_factory=dict)

    confidence: float = 0.0
    shared_context: dict[str, Any] = field(default_factory=dict)
    global_insights: list[Any] = field(default_factory=list)
    cross_subgoal_meta: dict[str, Any] = field(default_factory=dict)
    cross_subgoal_summary: str = ""
    early_stop_reason: str = ""
    last_rag_reason: str = ""

    trace_tool_lines: list[str] = field(default_factory=list)
    max_trace_lines: int = 120

    terminal_log: list[str] = field(default_factory=list)
    max_terminal_lines: int = 400

    def log(self, line: str) -> None:
        self.terminal_log.append(line)
        if len(self.terminal_log) > self.max_terminal_lines:
            self.terminal_log = self.terminal_log[-self.max_terminal_lines :]

    def _sync_working_into_open(self) -> None:
        merged = normalize_paths(
            list(self.working_set) + list(self.open_files), MAX_OPEN_FILES
        )
        self.open_files = merged

    def _sync_tabs(self) -> None:
        self._sync_working_into_open()
        tab_order = [t for t in self.tabs if t in self.open_files]
        for p in self.open_files:
            if p not in tab_order:
                tab_order.append(p)
        self.tabs = tab_order[:MAX_TABS]
        if not self.tabs:
            self.active_tab_index = 0
            return
        if self.active_file and self.active_file in self.tabs:
            self.active_tab_index = self.tabs.index(self.active_file)
        else:
            self.active_tab_index = min(self.active_tab_index, len(self.tabs) - 1)
            self.active_tab_index = max(0, self.active_tab_index)
            self.active_file = self.tabs[self.active_tab_index]

    def _normalize_all_paths(self) -> None:
        self.working_set = normalize_paths(self.working_set, MAX_WORKING_SET)
        self.open_files = normalize_paths(self.open_files, MAX_OPEN_FILES)
        self.touched_files = normalize_paths(self.touched_files, MAX_TOUCHED)
        self.tabs = normalize_paths(self.tabs, MAX_TABS)
        self._sync_tabs()

    def next_tab(self) -> None:
        if len(self.tabs) < 2:
            return
        self.active_tab_index = (self.active_tab_index + 1) % len(self.tabs)
        self.active_file = self.tabs[self.active_tab_index]
        self.diff_mode = bool(self.pending_diff.get(self.active_file))

    def apply_event(self, event: dict[str, Any]) -> None:
        t = event.get("type")
        if t == "final_answer":
            return
        if t == "agent_file_open":
            p = str(event.get("path") or "").replace("\\", "/").strip()
            if p:
                self.active_file = p
                if p not in self.open_files:
                    self.open_files.insert(0, p)
                self._normalize_all_paths()
                if p in self.tabs:
                    self.active_tab_index = self.tabs.index(p)
                self.diff_mode = bool(self.pending_diff.get(p))
        elif t == "diff_ready":
            fp = str(event.get("file") or "").replace("\\", "/").strip()
            diff = str(event.get("diff") or "")
            if fp:
                self.pending_diff[fp] = diff
                self.diff_mode = True
                self.active_file = fp
                if fp not in self.open_files:
                    self.open_files.insert(0, fp)
                if fp not in self.touched_files:
                    self.touched_files.insert(0, fp)
                self._normalize_all_paths()
                if fp in self.tabs:
                    self.active_tab_index = self.tabs.index(fp)
        elif t == "file_updated":
            fp = str(event.get("file") or "").replace("\\", "/").strip()
            if fp:
                self.log(f"file_updated: {fp}")
                self.pending_diff.pop(fp, None)
        elif t == "agent_status":
            self.agent_status = {
                "confidence": float(event.get("confidence") or 0),
                "files_read": int(event.get("files_read") or 0),
                "symbols": int(event.get("symbols") or 0),
                "weak_grounding": bool(event.get("weak_grounding")),
                "subgoal_id": event.get("subgoal_id"),
            }
            self.confidence = float(event.get("confidence") or 0)
        elif t == "subgoal_start":
            self.subgoal_timeline.append(
                {
                    "id": event.get("subgoal_id"),
                    "description": (event.get("description") or "")[:220],
                    "status": "running",
                    "priority": event.get("priority"),
                }
            )
            self.subgoal_timeline = self.subgoal_timeline[-24:]
        elif t == "subgoal_complete":
            sid = event.get("subgoal_id")
            for row in reversed(self.subgoal_timeline):
                if row.get("id") == sid and row.get("status") == "running":
                    row["status"] = "done"
                    row["confidence"] = event.get("confidence")
                    row["files_found"] = event.get("files_found")
                    break
        elif t == "state":
            gp = float(event.get("goal_progress") or 0)
            self.confidence = max(self.confidence, gp)
        elif t == "rag_escalation":
            r = event.get("reason") or "unknown"
            self.log(f"RAG escalation: {r}")
        elif t == "rag_triggered":
            r = event.get("reason") or "unknown"
            self.last_rag_reason = str(r)
            self.log(f"RAG triggered: {r}")
            trace = event.get("trace") or []
            if isinstance(trace, list):
                for row in trace[-6:]:
                    self.log(f"  · {row}")
        elif t == "react_session_end":
            esr = str(event.get("early_stop_reason") or "").strip()
            if esr:
                self.early_stop_reason = esr
        elif t == "step" and event.get("phase") == "result":
            tool = event.get("tool") or "?"
            action = event.get("action") or "?"
            st = event.get("status") or "?"
            summ = (event.get("summary") or "")[:80]
            line = f"{tool}.{action} → {st} {summ}".strip()
            self.trace_tool_lines.append(line)
            if len(self.trace_tool_lines) > self.max_trace_lines:
                self.trace_tool_lines = self.trace_tool_lines[-self.max_trace_lines :]

    def apply_meta(self, meta: dict[str, Any]) -> None:
        """After run completes — context continuity + ranking."""
        if not meta:
            return
        self.working_set = list(meta.get("active_working_set") or [])
        self.file_scores = {
            str(k).replace("\\", "/"): float(v)
            for k, v in (meta.get("file_scores") or {}).items()
            if k
        }
        if not self.file_scores:
            for i, p in enumerate(meta.get("react_priority_files") or []):
                self.file_scores[str(p).replace("\\", "/")] = 1.0 - i * 0.1
        self.shared_context = dict(meta.get("shared_context") or {})
        self.global_insights = list(meta.get("global_insights") or [])
        self.cross_subgoal_meta = dict(meta.get("cross_subgoal_meta") or {})
        self.cross_subgoal_summary = str(meta.get("cross_subgoal_summary") or "").strip()
        ac = meta.get("answer_confidence")
        if ac is not None:
            self.confidence = float(ac)
        esr = meta.get("early_stop_reason") or meta.get("react_early_stop_reason")
        if esr:
            self.early_stop_reason = str(esr)
            self.log(f"early_stop_reason: {esr}")
        rr = meta.get("rag_reason")
        if rr:
            self.last_rag_reason = str(rr)
            self.log(f"rag_reason (final): {rr}")
        rdt = meta.get("rag_decision_trace")
        if isinstance(rdt, list) and rdt:
            self.log("rag_decision_trace:")
            for row in rdt[-8:]:
                self.log(f"  · {row}")
        self._normalize_all_paths()

    def cross_subgoal_panel(self) -> str:
        m = self.cross_subgoal_meta or {}
        lines: list[str] = []
        if self.cross_subgoal_summary:
            lines.append(f"[dim]{self.cross_subgoal_summary[:320]}[/dim]\n")
        ag = m.get("agreements") or []
        cf = m.get("conflicts") or []
        gp = m.get("gaps") or []
        if ag:
            lines.append("[bold]Agreements[/bold]")
            for x in ag[:4]:
                lines.append(f" · {str(x)[:100]}")
            lines.append("")
        if cf:
            lines.append("[bold]Conflicts[/bold]")
            for x in cf[:5]:
                lines.append(f" ⚠ {str(x)[:100]}")
            lines.append("")
        if gp:
            lines.append("[bold]Gaps[/bold]")
            for x in gp[:5]:
                lines.append(f" · {str(x)[:100]}")
        return "\n".join(lines).strip() or "[dim](no synthesis meta)[/dim]"

    def prior_findings_text(self) -> str:
        if not self.global_insights:
            return ""
        lines = ["[bold]Previous findings[/bold]"]
        for x in self.global_insights[-6:]:
            lines.append(f" · {str(x)[:120]}")
        return "\n".join(lines)

    def explorer_text(self) -> str:
        lines: list[str] = []
        ws_set = set(self.working_set)
        ranked = sorted(
            self.file_scores.keys(),
            key=lambda p: self.file_scores.get(p, 0),
            reverse=True,
        )[:24]

        pf = self.prior_findings_text()
        if pf:
            lines.append(pf)
            lines.append("")

        cs = self.cross_subgoal_panel()
        if cs and cs != "[dim](no synthesis meta)[/dim]":
            lines.append("[bold]Cross-subgoal[/bold]")
            lines.append(cs)
            lines.append("")

        lines.append("[bold]> Active files (working set)[/bold]")
        if self.working_set:
            for p in self.working_set[:MAX_WORKING_SET]:
                sc = float(self.file_scores.get(p, 0.0))
                lines.append(
                    f" [green]▸[/green] {p}  [dim]score {sc:.2f}[/dim]  {_stars(min(4.0, sc))}"
                )
        else:
            lines.append(" [dim](empty — run exploration)[/dim]")

        lines.append("")
        lines.append("[bold]> Other ranked files[/bold]")
        shown = 0
        for p in ranked:
            if p in ws_set:
                continue
            sc = float(self.file_scores.get(p, 0))
            lines.append(f" · {p}  [dim]score {sc:.2f}[/dim]  {_stars(min(4.0, sc))}")
            shown += 1
            if shown >= 14:
                break
        if shown == 0:
            lines.append(" [dim](none yet)[/dim]")

        if self.touched_files:
            lines.append("")
            lines.append("[bold]Previously touched[/bold]")
            for p in self.touched_files[-10:]:
                lines.append(f" · {p}")
        return "\n".join(lines)

    def status_bar_text(self) -> str:
        a = self.agent_status
        c = float(a.get("confidence") or 0) if a else float(self.confidence or 0)
        if c < 0.5:
            cc = f"[red]conf={c:.2f}[/red]"
        elif c < 0.7:
            cc = f"[yellow]conf={c:.2f}[/yellow]"
        else:
            cc = f"[green]conf={c:.2f}[/green]"
        if not a:
            es = (
                f"  [dim]{rich_escape(self.early_stop_reason)}[/dim]"
                if self.early_stop_reason
                else ""
            )
            return f"{cc}{es}".strip()
        w = (
            "  [red]⚠ low grounding[/red]"
            if a.get("weak_grounding")
            else ""
        )
        es = ""
        if self.early_stop_reason:
            es = f"  [yellow]stop: {rich_escape(self.early_stop_reason)}[/yellow]"
        return (
            f"{cc}  files={int(a.get('files_read') or 0)}  "
            f"symbols={int(a.get('symbols') or 0)}{w}{es}"
        ).strip()

    def timeline_text(self) -> str:
        if not self.subgoal_timeline:
            return "[dim](no subgoals yet)[/dim]"
        lines = []
        for i, row in enumerate(self.subgoal_timeline[-14:], 1):
            st = row.get("status") or "?"
            sym = "✔" if st == "done" else ("⚠" if st == "running" else "○")
            sid = row.get("id") or "?"
            desc = (row.get("description") or "")[:56]
            lines.append(f"[{i}] {sym} [{sid}] {desc}")
        return "\n".join(lines)

    def trace_panel_text(self) -> str:
        parts = ["[bold]Structured trace[/bold]\n"]
        if self.early_stop_reason:
            parts.append(f"early_stop: {self.early_stop_reason}\n")
        if self.last_rag_reason:
            parts.append(f"last_rag: {self.last_rag_reason}\n")
        parts.append("\n[bold]Tool sequence[/bold]\n")
        if not self.trace_tool_lines:
            parts.append("[dim](no steps yet)[/dim]")
        else:
            for ln in self.trace_tool_lines[-40:]:
                parts.append(f" · {ln}")
        return "".join(parts)
