"""
Lightweight hierarchical decomposition: one ReAct loop per subgoal, then merged context.
"""
from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any

from app.agents.react_grounding import (
    compute_answer_confidence,
    has_strong_merged_lists,
    keyword_relevance,
)
from app.core.config import get_settings


@dataclass
class Subgoal:
    id: str
    description: str
    priority: int = 1


@dataclass
class HierarchicalPlan:
    goal: str
    subgoals: list[Subgoal] = field(default_factory=list)


def decompose_goal(query: str) -> HierarchicalPlan:
    q = (query or "").strip()
    low = q.lower()

    if "explain" in low or "architecture" in low or "how does" in low:
        return HierarchicalPlan(
            goal=q,
            subgoals=[
                Subgoal("entry", "Find entry points and main modules for: " + q[:120], 1),
                Subgoal("flow", "Trace main control or data flow for: " + q[:120], 2),
                Subgoal("deps", "Identify key dependencies and integrations for: " + q[:120], 3),
            ],
        )

    if "bug" in low or "error" in low or "fail" in low or "crash" in low:
        return HierarchicalPlan(
            goal=q,
            subgoals=[
                Subgoal("locate", "Locate error or failure source for: " + q[:120], 1),
                Subgoal("trace", "Trace execution path related to: " + q[:120], 2),
                Subgoal("cause", "Identify likely root cause for: " + q[:120], 3),
            ],
        )

    return HierarchicalPlan(
        goal=q,
        subgoals=[Subgoal("general", q[:400] or q, 1)],
    )


STRUCTURAL_SUBGOAL_IDS = frozenset(
    {"entry", "flow", "deps", "locate", "trace", "cause", "resolve_conflicts"}
)


def distinct_top_level_roots(paths: list[str]) -> int:
    roots: set[str] = set()
    for p in paths:
        parts = str(p).replace("\\", "/").split("/")
        if parts and parts[0].strip():
            roots.add(parts[0].strip())
    return len(roots)


def select_carryover_tool_results(
    parent_state: dict[str, Any],
    *,
    k: int = 3,
) -> list[dict[str, Any]]:
    raw = [
        dict(tr)
        for tr in (parent_state.get("tool_results") or [])
        if isinstance(tr, dict)
    ]
    pri = {
        "read_file": 4,
        "get_file_dependencies": 3,
        "get_symbol_dependencies": 3,
        "find_symbol": 2,
        "get_symbols": 2,
        "search": 1,
    }
    raw.sort(
        key=lambda t: -pri.get((t.get("action") or "").strip(), 0),
    )
    return raw[:k]


def select_carryover_symbols(
    parent_state: dict[str, Any],
    *,
    k: int = 10,
) -> list[str]:
    raw = parent_state.get("symbols_found") or []
    out: list[str] = []
    seen: set[str] = set()
    for s in raw:
        t = str(s).strip()
        if t and t not in seen:
            seen.add(t)
            out.append(t)
        if len(out) >= k:
            break
    return out


def select_carryover_insights(parent_state: dict[str, Any], *, k: int = 4) -> list[Any]:
    gi = list(parent_state.get("global_insights") or [])
    return gi[-k:] if gi else []


def select_carryover_file_scores(
    parent_state: dict[str, Any],
    *,
    k: int = 5,
) -> dict[str, float]:
    fs = parent_state.get("file_scores") or {}
    if not isinstance(fs, dict) or not fs:
        return {}
    items: list[tuple[str, float]] = []
    for p, v in fs.items():
        try:
            items.append((str(p).replace("\\", "/"), float(v)))
        except (TypeError, ValueError):
            continue
    items.sort(key=lambda kv: -kv[1])
    return dict(items[:k])


# Parent keys carried into each subgoal (identity + read-only context + tooling).
_SUBSTATE_KEYS = frozenset(
    {
        "query",
        "repo_id",
        "workspace_root",
        "trace",
        "embed_cache",
        "session_id",
        "mode",
        "write_enabled",
        "user_context",
        "classification",
        "adaptive_config",
        "memories",
        "base_query_embed",
        "sub_tasks",
        "stream_callback",
        "react_event_callback",
        "confirm_shell_callback",
        "long_run_mode",
        "react_cancel_event",
        "fallbacks_local",
        "dry_run",
        "auto_apply",
        "disable_aggressive_overrides",
        "max_retries",
        "hierarchical_plan",
        "execution_steps",
        "current_step_index",
        "step_history",
        "failure_signatures",
        "critic_retry_count",
        "latency_ms",
        "learned_file_bias",
        "prior_react_signal_paths",
    }
)


def create_substate(parent_state: dict[str, Any], subgoal: Subgoal) -> dict[str, Any]:
    """
    Partial isolation: no full parent copy — avoids router/tool bias leaking across subgoals.
    Keeps global read-only context (memories, classification); resets exploration.
    """
    s: dict[str, Any] = {k: parent_state[k] for k in _SUBSTATE_KEYS if k in parent_state}
    s.setdefault("memories", parent_state.get("memories") or [])
    s.setdefault("fallbacks_local", list(parent_state.get("fallbacks_local") or []))

    s["react_steps"] = []
    s["react_observations"] = []
    s["react_summary"] = ""
    s["files_read"] = []
    s["symbols_found"] = []
    s["goal_progress"] = 0.0
    s["react_useful_steps"] = 0
    s["react_failed_steps"] = 0
    s["react_force_rag"] = False
    s["tool_results"] = []
    s["file_scores"] = {}
    s["react_priority_files"] = []
    s["ranked_ctx"] = []
    sc = parent_state.get("shared_context") or {}
    s["hint_files"] = list(sc.get("top_files") or [])
    s["hint_symbols"] = list(sc.get("top_symbols") or [])
    s["prior_insights"] = list(parent_state.get("global_insights") or [])
    if parent_state.get("learned_file_bias"):
        s["learned_file_bias"] = list(parent_state["learned_file_bias"])
    s["carryover_tool_results"] = select_carryover_tool_results(
        parent_state, k=3
    )
    s["carryover_file_scores"] = select_carryover_file_scores(
        parent_state, k=5
    )
    s["carryover_symbols"] = select_carryover_symbols(parent_state, k=10)
    s["carryover_insights"] = select_carryover_insights(parent_state, k=4)
    cap_aws = int(getattr(get_settings(), "ACTIVE_WORKING_SET_MAX", 8) or 8)
    cap_aws = max(4, min(cap_aws, 16))
    s["active_working_set"] = list(parent_state.get("active_working_set") or [])[
        :cap_aws
    ]
    s["synthesis_hints"] = list(parent_state.get("next_subgoal_bias") or [])[:10]
    s["subgoal_id"] = subgoal.id
    s["subgoal_description"] = subgoal.description
    s["current_step"] = subgoal.description
    s["react_goal"] = subgoal.description

    # Neutralize router hint so subgoals are not locked to parent's tool choice.
    s["selected_tool"] = None
    s["router_confidence"] = 0.0
    s["router_source"] = "subgoal_neutral"
    s["tool_routing"] = {
        "selected_tool": None,
        "routed_candidate": None,
        "confidence": 0.0,
        "source": "subgoal_neutral",
        "bound": False,
    }

    return s


def estimate_dynamic_score(
    sg: Subgoal, subgoal_runs: list[tuple[Subgoal, dict[str, Any]]]
) -> float:
    """Higher = run sooner among remaining subgoals (adaptive ordering)."""
    files = merged_files_so_far(subgoal_runs)
    syms = merged_symbols_so_far(subgoal_runs)
    sid = sg.id.lower()
    desc = sg.description.lower()

    if sid == "entry" or "entry" in desc or sid == "locate" or "locate" in desc:
        return 0.9 if len(files) < 2 else 0.22
    if sid == "flow" or "flow" in desc or sid == "trace" or "trace" in desc:
        return 0.72 if len(files) >= 1 else 0.48
    if "dep" in sid or "dependency" in desc or sid == "cause" or "cause" in desc:
        return 0.82 if len(syms) < 2 else 0.38
    return 0.5 + (4 - min(sg.priority, 4)) * 0.08


def collect_read_file_snippets(
    subgoal_runs: list[tuple[Subgoal, dict[str, Any]]],
    *,
    max_chars: int = 900,
) -> dict[str, str]:
    """Best-effort path → content excerpt from read_file steps (zip with react_steps)."""
    snippets: dict[str, str] = {}
    for _sg, st in subgoal_runs:
        steps = list(st.get("react_steps") or [])
        tools = list(st.get("tool_results") or [])
        for step, tr in zip(steps, tools):
            if not isinstance(tr, dict) or tr.get("action") != "read_file":
                continue
            if not isinstance(step, dict):
                continue
            args = step.get("args") if isinstance(step.get("args"), dict) else {}
            path = args.get("path") or args.get("file_path")
            if not path:
                continue
            fn = str(path).replace("\\", "/").strip()
            data = tr.get("data") or {}
            if not isinstance(data, dict):
                continue
            content = str(data.get("content") or "")[:max_chars]
            if not content.strip():
                continue
            prev = snippets.get(fn, "")
            if len(content) > len(prev):
                snippets[fn] = content
    return snippets


def merge_weighted_files(
    subgoal_runs: list[tuple[Subgoal, dict[str, Any]]],
    *,
    top_n: int = 5,
    query: str | None = None,
    file_snippets: dict[str, str] | None = None,
) -> list[str]:
    scores: dict[str, float] = defaultdict(float)
    file_sources: dict[str, set[str]] = defaultdict(set)
    for idx, (sg, st) in enumerate(subgoal_runs):
        gp = float(st.get("goal_progress") or 0)
        sub_conf = compute_answer_confidence(st)
        conf_w = max(0.2, float(sub_conf))
        recency = 1.0 + 0.18 * float(idx)
        sid = str(sg.id)
        for f in st.get("files_read") or []:
            fn = str(f).replace("\\", "/").strip()
            if fn:
                file_sources[fn].add(sid)
                scores[fn] += (1.0 + gp) * recency * conf_w
    for fn, srcs in file_sources.items():
        c = len(srcs)
        if c >= 2:
            scores[fn] += 0.5 * float(c - 1)
    boost = structural_file_boost(subgoal_runs)
    for fn in list(scores.keys()):
        b = float(boost.get(fn, 0.0))
        if b > 0:
            scores[fn] *= 1.0 + min(b, 4.0) * 0.06
    q = (query or "").strip()
    snip = file_snippets or {}
    if q:
        for fn in list(scores.keys()):
            base = fn.replace("/", " ")
            leaf = fn.split("/")[-1]
            path_score = keyword_relevance(q, base, leaf)
            ex = snip.get(fn) or ""
            content_score = keyword_relevance(q, ex[:2000]) if ex else path_score
            blend = 0.4 * path_score + 0.6 * content_score
            scores[fn] *= 0.35 + 0.65 * blend
    if not scores:
        return []
    return sorted(scores.keys(), key=lambda x: scores[x], reverse=True)[:top_n]


def merge_weighted_symbols(
    subgoal_runs: list[tuple[Subgoal, dict[str, Any]]],
    *,
    top_n: int = 16,
    query: str | None = None,
) -> list[str]:
    scores: dict[str, float] = defaultdict(float)
    sym_sources: dict[str, set[str]] = defaultdict(set)
    for idx, (sg, st) in enumerate(subgoal_runs):
        gp = float(st.get("goal_progress") or 0)
        sub_conf = compute_answer_confidence(st)
        conf_w = max(0.2, float(sub_conf))
        recency = 1.0 + 0.14 * float(idx)
        sid = str(sg.id)
        for sy in st.get("symbols_found") or []:
            if sy:
                k = str(sy)
                sym_sources[k].add(sid)
                scores[k] += (1.0 + 0.5 * gp) * recency * conf_w
    for sym, srcs in sym_sources.items():
        c = len(srcs)
        if c >= 2:
            scores[sym] += 0.35 * float(c - 1)
    q = (query or "").strip()
    if q:
        for sym in list(scores.keys()):
            rel_kw = keyword_relevance(q, sym)
            scores[sym] *= 0.4 + 0.6 * rel_kw
    if not scores:
        return []
    return sorted(scores.keys(), key=lambda x: scores[x], reverse=True)[:top_n]


def _merge_files_by_frequency(file_lists: list[list[str]]) -> list[str]:
    flat: list[str] = []
    for lst in file_lists:
        flat.extend(lst)
    if not flat:
        return []
    cnt = Counter(flat)
    unique: list[str] = []
    seen: set[str] = set()
    for f in flat:
        n = f.replace("\\", "/")
        if n not in seen:
            seen.add(n)
            unique.append(n)
    unique.sort(key=lambda x: (-cnt[x], x))
    return unique


def _dedupe_symbols(sym_lists: list[list[str]]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for lst in sym_lists:
        for sym in lst:
            if sym and sym not in seen:
                seen.add(sym)
                out.append(sym)
    return out


def extract_key_findings(sub_state: dict[str, Any]) -> dict[str, Any]:
    """Compact cross-subgoal handoff (files/symbols + last observation)."""
    obs = list(sub_state.get("react_observations") or [])
    tail = str(obs[-1]) if obs else ""
    return {
        "files": [str(f).replace("\\", "/") for f in (sub_state.get("files_read") or [])[:5]],
        "symbols": [str(s) for s in (sub_state.get("symbols_found") or [])[:6]],
        "last_observation": tail[:420],
        "useful_steps": int(sub_state.get("react_useful_steps") or 0),
    }


def avg_exploration_confidence(
    subgoal_runs: list[tuple[Subgoal, dict[str, Any]]],
) -> float:
    if not subgoal_runs:
        return 0.0
    vals = [compute_answer_confidence(st) for _, st in subgoal_runs]
    return round(sum(vals) / max(1, len(vals)), 3)


def merge_react_subgoal_results(
    parent_state: dict[str, Any],
    *,
    prior_tool_results: list[dict[str, Any]],
    subgoal_runs: list[tuple[Subgoal, dict[str, Any]]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """
    Merge child states into parent_state keys used by RAG / AnswerAgent.
    Weighted file/symbol lists; returns subgoal_traces with per-subgoal confidence.
    """
    acc_tool: list[dict[str, Any]] = []
    subgoal_traces: list[dict[str, Any]] = []
    merged_scores: dict[str, float] = {}
    force_rag = False
    total_useful = 0
    total_failed = 0
    flat_steps: list[dict[str, Any]] = []
    flat_obs: list[str] = []

    for sg, sub in subgoal_runs:
        if bool(sub.get("react_force_rag")):
            force_rag = True
        total_useful += int(sub.get("react_useful_steps") or 0)
        total_failed += int(sub.get("react_failed_steps") or 0)
        sub_conf = compute_answer_confidence(sub)
        sub["subgoal_confidence"] = sub_conf
        for tr in sub.get("tool_results") or []:
            if isinstance(tr, dict):
                t2 = dict(tr)
                t2["_subgoal_id"] = sg.id
                acc_tool.append(t2)
        for k, v in (sub.get("file_scores") or {}).items():
            try:
                kn = str(k).replace("\\", "/")
                merged_scores[kn] = max(merged_scores.get(kn, 0.0), float(v))
            except (TypeError, ValueError):
                pass
        steps = list(sub.get("react_steps") or [])
        obs = list(sub.get("react_observations") or [])
        subgoal_traces.append(
            {
                "id": sg.id,
                "description": sg.description,
                "priority": sg.priority,
                "confidence": sub_conf,
                "steps": steps[-12:],
                "observations": obs[-10:],
                "useful_steps": int(sub.get("react_useful_steps") or 0),
                "failed_steps": int(sub.get("react_failed_steps") or 0),
                "goal_progress": float(sub.get("goal_progress") or 0),
            }
        )
        for st in steps:
            if isinstance(st, dict):
                fs = dict(st)
                fs["subgoal_id"] = sg.id
                flat_steps.append(fs)
        flat_obs.extend(obs)

    if total_useful > 0:
        force_rag = False

    qtxt = (parent_state.get("query") or "").strip()
    file_snips = collect_read_file_snippets(subgoal_runs)
    parent_state["files_read"] = merge_weighted_files(
        subgoal_runs,
        top_n=5,
        query=qtxt,
        file_snippets=file_snips,
    )
    parent_state["symbols_found"] = merge_weighted_symbols(
        subgoal_runs, top_n=16, query=qtxt
    )
    parent_state["tool_results"] = list(prior_tool_results) + acc_tool
    wide_f = merge_weighted_files(
        subgoal_runs, top_n=32, query=qtxt, file_snippets=file_snips
    )
    wide_s = merge_weighted_symbols(subgoal_runs, top_n=32, query=qtxt)
    if has_strong_merged_lists(wide_f, wide_s):
        force_rag = False
    parent_state["react_force_rag"] = force_rag
    parent_state["react_useful_steps"] = total_useful
    parent_state["react_failed_steps"] = total_failed
    parent_state["goal_progress"] = max(
        (float(s.get("goal_progress") or 0) for _, s in subgoal_runs),
        default=0.0,
    )
    parent_state["file_scores"] = {
        k: round(v, 4) for k, v in sorted(merged_scores.items(), key=lambda kv: -kv[1])
    }
    top_files = sorted(merged_scores.keys(), key=lambda p: merged_scores[p], reverse=True)[:2]
    parent_state["react_priority_files"] = top_files
    parent_state["react_steps"] = flat_steps[-24:]
    parent_state["react_observations"] = flat_obs[-32:]
    parent_state["subgoal_traces"] = subgoal_traces
    parent_state["subgoal_confidence"] = {
        t["id"]: float(t["confidence"]) for t in subgoal_traces
    }
    q_ws = (parent_state.get("query") or parent_state.get("current_step") or "").strip()
    parent_state["active_working_set"] = build_active_working_set(
        subgoal_runs, query=q_ws
    )
    prior_union: set[str] = set()
    for p in parent_state.get("prior_react_signal_paths") or []:
        kn = str(p).replace("\\", "/").strip().lstrip("./")
        if kn:
            prior_union.add(kn)
    for _sg, sub in subgoal_runs:
        for p in sub.get("prior_react_signal_paths") or []:
            kn = str(p).replace("\\", "/").strip().lstrip("./")
            if kn:
                prior_union.add(kn)
    parent_state["prior_react_signal_paths"] = list(prior_union)[:120]
    if subgoal_runs:
        _last_sub = subgoal_runs[-1][1]
        r = str(_last_sub.get("react_early_stop_reason") or "").strip()
        if r:
            parent_state["react_early_stop_reason"] = r
    return parent_state, subgoal_traces


def _symbol_related_paths_from_tool_results(
    subgoal_runs: list[tuple[Subgoal, dict[str, Any]]],
) -> set[str]:
    """Paths tied to symbol tools (definitions / dependency bundles), not filename substring guesses."""
    out: set[str] = set()
    for _sg, st in subgoal_runs:
        for tr in st.get("tool_results") or []:
            if not isinstance(tr, dict) or not tr.get("success"):
                continue
            act = tr.get("action")
            data = tr.get("data") or {}
            if not isinstance(data, dict):
                continue
            if act == "find_symbol":
                for m in data.get("matches") or []:
                    if isinstance(m, dict):
                        fp = m.get("file")
                        if isinstance(fp, str) and fp.strip():
                            out.add(fp.replace("\\", "/"))
            elif act == "get_symbol_dependencies":
                pf = data.get("primary_file")
                if isinstance(pf, str) and pf.strip():
                    out.add(pf.replace("\\", "/"))
                for m in data.get("matches") or []:
                    if isinstance(m, dict):
                        fp = m.get("file")
                        if isinstance(fp, str) and fp.strip():
                            out.add(fp.replace("\\", "/"))
                g = data.get("graph") or {}
                if isinstance(g, dict):
                    for p in g:
                        if isinstance(p, str) and p.strip():
                            out.add(p.replace("\\", "/"))
                    for bundle in g.values():
                        if not isinstance(bundle, dict):
                            continue
                        for k in ("imports", "imported_by"):
                            for p in bundle.get(k) or []:
                                if isinstance(p, str) and p.strip():
                                    out.add(p.replace("\\", "/"))
    return out


def _norm_ws_path_key(p: str) -> str:
    return str(p).replace("\\", "/").strip().lstrip("./").lower()


def build_active_working_set(
    subgoal_runs: list[tuple[Subgoal, dict[str, Any]]],
    *,
    max_files: int | None = None,
    query: str = "",
) -> list[str]:
    """Recent subgoal files first, then merged frequency list + dep neighbors (capped)."""
    stg = get_settings()
    cap = max_files if max_files is not None else int(
        getattr(stg, "ACTIVE_WORKING_SET_MAX", 8) or 8
    )
    cap = max(4, min(cap, 16))
    recent: list[str] = []
    if subgoal_runs:
        _last_sg, last_st = subgoal_runs[-1]
        recent = [
            str(f).replace("\\", "/")
            for f in (last_st.get("files_read") or [])[:5]
            if f
        ]
    files = merged_files_so_far(subgoal_runs)
    neigh: set[str] = set()
    for _sg, st in subgoal_runs:
        for tr in st.get("tool_results") or []:
            if not isinstance(tr, dict) or tr.get("action") != "get_file_dependencies":
                continue
            data = tr.get("data") or {}
            if not isinstance(data, dict):
                continue
            for k in ("imports", "imported_by"):
                for p in data.get(k) or []:
                    if isinstance(p, str) and p.strip():
                        neigh.add(p.replace("\\", "/"))
    ordered: list[str] = []
    seen: set[str] = set()
    for p in recent + files:
        if not p or p in seen:
            continue
        seen.add(p)
        ordered.append(p)
    head_n = max(3, cap // 2)
    out: list[str] = list(ordered[:head_n])
    seen_out: set[str] = set(out)
    for p in sorted(neigh):
        if len(out) >= cap:
            break
        if p not in seen_out:
            seen_out.add(p)
            out.append(p)
    out = out[:cap]
    qtxt = (query or "").strip()
    sym_tokens = [
        str(s).strip().lower()
        for s in merged_symbols_so_far(subgoal_runs)
        if isinstance(s, str) and len(str(s).strip()) > 1
    ]
    sym_path_keys = {
        _norm_ws_path_key(x) for x in _symbol_related_paths_from_tool_results(subgoal_runs)
    }
    if qtxt or sym_tokens or sym_path_keys:
        scored: list[tuple[float, int, str]] = []
        for idx, p in enumerate(out):
            base = p.split("/")[-1] if p else ""
            rel = (
                keyword_relevance(qtxt, p, base.replace("_", " ")) if qtxt else 0.0
            )
            sym_boost = 0.0
            if _norm_ws_path_key(p) in sym_path_keys:
                sym_boost = max(sym_boost, 2.5)
            pl = p.lower()
            for sym in sym_tokens:
                if len(sym) < 3:
                    continue
                if sym in pl or sym.replace("_", "") in pl.replace("_", ""):
                    sym_boost = max(sym_boost, 2.0)
                    break
            scored.append((rel + sym_boost, idx, p))
        scored.sort(key=lambda t: (-t[0], t[1]))
        out = [t[2] for t in scored][:cap]
    return out


def dependency_graph_edge_total(
    subgoal_runs: list[tuple[Subgoal, dict[str, Any]]],
) -> int:
    """Total import edges observed (proxy for dependency traversal depth)."""
    total = 0
    for _sg, st in subgoal_runs:
        for tr in st.get("tool_results") or []:
            if not isinstance(tr, dict) or tr.get("action") != "get_file_dependencies":
                continue
            data = tr.get("data") or {}
            if not isinstance(data, dict):
                continue
            total += len(data.get("imports") or []) + len(data.get("imported_by") or [])
    return total


def structural_file_boost(
    subgoal_runs: list[tuple[Subgoal, dict[str, Any]]],
) -> dict[str, float]:
    """
    Dependency degree + search hit frequency as a soft merge prior
    (reduces domination by irrelevant high-frequency mentions).
    """
    dep_degree: dict[str, float] = defaultdict(float)
    edge_cnt: dict[str, int] = defaultdict(int)
    ref_hits: dict[str, float] = defaultdict(float)
    for _sg, st in subgoal_runs:
        for tr in st.get("tool_results") or []:
            if not isinstance(tr, dict):
                continue
            data = tr.get("data") or {}
            if not isinstance(data, dict):
                continue
            act = tr.get("action")
            if act == "get_file_dependencies":
                fn = str(data.get("file") or "").replace("\\", "/").strip()
                imp = data.get("imports") or []
                ib = data.get("imported_by") or []
                deg = float(len(imp) + len(ib))
                if fn:
                    dep_degree[fn] += 0.5 * deg
                    edge_cnt[fn] += int(len(imp) + len(ib))
                for p in imp:
                    if isinstance(p, str) and p.strip():
                        pn = p.replace("\\", "/")
                        dep_degree[pn] += 0.35
                        edge_cnt[pn] += 1
                for p in ib:
                    if isinstance(p, str) and p.strip() and fn:
                        dep_degree[fn] += 0.3
                        edge_cnt[fn] += 1
            elif act == "search":
                for m in data.get("matches") or []:
                    if not isinstance(m, dict):
                        continue
                    fp = m.get("file")
                    if isinstance(fp, str) and fp.strip():
                        ref_hits[fp.replace("\\", "/")] += 0.3
    out: dict[str, float] = defaultdict(float)
    for p, v in dep_degree.items():
        ec = int(edge_cnt.get(p, 0))
        out[p] += v + 0.35 * math.log1p(min(ec, 120))
    for p, v in ref_hits.items():
        out[p] += v
    return dict(out)


def partial_subgoal_traces_for_synthesis(
    subgoal_runs: list[tuple[Subgoal, dict[str, Any]]],
) -> list[dict[str, Any]]:
    """Lightweight traces for mid-hierarchy synthesis (same shape as post-merge)."""
    traces: list[dict[str, Any]] = []
    for sg, st in subgoal_runs:
        obs = list(st.get("react_observations") or [])
        traces.append(
            {
                "id": sg.id,
                "description": sg.description,
                "priority": sg.priority,
                "confidence": compute_answer_confidence(st),
                "steps": list(st.get("react_steps") or [])[-12:],
                "observations": obs[-10:],
                "useful_steps": int(st.get("react_useful_steps") or 0),
                "failed_steps": int(st.get("react_failed_steps") or 0),
                "goal_progress": float(st.get("goal_progress") or 0),
            }
        )
    return traces


def subgoal_synthesis_bias_score(subgoal: Subgoal, hints: list[str]) -> float:
    if not hints:
        return 0.0
    blob = f"{subgoal.id} {subgoal.description}".lower()
    score = 0.0
    for h in hints[:14]:
        low = str(h).lower()
        for token in low.replace(",", " ").split():
            if len(token) > 3 and token in blob:
                score += 0.11
        if len(low) > 10 and low[:72] in blob:
            score += 0.18
    return min(score, 1.6)


def merged_files_so_far(subgoal_runs: list[tuple[Subgoal, dict[str, Any]]]) -> list[str]:
    return _merge_files_by_frequency(
        [list(s.get("files_read") or []) for _, s in subgoal_runs]
    )


def merged_symbols_so_far(subgoal_runs: list[tuple[Subgoal, dict[str, Any]]]) -> list[str]:
    return _dedupe_symbols(
        [list(s.get("symbols_found") or []) for _, s in subgoal_runs]
    )


def total_useful_so_far(subgoal_runs: list[tuple[Subgoal, dict[str, Any]]]) -> int:
    return sum(int(s.get("react_useful_steps") or 0) for _, s in subgoal_runs)


def plan_needs_dependency_subgoal(plan_subgoals: list[Subgoal]) -> bool:
    return any(sg.id == "deps" for sg in plan_subgoals)


def has_dependency_coverage(
    subgoal_runs: list[tuple[Subgoal, dict[str, Any]]],
    completed_ids: list[str],
    *,
    needs_dep: bool,
) -> bool:
    if not needs_dep:
        return True
    if "deps" in completed_ids:
        return True
    for _sg, st in subgoal_runs:
        for tr in st.get("tool_results") or []:
            if not isinstance(tr, dict):
                continue
            if tr.get("action") in (
                "get_file_dependencies",
                "get_symbol_dependencies",
            ):
                return True
    return False


def subgoal_success(sub_state: dict[str, Any]) -> bool:
    if sub_state.get("files_read") or sub_state.get("symbols_found"):
        return True
    if int(sub_state.get("react_useful_steps") or 0) >= 1:
        return True
    if float(sub_state.get("goal_progress") or 0) >= 0.2:
        return True
    return False


def should_stop_subgoal_loop(
    *,
    plan_size: int,
    completed_ids: list[str],
    subgoal_runs: list[tuple[Subgoal, dict[str, Any]]],
    strong_merged_predicate,
    plan_subgoals: list[Subgoal] | None = None,
) -> bool:
    """
    Stop early only when enough subgoals succeeded AND merged context is strong,
    or exploration budget hit with at least one success.
    """
    snap_files = merged_files_so_far(subgoal_runs)
    snap_syms = merged_symbols_so_far(subgoal_runs)
    useful = total_useful_so_far(subgoal_runs)
    avg_conf = avg_exploration_confidence(subgoal_runs)
    psg = plan_subgoals or []
    needs_dep = plan_needs_dependency_subgoal(psg)
    stg = get_settings()
    min_edges = int(
        getattr(stg, "HIERARCHICAL_MIN_DEP_EDGES_FOR_EARLY_STOP", 4) or 4
    )
    dep_edges = dependency_graph_edge_total(subgoal_runs)

    if plan_size <= 1:
        return False

    if avg_conf < 0.4 and len(completed_ids) < plan_size:
        return False

    if len(completed_ids) >= 2 and strong_merged_predicate(snap_files, snap_syms):
        if avg_conf < 0.6:
            return False
        if plan_size >= 3:
            if not STRUCTURAL_SUBGOAL_IDS.intersection(set(completed_ids)):
                return False
            roots = distinct_top_level_roots(snap_files)
            if roots < 2 and len(snap_files) < 3:
                return False
            if not has_dependency_coverage(
                subgoal_runs, completed_ids, needs_dep=needs_dep
            ):
                return False
            if needs_dep and dep_edges < min_edges:
                return False
        return True

    if (
        should_early_stop_merged(snap_files, snap_syms, useful)
        and len(completed_ids) >= 1
    ):
        if plan_size >= 3 and not STRUCTURAL_SUBGOAL_IDS.intersection(
            set(completed_ids)
        ):
            return False
        if not has_dependency_coverage(
            subgoal_runs, completed_ids, needs_dep=needs_dep
        ):
            return False
        if needs_dep and dep_edges < min_edges:
            return False
        return avg_conf >= 0.5

    return False


def should_early_stop_merged(
    files_read: list[str],
    symbols_found: list[str],
    total_useful_steps: int,
) -> bool:
    if len(files_read) >= 4:
        return True
    if len(symbols_found) >= 2 and len(files_read) >= 1:
        return True
    if total_useful_steps >= 5:
        return True
    return False
