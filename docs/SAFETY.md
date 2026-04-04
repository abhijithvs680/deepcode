# DeepCode Pipeline Safety Constraints (V1.0.0)

DeepCode Agent strictly defines FileWrite extraction layers cleanly enforcing non-destructive principles structurally.

## Branch Generation Constraints (Phase 3)
1. **Strict Feature Branches**: Absolutely no commit arrays execute randomly inside your currently active `HEAD`. The pipeline enforces `git checkout -b deepcode/patch-<uuid>` implicitly preventing active logic overrides destructively. 
2. **Git Auto-Merge Disabled**: Even after successful sandbox cycles gracefully tracking complete validation limits natively, the pipeline intentionally aborts Auto-Merging processes. The user retains complete authority via local `git diff branch` evaluating precisely before explicit `git merge` execution natively. 

## Trajectory & Loop Halts (Phases 2 & 4)
- **Max Iterations Limit Check**: The `ExecutionPlanner` limits explicit workflow tracks logically mapping `MAX_TOTAL_ITERATIONS = 6` strictly inside native Orchestrator logic.
- **Trajectory Fingerprinting**: Generative LLM hallucination execution is fundamentally avoided dynamically tracking Bash failure signatures (e.g. `AssertionError: ...` string footprint) terminating loops early gracefully natively mapping exactly if the same trace repeats `3` times seamlessly preventing unbounded loop hangs!

## Mode Safeties (Validation Check)
- Explicit scope restrictions reject ANY execution path mapping `.write()` mutations naturally outside of `mode == "build"`. Even if the generative LLM maliciously injects `<patch>` schemas natively, the Orhestrator's execution DAG dynamically bypasses evaluation limits silently preserving system environments accurately!
