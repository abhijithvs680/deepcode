# DeepCode Agent Architecture (V1.0.0)

DeepCode Agent is a stateful, tool-aware, execution-capable AI coding system designed for controlled code intelligence and autonomous execution loop safety.

## 1. Core Component Layering

The system isolates layers explicitly preventing orchestration loops from mutating presentation boundaries and vice versa.

### Execution Engine (Orchestrator Control Plane)
- **`app/agents/orchestrator.py`**: The central DAG routing execution across sub-agents dynamically tracking state sequences.

### Agents
- **Planner Agent (`planner_agent.py`)**: Focuses on sub-task extraction and tool heuristic routing mapped accurately against Query logic domains.
- **Execution Planner (`execution_planner.py`)**: Breaks raw complex bounds into step-sized executions (Array `1` -> `5`) dynamically.
- **Answer Agent (`answer_agent.py`)**: The primary payload constructor. Combines RAG bounds, Tools, Memory, and System Prompts converting bounds into `markdown` or specific `<patch>` arrays!
- **Critic Agent (`critic_agent.py` & `critic_executor.py`)**: Heuristic Validation. Extracts logical feedback tracking error signatures natively preventing infinite loop hallucinations automatically!

### Retrieval Engine (Context)
- **ParallelRetrieval**: Queries internal PGVector memory arrays bounding explicit Focus Entities (`symbol`, `import`, `filepath`) dynamically against Hybrid BM25+Embeddings natively mapping scores (`0.45*semantic`, `0.2*keyword`, etc.).
- **Memory Intelligence (`intel_store.py`)**: Pulls persistent past successes, user bugs, and historical query vectors explicitly injecting summaries directly back into the Planner loops safely tracking past learning inherently!

### Presentation (Textual TUI)
- **Streaming Pipeline**: Bypasses heavy local context strings by allowing Ollama and LLM endpoints to yield NDJSON directly onto explicitly updating Static Widgets dynamically mapping chunks at `0.05` throttle speeds explicitly preventing execution UI freezing natively!

## 2. DAG Loop Triggers

DeepCode executes a strict Pipeline bounding state logic:
1. Classification & Planning extraction
2. Sequential Steps Generation
3. `while(steps < max_steps)`
   - Planner Route
   - (Tool Execution `||` Semantic RAG bounds)
   - Answer Payload Streaming
   - Sandbox Diff Validate -> Git Backup -> Apply
   - Background Diagnostics Tests (Python/JS tests)
   - Critic Halt Break / Retry Break
4. Long Term Memory Flush  
