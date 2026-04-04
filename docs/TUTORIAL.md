# DeepCode Agent: Operator Tutorial

Welcome to the DeepCode Agent environment framework! This guide dictates explicitly how developers integrate native execution targets bounding standard limits locally.

## Prerequisite Capabilities

Ensure your machine is explicitly configured executing these dependencies securely:
1. **Python `3.11+`**: Powers the dynamic DAG pipeline constraints Native OS.
2. **Docker / Docker Compose**: Tracks the explicit PostgreSQL database embedding vector structures globally.
3. **Ollama**: Exposes local dependencies binding `llama` models efficiently alongside `nomic-embed-text` variables.

---

## 1. Environment Spin-Up

### Bootstrap the DB Vector Nodes
Leverage `docker-compose` pulling local database execution payloads correctly formatting schemas inherently bypassing local network restrictions:
```bash
docker-compose up -d
```
This isolates PostgreSQL configurations scaling `1536`-dimension embedding architectures directly onto port `5432`.

### Install Virtual Dependencies
To deploy the CLI securely, globally, and correctly avoiding host system environments natively:

```bash
# Strongly Recommended!
pip install pipx
pipx ensurepath

pipx install git+https://github.com/your-org/deepcode-agent.git
```
*⚠ Using `virtualenv` manually is NOT recommended for end CLI operators globally.*

### Apply Table Schemas
You explicitly bind initial `pgvector` parameters via our explicit CLI endpoint:
```bash
deepcode init-db
```

---

## 2. Ingesting Semantic Target Matrices 

Before the DeepCode Agent yields proper retrieval, initialize local constraints explicitly scanning your folder targets extracting memory boundaries:

```bash
# Ingest local repos
deepcode repo ./ --chunk-size 400 --overlap 100
```
*Note: Overlaps correctly string contiguous chunks reducing LLM cutoff vectors limiting hallucination behaviors natively.*

---

## 3. Querying & Architecting

The Agent framework utilizes 4 core execution behaviors matching explicit capabilities explicitly:

### Option A: The Textual UI (TUI) & Modes
Launch the real-time Textual UI natively encapsulating all cognitive states precisely:
```bash
deepcode ui
```

Inside the CLI, swap contexts matching your active requirements explicitly:
1. **/ask** `explain the auth flow`: Standard vector retrieval tracking bounds strictly without sandbox mutations natively.
2. **/plan** `how do we refactor the user routes`: Enables the Intelligence Execution Planner explicitly splitting trajectories cleanly.
3. **/build** `fix the failing tests under auth`: Activates Sandbox write tracks! Checks out branches (`deepcode/patch-...`), applies Diff arrays, executes node tests via `TestRunner` and evaluates loops.

**Health Check Diagnostics:**
```bash
deepcode doctor
```

### Option B: Raw API Engine Mappings
Integrate any frontend dynamically parsing structural JSON. Start the explicit FastAPI Web worker:
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

Use HTTP targets binding query variables securely:
```bash
curl -X POST http://127.0.0.1:8000/ask \
    -H "Content-Type: application/json" \
    -d '{
        "question": "Who recently broke memory processing mapping functions?", 
        "repo_id": "DEEPCODE-AGENT"
    }'
```

### Option C: E2E Python Integrations
Bind tests evaluating internal `Planner -> Tool Executor -> Retriever -> Answer -> Critic` routines natively:
```python
from app.agents.orchestrator import AgentOrchestrator
from app.core.database import async_session_factory
import asyncio, uuid

async def run():
    async with async_session_factory() as db:
        orch = AgentOrchestrator(db=db)
        # Bypasses RAG triggering execution mapping git parameters
        res = await orch.run("Who last changed tests?", str(uuid.uuid4()), "DEEPCODE-AGENT")
        print(res["answer"])

asyncio.run(run())
```

---

## 4. Telemetry and Tuning Operations

DeepCode Agent logs tracking constraints mapping Evaluator Hallucination overrides dynamically! Inspect the traces binding executions precisely via:
```text
tail -f .ai_assistant/logs/trace_<SESSION_UUID>_<TIMESTAMP>.json
```

These matrices isolate the Critic fallbacks measuring precisely if the agent safely bypassed generic RAG limits tracking deterministic Python tools dynamically seamlessly.
