# DeepCode Agent (V1.0.0 Intelligent Execution System)

An advanced, privacy-first **Local AI Coding Assistant** engineered natively onto `pgvector` dependencies executing purely deterministic **Multi-Agent Orchestrator bindings**.

DeepCode Agent doesn't just evaluate spatial token boundaries via naive RAG lookups; it actively manages dynamic execution tools mapping standard explicit `Git`, `Log`, and `File` dependencies bridging complex code trajectories securely.

---

## 🚀 Engine Architecture (V1.0.0)

The framework relies upon a modular Node DAG architecture bridging context dependencies through deterministic iterative loops:

- **1️⃣ Execution Planner:** Dynamically breaks complex user intents into tightly bounded `1-5` step task pipelines cleanly tracking state limits natively!
- **2️⃣ PlannerAgent:** Binds heuristic tool targets targeting explicit subsets precisely without LLM overhead.
- **3️⃣ ToolExecutionAgent:** Scrapes exact bounds executing Git blames, local file reads, and structured Log exceptions cleanly over `subprocess` arrays.
- **4️⃣ RetrieverAgent:** Extrapolates spatial vector graphs mapping semantic overlaps cleanly through Ollama embedded matrices. By-passed entirely if Router Confidence `> 0.9`. 
- **5️⃣ Answer & Critic Loop:** Constructs precise `<patch>` boundaries safely validating outputs against explicit heuristical Bash signatures ensuring logic never hallucination loops infinitely.
- **6️⃣ FileWrite Test Sandbox:** Triggers autonomous `npm test` or `pytest` bindings natively testing File diff patches dynamically executing `git checkout -b` limits reliably securely!

---

## 🛠 Features

- **Decoupled Evaluators:** Generates End-To-End metrics strictly evaluating hallucination tracking dynamically scaling Tool usage heuristics across telemetry footprints (`.ai_assistant/logs/`).
- **Autonomous Cognitive Memory:** Standard `pgvector` matrices continuously log your codebase evolution mapping Developer Intent, Design Patterns, and Context cleanly alongside standard spatial arrays!
- **Dynamic Context Building:** Extracts explicit Python AST metadata matching explicit class dependencies natively.
- **Privacy-First Native LLMs:** Operates explicitly across standard local instances mapping targets onto `nomic-embed-text` and `llama` variations via Ollama bindings!

## ⚖️ Why DeepCode? (Advantages vs Copilot, Cursor, & Aider)

While tools like GitHub Copilot, Cursor, and OpenDevin provide excellent LLM-based coding assistance, DeepCode is engineered specifically with these core advantages:

1. **Absolute Data Privacy (100% Local)**: DeepCode executes entirely on your own hardware. Your proprietary source code never leaves your network and is never transmitted to OpenAI, Microsoft, or Anthropic. 
2. **Persistent Cognitive Memory (`pgvector`)**: Standard editor tools rely on tiny spatial context windows. DeepCode natively utilizes asynchronous PostgreSQL instances mapping structured memory vectors dynamically across your whole architecture. 
3. **Execution Sandbox Boundaries**: Unlike naive agents that hallucinate brute-force terminal destruction, DeepCode strictly limits explicit operations mapping Git diffs, log traces, and File manipulations within validated sandboxes natively.
4. **Deterministic Critic Loops**: Instead of "generation -> pass" paradigms, DeepCode intrinsically mounts a Critic validation cycle checking explicit output logic before execution, plummeting hallucinated outputs reliably.

---

## ⚙️ System Dependencies

DeepCode operates locally but requires specific engine primitives to run. Ensure you have the following installed before bootstrapping:

1. **Python 3.11+**: The core execution engine.
2. **PostgreSQL & `pgvector`**: Drives all semantic storage, memory bounds, and code indexing algorithms natively. We highly recommend docker-compose for this setup.
3. **Ollama**: Acts as the local inference provider mapping dependencies to open-source models like `llama3.1:latest` and `nomic-embed-text` safely!

---

## 🔧 Installation (Recommended)

### Spin up the Database & LLM Endpoints
First, ensure your memory engine (`pgvector`) and local models are actively running:
```bash
# 1. Spin up the asynchronous pgvector database
docker-compose up -d

# 2. Ensure Ollama models are pulled and available
ollama run llama3.1
ollama pull nomic-embed-text
```

### Install globally using `pipx`
`pipx` ensures DeepCode operates safely globally across your entire OS avoiding local standard `venv` conflicts.

```bash
pip install pipx
pipx ensurepath

# Install strictly binding the standard Git repository:
pipx install git+https://github.com/abhijithvs680/deepcode.git
```

### Apply Table Schemas & Init
You explicitly bind initial vector parameters by executing:
```bash
deepcode init-db
```

### Run from anywhere
Executing the Intelligence layers is now absolutely global:
```bash
deepcode ui
deepcode ask "why is auth failing here?"
```

### Diagnostics and Upgrades
```bash
deepcode doctor
pipx upgrade deepcode
```

> [!WARNING]
> ⚠ Using `virtualenv` manually is NOT recommended for CLI end-usage natively! Explicitly rely on `pipx` for safe bounding globally.
