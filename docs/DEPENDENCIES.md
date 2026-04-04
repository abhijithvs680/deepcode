# DeepCode Agent Dependencies

This document provides a detailed breakdown of the system dependencies required to operate the **DeepCode Agent** natively and explains why they are needed. 

DeepCode operates 100% locally to guarantee zero-knowledge privacy. Because of this, you must run the persistence and intelligence layers on your own hardware.

---

## 1. Local Database Memory (`PostgreSQL` + `pgvector`)

Unlike simplistic CLI tools that only read files on demand, DeepCode builds an asynchronous graph of your codebase for persistent associative memory. Long-term memory tracking, AST tokenization, and context embeddings are continuously vectorized and stored.

To run this safely, DeepCode depends exclusively on **PostgreSQL** configured with the `pgvector` extension. 

### Why PostgreSQL + pgvector?
- **Speed**: Asynchronous pgDB connections using `asyncpg` stream vectors directly into the planning agents dynamically.
- **Scale**: Explicit HNSW indexes (`vector_cosine_ops`) bind hundreds of thousands of context files accurately without LLM window crashes.
- **Persistence**: DeepCode maps explicitly back into the database to recall previous bugs, query results, and learned heuristics across your codebase seamlessly natively.

### Installation via Docker Compose (Recommended)
You do not need to install Postgres globally on your host OS. DeepCode includes a strict `docker-compose.yml` natively configured for this logic:
```bash
docker-compose up -d
```
This spins up a secure, local-only node of PostgreSQL 15+ preconfigured with the pgvector extension on port `5432`.

---

## 2. Intelligence Provider (`Ollama`)

As a privacy-focused execution environment, DeepCode does NOT bounce your source code off of OpenAI or Anthropic servers unless explicitly configured. Instead, we rely entirely on **Ollama** explicitly hosting `llama3.1` and `nomic-embed-text` safely!

### Why Ollama?
- **Local Embedded Tensors**: Uses `nomic-embed-text` to explicitly chunk code vectors onto the PostgreSQL layer dynamically. 
- **Offline Reasoner Pipeline**: Leverages open-weight LLMs (`llama3.1`) passing execution signatures right through the internal Multi-Agent loop natively bypassing network telemetry loops.

### Installation & Initialization
Install Ollama native to your OS structure directly from their primary source boundary. 

Once Ollama is installed natively on your OS, pull the mandatory execution models explicitly:
```bash
# 1. Pull the primary generation reasoning model
ollama run llama3.1

# 2. Pull the designated text embedding model
ollama pull nomic-embed-text
```

Ensure the Ollama API is actively listening on `http://localhost:11434` before triggering DeepCode.

---

## 3. General Requirements

### Python 3.11+
The entire DeepCode Agent stack uses strict `asyncio` bounds dynamically formatting execution traces natively. Be certain your OS explicitly defaults to Python 3.11+ implicitly before executing pip configurations.

### Virtual Environments (`pipx` Recommended)
Do not install DeepCode globally bridging onto system Python boundaries. Rely explicitly on `pipx` to map the binary securely cleanly:
```bash
pipx install git+https://github.com/abhijithvs680/deepcode.git
```
