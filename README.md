# Sentinel Graph — Agentic Fraud Investigation

An evidence-first fraud investigation agent for the TigerGraph Hacker House Goa benchmark. It reconstructs canonical cards, learns from 5,565 closed cases, queries graph context through TigerGraph MCP, follows the supplied approval policy, writes case memory back to the graph, and emits the required 20 schema-validated answers.

## What is included

- A bounded LangGraph workflow with deterministic evidence simulation and policy guardrails.
- TigerGraph schema, loading-job, installed-query, vector-search, and MCP adapters.
- A leakage-aware DuckDB development backend that preserves as-of-time investigation semantics.
- A sparse mixed-type fraud classifier trained only on the supplied closed-case history.
- Twenty JSON answers in `cases/`, validated against strict Pydantic and policy invariants.
- A Streamlit analyst console, architecture notes, runbook, and submission copy.

## Quick start

Requires Python 3.12 or newer.

```powershell
python -m pip install -e ".[dev,tigergraph]"
Copy-Item .env.example .env
python -m fraud_agent profile --source "Drive Files"
python -m fraud_agent prepare --source "Drive Files"
python -m fraud_agent train --source "Drive Files"
python -m fraud_agent batch --source "Drive Files" --backend local --no-llm
python -m fraud_agent validate --cases-dir cases --expected-count 20
python -m streamlit run app.py
```

`Drive Files/`, `.env`, the DuckDB database, and trained model are ignored intentionally. The source benchmark is never copied into Git.

## Savanna / TigerGraph

Populate `TG_HOST`, `TG_GRAPHNAME`, and either `TG_SECRET` (the recommended Savanna Database Secret) or `TG_API_TOKEN` (an already-minted REST++ bearer token) in `.env`. Then deploy and populate the graph through one persistent MCP session:

```powershell
python -m fraud_agent graph-assets --out graph --graph-name FraudGraph
python -m fraud_agent graph-prepare --source "Drive Files"
python -m fraud_agent graph-deploy --assets graph --replace
python -m fraud_agent graph-load
python -m fraud_agent graph-knowledge --source "Drive Files/README.md"
python -m fraud_agent batch --source "Drive Files" --backend mcp
```

`--replace` permanently drops the configured graph before recreating it; use it only when replacing an incorrect or disposable schema. Omit it for a fresh database with no conflicting graph.

Without those credentials, `--backend auto` deliberately falls back to the contract-compatible DuckDB graph adapter. See [the runbook](docs/runbook.md) for deployment details and [the architecture](docs/architecture.md) for the trust boundaries.

## Verification

```powershell
python -m ruff check .
python -m mypy src
python -m pytest
python -m fraud_agent validate --cases-dir cases --expected-count 20
```

The optional Anthropic-compatible narrator may rewrite only analyst-facing summary and SAR prose. It cannot change probabilities, evidence, verdicts, actions, approval routes, or entity IDs; malformed responses fall back to deterministic prose.
