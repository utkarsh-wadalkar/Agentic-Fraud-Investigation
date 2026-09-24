# Runbook

## Local benchmark run

1. Place the supplied files in `Drive Files/`; do not rename or commit that directory.
2. Install the package with `python -m pip install -e ".[dev,tigergraph]"`.
3. Run `python -m fraud_agent profile --source "Drive Files"`. Expected counts are 590,742 transactions, 144,432 identity rows, 5,565 closed cases, and 20 benchmark cases.
4. Run `python -m fraud_agent prepare --source "Drive Files"` and `python -m fraud_agent train --source "Drive Files"`.
5. Generate answers with `python -m fraud_agent batch --source "Drive Files" --backend local --no-llm`.
6. Validate with `python -m fraud_agent validate --cases-dir cases --expected-count 20`.
7. Launch the console with `python -m streamlit run app.py`.

## Savanna deployment

Create a Savanna workspace and graph, copy `.env.example` to `.env`, and provide `TG_HOST`, `TG_GRAPHNAME`, plus `TG_SECRET` from **Database Secrets**. `TG_API_TOKEN` is only for a REST++ bearer token that has already been minted from a database secret; do not put a Savanna organization API key there.

Run `graph-assets`, inspect the generated GSQL, then execute `graph-prepare`, `graph-deploy`, `graph-load`, and `graph-knowledge` in that order. `graph-prepare` streams the canonical DuckDB data into compact normalized CSVs under `.artifacts/graph-load`; `graph-load` uploads every vertex and edge file through `load_fraud_graph`; `graph-knowledge` creates deterministic 384-dimensional benchmark-document embeddings. Use `graph-deploy --replace` only to replace an incorrect disposable schema because it permanently drops the configured graph first.

The MCP process is kept open for a whole batch, so connection setup is not repeated per query. A case write is idempotent within the session. If credentials are missing, explicit MCP selection fails instead of silently claiming a live graph write.

## LLM narration

Set `ANTHROPIC_BASE_URL`, `ANTHROPIC_AUTH_TOKEN`, and `ANTHROPIC_MODEL` to enable narration. Use `--no-llm` for a deterministic run. Endpoint failure or malformed JSON falls back safely.

## Troubleshooting

- Missing transaction: rebuild `.artifacts/fraud.duckdb` from the unmodified source files.
- TigerGraph query failure: confirm the graph name, install all queries in `graph/queries.gsql`, and verify the token has graph access.
- Model warning or stale behavior: delete only `.artifacts/fraud-model.joblib` and rerun `train`.
- Invalid answer: run the validator; it reports filename, schema, action-route, SAR, and exposure inconsistencies.
- Dashboard has no cases: generate the `cases/HHG-*.json` files first.

## Data safety

Before any commit, run `git status --short --ignored` and confirm `Drive Files/`, `.env`, `.artifacts/`, databases, and models are ignored. Never add them with `git add -f`.
