# Agentic Fraud Investigation Implementation Plan

## Goal

Build the complete benchmark-first submission described in `Drive Files/README.md`: a guarded fraud-investigation agent backed by TigerGraph MCP, twenty validated case answers, a Streamlit demonstration UI, and submission documentation. Source data and secrets must never be committed.

## Task 1: Scaffold and domain contracts

- Write failing tests for answer validation, exposure consistency, SAR/action consistency, and policy approval routing.
- Create the Python package, configuration model, Pydantic answer models, and policy engine.
- Expected: focused tests and the full suite pass.

## Task 2: Data preparation and feature extraction

- Write failing tests for authoritative card mapping, device fingerprints, temporal filtering, and canonical dataset preparation.
- Implement streaming CSV preparation into DuckDB/Parquet-ready records without modifying source files.
- Implement explainable feature extraction and chronological model-training interfaces.
- Expected: fixture tests pass and a source-data smoke check reports the documented row counts.

## Task 3: TigerGraph schema, queries, vectors, and MCP adapter

- Write failing tests for graph schema artifacts, query contracts, persistent MCP request accounting, and idempotent case upserts.
- Implement GSQL schema/loading/query assets and an MCP adapter for graph reads, vector retrieval, and case-memory writes.
- Provide an in-memory adapter with the same interface for tests and offline development.
- Expected: contract tests pass; live smoke checks run only when TigerGraph credentials exist.

## Task 4: Guarded investigation workflow and batch answers

- Write failing tests for pattern detection, probability updates, evidence simulation, chronological batching, and schema-valid answers.
- Implement the bounded investigation workflow, optional Anthropic synthesis, deterministic fallback, policy validation, metrics, and graph writes.
- Generate twenty answer files when prepared data and a graph adapter are available.
- Expected: workflow tests pass and all generated answers pass invariant validation.

## Task 5: Streamlit UI and submission package

- Write failing smoke tests for dashboard view-model behavior and artifact generation.
- Implement the analyst dashboard, architecture documentation, runbook, technical post, social post, and demo script.
- Expected: UI import/smoke checks and document-generation tests pass.

## Task 6: End-to-end validation and Git handoff

- Run unit, integration, lint, type, CLI, data-integrity, output-validation, and UI smoke checks.
- Review the complete branch with a fresh reviewer and address Critical/Important findings using RED-GREEN tests.
- Commit only intended project files and generated submission artifacts; never push or stage `Drive Files`.
- Expected: focused Git diff, green verification, and no ignored source data or secrets in the index.

