# Technical Post

## Sentinel Graph: evidence first, LLM second

Fraud investigations are graph problems disguised as classification problems. A suspicious transaction matters because of the card's history, the device behind it, other cards sharing that origin, and how earlier analysts resolved similar cases. Sentinel Graph turns that context into a bounded investigation rather than asking a language model to guess from a row.

The system prepares all 590,742 supplied transactions without altering the source data, reconstructs canonical card identities, and trains a mixed numeric/categorical classifier only from the 5,565 historical closed cases. Every benchmark case is evaluated as of its opening time. That small detail prevents future activity from leaking into an earlier decision.

TigerGraph is the evidence and memory layer. Installed queries retrieve card timelines, device neighbors, prior cases, and policy knowledge through one persistent MCP session. Once an investigation finishes, its case vertex and evidence relationships are written back, giving the next investigation an expanding institutional memory.

The LangGraph workflow is intentionally bounded. Explainable detectors, the historical model, and the source score contribute to probability; no one signal is treated as truth. Ambiguous cases request simulated customer or authentication evidence and preserve both the initial and final actions. A deterministic policy engine assigns `auto`, `L1`, or `L2` approval and keeps SAR decisions consistent with the final action set.

An Anthropic-compatible model is optional and deliberately untrusted. It can improve only the analyst summary and SAR narrative after the structured decision exists. Entity IDs, verdicts, probabilities, exposure, actions, and routes remain code-owned. If narration fails validation, deterministic prose wins.

The result is a complete set of twenty schema-valid answer files, a case-memory write path, and a Streamlit console that lets an analyst inspect the evidence trail, graph neighborhood, action routing, and report. The central design choice is simple: make every consequential decision traceable to evidence and policy, then use generation where it is strongest—clear communication.
