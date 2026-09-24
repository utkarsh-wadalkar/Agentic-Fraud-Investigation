# Demo Script

## 0:00 — Frame the problem

Open the Streamlit overview. Explain that a risk score starts an investigation but never decides it. Point to the case, fraud, legitimate, exposure, and SAR metrics.

## 0:30 — Follow one investigation

Select a case with an evidence request. Show its initial probability, graph-derived pattern evidence, and the recorded simulated response. Compare initial and final action sets and call out the approval route.

## 1:30 — Show graph context

Open a case with a connected device or card. Use the neighborhood panel to explain how TigerGraph turns a transaction into a cross-card investigation and how `MONITOR_CONNECTED_CARDS` follows policy R6.

## 2:15 — Show regulatory separation

Open the SAR tab. Explain that case creation and external reporting are distinct. Show that a SAR appears only when `FILE_REPORT` is in the final action set and that the narrative stands on its own.

## 3:00 — Show engineering guardrails

Switch briefly to `docs/architecture.md`: as-of queries prevent temporal leakage, the LangGraph run is bounded, action routing is deterministic, and the LLM can rewrite prose only. Mention that every answer is validated before being written.

## 3:40 — Close

Show the `cases/` directory with twenty artifacts and run `python -m fraud_agent validate --cases-dir cases --expected-count 20`. End on the idea that Sentinel Graph builds reusable case memory instead of producing an isolated score.
