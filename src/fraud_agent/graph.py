"""TigerGraph schema assets and graph-store adapters."""

# ruff: noqa: E501 -- GSQL source lines intentionally mirror deployable assets.

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from sklearn.feature_extraction.text import HashingVectorizer

from fraud_agent.models import InvestigationAnswer


@dataclass(frozen=True)
class GraphDefinition:
    vertices: tuple[str, ...]
    edges: tuple[str, ...]


def graph_definition() -> GraphDefinition:
    return GraphDefinition(
        vertices=(
            "Customer",
            "Card",
            "Transaction",
            "DeviceProfile",
            "EmailDomain",
            "BillingRegion",
            "FraudCase",
            "KnowledgeChunk",
        ),
        edges=(
            "OWNS",
            "MADE",
            "FROM_DEVICE",
            "PURCHASER_EMAIL",
            "RECIPIENT_EMAIL",
            "BILLED_IN",
            "NEXT",
            "INVOLVES",
            "ON_CARD",
            "CONNECTED_TO",
            "CITES_PRIOR_CASE",
            "CITES_KNOWLEDGE",
        ),
    )


SCHEMA_TEMPLATE = """\
CREATE VERTEX Customer(PRIMARY_ID customer_id STRING) WITH primary_id_as_attribute="true";
CREATE VERTEX Card(PRIMARY_ID card_id STRING, network STRING, card_type STRING) WITH primary_id_as_attribute="true";
CREATE VERTEX Transaction(PRIMARY_ID txn_id STRING, ts DATETIME, amount DOUBLE, product_cd STRING, channel STRING, risk_score DOUBLE, region STRING, device_new BOOL, proxy_present BOOL, match_status STRING) WITH primary_id_as_attribute="true";
CREATE VERTEX DeviceProfile(PRIMARY_ID device_id STRING, label STRING, device_type STRING, os STRING, browser STRING, screen STRING) WITH primary_id_as_attribute="true";
CREATE VERTEX EmailDomain(PRIMARY_ID domain STRING) WITH primary_id_as_attribute="true";
CREATE VERTEX BillingRegion(PRIMARY_ID region_id STRING) WITH primary_id_as_attribute="true";
CREATE VERTEX FraudCase(PRIMARY_ID case_id STRING, opened_at DATETIME, status STRING, verdict STRING, fraud_probability DOUBLE, pattern STRING, exposure_usd DOUBLE, summary STRING, source STRING) WITH primary_id_as_attribute="true";
CREATE VERTEX KnowledgeChunk(PRIMARY_ID chunk_id STRING, text STRING, source STRING, source_ref STRING, embedding LIST<DOUBLE>) WITH primary_id_as_attribute="true";

CREATE DIRECTED EDGE OWNS(FROM Customer, TO Card);
CREATE DIRECTED EDGE MADE(FROM Card, TO Transaction);
CREATE DIRECTED EDGE FROM_DEVICE(FROM Transaction, TO DeviceProfile);
CREATE DIRECTED EDGE PURCHASER_EMAIL(FROM Transaction, TO EmailDomain);
CREATE DIRECTED EDGE RECIPIENT_EMAIL(FROM Transaction, TO EmailDomain);
CREATE DIRECTED EDGE BILLED_IN(FROM Transaction, TO BillingRegion);
CREATE DIRECTED EDGE NEXT(FROM Transaction, TO Transaction);
CREATE DIRECTED EDGE INVOLVES(FROM FraudCase, TO Transaction);
CREATE DIRECTED EDGE ON_CARD(FROM FraudCase, TO Card);
CREATE DIRECTED EDGE CONNECTED_TO(FROM FraudCase, TO Card);
CREATE DIRECTED EDGE CITES_PRIOR_CASE(FROM FraudCase, TO FraudCase);
CREATE DIRECTED EDGE CITES_KNOWLEDGE(FROM FraudCase, TO KnowledgeChunk);

CREATE GRAPH {graph_name}(*)
"""

QUERY_TEMPLATE = """\
USE GRAPH {graph_name}

CREATE QUERY transaction_context(STRING txn_id, DATETIME as_of) FOR GRAPH {graph_name} SYNTAX v2 {{
  Seed = {{Transaction.*}};
  Result = SELECT t FROM Seed:t WHERE t.txn_id == txn_id AND t.ts <= as_of;
  PRINT Result;
}}

CREATE QUERY card_history(STRING card_id, DATETIME as_of, INT limit_n = 100) FOR GRAPH {graph_name} SYNTAX v2 {{
  Cards = {{Card.*}};
  Target = SELECT c FROM Cards:c WHERE c.card_id == card_id;
  Txns = SELECT t FROM Target:c -(MADE:e)-> Transaction:t
         WHERE t.ts <= as_of ORDER BY t.ts DESC LIMIT limit_n;
  PRINT Txns;
}}

CREATE QUERY device_neighbors(STRING device_id, DATETIME as_of) FOR GRAPH {graph_name} SYNTAX v2 {{
  Devices = {{DeviceProfile.*}};
  Target = SELECT d FROM Devices:d WHERE d.device_id == device_id;
  Txns = SELECT t FROM Target:d <-(FROM_DEVICE:e)- Transaction:t WHERE t.ts <= as_of;
  Cards = SELECT c FROM Txns:t <-(MADE:e)- Card:c;
  PRINT Txns, Cards;
}}

CREATE QUERY prior_cases(STRING card_id, DATETIME as_of, INT limit_n = 10) FOR GRAPH {graph_name} SYNTAX v2 {{
  Cards = {{Card.*}};
  Target = SELECT c FROM Cards:c WHERE c.card_id == card_id;
  Cases = SELECT fc FROM Target:c <-(ON_CARD:e)- FraudCase:fc
          WHERE fc.opened_at <= as_of ORDER BY fc.opened_at DESC LIMIT limit_n;
  PRINT Cases;
}}
"""

LOADING_TEMPLATE = """\
USE GRAPH {graph_name}

CREATE LOADING JOB load_fraud_graph FOR GRAPH {graph_name} {{
  DEFINE FILENAME transactions_file;
  DEFINE FILENAME identity_file;
  DEFINE FILENAME closed_cases_file;
  LOAD transactions_file TO VERTEX Transaction VALUES($"TransactionID", $"ts", $"TransactionAmt", $"ProductCD", $"channel", $"risk_score", $"addr1", false, false, "") USING HEADER="true", SEPARATOR=",";
  LOAD closed_cases_file TO VERTEX FraudCase VALUES($"case_id", $"opened_at", "closed", $"outcome", 0.0, $"pattern", $"exposure_usd", $"analyst_notes", "history") USING HEADER="true", SEPARATOR=",";
}}
"""


def render_graph_assets(graph_name: str = "FraudGraph") -> dict[str, str]:
    if not graph_name.replace("_", "").isalnum():
        raise ValueError("graph name must contain only letters, numbers, and underscores")
    return {
        "schema.gsql": SCHEMA_TEMPLATE.format(graph_name=graph_name),
        "queries.gsql": QUERY_TEMPLATE.format(graph_name=graph_name),
        "loading.gsql": LOADING_TEMPLATE.format(graph_name=graph_name),
    }


def write_graph_assets(directory: Path, graph_name: str = "FraudGraph") -> list[Path]:
    directory.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for filename, content in render_graph_assets(graph_name).items():
        path = directory / filename
        path.write_text(content, encoding="utf-8")
        written.append(path)
    return written


class HashEmbedding:
    """Dependency-light deterministic embeddings suitable for TigerGraph vector search."""

    def __init__(self, dimensions: int = 384) -> None:
        self.dimensions = dimensions
        self._vectorizer = HashingVectorizer(
            n_features=dimensions,
            alternate_sign=False,
            norm="l2",
            ngram_range=(1, 2),
        )

    def embed(self, text: str) -> list[float]:
        vector = self._vectorizer.transform([text]).toarray()[0].astype(float)
        norm = math.sqrt(float(vector @ vector))
        if norm == 0:
            vector[0] = 1.0
        return [float(value) for value in vector.tolist()]


class ToolClient(Protocol):
    async def call_tool(self, name: str, arguments: dict[str, Any]) -> Any: ...


def _parse_tool_result(result: Any) -> Any:
    if isinstance(result, dict):
        if result.get("success") is False:
            raise RuntimeError(str(result.get("error") or result))
        return result.get("data", result)
    content = getattr(result, "content", None)
    if content:
        text = getattr(content[0], "text", None)
        if text:
            parsed = json.loads(text)
            if isinstance(parsed, dict) and parsed.get("success") is False:
                raise RuntimeError(str(parsed.get("error") or parsed))
            return parsed.get("data", parsed) if isinstance(parsed, dict) else parsed
    raise TypeError(f"unsupported MCP result type: {type(result)!r}")


class MCPGraphStore:
    """Graph operations over one caller-owned persistent TigerGraph MCP session."""

    def __init__(self, client: ToolClient, graph_name: str) -> None:
        self.client = client
        self.graph_name = graph_name
        self.tool_calls = 0
        self._written_cases: set[str] = set()

    async def _call(self, name: str, arguments: dict[str, Any]) -> Any:
        self.tool_calls += 1
        return _parse_tool_result(await self.client.call_tool(name, arguments))

    async def run_query(self, query_name: str, params: dict[str, Any]) -> Any:
        return await self._call(
            "tigergraph__run_installed_query",
            {"graph_name": self.graph_name, "query_name": query_name, "params": params},
        )

    async def search_knowledge(self, vector: list[float], top_k: int = 5) -> Any:
        return await self._call(
            "tigergraph__search_top_k_similarity",
            {
                "graph_name": self.graph_name,
                "vertex_type": "KnowledgeChunk",
                "vector_attribute": "embedding",
                "query_vector": vector,
                "top_k": top_k,
            },
        )

    async def write_case(self, answer: InvestigationAnswer) -> str:
        if answer.case_id in self._written_cases:
            return answer.case_id
        attributes = {
            "status": answer.case.status,
            "verdict": answer.case.verdict,
            "fraud_probability": answer.case.fraud_probability,
            "pattern": answer.case.pattern,
            "exposure_usd": answer.case.exposure_usd,
            "summary": answer.case.summary,
            "source": "benchmark",
        }
        await self._call(
            "tigergraph__add_node",
            {
                "graph_name": self.graph_name,
                "vertex_type": "FraudCase",
                "vertex_id": answer.case_id,
                "attributes": attributes,
            },
        )
        self._written_cases.add(answer.case_id)
        return answer.case_id


class InMemoryGraphStore:
    def __init__(self, query_results: dict[str, Any] | None = None) -> None:
        self.query_results = query_results or {}
        self.cases: dict[str, InvestigationAnswer] = {}
        self.knowledge: list[dict[str, Any]] = []
        self.tool_calls = 0

    async def run_query(self, query_name: str, params: dict[str, Any]) -> Any:
        self.tool_calls += 1
        value = self.query_results.get(query_name, {})
        return value(params) if callable(value) else value

    async def search_knowledge(self, vector: list[float], top_k: int = 5) -> list[dict[str, Any]]:
        self.tool_calls += 1
        scored: list[tuple[float, dict[str, Any]]] = []
        for item in self.knowledge:
            candidate = item.get("embedding", [])
            score = sum(a * b for a, b in zip(vector, candidate, strict=False))
            scored.append((score, item))
        return [item for _, item in sorted(scored, key=lambda pair: pair[0], reverse=True)[:top_k]]

    async def write_case(self, answer: InvestigationAnswer) -> str:
        if answer.case_id not in self.cases:
            self.tool_calls += 1
            self.cases[answer.case_id] = answer
        return answer.case_id
