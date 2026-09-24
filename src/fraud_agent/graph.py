"""TigerGraph schema assets and graph-store adapters."""

# ruff: noqa: E501 -- GSQL source lines intentionally mirror deployable assets.

from __future__ import annotations

import json
import math
import shutil
import tempfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Literal, Protocol

import duckdb
from sklearn.feature_extraction.text import HashingVectorizer

from fraud_agent.data import DEVICE_FIELDS, device_profile_id
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
CREATE VERTEX Customer(PRIMARY_ID customer_id STRING) WITH primary_id_as_attribute="true"
CREATE VERTEX Card(PRIMARY_ID card_id STRING, network STRING, card_type STRING) WITH primary_id_as_attribute="true"
CREATE VERTEX Transaction(PRIMARY_ID txn_id STRING, ts DATETIME, amount DOUBLE, product_cd STRING, channel STRING, risk_score DOUBLE, region STRING, device_new BOOL, proxy_present BOOL, match_status STRING) WITH primary_id_as_attribute="true"
CREATE VERTEX DeviceProfile(PRIMARY_ID device_id STRING, label STRING, device_type STRING, os STRING, browser STRING, screen STRING) WITH primary_id_as_attribute="true"
CREATE VERTEX EmailDomain(PRIMARY_ID domain STRING) WITH primary_id_as_attribute="true"
CREATE VERTEX BillingRegion(PRIMARY_ID region_id STRING) WITH primary_id_as_attribute="true"
CREATE VERTEX FraudCase(PRIMARY_ID case_id STRING, opened_at DATETIME, status STRING, verdict STRING, fraud_probability DOUBLE, pattern STRING, exposure_usd DOUBLE, summary STRING, source STRING) WITH primary_id_as_attribute="true"
CREATE VERTEX KnowledgeChunk(PRIMARY_ID chunk_id STRING, text STRING, source STRING, source_ref STRING) WITH primary_id_as_attribute="true"

CREATE DIRECTED EDGE OWNS(FROM Customer, TO Card)
CREATE DIRECTED EDGE MADE(FROM Card, TO Transaction)
CREATE DIRECTED EDGE FROM_DEVICE(FROM Transaction, TO DeviceProfile)
CREATE DIRECTED EDGE PURCHASER_EMAIL(FROM Transaction, TO EmailDomain)
CREATE DIRECTED EDGE RECIPIENT_EMAIL(FROM Transaction, TO EmailDomain)
CREATE DIRECTED EDGE BILLED_IN(FROM Transaction, TO BillingRegion)
CREATE DIRECTED EDGE NEXT(FROM Transaction, TO Transaction)
CREATE DIRECTED EDGE INVOLVES(FROM FraudCase, TO Transaction)
CREATE DIRECTED EDGE ON_CARD(FROM FraudCase, TO Card)
CREATE DIRECTED EDGE CONNECTED_TO(FROM FraudCase, TO Card)
CREATE DIRECTED EDGE CITES_PRIOR_CASE(FROM FraudCase, TO FraudCase)
CREATE DIRECTED EDGE CITES_KNOWLEDGE(FROM FraudCase, TO KnowledgeChunk)

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
  MatchingCards = SELECT c FROM Cards:c WHERE c.card_id == card_id;
  Txns = SELECT t FROM MatchingCards:c -(MADE>)- Transaction:t
         WHERE t.ts <= as_of ORDER BY t.ts DESC LIMIT limit_n;
  PRINT Txns;
}}

CREATE QUERY device_neighbors(STRING device_id, DATETIME window_start, DATETIME as_of) FOR GRAPH {graph_name} SYNTAX v2 {{
  Devices = {{DeviceProfile.*}};
  MatchingDevices = SELECT d FROM Devices:d WHERE d.device_id == device_id;
  Txns = SELECT t FROM MatchingDevices:d -(<FROM_DEVICE)- Transaction:t
         WHERE t.ts >= window_start AND t.ts <= as_of;
  FraudTxns = SELECT t FROM Txns:t -(<INVOLVES)- FraudCase:fc
              WHERE fc.verdict == "fraud";
  Cards = SELECT c FROM FraudTxns:t -(<MADE)- Card:c;
  PRINT Cards AS confirmed_fraud_cards;
}}

CREATE QUERY prior_cases(STRING card_id, DATETIME as_of, INT limit_n = 10) FOR GRAPH {graph_name} SYNTAX v2 {{
  Cards = {{Card.*}};
  MatchingCards = SELECT c FROM Cards:c WHERE c.card_id == card_id;
  Cases = SELECT fc FROM MatchingCards:c -(<ON_CARD)- FraudCase:fc
          WHERE fc.opened_at <= as_of ORDER BY fc.opened_at DESC LIMIT limit_n;
  PRINT Cases;
}}

INSTALL QUERY ALL
"""

LOADING_TEMPLATE = """\
USE GRAPH {graph_name}

CREATE LOADING JOB load_fraud_graph FOR GRAPH {graph_name} {{
  DEFINE FILENAME customers_file;
  DEFINE FILENAME cards_file;
  DEFINE FILENAME transactions_file;
  DEFINE FILENAME devices_file;
  DEFINE FILENAME transaction_devices_file;
  DEFINE FILENAME email_domains_file;
  DEFINE FILENAME purchaser_emails_file;
  DEFINE FILENAME recipient_emails_file;
  DEFINE FILENAME billing_regions_file;
  DEFINE FILENAME transaction_regions_file;
  DEFINE FILENAME next_transactions_file;
  DEFINE FILENAME closed_cases_file;
  DEFINE FILENAME case_transactions_file;
  DEFINE FILENAME case_connected_cards_file;

  LOAD customers_file TO VERTEX Customer VALUES($0) USING HEADER="false", SEPARATOR=",";
  LOAD cards_file TO VERTEX Card VALUES($0, $1, $2), TO EDGE OWNS VALUES($3, $0) USING HEADER="false", SEPARATOR=",";
  LOAD transactions_file TO VERTEX Transaction VALUES($0, $1, $2, $3, $4, $5, $6, $7, $8, $9), TO EDGE MADE VALUES($10, $0) USING HEADER="false", SEPARATOR=",";
  LOAD devices_file TO VERTEX DeviceProfile VALUES($0, $1, $2, $3, $4, $5) USING HEADER="false", SEPARATOR=",";
  LOAD transaction_devices_file TO EDGE FROM_DEVICE VALUES($0, $1) USING HEADER="false", SEPARATOR=",";
  LOAD email_domains_file TO VERTEX EmailDomain VALUES($0) USING HEADER="false", SEPARATOR=",";
  LOAD purchaser_emails_file TO EDGE PURCHASER_EMAIL VALUES($0, $1) USING HEADER="false", SEPARATOR=",";
  LOAD recipient_emails_file TO EDGE RECIPIENT_EMAIL VALUES($0, $1) USING HEADER="false", SEPARATOR=",";
  LOAD billing_regions_file TO VERTEX BillingRegion VALUES($0) USING HEADER="false", SEPARATOR=",";
  LOAD transaction_regions_file TO EDGE BILLED_IN VALUES($0, $1) USING HEADER="false", SEPARATOR=",";
  LOAD next_transactions_file TO EDGE NEXT VALUES($0, $1) USING HEADER="false", SEPARATOR=",";
  LOAD closed_cases_file TO VERTEX FraudCase VALUES($0, $1, $2, $3, $4, $5, $6, $7, $8), TO EDGE ON_CARD VALUES($0, $9) USING HEADER="false", SEPARATOR=",";
  LOAD case_transactions_file TO EDGE INVOLVES VALUES($0, $1) USING HEADER="false", SEPARATOR=",";
  LOAD case_connected_cards_file TO EDGE CONNECTED_TO VALUES($0, $1) USING HEADER="false", SEPARATOR=",";
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


EdgeArgumentStyle = Literal["source", "from"]


def _decode_mcp_text(text: str) -> Any:
    stripped = text.strip()
    if stripped.startswith("```"):
        first_newline = stripped.find("\n")
        closing_fence = stripped.find("\n```", first_newline + 1)
        if first_newline != -1 and closing_fence != -1:
            stripped = stripped[first_newline + 1 : closing_fence].strip()
    return json.JSONDecoder().raw_decode(stripped)[0]


def _parse_tool_result(result: Any) -> Any:
    if isinstance(result, dict):
        if result.get("success") is False:
            raise RuntimeError(str(result.get("error") or result))
        return result.get("data", result)
    content = getattr(result, "content", None)
    if content:
        text = getattr(content[0], "text", None)
        if text:
            parsed = _decode_mcp_text(text)
            if isinstance(parsed, dict) and parsed.get("success") is False:
                raise RuntimeError(str(parsed.get("error") or parsed))
            return parsed.get("data", parsed) if isinstance(parsed, dict) else parsed
    raise TypeError(f"unsupported MCP result type: {type(result)!r}")


def _normalize_vertex(item: Any) -> dict[str, Any]:
    if not isinstance(item, dict):
        return {}
    attributes = item.get("attributes")
    row = dict(attributes) if isinstance(attributes, dict) else dict(item)
    vertex_id = item.get("v_id")
    if vertex_id is not None:
        row.setdefault("v_id", vertex_id)
    if "txn_id" in row or item.get("v_type") == "Transaction":
        row.setdefault("TransactionID", row.get("txn_id") or vertex_id)
        row.setdefault("TransactionAmt", row.get("amount"))
        row.setdefault("ProductCD", row.get("product_cd"))
        row.setdefault("addr1", row.get("region"))
    if "case_id" not in row and item.get("v_type") == "FraudCase":
        row["case_id"] = vertex_id
    return row


def _printed_rows(result: Any, key: str) -> list[dict[str, Any]]:
    value = result
    if isinstance(result, list) and len(result) == 1 and isinstance(result[0], dict):
        value = result[0].get(key, result)
    if isinstance(value, dict):
        value = value.get(key, [])
    if not isinstance(value, list):
        return []
    return [_normalize_vertex(item) for item in value]


class MCPGraphStore:
    """Graph operations over one caller-owned persistent TigerGraph MCP session."""

    writes_to_tigergraph = True

    def __init__(
        self,
        client: ToolClient,
        graph_name: str,
        edge_argument_style: EdgeArgumentStyle = "source",
    ) -> None:
        self.client = client
        self.graph_name = graph_name
        self.edge_argument_style = edge_argument_style
        self.tool_calls = 0
        self._written_cases: set[str] = set()

    def _edge_arguments(
        self,
        from_vertex_type: str,
        from_vertex_id: str,
        edge_type: str,
        to_vertex_type: str,
        to_vertex_id: str,
    ) -> dict[str, Any]:
        arguments: dict[str, Any] = {
            "graph_name": self.graph_name,
            "edge_type": edge_type,
            "attributes": {},
        }
        if self.edge_argument_style == "from":
            arguments.update(
                {
                    "from_vertex_type": from_vertex_type,
                    "from_vertex_id": from_vertex_id,
                    "to_vertex_type": to_vertex_type,
                    "to_vertex_id": to_vertex_id,
                }
            )
        else:
            arguments.update(
                {
                    "source_vertex_type": from_vertex_type,
                    "source_vertex_id": from_vertex_id,
                    "target_vertex_type": to_vertex_type,
                    "target_vertex_id": to_vertex_id,
                }
            )
        return arguments

    async def _call(self, name: str, arguments: dict[str, Any]) -> Any:
        self.tool_calls += 1
        return _parse_tool_result(await self.client.call_tool(name, arguments))

    async def run_query(self, query_name: str, params: dict[str, Any]) -> Any:
        data = await self._call(
            "tigergraph__run_installed_query",
            {"graph_name": self.graph_name, "query_name": query_name, "params": params},
        )
        result = data.get("result", data) if isinstance(data, dict) else data
        if query_name == "transaction_context":
            rows = _printed_rows(result, "Result")
            return rows[0] if rows else {}
        if query_name == "card_history":
            return _printed_rows(result, "Txns")
        if query_name == "prior_cases":
            return _printed_rows(result, "Cases")
        if query_name == "device_neighbors":
            cards = _printed_rows(result, "confirmed_fraud_cards")
            return {
                "confirmed_fraud_card_ids": [
                    str(card.get("card_id") or card.get("v_id"))
                    for card in cards
                    if card.get("card_id") or card.get("v_id")
                ]
            }
        return result

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

    async def execute_gsql(self, command: str) -> Any:
        return await self._call(
            "tigergraph__gsql",
            {"command": command, "graph_name": self.graph_name},
        )

    async def load_file(
        self,
        path: Path,
        file_tag: str,
        job_name: str = "load_fraud_graph",
    ) -> Any:
        resolved = path.resolve()
        with resolved.open("rb") as source:
            source.readline()
            with tempfile.NamedTemporaryFile(
                mode="wb",
                delete=False,
                dir=resolved.parent,
                prefix=f".{resolved.stem}-",
                suffix=".upload.csv",
            ) as upload:
                shutil.copyfileobj(source, upload)
                upload_path = Path(upload.name)

        try:
            size_limit = max(128_000_000, upload_path.stat().st_size + 1_000_000)
            return await self._call(
                "tigergraph__run_loading_job_with_file",
                {
                    "graph_name": self.graph_name,
                    "file_path": str(upload_path),
                    "file_tag": file_tag,
                    "job_name": job_name,
                    "timeout": 0,
                    "size_limit": size_limit,
                },
            )
        finally:
            upload_path.unlink(missing_ok=True)

    async def drop_graph(self) -> Any:
        if not self.graph_name.replace("_", "").isalnum():
            raise ValueError("graph name must contain only letters, numbers, and underscores")
        return await self._call(
            "tigergraph__gsql",
            {
                "command": f"DROP GRAPH {self.graph_name} CASCADE",
                "graph_name": self.graph_name,
            },
        )

    async def configure_knowledge_vectors(self, dimension: int = 384) -> Any:
        return await self._call(
            "tigergraph__add_vector_attribute",
            {
                "graph_name": self.graph_name,
                "vertex_type": "KnowledgeChunk",
                "vector_name": "embedding",
                "dimension": dimension,
                "metric": "COSINE",
            },
        )

    async def upsert_knowledge_vectors(self, vectors: list[dict[str, Any]]) -> Any:
        return await self._call(
            "tigergraph__upsert_vectors",
            {
                "graph_name": self.graph_name,
                "vertex_type": "KnowledgeChunk",
                "vector_attribute": "embedding",
                "vectors": vectors,
            },
        )

    async def write_case(
        self,
        answer: InvestigationAnswer,
        card_id: str = "",
        opened_at: datetime | None = None,
    ) -> str:
        if answer.case_id in self._written_cases:
            return answer.case_id
        attributes = {
            "opened_at": opened_at.isoformat(sep=" ") if opened_at else "",
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
        if card_id:
            await self._call(
                "tigergraph__add_edge",
                self._edge_arguments(
                    "FraudCase", answer.case_id, "ON_CARD", "Card", card_id
                ),
            )
        for txn_id in answer.case.affected_txn_ids:
            await self._call(
                "tigergraph__add_edge",
                self._edge_arguments(
                    "FraudCase", answer.case_id, "INVOLVES", "Transaction", txn_id
                ),
            )
        for card_id in answer.case.connected_card_ids:
            await self._call(
                "tigergraph__add_edge",
                self._edge_arguments(
                    "FraudCase", answer.case_id, "CONNECTED_TO", "Card", card_id
                ),
            )
        for prior_case_id in answer.case.similar_prior_cases:
            await self._call(
                "tigergraph__add_edge",
                self._edge_arguments(
                    "FraudCase",
                    answer.case_id,
                    "CITES_PRIOR_CASE",
                    "FraudCase",
                    prior_case_id,
                ),
            )
        self._written_cases.add(answer.case_id)
        return answer.case_id


class InMemoryGraphStore:
    writes_to_tigergraph = False

    def __init__(self, query_results: dict[str, Any] | None = None) -> None:
        self.query_results = query_results or {}
        self.cases: dict[str, InvestigationAnswer] = {}
        self._case_context: dict[str, tuple[str, datetime]] = {}
        self.knowledge: list[dict[str, Any]] = []
        self.tool_calls = 0

    async def run_query(self, query_name: str, params: dict[str, Any]) -> Any:
        self.tool_calls += 1
        value = self.query_results.get(query_name, {})
        result = value(params) if callable(value) else value
        if query_name != "prior_cases" or not isinstance(result, list):
            return result
        memory = [
            {
                "case_id": case_id,
                "pattern": self.cases[case_id].case.pattern,
                "opened_at": opened_at.isoformat(sep=" "),
            }
            for case_id, (card_id, opened_at) in self._case_context.items()
            if card_id == params["card_id"]
            and opened_at <= datetime.fromisoformat(str(params["as_of"]))
        ]
        return sorted(
            [*memory, *result],
            key=lambda item: str(item.get("opened_at") or item.get("closed_at") or ""),
            reverse=True,
        )[: int(params.get("limit_n", 5))]

    async def search_knowledge(self, vector: list[float], top_k: int = 5) -> list[dict[str, Any]]:
        self.tool_calls += 1
        scored: list[tuple[float, dict[str, Any]]] = []
        for item in self.knowledge:
            candidate = item.get("embedding", [])
            score = sum(a * b for a, b in zip(vector, candidate, strict=False))
            scored.append((score, item))
        return [item for _, item in sorted(scored, key=lambda pair: pair[0], reverse=True)[:top_k]]

    async def write_case(
        self,
        answer: InvestigationAnswer,
        card_id: str = "",
        opened_at: datetime | None = None,
    ) -> str:
        if answer.case_id not in self.cases:
            self.tool_calls += 1
            self.cases[answer.case_id] = answer
            if card_id and opened_at:
                self._case_context[answer.case_id] = (card_id, opened_at)
        return answer.case_id


def _rows(cursor: duckdb.DuckDBPyConnection) -> list[dict[str, Any]]:
    columns = [item[0] for item in cursor.description]
    return [dict(zip(columns, row, strict=True)) for row in cursor.fetchall()]


class DuckDBGraphStore:
    """Offline development adapter with the same evidence contract as TigerGraph MCP."""

    writes_to_tigergraph = False

    def __init__(self, database_path: Path) -> None:
        self.database_path = database_path
        self.tool_calls = 0
        self._written_cases: set[str] = set()
        self._device_components: dict[str, tuple[str, ...]] = {}
        self._embedder = HashEmbedding()
        self._ensure_case_tables()

    def _ensure_case_tables(self) -> None:
        with duckdb.connect(str(self.database_path)) as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS investigated_cases(
                    case_id VARCHAR PRIMARY KEY,
                    card_id VARCHAR,
                    opened_at VARCHAR,
                    verdict VARCHAR,
                    probability DOUBLE,
                    pattern VARCHAR,
                    exposure DOUBLE,
                    answer_json VARCHAR
                )
                """
            )
            connection.execute(
                "ALTER TABLE investigated_cases ADD COLUMN IF NOT EXISTS card_id VARCHAR"
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS investigated_case_transactions(
                    case_id VARCHAR,
                    txn_id VARCHAR,
                    PRIMARY KEY(case_id, txn_id)
                )
                """
            )

    async def run_query(self, query_name: str, params: dict[str, Any]) -> Any:
        self.tool_calls += 1
        with duckdb.connect(str(self.database_path)) as connection:
            if query_name == "transaction_context":
                cursor = connection.execute(
                    """
                    SELECT t.*, i.* EXCLUDE ("TransactionID")
                    FROM transactions t
                    LEFT JOIN identity i USING ("TransactionID")
                    WHERE t."TransactionID" = ? AND t.ts <= ?
                    LIMIT 1
                    """,
                    [params["txn_id"], params["as_of"]],
                )
                rows = _rows(cursor)
                if not rows:
                    return {}
                row = rows[0]
                profile_id = device_profile_id(row)
                row["device_profile_id"] = profile_id
                if profile_id:
                    self._device_components[profile_id] = tuple(
                        str(row.get(field) or "") for field in DEVICE_FIELDS
                    )
                return row
            if query_name == "card_history":
                cursor = connection.execute(
                    """
                    SELECT t.*, i.id_15, i.id_23, i.id_30, i.id_31, i.id_33,
                           i.id_34, i.DeviceType, i.DeviceInfo
                    FROM transactions t
                    LEFT JOIN identity i USING ("TransactionID")
                    WHERE t.card_id = ? AND t.ts <= ?
                    ORDER BY t.ts
                    """,
                    [params["card_id"], params["as_of"]],
                )
                return _rows(cursor)
            if query_name == "prior_cases":
                cursor = connection.execute(
                    """
                    SELECT case_id, card_id, event_at AS closed_at, pattern, analyst_notes
                    FROM (
                      SELECT case_id, card_id, closed_at AS event_at, pattern,
                             analyst_notes
                      FROM closed_cases
                      WHERE card_id = ? AND closed_at <= ?
                      UNION ALL
                      SELECT case_id, card_id, opened_at AS event_at, pattern,
                             answer_json AS analyst_notes
                      FROM investigated_cases
                      WHERE card_id = ? AND opened_at <= ?
                    )
                    ORDER BY event_at DESC LIMIT ?
                    """,
                    [
                        params["card_id"],
                        params["as_of"],
                        params["card_id"],
                        params["as_of"],
                        params.get("limit_n", 5),
                    ],
                )
                return _rows(cursor)
            if query_name == "device_neighbors":
                components = self._device_components.get(str(params["device_id"]))
                if not components:
                    return {"connected_card_ids": []}
                cursor = connection.execute(
                    """
                    SELECT DISTINCT t.card_id
                    FROM identity i
                    JOIN transactions t USING ("TransactionID")
                    JOIN investigated_case_transactions ict
                      ON ict.txn_id = t."TransactionID"
                    JOIN investigated_cases ic ON ic.case_id = ict.case_id
                    WHERE coalesce(i.DeviceInfo, '') = ?
                      AND coalesce(i.id_30, '') = ?
                      AND coalesce(i.id_31, '') = ?
                      AND coalesce(i.id_33, '') = ?
                      AND coalesce(i.DeviceType, '') = ?
                      AND ic.verdict = 'fraud'
                      AND t.ts >= ?
                      AND t.ts <= ?
                    ORDER BY t.card_id
                    """,
                    [*components, params["window_start"], params["as_of"]],
                )
                return {"confirmed_fraud_card_ids": [row[0] for row in cursor.fetchall()]}
        raise ValueError(f"unsupported query: {query_name}")

    async def search_knowledge(self, vector: list[float], top_k: int = 5) -> list[dict[str, Any]]:
        self.tool_calls += 1
        policy = [
            {
                "chunk_id": "POLICY-R1",
                "text": "R1: verify a single weak signal before blocking when probability is below 0.70.",
            },
            {
                "chunk_id": "POLICY-R2",
                "text": "R2: customer denial requires blocking the card and creating a case.",
            },
            {
                "chunk_id": "POLICY-R5",
                "text": "R5: three tiny online authorizations followed by a larger purchase is card testing.",
            },
            {
                "chunk_id": "POLICY-R6",
                "text": "R6: shared fraudulent origins require a case, report, and connected-card monitoring.",
            },
        ]
        scored = []
        for item in policy:
            embedded = self._embedder.embed(item["text"])
            score = sum(a * b for a, b in zip(vector, embedded, strict=True))
            scored.append((score, item))
        return [item for _, item in sorted(scored, reverse=True)[:top_k]]

    async def write_case(
        self,
        answer: InvestigationAnswer,
        card_id: str = "",
        opened_at: datetime | None = None,
    ) -> str:
        if answer.case_id in self._written_cases:
            return answer.case_id
        with duckdb.connect(str(self.database_path)) as connection:
            connection.execute(
                """
                INSERT OR REPLACE INTO investigated_cases(
                    case_id, card_id, opened_at, verdict, probability,
                    pattern, exposure, answer_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    answer.case_id,
                    card_id,
                    opened_at.isoformat(sep=" ") if opened_at else "",
                    answer.case.verdict,
                    answer.case.fraud_probability,
                    answer.case.pattern,
                    answer.case.exposure_usd,
                    answer.model_dump_json(),
                ],
            )
            connection.execute(
                "DELETE FROM investigated_case_transactions WHERE case_id = ?",
                [answer.case_id],
            )
            if answer.case.affected_txn_ids:
                connection.executemany(
                    "INSERT INTO investigated_case_transactions VALUES (?, ?)",
                    [(answer.case_id, txn_id) for txn_id in answer.case.affected_txn_ids],
                )
        self.tool_calls += 1
        self._written_cases.add(answer.case_id)
        return answer.case_id
