from __future__ import annotations

# ruff: noqa: E501 -- SQL fixture declarations are clearer on single lines.
import asyncio
import importlib
import json
import math
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import duckdb

from tests.test_domain import make_answer


def module():
    return importlib.import_module("fraud_agent.graph")


def test_graph_definition_exposes_required_entities_and_writes_assets(tmp_path) -> None:
    graph = module()

    definition = graph.graph_definition()
    written = graph.write_graph_assets(tmp_path, graph_name="FraudGraph")

    assert {
        "Customer",
        "Card",
        "Transaction",
        "DeviceProfile",
        "EmailDomain",
        "BillingRegion",
        "FraudCase",
        "KnowledgeChunk",
    } <= set(definition.vertices)
    assert {"OWNS", "MADE", "FROM_DEVICE", "INVOLVES", "CITES_PRIOR_CASE"} <= set(definition.edges)
    assert {path.name for path in written} == {"schema.gsql", "queries.gsql", "loading.gsql"}
    assert "CREATE GRAPH FraudGraph" in (tmp_path / "schema.gsql").read_text(encoding="utf-8")
    schema = (tmp_path / "schema.gsql").read_text(encoding="utf-8")
    queries = (tmp_path / "queries.gsql").read_text(encoding="utf-8")
    assert "KnowledgeChunk" in schema
    assert "embedding LIST<DOUBLE>" not in schema
    assert "Target =" not in queries
    assert "-(MADE>)-" in queries
    assert "-(<FROM_DEVICE)-" in queries
    assert "-(<ON_CARD)-" in queries
    assert queries.count("INSTALL QUERY") == 1
    assert "INSTALL QUERY ALL" in queries
    assert all(
        not line.endswith(";")
        for line in schema.splitlines()
        if line.startswith(("CREATE VERTEX", "CREATE DIRECTED EDGE"))
    )


def test_hash_embedding_is_deterministic_normalized_and_384_dimensions() -> None:
    embedder = module().HashEmbedding(dimensions=384)

    first = embedder.embed("R5 card testing small authorizations")
    second = embedder.embed("R5 card testing small authorizations")

    assert first == second
    assert len(first) == 384
    assert math.isclose(sum(value * value for value in first), 1.0, rel_tol=1e-6)


def test_mcp_result_parser_accepts_markdown_fenced_structured_response() -> None:
    result = SimpleNamespace(
        content=[
            SimpleNamespace(
                text='```json\n{"success": true, "data": {"graphs": ["FraudGraph"]}}\n```\n'
                "\n**Found 1 graph**"
            )
        ]
    )

    assert module()._parse_tool_result(result) == {"graphs": ["FraudGraph"]}


def test_mcp_result_parser_ignores_fences_inside_json_string_values() -> None:
    payload = {
        "success": True,
        "data": {"text": "Example:\n```json\n{\"verdict\": \"fraud\"}\n```"},
    }
    result = SimpleNamespace(
        content=[
            SimpleNamespace(
                text=f"```json\n{json.dumps(payload)}\n```\n\n**Suggestions:**\n1. Continue"
            )
        ]
    )

    assert module()._parse_tool_result(result) == payload["data"]


def test_in_memory_graph_store_upserts_case_idempotently() -> None:
    store = module().InMemoryGraphStore()
    answer = make_answer()

    first = asyncio.run(store.write_case(answer))
    second = asyncio.run(store.write_case(answer))

    assert first == second == "HHG-001"
    assert list(store.cases) == ["HHG-001"]
    assert store.tool_calls == 1


class FakeToolClient:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict]] = []
        self.uploaded_data = ""

    async def call_tool(self, name: str, arguments: dict):
        if name == "tigergraph__run_loading_job_with_file":
            self.uploaded_data = Path(arguments["file_path"]).read_text(encoding="utf-8")
        if name == "tigergraph__add_edge" and "source_vertex_type" in arguments:
            edge_tools = importlib.import_module("tigergraph_mcp.tools.edge_tools")
            edge_tools.AddEdgeToolInput.model_validate(arguments)
        self.calls.append((name, arguments))
        if name == "tigergraph__run_installed_query":
            if arguments["query_name"] == "device_neighbors":
                return {
                    "success": True,
                    "data": {
                        "query_name": "device_neighbors",
                        "result": [
                            {
                                "confirmed_fraud_cards": [
                                    {"v_id": "C2-K1"},
                                    {"v_id": "C3-K1"},
                                ]
                            }
                        ],
                    },
                }
            return {
                "success": True,
                "data": {
                    "query_name": "transaction_context",
                    "result": [{"TransactionID": "3514030"}],
                },
            }
        return {"success": True, "data": None}


def test_mcp_graph_store_counts_and_parses_persistent_session_calls() -> None:
    client = FakeToolClient()
    store = module().MCPGraphStore(client=client, graph_name="FraudGraph")

    result = asyncio.run(
        store.run_query("transaction_context", {"txn_id": "3514030", "as_of": "2016-12-05"})
    )

    assert result == {"TransactionID": "3514030"}
    assert store.tool_calls == 1
    assert client.calls == [
        (
            "tigergraph__run_installed_query",
            {
                "graph_name": "FraudGraph",
                "query_name": "transaction_context",
                "params": {"txn_id": "3514030", "as_of": "2016-12-05"},
            },
        )
    ]


def test_mcp_graph_store_uploads_loading_file_with_absolute_path(tmp_path) -> None:
    data_file = tmp_path / "transactions.csv"
    data_file.write_text("txn_id\nT1\n", encoding="utf-8")
    client = FakeToolClient()
    store = module().MCPGraphStore(client=client, graph_name="FraudGraph")

    asyncio.run(store.load_file(data_file, "transactions_file"))

    assert client.uploaded_data == "T1\n"
    assert len(client.calls) == 1
    name, arguments = client.calls[0]
    assert name == "tigergraph__run_loading_job_with_file"
    assert arguments["graph_name"] == "FraudGraph"
    assert arguments["file_tag"] == "transactions_file"
    assert arguments["job_name"] == "load_fraud_graph"
    assert arguments["timeout"] == 0
    assert arguments["size_limit"] == 128_000_000
    assert not Path(arguments["file_path"]).exists()


def test_mcp_graph_store_configures_and_upserts_knowledge_vectors() -> None:
    client = FakeToolClient()
    store = module().MCPGraphStore(client=client, graph_name="FraudGraph")
    vectors = [
        {
            "vertex_id": "K-1",
            "vector": [0.0, 1.0],
            "attributes": {"text": "Rule R5", "source": "benchmark", "source_ref": "R5"},
        }
    ]

    asyncio.run(store.configure_knowledge_vectors(dimension=2))
    asyncio.run(store.upsert_knowledge_vectors(vectors))

    assert client.calls == [
        (
            "tigergraph__add_vector_attribute",
            {
                "graph_name": "FraudGraph",
                "vertex_type": "KnowledgeChunk",
                "vector_name": "embedding",
                "dimension": 2,
                "metric": "COSINE",
            },
        ),
        (
            "tigergraph__upsert_vectors",
            {
                "graph_name": "FraudGraph",
                "vertex_type": "KnowledgeChunk",
                "vector_attribute": "embedding",
                "vectors": vectors,
            },
        ),
    ]


def test_mcp_graph_store_drops_graph_and_dependencies_with_cascade() -> None:
    client = FakeToolClient()
    store = module().MCPGraphStore(client=client, graph_name="FraudGraph")

    asyncio.run(store.drop_graph())

    assert client.calls == [
        (
            "tigergraph__gsql",
            {
                "command": "DROP GRAPH FraudGraph CASCADE",
                "graph_name": "FraudGraph",
            },
        )
    ]

def test_mcp_device_neighbors_normalize_confirmed_fraud_card_ids() -> None:
    store = module().MCPGraphStore(client=FakeToolClient(), graph_name="FraudGraph")

    result = asyncio.run(
        store.run_query(
            "device_neighbors",
            {
                "device_id": "DEV-1",
                "window_start": "2016-11-29 12:00:00",
                "as_of": "2016-12-01 12:00:00",
            },
        )
    )

    assert result == {"confirmed_fraud_card_ids": ["C2-K1", "C3-K1"]}


def test_mcp_case_write_is_idempotent_within_session() -> None:
    client = FakeToolClient()
    store = module().MCPGraphStore(client=client, graph_name="FraudGraph")
    answer = make_answer()

    asyncio.run(store.write_case(answer))
    asyncio.run(store.write_case(answer))

    assert store.tool_calls == 4
    assert [name for name, _ in client.calls] == [
        "tigergraph__add_node",
        "tigergraph__add_edge",
        "tigergraph__add_edge",
        "tigergraph__add_edge",
    ]
    assert [call[1]["edge_type"] for call in client.calls[1:]] == [
        "INVOLVES",
        "INVOLVES",
        "CITES_PRIOR_CASE",
    ]


def test_mcp_case_write_stores_opened_at_and_primary_card_relationship() -> None:
    client = FakeToolClient()
    store = module().MCPGraphStore(client=client, graph_name="FraudGraph")

    asyncio.run(
        store.write_case(
            make_answer(),
            card_id="C1-K1",
            opened_at=datetime(2016, 12, 1, 12, 5),
        )
    )

    node_attributes = client.calls[0][1]["attributes"]
    assert node_attributes["opened_at"] == "2016-12-01 12:05:00"
    assert any(
        name == "tigergraph__add_edge"
        and arguments["edge_type"] == "ON_CARD"
        and arguments["target_vertex_id"] == "C1-K1"
        for name, arguments in client.calls
    )


def test_mcp_case_write_supports_from_to_edge_arguments() -> None:
    client = FakeToolClient()
    store = module().MCPGraphStore(
        client=client,
        graph_name="FraudGraph",
        edge_argument_style="from",
    )

    asyncio.run(
        store.write_case(
            make_answer(),
            card_id="C1-K1",
            opened_at=datetime(2016, 12, 1, 12, 5),
        )
    )

    edge_arguments = [
        arguments for name, arguments in client.calls if name == "tigergraph__add_edge"
    ]
    assert edge_arguments
    assert all(
        {
            "from_vertex_type",
            "from_vertex_id",
            "to_vertex_type",
            "to_vertex_id",
        }
        <= arguments.keys()
        for arguments in edge_arguments
    )
    assert all("source_vertex_type" not in arguments for arguments in edge_arguments)


def test_mcp_gsql_uses_documented_command_argument() -> None:
    client = FakeToolClient()
    store = module().MCPGraphStore(client=client, graph_name="FraudGraph")

    asyncio.run(store.execute_gsql("LS"))

    assert client.calls == [("tigergraph__gsql", {"command": "LS", "graph_name": "FraudGraph"})]


def test_duckdb_development_store_serves_as_of_context_and_case_memory(tmp_path) -> None:
    database = tmp_path / "fraud.duckdb"
    with duckdb.connect(str(database)) as connection:
        connection.execute(
            """
            CREATE TABLE transactions(
                "TransactionID" VARCHAR, "TransactionAmt" VARCHAR, ts VARCHAR,
                channel VARCHAR, addr1 VARCHAR, risk_score VARCHAR,
                customer_id VARCHAR, card_id VARCHAR, "ProductCD" VARCHAR
            )
            """
        )
        connection.execute(
            "INSERT INTO transactions VALUES ('T1','10','2016-11-01 10:00:00','online','100','0.2','C1','C1-K1','C'), ('T2','50','2016-12-01 10:00:00','online','100','0.8','C1','C1-K1','C')"
        )
        connection.execute(
            'CREATE TABLE identity("TransactionID" VARCHAR, id_01 VARCHAR, id_15 VARCHAR, id_23 VARCHAR, id_30 VARCHAR, id_31 VARCHAR, id_33 VARCHAR, id_34 VARCHAR, DeviceType VARCHAR, DeviceInfo VARCHAR)'
        )
        connection.execute(
            "INSERT INTO identity VALUES ('T2','-5.0','New','','Windows 10','edge 16','1366x768','match_status:2','desktop','Windows')"
        )
        connection.execute(
            "CREATE TABLE closed_cases(case_id VARCHAR, card_id VARCHAR, closed_at VARCHAR, pattern VARCHAR, analyst_notes VARCHAR)"
        )
        connection.execute(
            "INSERT INTO closed_cases VALUES ('CC-1','C1-K1','2016-10-01 00:00:00','card_not_present_fraud','Prior fraud')"
        )
        connection.execute("CREATE TABLE case_pack(case_id VARCHAR)")
    store = module().DuckDBGraphStore(database)

    flagged = asyncio.run(
        store.run_query("transaction_context", {"txn_id": "T2", "as_of": "2016-12-01 10:00:00"})
    )
    history = asyncio.run(
        store.run_query("card_history", {"card_id": "C1-K1", "as_of": "2016-11-15 00:00:00"})
    )
    prior = asyncio.run(
        store.run_query("prior_cases", {"card_id": "C1-K1", "as_of": "2016-12-01", "limit_n": 5})
    )
    answer = make_answer()
    answer.case_id = "HHG-LOCAL"
    answer.case.graph_case_id = "HHG-LOCAL"
    asyncio.run(
        store.write_case(
            answer,
            card_id="C1-K1",
            opened_at=datetime(2016, 11, 20, 12, 0),
        )
    )
    prior_after_write = asyncio.run(
        store.run_query(
            "prior_cases",
            {"card_id": "C1-K1", "as_of": "2016-12-01", "limit_n": 5},
        )
    )

    assert flagged["TransactionID"] == "T2"
    assert flagged["id_15"] == "New"
    assert flagged["id_01"] == "-5.0"
    assert flagged["device_profile_id"].startswith("DEV-")
    assert [row["TransactionID"] for row in history] == ["T1"]
    assert [row["case_id"] for row in prior] == ["CC-1"]
    assert [row["case_id"] for row in prior_after_write] == ["HHG-LOCAL", "CC-1"]
    with duckdb.connect(str(database), read_only=True) as connection:
        stored = connection.execute(
            "SELECT case_id FROM investigated_cases WHERE case_id='HHG-LOCAL'"
        ).fetchone()
    assert stored == ("HHG-LOCAL",)


def test_duckdb_device_neighbors_return_only_recent_confirmed_fraud_cards(tmp_path) -> None:
    database = tmp_path / "fraud.duckdb"
    with duckdb.connect(str(database)) as connection:
        connection.execute(
            """
            CREATE TABLE transactions(
                "TransactionID" VARCHAR, "TransactionAmt" VARCHAR, ts VARCHAR,
                channel VARCHAR, addr1 VARCHAR, risk_score VARCHAR,
                customer_id VARCHAR, card_id VARCHAR, "ProductCD" VARCHAR
            )
            """
        )
        connection.execute(
            """
            INSERT INTO transactions VALUES
              ('T1','80','2016-12-01 10:00:00','online','100','0.5','C1','C1-K1','C'),
              ('T2','90','2016-12-01 09:00:00','online','100','0.5','C2','C2-K1','C'),
              ('T3','40','2016-12-01 09:30:00','online','100','0.5','C3','C3-K1','C')
            """
        )
        connection.execute(
            """
            CREATE TABLE identity(
                "TransactionID" VARCHAR, id_15 VARCHAR, id_23 VARCHAR,
                id_30 VARCHAR, id_31 VARCHAR, id_33 VARCHAR, id_34 VARCHAR,
                DeviceType VARCHAR, DeviceInfo VARCHAR
            )
            """
        )
        connection.execute(
            """
            INSERT INTO identity VALUES
              ('T1','New','','Windows','edge','1366x768','','desktop','Shared'),
              ('T2','New','','Windows','edge','1366x768','','desktop','Shared'),
              ('T3','New','','Windows','edge','1366x768','','desktop','Shared')
            """
        )
        connection.execute(
            """
            CREATE TABLE closed_cases(
                case_id VARCHAR, card_id VARCHAR, closed_at VARCHAR,
                pattern VARCHAR, analyst_notes VARCHAR
            )
            """
        )
    store = module().DuckDBGraphStore(database)
    flagged = asyncio.run(
        store.run_query(
            "transaction_context", {"txn_id": "T1", "as_of": "2016-12-01 10:05:00"}
        )
    )
    confirmed = make_answer()
    confirmed.case_id = "HHG-CONFIRMED"
    confirmed.case.affected_txn_ids = ["T2"]
    asyncio.run(
        store.write_case(
            confirmed,
            card_id="C2-K1",
            opened_at=datetime(2016, 12, 1, 9, 5),
        )
    )

    neighbors = asyncio.run(
        store.run_query(
            "device_neighbors",
            {
                "device_id": flagged["device_profile_id"],
                "window_start": "2016-11-29 10:05:00",
                "as_of": "2016-12-01 10:05:00",
            },
        )
    )

    assert neighbors == {"confirmed_fraud_card_ids": ["C2-K1"]}
