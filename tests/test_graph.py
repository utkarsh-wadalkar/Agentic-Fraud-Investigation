from __future__ import annotations

import asyncio
import importlib
import math

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
    assert {"OWNS", "MADE", "FROM_DEVICE", "INVOLVES", "CITES_PRIOR_CASE"} <= set(
        definition.edges
    )
    assert {path.name for path in written} == {"schema.gsql", "queries.gsql", "loading.gsql"}
    assert "CREATE GRAPH FraudGraph" in (tmp_path / "schema.gsql").read_text(encoding="utf-8")


def test_hash_embedding_is_deterministic_normalized_and_384_dimensions() -> None:
    embedder = module().HashEmbedding(dimensions=384)

    first = embedder.embed("R5 card testing small authorizations")
    second = embedder.embed("R5 card testing small authorizations")

    assert first == second
    assert len(first) == 384
    assert math.isclose(sum(value * value for value in first), 1.0, rel_tol=1e-6)


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

    async def call_tool(self, name: str, arguments: dict):
        self.calls.append((name, arguments))
        return {"success": True, "data": {"rows": [{"TransactionID": "3514030"}]}}


def test_mcp_graph_store_counts_and_parses_persistent_session_calls() -> None:
    client = FakeToolClient()
    store = module().MCPGraphStore(client=client, graph_name="FraudGraph")

    result = asyncio.run(
        store.run_query("transaction_context", {"txn_id": "3514030", "as_of": "2016-12-05"})
    )

    assert result == {"rows": [{"TransactionID": "3514030"}]}
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


def test_mcp_case_write_is_idempotent_within_session() -> None:
    client = FakeToolClient()
    store = module().MCPGraphStore(client=client, graph_name="FraudGraph")
    answer = make_answer()

    asyncio.run(store.write_case(answer))
    asyncio.run(store.write_case(answer))

    assert store.tool_calls == 1
    assert client.calls[0][0] == "tigergraph__add_node"

