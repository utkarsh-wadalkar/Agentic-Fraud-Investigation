from __future__ import annotations

import importlib
from types import SimpleNamespace

from fraud_agent.config import Settings


def module():
    return importlib.import_module("fraud_agent.mcp_runtime")


def test_server_environment_maps_tigergraph_settings_without_inheriting_unrelated_values() -> None:
    settings = Settings(
        tg_host="https://example.i.tgcloud.io",
        tg_graphname="FraudGraph",
        tg_secret="",
        tg_api_token="secret",
        tg_tgcloud=True,
    )

    environment = module().server_environment(settings, base={"PATH": "bin", "UNRELATED": "x"})

    assert environment == {
        "PATH": "bin",
        "UNRELATED": "x",
        "TG_HOST": "https://example.i.tgcloud.io",
        "TG_GRAPHNAME": "FraudGraph",
        "TG_API_TOKEN": "secret",
        "TG_SECRET": "",
        "TG_TGCLOUD": "true",
    }


def test_server_environment_supports_database_secret_authentication() -> None:
    settings = Settings(
        tg_host="https://example.i.tgcloud.io",
        tg_graphname="FraudGraph",
        tg_secret="database-secret",
        tg_tgcloud=True,
    )

    environment = module().server_environment(settings, base={"PATH": "bin"})

    assert environment == {
        "PATH": "bin",
        "TG_HOST": "https://example.i.tgcloud.io",
        "TG_GRAPHNAME": "FraudGraph",
        "TG_API_TOKEN": "",
        "TG_SECRET": "database-secret",
        "TG_TGCLOUD": "true",
    }
    assert settings.tigergraph_ready is True


def test_edge_argument_style_follows_advertised_mcp_tool_schema() -> None:
    from_to_tool = SimpleNamespace(
        name="tigergraph__add_edge",
        inputSchema={
            "properties": {
                "from_vertex_type": {},
                "from_vertex_id": {},
                "to_vertex_type": {},
                "to_vertex_id": {},
            }
        },
    )
    source_target_tool = SimpleNamespace(
        name="tigergraph__add_edge",
        inputSchema={
            "properties": {
                "source_vertex_type": {},
                "source_vertex_id": {},
                "target_vertex_type": {},
                "target_vertex_id": {},
            }
        },
    )

    assert module().edge_argument_style([from_to_tool]) == "from"
    assert module().edge_argument_style([source_target_tool]) == "source"
