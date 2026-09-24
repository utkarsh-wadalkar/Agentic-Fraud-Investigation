"""Persistent TigerGraph MCP session lifecycle."""

from __future__ import annotations

import sys
from collections.abc import AsyncIterator, Mapping
from contextlib import asynccontextmanager
from typing import Any

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import get_default_environment, stdio_client

from fraud_agent.config import Settings
from fraud_agent.graph import EdgeArgumentStyle, MCPGraphStore


def edge_argument_style(tools: list[Any]) -> EdgeArgumentStyle:
    """Choose edge argument names from the MCP server's advertised schema."""
    for tool in tools:
        if getattr(tool, "name", None) != "tigergraph__add_edge":
            continue
        schema = getattr(tool, "inputSchema", None)
        if not isinstance(schema, dict):
            schema = getattr(tool, "input_schema", {})
        properties = schema.get("properties", {}) if isinstance(schema, dict) else {}
        property_names = set(properties)
        if {
            "from_vertex_type",
            "from_vertex_id",
            "to_vertex_type",
            "to_vertex_id",
        } <= property_names:
            return "from"
        if {
            "source_vertex_type",
            "source_vertex_id",
            "target_vertex_type",
            "target_vertex_id",
        } <= property_names:
            return "source"
        raise RuntimeError("tigergraph__add_edge exposes an unsupported input schema")
    raise RuntimeError("TigerGraph MCP server did not advertise tigergraph__add_edge")


def server_environment(settings: Settings, base: Mapping[str, str] | None = None) -> dict[str, str]:
    environment = dict(base if base is not None else get_default_environment())
    environment.update({
        "TG_HOST": settings.tg_host,
        "TG_GRAPHNAME": settings.tg_graphname,
        "TG_TGCLOUD": str(settings.tg_tgcloud).lower(),
    })
    secret = settings.tg_secret.get_secret_value()
    if secret:
        environment["TG_SECRET"] = secret
        environment["TG_API_TOKEN"] = ""
    else:
        environment["TG_API_TOKEN"] = settings.tg_api_token.get_secret_value()
        environment["TG_SECRET"] = ""
    return environment


@asynccontextmanager
async def open_mcp_graph_store(settings: Settings) -> AsyncIterator[MCPGraphStore]:
    if not settings.tigergraph_ready:
        raise ValueError("TG_HOST, TG_GRAPHNAME, and TG_SECRET or TG_API_TOKEN are required")
    parameters = StdioServerParameters(
        command=sys.executable,
        args=["-m", "tigergraph_mcp.main"],
        env=server_environment(settings),
    )
    async with stdio_client(parameters) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            available_tools = await session.list_tools()
            yield MCPGraphStore(
                session,
                settings.tg_graphname,
                edge_argument_style=edge_argument_style(available_tools.tools),
            )
