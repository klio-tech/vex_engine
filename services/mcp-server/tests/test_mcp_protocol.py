"""MCP protocol-level tests for the Klio tool surface.

These tests exercise the MCP machinery through the SDK's own client,
not by inspecting Python dicts:

* The protocol assertions (``initialize`` handshake, ``tools/list``)
  run over the SDK's in-memory transport via
  :func:`mcp.shared.memory.create_connected_server_and_client_session`,
  which connects a real :class:`mcp.ClientSession` to our ``FastMCP``
  server.  This proves ``tools/list`` returns the seven Klio tools
  through the genuine request/response round-trip.

* A separate smoke test mounts the Streamable-HTTP ASGI app under the
  FastAPI app and confirms the ``/mcp`` route is wired and responding
  (i.e. the transport is actually mounted, not just constructed).
"""

from mcp.shared.memory import create_connected_server_and_client_session

from app.mcp_app import EXPECTED_TOOL_NAMES, build_mcp_server

# Required input-schema properties each tool must expose.  Only the
# load-bearing required arg is asserted here; the full schema lives in
# ``mcp_app.py`` and is verified structurally below.
_REQUIRED_SCHEMA_FIELD = {
    "recall": "query",
    "remember": "content",
    "observe": "content",
    "plan": "content",
    "decide": "content",
    "note": "content",
    # `kind`, not `title`: the confirm branch of the two-step upload takes its
    # title and summary from the pending row, so requiring them in the schema
    # made the tool's own documented call fail before it ever ran. `kind` is
    # the one argument every branch needs.
    "artifact_put": "kind",
    "artifact_get": "artifact_id",
    "space": "action",
    "forget": "memory_id",
    "share": "memory_id",
    "project_create": "name",
    # project_list takes no arguments at all — None means "assert nothing is
    # required" rather than dropping it from the map, so a tool that silently
    # GAINS a mandatory argument still shows up here.
    "project_list": None,
    "project_members": "project",
    "project_grant": "project",
    "project_scope": "project",
    "klio_retrieve": "handle",
    "compress": "content",
    "explore": "entity",
    "init": "phase",
    "claim": "description",
    "release": "claim_id",
}


async def test_initialize_succeeds():
    """The MCP ``initialize`` handshake completes and returns server info."""
    mcp = build_mcp_server()
    async with create_connected_server_and_client_session(mcp) as session:
        # ``create_connected_server_and_client_session`` calls
        # ``initialize()`` internally; if it returned a session the
        # handshake succeeded.  Re-issue a ping to confirm the session is
        # live and bidirectional.
        await session.send_ping()


async def test_tools_list_returns_expected_tools():
    """``tools/list`` returns exactly the expected Klio tools."""
    mcp = build_mcp_server()
    async with create_connected_server_and_client_session(mcp) as session:
        result = await session.list_tools()

    names = {tool.name for tool in result.tools}
    assert names == set(EXPECTED_TOOL_NAMES), (
        f"expected {sorted(EXPECTED_TOOL_NAMES)}, got {sorted(names)}"
    )
    # No duplicates or extras.
    assert len(result.tools) == len(EXPECTED_TOOL_NAMES)


async def test_tool_has_description_and_schema():
    """Every tool advertises a non-empty description and a valid input
    schema containing its required field."""
    mcp = build_mcp_server()
    async with create_connected_server_and_client_session(mcp) as session:
        result = await session.list_tools()

    by_name = {tool.name for tool in result.tools}
    assert by_name == set(EXPECTED_TOOL_NAMES)

    for tool in result.tools:
        assert tool.description, f"{tool.name} has an empty description"
        schema = tool.inputSchema
        assert isinstance(schema, dict), f"{tool.name} schema is not an object"
        assert schema.get("type") == "object", f"{tool.name} schema is not object-typed"

        required_field = _REQUIRED_SCHEMA_FIELD[tool.name]
        if required_field is None:
            assert not schema.get("required"), (
                f"{tool.name} is supposed to take no required arguments"
            )
            continue
        props = schema.get("properties", {})
        assert required_field in props, (
            f"{tool.name} schema missing property '{required_field}'"
        )
        assert required_field in schema.get("required", []), (
            f"{tool.name} schema does not mark '{required_field}' as required"
        )


async def test_recall_description_mentions_agent_header():
    """The ``recall`` tool guides agents to set the X-Vex-Agent header."""
    mcp = build_mcp_server()
    async with create_connected_server_and_client_session(mcp) as session:
        result = await session.list_tools()

    recall = next(t for t in result.tools if t.name == "recall")
    assert "X-Vex-Agent" in recall.description


def test_mcp_route_mounted_on_fastapi():
    """The Streamable-HTTP transport is mounted under the FastAPI app.

    A bare GET to ``/mcp`` (no MCP session) must NOT 404 — the route
    exists and the transport responds.  An unmounted path would yield a
    404; a mounted-but-misused Streamable-HTTP endpoint yields a 4xx MCP
    error (e.g. 400/406) instead, which is what proves the wiring.
    """
    from fastapi.testclient import TestClient

    from app.main import create_app

    app = create_app()
    with TestClient(app) as client:
        resp = client.get("/mcp")
        assert resp.status_code != 404, (
            "/mcp not mounted — Streamable-HTTP transport missing"
        )


async def test_tools_list_over_streamable_http_transport():
    """End-to-end: drive ``tools/list`` through the SDK's Streamable-HTTP
    client against the route mounted on the real FastAPI app.

    Unlike the in-memory helper above, this exercises the full mounted
    ASGI transport (session manager + ``/mcp`` route), proving the wiring
    in ``app.main`` actually serves the protocol — not just that the
    server object registers the tools.
    """
    import httpx
    from mcp import ClientSession
    from mcp.client.streamable_http import streamable_http_client

    from app.main import create_app

    app = create_app()
    # Route the SDK client's HTTP calls through the in-process ASGI app
    # rather than a real socket.
    http_client = httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://testserver",
    )

    # ``lifespan_context`` starts the MCP session manager (required for
    # the transport to service requests).
    async with app.router.lifespan_context(app):
        async with http_client:
            async with streamable_http_client(
                "http://testserver/mcp", http_client=http_client
            ) as (read, write, _get_session_id):
                async with ClientSession(read, write) as session:
                    init = await session.initialize()
                    assert init.serverInfo.name == "klio-memory"
                    assert "X-Vex-Agent" in (init.instructions or "")

                    result = await session.list_tools()
                    names = {tool.name for tool in result.tools}
                    assert names == set(EXPECTED_TOOL_NAMES)


def test_server_instructions_state_proactive_write_protocol():
    """SERVER_INSTRUCTIONS must encode the proactive when-to-write protocol."""
    from app.mcp_app import SERVER_INSTRUCTIONS

    low = SERVER_INSTRUCTIONS.lower()
    assert "proactively" in low
    assert "decision" in low or "decide" in low
    # Steers away from noise, consistent with the recall nudge + cleanup.
    assert "not" in low and ("tool call" in low or "tool calls" in low)


async def test_remember_description_is_proactive():
    """The remember tool must instruct proactive writing, not passive storage."""
    mcp = build_mcp_server()
    async with create_connected_server_and_client_session(mcp) as session:
        result = await session.list_tools()

    remember = next(t for t in result.tools if t.name == "remember")
    desc = remember.description.lower()
    assert "proactively" in desc or "whenever" in desc
