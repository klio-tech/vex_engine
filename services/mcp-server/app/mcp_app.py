"""MCP protocol layer for the Klio memory product.

Builds a :class:`mcp.server.fastmcp.FastMCP` server over the
Streamable-HTTP transport and registers the seven Klio memory tools.

In D1 the tool *bodies* are stubs (they return a
``not_implemented_yet`` marker) — the real memory logic lands in
D2/D3/D4.  The tool *schemas*, however, are the real ones: each tool's
signature and argument descriptions define the JSON input schema that
``tools/list`` advertises, so the later tasks only need to fill in the
bodies without changing the public contract.

Transport notes (mcp SDK 1.27.x):

* :meth:`FastMCP.streamable_http_app` returns a Starlette ASGI app that
  serves the Streamable-HTTP endpoint at ``streamable_http_path``
  (default ``/mcp``).  ``app.main`` mounts this under the FastAPI app.
* The Streamable-HTTP transport requires its session manager to be
  running for the lifetime of the app.  ``app.main`` enters
  ``mcp.session_manager.run()`` from the FastAPI lifespan — without that
  context the mounted route accepts connections but cannot service MCP
  requests.

Why set the ``X-Vex-Agent`` header: a client that omits it maps to a
single shared default agent (``mcp-default`` — see ``app.auth``), so its
memories pool under one agent_id.  The server instructions and the
``recall``/``remember`` tool descriptions tell agents to set the header
to scope memory to a distinct agent.
"""

from typing import Annotated, Any, Literal, Optional

from mcp.server.fastmcp import FastMCP
from mcp.server.transport_security import TransportSecuritySettings
from pydantic import Field

from app.config import (
    MCP_ALLOWED_HOSTS,
    MCP_ALLOWED_ORIGINS,
    MCP_DNS_REBINDING_PROTECTION,
)

# Canonical, ordered list of the Klio tool names.  Tests assert the
# registered tool set matches this exactly.
EXPECTED_TOOL_NAMES: tuple[str, ...] = (
    "recall",
    "remember",
    "observe",
    "plan",
    "decide",
    "note",
    "space",
)

# Stub payload returned by every tool body in D1.  Real implementations
# (D2-D4) replace these return values.
_STUB_RESULT: dict[str, str] = {"status": "not_implemented_yet"}

# Server-level guidance surfaced to MCP clients during ``initialize``.
# Reinforces the X-Vex-Agent contract at the protocol level (in addition
# to the per-tool descriptions).
SERVER_INSTRUCTIONS = (
    "Klio memory tools for AI agents. Memory is scoped to the org "
    "resolved from your X-Vex-Key, and to an agent resolved from the "
    "X-Vex-Agent header. Set the X-Vex-Agent header to a stable "
    "per-agent identifier; clients that omit it share a single default "
    "agent and their memories pool together. Use `recall` to retrieve "
    "context before acting, and the write tools (`remember`, `observe`, "
    "`plan`, `decide`, `note`) to persist what matters."
)


def build_mcp_server() -> FastMCP:
    """Construct a fresh :class:`FastMCP` with the seven Klio tools.

    Returns a new instance on every call so tests get isolated servers
    and so the app factory controls the lifecycle.  Tool bodies are D1
    stubs; schemas are final.
    """
    # Keep the default ``streamable_http_path="/mcp"``.  ``app.main``
    # mounts this server's ASGI app at the application root, so the
    # transport endpoint lands exactly at ``/mcp`` (no trailing-slash
    # redirect).  Mounting at a ``/mcp`` prefix with a ``"/"`` transport
    # path instead would force Starlette to serve at ``/mcp/`` and 307
    # redirect bare ``/mcp`` POSTs, which the MCP client does not follow.
    #
    # Pass ``transport_security`` EXPLICITLY: when it is left ``None``,
    # FastMCP auto-enables DNS-rebinding protection with a hardcoded
    # localhost allow-list (``localhost:*`` etc.), which rejects
    # legitimate non-localhost traffic with HTTP 421.  We drive it from
    # config instead (see ``app.config`` for the rationale — auth is the
    # real boundary and the deployment Host is environment-specific).
    mcp = FastMCP(
        name="klio-memory",
        instructions=SERVER_INSTRUCTIONS,
        transport_security=_build_transport_security(),
    )
    _register_tools(mcp)
    return mcp


def _build_transport_security() -> TransportSecuritySettings:
    """Construct the SDK transport-security settings from config.

    Defaults to DNS-rebinding protection OFF (see ``app.config``).  When
    enabled, the configured host/origin allow-lists are applied verbatim,
    supporting the SDK's ``host:*`` wildcard-port syntax.
    """
    return TransportSecuritySettings(
        enable_dns_rebinding_protection=MCP_DNS_REBINDING_PROTECTION,
        allowed_hosts=list(MCP_ALLOWED_HOSTS),
        allowed_origins=list(MCP_ALLOWED_ORIGINS),
    )


def _register_tools(mcp: FastMCP) -> None:
    """Register all seven Klio tools on the given server.

    Each tool's typed signature drives the JSON input schema advertised
    by ``tools/list``.  Bodies are stubs in D1.
    """

    @mcp.tool(
        name="recall",
        description=(
            "Retrieve relevant memories for the current org/team. "
            "Args: query (required); optional scope ('org' default | "
            "'agent'), space, project. Set the X-Vex-Agent header to "
            "scope by agent."
        ),
    )
    def recall(
        query: Annotated[
            str, Field(description="What to search the team's memory for.")
        ],
        scope: Annotated[
            Literal["org", "agent"],
            Field(description="Memory scope: 'org' (default) or 'agent'."),
        ] = "org",
        space: Annotated[
            Optional[str],
            Field(description="Optional space slug to scope the search."),
        ] = None,
        project: Annotated[
            Optional[str],
            Field(description="Optional project identifier to scope the search."),
        ] = None,
    ) -> dict[str, Any]:
        return dict(_STUB_RESULT)

    @mcp.tool(
        name="remember",
        description=(
            "Store a durable fact/preference for the team. Arg: content. "
            "Set the X-Vex-Agent header to attribute it to a specific "
            "agent."
        ),
    )
    def remember(
        content: Annotated[
            str, Field(description="The durable fact or preference to store.")
        ],
    ) -> dict[str, Any]:
        return dict(_STUB_RESULT)

    @mcp.tool(
        name="observe",
        description="Log something the agent saw/did this session. Arg: content.",
    )
    def observe(
        content: Annotated[
            str, Field(description="What the agent observed or did this session.")
        ],
    ) -> dict[str, Any]:
        return dict(_STUB_RESULT)

    @mcp.tool(
        name="plan",
        description="Record a forward-looking plan. Arg: content.",
    )
    def plan(
        content: Annotated[
            str, Field(description="The forward-looking plan to record.")
        ],
    ) -> dict[str, Any]:
        return dict(_STUB_RESULT)

    @mcp.tool(
        name="decide",
        description="Record a decision + rationale. Args: content, optional rationale.",
    )
    def decide(
        content: Annotated[
            str, Field(description="The decision that was made.")
        ],
        rationale: Annotated[
            Optional[str],
            Field(description="Optional rationale behind the decision."),
        ] = None,
    ) -> dict[str, Any]:
        return dict(_STUB_RESULT)

    @mcp.tool(
        name="note",
        description="Free-form note. Arg: content.",
    )
    def note(
        content: Annotated[
            str, Field(description="The free-form note to store.")
        ],
    ) -> dict[str, Any]:
        return dict(_STUB_RESULT)

    @mcp.tool(
        name="space",
        description=(
            "Manage spaces. Args: action (list|create|info|switch), "
            "optional name/slug."
        ),
    )
    def space(
        action: Annotated[
            Literal["list", "create", "info", "switch"],
            Field(description="The space operation to perform."),
        ],
        name: Annotated[
            Optional[str],
            Field(description="Human-readable space name (for 'create')."),
        ] = None,
        slug: Annotated[
            Optional[str],
            Field(description="Space slug (for 'info'/'switch'/'create')."),
        ] = None,
    ) -> dict[str, Any]:
        return dict(_STUB_RESULT)
