"""MCP protocol layer for the Klio memory product.

Builds a :class:`mcp.server.fastmcp.FastMCP` server over the
Streamable-HTTP transport and registers the seven Klio memory tools.

As of D2 the five *write* tools (``remember``, ``observe``, ``plan``,
``decide``, ``note``) are fully implemented: they resolve the tenant
principal, derive a stable brain ``session_id``, and persist via
:func:`shared.brain.write_scoped_memory`.  The ``recall`` and ``space``
tool *bodies* remain stubs (they return a ``not_implemented_yet``
marker) — that logic lands in D3/D4.  The tool *schemas* are the real
ones throughout: each tool's signature and argument descriptions define
the JSON input schema that ``tools/list`` advertises, so the later tasks
only fill in the remaining bodies without changing the public contract.

The write tools take a FastMCP ``Context`` as their first parameter;
FastMCP injects it and EXCLUDES it from the public input schema, so the
``tools/list`` contract is unchanged.

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

import logging
from typing import Annotated, Any, Literal, Optional

from fastapi import HTTPException
from mcp.server.fastmcp import Context, FastMCP
from mcp.server.transport_security import TransportSecuritySettings
from pydantic import Field
from shared import brain

from app.auth import resolve_principal
from app.config import (
    MCP_ALLOWED_HOSTS,
    MCP_ALLOWED_ORIGINS,
    MCP_DNS_REBINDING_PROTECTION,
)
from app.db import SessionLocal
from app.session import brain_session_id

logger = logging.getLogger("agentguard.mcp.tools")

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

# Stub payload returned by the still-unimplemented read/management tools
# (``recall``, ``space``).  The five write tools are implemented in D2; the
# remaining bodies replace this in D3/D4.
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


async def _perform_write(
    ctx: Context,
    *,
    content: str,
    memory_type: str,
    metadata: Optional[dict] = None,
) -> dict[str, Any]:
    """Shared body for the five Klio write tools.

    Resolves the tenant principal from the transport request, derives a stable
    brain ``session_id``, opens a short-lived DB session, and delegates to
    :func:`shared.brain.write_scoped_memory` (which dedups, inserts, and
    promotes the row to the org brain scope).

    Auth failures from :func:`resolve_principal` surface as Starlette
    ``HTTPException``; we translate them into a clean ``unauthorized`` result
    dict rather than letting the exception leak through the MCP layer (the MCP
    transport has no notion of HTTP status codes for tool errors).

    Args:
        ctx: FastMCP-injected context; ``ctx.request_context.request`` is the
            Starlette request for the Streamable-HTTP transport.
        content: The memory text to store.
        memory_type: The brain memory type for this tool.
        metadata: Optional metadata persisted on the row.

    Returns:
        The result dict from :func:`shared.brain.write_scoped_memory`, or a
        clean error dict on auth failure / unexpected error.
    """
    request = ctx.request_context.request
    if request is None:
        # The Streamable-HTTP transport always attaches the Starlette request
        # (mcp 1.27.x), so this only fires under a transport that does not —
        # there is no request to authenticate, so deny rather than guess a
        # principal.
        logger.error("brain write got no transport request; cannot authenticate")
        return {
            "stored": False,
            "error": "unauthorized",
            "detail": "no request context",
        }

    try:
        org_id, agent_id = resolve_principal(request)
    except HTTPException as exc:
        return {"stored": False, "error": "unauthorized", "detail": str(exc.detail)}

    session_id = brain_session_id(request, org_id, agent_id)

    db = SessionLocal()
    try:
        return await brain.write_scoped_memory(
            db,
            org_id=org_id,
            agent_id=agent_id,
            content=content,
            memory_type=memory_type,
            session_id=session_id,
            metadata=metadata,
        )
    except Exception:
        logger.exception("brain write failed (memory_type=%s)", memory_type)
        return {"stored": False, "error": "write_failed"}
    finally:
        db.close()


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
    async def remember(
        ctx: Context,
        content: Annotated[
            str, Field(description="The durable fact or preference to store.")
        ],
    ) -> dict[str, Any]:
        return await _perform_write(
            ctx, content=content, memory_type="memory", metadata={"source": "mcp"}
        )

    @mcp.tool(
        name="observe",
        description="Log something the agent saw/did this session. Arg: content.",
    )
    async def observe(
        ctx: Context,
        content: Annotated[
            str, Field(description="What the agent observed or did this session.")
        ],
    ) -> dict[str, Any]:
        return await _perform_write(
            ctx, content=content, memory_type="observation", metadata={"source": "mcp"}
        )

    @mcp.tool(
        name="plan",
        description="Record a forward-looking plan. Arg: content.",
    )
    async def plan(
        ctx: Context,
        content: Annotated[
            str, Field(description="The forward-looking plan to record.")
        ],
    ) -> dict[str, Any]:
        return await _perform_write(
            ctx, content=content, memory_type="plan", metadata={"source": "mcp"}
        )

    @mcp.tool(
        name="decide",
        description="Record a decision + rationale. Args: content, optional rationale.",
    )
    async def decide(
        ctx: Context,
        content: Annotated[
            str, Field(description="The decision that was made.")
        ],
        rationale: Annotated[
            Optional[str],
            Field(description="Optional rationale behind the decision."),
        ] = None,
    ) -> dict[str, Any]:
        metadata = {"source": "mcp"}
        if rationale is not None:
            metadata["rationale"] = rationale
        return await _perform_write(
            ctx, content=content, memory_type="decision", metadata=metadata
        )

    @mcp.tool(
        name="note",
        description="Free-form note. Arg: content.",
    )
    async def note(
        ctx: Context,
        content: Annotated[
            str, Field(description="The free-form note to store.")
        ],
    ) -> dict[str, Any]:
        return await _perform_write(
            ctx, content=content, memory_type="note", metadata={"source": "mcp"}
        )

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
