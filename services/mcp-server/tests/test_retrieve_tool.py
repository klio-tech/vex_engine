"""klio_retrieve returns an original, to its owner and nobody else.

Two halves, deliberately:

* Wiring and response shape use the mocked FastMCP-registry harness of
  ``test_recall_tool.py`` — the tool callable off ``server._tool_manager``,
  a hand-built Context, and the DB seam patched at the ``app.mcp_app``
  boundary.
* The ownership boundary is exercised against a REAL Postgres connection.
  A mocked session returns whatever the test queued, so it would prove
  ``not_found`` was returned without proving the SQL refused anything. These
  mirror ``test_compression_cache_isolation.py``'s fixture.
"""

import uuid
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from sqlalchemy import create_engine, text

from app.compression_cache import store_original
from app.mcp_app import build_mcp_server
from shared.artifacts import TEXT_PAGE_CHARS
from shared.testing import database_url, requires_db

DB_URL = database_url()

ALICE = "11111111-1111-1111-1111-111111111111"
BOB = "22222222-2222-2222-2222-222222222222"


needs_db = requires_db("the klio_retrieve boundary tests")


def _fake_ctx(headers: dict | None = None):
    request = SimpleNamespace(headers=headers if headers is not None else {})
    return SimpleNamespace(request_context=SimpleNamespace(request=request))


def _retrieve_fn():
    server = build_mcp_server()
    tool = server._tool_manager.get_tool("klio_retrieve")
    assert tool is not None, "tool 'klio_retrieve' not registered"
    return tool.fn


@pytest.fixture()
def db():
    """A real Connection, with every row this test created removed afterwards."""
    engine = create_engine(DB_URL)
    conn = engine.connect()
    orgs: set[str] = set()
    conn.created_org_ids = orgs  # type: ignore[attr-defined]
    try:
        yield conn
    finally:
        try:
            for org_id in orgs:
                conn.execute(
                    text("DELETE FROM compression_cache WHERE org_id = :o"),
                    {"o": org_id},
                )
            conn.commit()
        finally:
            conn.close()
            engine.dispose()


@pytest.fixture()
def acme(db):
    org_id = f"acme_{uuid.uuid4().hex[:8]}"
    db.created_org_ids.add(org_id)
    return org_id


def _live(db, *, org_id: str, user_id: str | None):
    """Patch the tool's principal + DB seams onto a real connection.

    ``SessionLocal`` is replaced by a factory returning the live Connection
    with ``close`` neutered — the fixture owns its lifetime, and the tool
    closing it would break the cleanup that follows.
    """
    live = MagicMock(wraps=db)
    live.execute = db.execute
    live.commit = db.commit
    live.close = MagicMock()
    return (
        patch(
            "app.mcp_app.resolve_active_principal",
            return_value=(org_id, "agent-y", "free", None),
        ),
        patch("app.mcp_app.resolve_request_user_id", return_value=user_id),
        patch("app.mcp_app.SessionLocal", MagicMock(return_value=live)),
    )


# ---------------------------------------------------------------------------
# The boundary, against real Postgres.
# ---------------------------------------------------------------------------


@needs_db
@pytest.mark.asyncio
async def test_owner_retrieves_the_original(db, acme):
    """Positive control: without it, a tool that never returns anything would
    pass every refusal test below."""
    handle = store_original(db, org_id=acme, user_id=ALICE, text="secret log")

    a, b, c = _live(db, org_id=acme, user_id=ALICE)
    with a, b, c:
        result = await _retrieve_fn()(_fake_ctx(), handle)

    assert result["text"] == "secret log"
    assert result["offset"] == 0
    assert result["has_more"] is False


@needs_db
@pytest.mark.asyncio
async def test_another_users_handle_is_not_found_not_forbidden(db, acme):
    """`forbidden` would confirm the handle exists. Refusals must be
    indistinguishable from a handle that was never issued."""
    handle = store_original(db, org_id=acme, user_id=ALICE, text="secret log")

    a, b, c = _live(db, org_id=acme, user_id=BOB)
    with a, b, c:
        stolen = await _retrieve_fn()(_fake_ctx(), handle)
        never_issued = await _retrieve_fn()(_fake_ctx(), "kc_0000000000000000")

    assert stolen == {"error": "not_found"}
    # Byte-identical to the response for a handle that never existed: the
    # refusal must carry no signal that this one does.
    assert stolen == never_issued
    assert "secret log" not in str(stolen)


@needs_db
@pytest.mark.asyncio
async def test_an_unattributed_caller_gets_not_found(db, acme):
    """No resolvable user means no user-scoped read can succeed."""
    handle = store_original(db, org_id=acme, user_id=ALICE, text="secret log")

    a, b, c = _live(db, org_id=acme, user_id=None)
    with a, b, c:
        result = await _retrieve_fn()(_fake_ctx(), handle)

    assert result == {"error": "not_found"}


@needs_db
@pytest.mark.asyncio
async def test_expired_handle_says_expired(db, acme):
    """Silence would look like an empty original. Name the reason."""
    handle = store_original(
        db, org_id=acme, user_id=ALICE, text="secret log", ttl_seconds=-1
    )

    a, b, c = _live(db, org_id=acme, user_id=ALICE)
    with a, b, c:
        result = await _retrieve_fn()(_fake_ctx(), handle)

    assert result["error"] == "expired"
    assert result["detail"], "an expired handle must name the reason"
    # The reason is told only to the owner, and never leaks the content.
    assert "secret log" not in str(result)


@needs_db
@pytest.mark.asyncio
async def test_another_users_expired_handle_is_still_only_not_found(db, acme):
    """`expired` is itself a confirmation that the handle was real."""
    handle = store_original(
        db, org_id=acme, user_id=ALICE, text="secret log", ttl_seconds=-1
    )

    a, b, c = _live(db, org_id=acme, user_id=BOB)
    with a, b, c:
        result = await _retrieve_fn()(_fake_ctx(), handle)

    assert result == {"error": "not_found"}


@needs_db
@pytest.mark.asyncio
async def test_paging_matches_the_artifact_get_contract(db, acme):
    """Same contract as artifact_get, so an agent that knows one knows the
    other: TEXT_PAGE_CHARS per page, `offset` to continue, `has_more` to stop.

    Reassembling every page must reproduce the original byte for byte — a
    recovery contract that returns a mangled original is not a recovery
    contract.
    """
    # The inline ceiling is 64KB (INLINE_MAX_BYTES, mirroring artifacts) and a
    # page is 51,200 characters, so an inline original spans at most two
    # pages. That is the whole of what this storage path can produce; a
    # deeper multi-page walk arrives with the object-storage offload in
    # stage 4.
    original = "".join(f"line {i:05d}\n" for i in range(5000))
    assert TEXT_PAGE_CHARS < len(original) <= 64 * 1024, "fixture must span 2 pages"
    handle = store_original(db, org_id=acme, user_id=ALICE, text=original)

    pages: list[str] = []
    offset = 0
    a, b, c = _live(db, org_id=acme, user_id=ALICE)
    with a, b, c:
        while True:
            page = await _retrieve_fn()(_fake_ctx(), handle, offset=offset)
            assert page["offset"] == offset
            assert len(page["text"]) <= TEXT_PAGE_CHARS
            pages.append(page["text"])
            if not page["has_more"]:
                break
            offset += len(page["text"])
            assert len(pages) < 10, "paging did not terminate"

    assert len(pages) == 2
    assert "".join(pages) == original


@needs_db
@pytest.mark.asyncio
async def test_offset_past_the_end_is_empty_not_an_error(db, acme):
    handle = store_original(db, org_id=acme, user_id=ALICE, text="short")

    a, b, c = _live(db, org_id=acme, user_id=ALICE)
    with a, b, c:
        result = await _retrieve_fn()(_fake_ctx(), handle, offset=10_000)

    assert result["text"] == ""
    assert result["has_more"] is False


# ---------------------------------------------------------------------------
# Wiring, mocked.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_retrieve_auth_failure_returns_clean_error():
    """An auth failure is a clean dict, never an HTTP status through MCP."""
    from fastapi import HTTPException

    session_local = MagicMock()
    with (
        patch(
            "app.mcp_app.resolve_active_principal",
            side_effect=HTTPException(status_code=401, detail="Missing X-Vex-Key"),
        ),
        patch("app.mcp_app.SessionLocal", session_local),
    ):
        result = await _retrieve_fn()(_fake_ctx(), "kc_abc")

    assert result["error"] == "unauthorized"
    # The session opens before resolution now (the active-node resolver
    # needs it); the guarantee on a refusal is that it was CLOSED, unused.
    session_local.assert_called_once()
    session_local.return_value.close.assert_called_once()


@pytest.mark.asyncio
async def test_retrieve_without_a_request_context_denies():
    ctx = SimpleNamespace(request_context=SimpleNamespace(request=None))
    session_local = MagicMock()
    with patch("app.mcp_app.SessionLocal", session_local):
        result = await _retrieve_fn()(ctx, "kc_abc")

    assert result["error"] == "unauthorized"
    session_local.assert_not_called()


@pytest.mark.asyncio
async def test_a_cache_failure_is_an_internal_error_not_a_crash():
    """The tool must not propagate an exception through the MCP layer."""
    session = MagicMock()
    with (
        patch(
            "app.mcp_app.resolve_active_principal",
            return_value=("org-x", "agent-y", "free", None),
        ),
        patch("app.mcp_app.resolve_request_user_id", return_value=ALICE),
        patch("app.mcp_app.SessionLocal", MagicMock(return_value=session)),
        patch("app.mcp_app.load_original", side_effect=RuntimeError("db down")),
    ):
        result = await _retrieve_fn()(_fake_ctx(), "kc_abc")

    assert result == {"error": "internal_error"}
    session.close.assert_called_once()


@pytest.mark.asyncio
async def test_schema_advertises_handle_and_offset():
    from mcp.shared.memory import create_connected_server_and_client_session

    mcp = build_mcp_server()
    async with create_connected_server_and_client_session(mcp) as session:
        result = await session.list_tools()

    tool = next(t for t in result.tools if t.name == "klio_retrieve")
    props = tool.inputSchema.get("properties", {})
    assert "handle" in props
    assert "offset" in props
