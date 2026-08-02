"""The `compress` tool: fewer tokens, and a way back when it costs something.

The contract that matters is the pairing. ``lossy=true`` means the caller was
handed something it cannot reconstruct, so it must also be handed a handle
that ``klio_retrieve`` resolves. ``lossy=false`` means there is nothing to
recover and no handle is issued. Any other combination is a broken promise:
a handle with nothing behind it, or lossy output with no way back.

The handle is a separate field. It is never a marker inside the text — models
echo those back, and an echoed marker inside a diff or a source file corrupts
it.

Ownership is exercised against real Postgres, like ``test_retrieve_tool.py``:
a mocked session would prove a handle came back without proving anything about
who can spend it.
"""

import json
import os
import uuid
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from sqlalchemy import create_engine, text

from app.mcp_app import build_mcp_server

DB_URL = os.environ.get(
    "DATABASE_URL",
    "postgresql://agentguard:agentguard_dev@localhost:5432/agentguard",
)

ALICE = "11111111-1111-1111-1111-111111111111"
BOB = "22222222-2222-2222-2222-222222222222"


def _db_reachable() -> bool:
    try:
        engine = create_engine(DB_URL)
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        engine.dispose()
        return True
    except Exception:
        return False


_DB_UP = _db_reachable()
requires_db = pytest.mark.skipif(
    not _DB_UP, reason="Postgres not reachable; the compress handle tests require it"
)


LONG_PYTHON = "import os\n\n\n" + "".join(
    f"def handler_{i}(request, timeout=30):\n"
    f"    payload = request.json()\n"
    f"    result = process(payload, timeout)\n"
    f"    return result\n\n\n"
    for i in range(40)
)

LONG_JSON = json.dumps(
    {
        "memories": [
            {
                "id": f"row-{i}",
                "content": f"memory number {i} about the deployment and the pool",
                "scope": "org",
                "provenance": "EXTRACTED",
                "confidence": None,
            }
            for i in range(40)
        ]
    },
    indent=2,
)

SHORT = "def f(a):\n    return a + 1\n"


def _fake_ctx(headers: dict | None = None):
    request = SimpleNamespace(headers=headers if headers is not None else {})
    return SimpleNamespace(request_context=SimpleNamespace(request=request))


def _tool_fn(name: str):
    server = build_mcp_server()
    tool = server._tool_manager.get_tool(name)
    assert tool is not None, f"tool {name!r} not registered"
    return tool.fn


@pytest.fixture()
def db():
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
    live = MagicMock(wraps=db)
    live.execute = db.execute
    live.commit = db.commit
    live.close = MagicMock()
    return (
        patch(
            "app.mcp_app.resolve_principal_plan",
            return_value=(org_id, "agent-y", "free", None),
        ),
        patch("app.mcp_app.resolve_request_user_id", return_value=user_id),
        patch("app.mcp_app.SessionLocal", MagicMock(return_value=live)),
    )


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


def test_compress_is_a_registered_tool():
    from app.mcp_app import EXPECTED_TOOL_NAMES

    assert "compress" in EXPECTED_TOOL_NAMES
    assert _tool_fn("compress") is not None


# ---------------------------------------------------------------------------
# Shape
# ---------------------------------------------------------------------------


@requires_db
@pytest.mark.asyncio
async def test_lossless_compression_returns_no_handle(db, acme):
    """Nothing was lost, so there is nothing to recover and no row to write."""
    a, b, c = _live(db, org_id=acme, user_id=ALICE)
    with a, b, c:
        result = await _tool_fn("compress")(_fake_ctx(), LONG_JSON)

    assert result["lossy"] is False
    assert "handle" not in result
    assert result["compressed_tokens"] < result["original_tokens"]
    assert result["original_tokens"] > 0


@requires_db
@pytest.mark.asyncio
async def test_lossy_is_refused_unless_the_caller_asks(db, acme):
    """`lossy=false` is the default, and it is a promise, not a preference."""
    a, b, c = _live(db, org_id=acme, user_id=ALICE)
    with a, b, c:
        result = await _tool_fn("compress")(_fake_ctx(), LONG_PYTHON)

    assert result["lossy"] is False
    assert result["compressed"] == LONG_PYTHON
    assert "handle" not in result


@requires_db
@pytest.mark.asyncio
async def test_lossy_compression_returns_a_recoverable_handle(db, acme):
    a, b, c = _live(db, org_id=acme, user_id=ALICE)
    with a, b, c:
        result = await _tool_fn("compress")(_fake_ctx(), LONG_PYTHON, lossy=True)

        assert result["lossy"] is True
        assert result["handle"].startswith("kc_")
        assert result["compressed_tokens"] < result["original_tokens"]
        assert "def handler_0(request, timeout=30):" in result["compressed"]
        assert "request.json()" not in result["compressed"]

        recovered = await _tool_fn("klio_retrieve")(_fake_ctx(), result["handle"])

    assert recovered["text"] == LONG_PYTHON


@requires_db
@pytest.mark.asyncio
async def test_the_handle_is_never_a_marker_inside_the_text(db, acme):
    """An inline marker gets echoed back by a model and corrupts code."""
    a, b, c = _live(db, org_id=acme, user_id=ALICE)
    with a, b, c:
        result = await _tool_fn("compress")(_fake_ctx(), LONG_PYTHON, lossy=True)

    assert result["handle"] not in result["compressed"]
    assert "kc_" not in result["compressed"]


@requires_db
@pytest.mark.asyncio
async def test_content_under_the_floor_is_returned_untouched(db, acme):
    a, b, c = _live(db, org_id=acme, user_id=ALICE)
    with a, b, c:
        result = await _tool_fn("compress")(_fake_ctx(), SHORT, lossy=True)

    assert result["compressed"] == SHORT
    assert result["lossy"] is False
    assert "handle" not in result


# ---------------------------------------------------------------------------
# Ownership
# ---------------------------------------------------------------------------


@requires_db
@pytest.mark.asyncio
async def test_another_user_cannot_spend_the_handle(db, acme):
    """The originals cache is a tenancy boundary. A handle is not a bearer
    token: it resolves for its owner and for nobody else."""
    a, b, c = _live(db, org_id=acme, user_id=ALICE)
    with a, b, c:
        result = await _tool_fn("compress")(_fake_ctx(), LONG_PYTHON, lossy=True)
    handle = result["handle"]

    a, b, c = _live(db, org_id=acme, user_id=BOB)
    with a, b, c:
        stolen = await _tool_fn("klio_retrieve")(_fake_ctx(), handle)

    assert stolen == {"error": "not_found"}


@requires_db
@pytest.mark.asyncio
async def test_another_org_cannot_spend_the_handle(db, acme):
    a, b, c = _live(db, org_id=acme, user_id=ALICE)
    with a, b, c:
        result = await _tool_fn("compress")(_fake_ctx(), LONG_PYTHON, lossy=True)
    handle = result["handle"]

    other_org = f"other_{uuid.uuid4().hex[:8]}"
    db.created_org_ids.add(other_org)
    a, b, c = _live(db, org_id=other_org, user_id=ALICE)
    with a, b, c:
        stolen = await _tool_fn("klio_retrieve")(_fake_ctx(), handle)

    assert stolen == {"error": "not_found"}


# ---------------------------------------------------------------------------
# Fail-open
# ---------------------------------------------------------------------------


@requires_db
@pytest.mark.asyncio
async def test_an_unstorable_original_is_never_returned_lossy(db, acme):
    """If the original cannot be cached there is no way back, so the caller
    must get its content unchanged rather than a lossy result it can never
    recover."""
    from app.compression_cache import OriginalTooLarge

    a, b, c = _live(db, org_id=acme, user_id=ALICE)
    with (
        a,
        b,
        c,
        patch("app.mcp_app.store_original", side_effect=OriginalTooLarge("too big")),
    ):
        result = await _tool_fn("compress")(_fake_ctx(), LONG_PYTHON, lossy=True)

    assert result["compressed"] == LONG_PYTHON
    assert result["lossy"] is False
    assert "handle" not in result


@requires_db
@pytest.mark.asyncio
async def test_a_compressor_that_raises_returns_the_content_unchanged(db, acme):
    a, b, c = _live(db, org_id=acme, user_id=ALICE)
    with (
        a,
        b,
        c,
        patch(
            "app.mcp_app.route_compress", side_effect=RuntimeError("grammar exploded")
        ),
    ):
        result = await _tool_fn("compress")(_fake_ctx(), LONG_PYTHON, lossy=True)

    assert result["compressed"] == LONG_PYTHON
    assert result["lossy"] is False
    assert "handle" not in result


@pytest.mark.asyncio
async def test_auth_failure_returns_a_clean_error():
    from fastapi import HTTPException

    with patch(
        "app.mcp_app.resolve_principal_plan",
        side_effect=HTTPException(status_code=401, detail="bad key"),
    ):
        result = await _tool_fn("compress")(_fake_ctx(), LONG_PYTHON)

    assert result["error"] == "unauthorized"
    assert "401" not in str(result)


# ---------------------------------------------------------------------------
# The correctness floor, through the tool
# ---------------------------------------------------------------------------


@requires_db
@pytest.mark.asyncio
async def test_a_stack_trace_survives_the_tool_byte_for_byte(db, acme):
    trace = (
        "Traceback (most recent call last):\n"
        '  File "/Users/dev/app/main.py", line 42, in handler\n'
        "    return loader.load(path)\n"
        "ValueError: empty\n"
    ) * 12

    a, b, c = _live(db, org_id=acme, user_id=ALICE)
    with a, b, c:
        result = await _tool_fn("compress")(_fake_ctx(), trace, lossy=True)

    assert "/Users/dev/app/main.py" in result["compressed"]
    assert "loader.load(path)" in result["compressed"]
