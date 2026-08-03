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
import uuid
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from sqlalchemy import create_engine, text

from app.mcp_app import build_mcp_server
from shared.testing import database_url, requires_db

DB_URL = database_url()

ALICE = "11111111-1111-1111-1111-111111111111"
BOB = "22222222-2222-2222-2222-222222222222"


needs_db = requires_db("the compress handle tests")


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

LONG_PROSE = (
    "The deployment failed because the database connection pool was exhausted "
    "and we should probably look into raising the pool size at some point. "
    "The error surfaced in the checkout service during the evening peak, and "
    "it is the sort of thing that will happen again if nobody changes it. "
) * 8


def _code_compressor_available() -> bool:
    """Whether the tree-sitter grammars are installed here.

    `shared[code]` is an optional extra. Without it `detect_language` returns
    None, the router never selects the code compressor, `lossy` comes back
    False and no handle is issued -- so every test below that needs a handle
    fails on `assert False is True`, which says nothing about what is wrong.
    An optional dependency silently changing test outcomes is the same disease
    as a tool description silently over-promising: the failure is illegible.
    A named skip is not a pass, but it is the truth.
    """
    from shared.compression.code import detect_language

    return detect_language(LONG_PYTHON) is not None


requires_code_extra = pytest.mark.skipif(
    not _code_compressor_available(),
    reason=(
        "the tree-sitter grammars (shared[code]) are not installed, so code "
        "compression degrades to passthrough and issues no handle. Production "
        "installs them; see services/mcp-server/Dockerfile."
    ),
)


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


@needs_db
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


@needs_db
@pytest.mark.asyncio
async def test_lossy_is_refused_unless_the_caller_asks(db, acme):
    """`lossy=false` is the default, and it is a promise, not a preference."""
    a, b, c = _live(db, org_id=acme, user_id=ALICE)
    with a, b, c:
        result = await _tool_fn("compress")(_fake_ctx(), LONG_PYTHON)

    assert result["lossy"] is False
    assert result["compressed"] == LONG_PYTHON
    assert "handle" not in result


@needs_db
@requires_code_extra
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


@needs_db
@requires_code_extra
@pytest.mark.asyncio
async def test_the_handle_is_never_a_marker_inside_the_text(db, acme):
    """An inline marker gets echoed back by a model and corrupts code."""
    a, b, c = _live(db, org_id=acme, user_id=ALICE)
    with a, b, c:
        result = await _tool_fn("compress")(_fake_ctx(), LONG_PYTHON, lossy=True)

    assert result["handle"] not in result["compressed"]
    assert "kc_" not in result["compressed"]


@needs_db
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


@needs_db
@requires_code_extra
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


@needs_db
@requires_code_extra
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


@needs_db
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


@needs_db
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


@needs_db
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


# ---------------------------------------------------------------------------
# The unattributed caller
# ---------------------------------------------------------------------------


@needs_db
@requires_code_extra
@pytest.mark.asyncio
async def test_an_unattributed_caller_can_redeem_its_handle(db, acme):
    """`resolve_request_user_id` returns None by design, for legacy API keys.

    That caller still calls `compress`, and the lossy compressor has already
    dropped its function bodies by the time it finds out. A handle it cannot
    redeem is precisely the broken promise `_perform_compress` exists to
    prevent — two lines above the code that made it.
    """
    a, b, c = _live(db, org_id=acme, user_id=None)
    with a, b, c:
        result = await _tool_fn("compress")(_fake_ctx(), LONG_PYTHON, lossy=True)
        assert result["lossy"] is True, "the premise of this test"
        assert result["handle"].startswith("kc_")

        recovered = await _tool_fn("klio_retrieve")(_fake_ctx(), result["handle"])

    assert recovered.get("text") == LONG_PYTHON, f"unredeemable handle: {recovered!r}"


@needs_db
@requires_code_extra
@pytest.mark.asyncio
async def test_an_attributed_user_cannot_spend_an_unattributed_handle(db, acme):
    """Making the handle redeemable must not widen who can redeem it."""
    a, b, c = _live(db, org_id=acme, user_id=None)
    with a, b, c:
        result = await _tool_fn("compress")(_fake_ctx(), LONG_PYTHON, lossy=True)
        handle = result["handle"]

    a, b, c = _live(db, org_id=acme, user_id=ALICE)
    with a, b, c:
        recovered = await _tool_fn("klio_retrieve")(_fake_ctx(), handle)

    assert recovered == {"error": "not_found"}


@needs_db
@requires_code_extra
@pytest.mark.asyncio
async def test_an_unattributed_caller_cannot_spend_an_attributed_handle(db, acme):
    a, b, c = _live(db, org_id=acme, user_id=ALICE)
    with a, b, c:
        result = await _tool_fn("compress")(_fake_ctx(), LONG_PYTHON, lossy=True)
        handle = result["handle"]

    a, b, c = _live(db, org_id=acme, user_id=None)
    with a, b, c:
        recovered = await _tool_fn("klio_retrieve")(_fake_ctx(), handle)

    assert recovered == {"error": "not_found"}


# ---------------------------------------------------------------------------
# Legibility of what actually happened
# ---------------------------------------------------------------------------


@needs_db
@pytest.mark.asyncio
async def test_the_response_says_which_compressor_ran(db, acme):
    a, b, c = _live(db, org_id=acme, user_id=ALICE)
    with a, b, c:
        result = await _tool_fn("compress")(_fake_ctx(), LONG_JSON)

    assert result["compressor"] == "structural"
    assert result["selected_compressor"] == "structural"
    assert "note" not in result


@needs_db
@pytest.mark.asyncio
async def test_an_unavailable_compressor_is_named_rather_than_hidden(db, acme):
    """The shipped image omits the prose model on purpose.

    The behaviour is honest -- content back verbatim, `lossy:false`, no
    handle, nothing invented. It is not legible: a caller cannot tell that
    from "already optimal", and the only rational inference it can draw is the
    wrong one. The response now names the compressor the router chose and says
    it was unavailable.
    """
    a, b, c = _live(db, org_id=acme, user_id=ALICE)
    with (
        a,
        b,
        c,
        patch("shared.compression.router.prose_model_available", lambda: False),
    ):
        result = await _tool_fn("compress")(_fake_ctx(), LONG_PROSE, lossy=True)

    assert result["compressed"] == LONG_PROSE
    assert result["lossy"] is False
    assert "handle" not in result
    assert result["compressor"] == "passthrough"
    assert result["selected_compressor"] == "prose"
    assert "not available" in result["note"]


@needs_db
@pytest.mark.asyncio
async def test_content_below_the_floor_is_not_reported_as_unavailable(db, acme):
    a, b, c = _live(db, org_id=acme, user_id=ALICE)
    with a, b, c:
        result = await _tool_fn("compress")(_fake_ctx(), SHORT, lossy=True)

    assert result["compressor"] == "passthrough"
    assert result["selected_compressor"] == "passthrough"
    assert "note" not in result
