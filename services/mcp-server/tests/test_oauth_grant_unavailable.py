"""Pool exhaustion must surface as a retryable 503, not `403 not_connected`.

``GrantResolver.resolve`` wrapped ``_lookup`` in ``except Exception: return
None``, and the middleware maps ``None`` to ``403 not_connected`` — a message
telling a correctly-connected user to go re-authorize their workspace at
app.klio.tech. The Supabase engine is a 5+10 pool, so at 16 concurrent Bearer
resolutions the 16th queued for ``pool_timeout`` and then raised
``sqlalchemy.exc.TimeoutError``, which became exactly that message.

The distinction already exists one rung up the ladder: an unreachable JWKS
endpoint raises ``VerifierUnavailable`` and the middleware answers 503 +
Retry-After rather than 401. Grant lookup now has the same shape.

403 stays for a grant that genuinely is not there — that is authorization
working, and it must not be softened into "try again".
"""

from unittest.mock import MagicMock, patch

import pytest
import sqlalchemy.exc
from app import oauth_middleware
from app.oauth import GrantResolver, GrantStoreUnavailable, OAuthIdentity


def _identity(user_id="user-1", client_id="client-1"):
    return OAuthIdentity(user_id=user_id, client_id=client_id)


def _resolver():
    return GrantResolver(supabase_engine=MagicMock(), tsdb_engine=MagicMock())


def _pool_timeout():
    return sqlalchemy.exc.TimeoutError(
        "QueuePool limit of size 5 overflow 10 reached, connection timed out, "
        "timeout 30.00"
    )


# ── The resolver's own contract ───────────────────────────────────────


def test_a_genuinely_absent_grant_still_resolves_to_none():
    """Positive control: 403 must survive. This is authorization working."""
    resolver = _resolver()
    resolver._lookup = lambda identity: None
    assert resolver.resolve(_identity()) is None


def test_a_real_principal_is_still_returned():
    """Positive control: the happy path is untouched."""
    resolver = _resolver()
    sentinel = object()
    resolver._lookup = lambda identity: sentinel
    assert resolver.resolve(_identity()) is sentinel


def test_pool_exhaustion_raises_rather_than_denying():
    resolver = _resolver()

    def _boom(identity):
        raise _pool_timeout()

    resolver._lookup = _boom
    with pytest.raises(GrantStoreUnavailable):
        resolver.resolve(_identity())


def test_a_dropped_connection_raises_rather_than_denying():
    resolver = _resolver()

    def _boom(identity):
        raise sqlalchemy.exc.OperationalError("SELECT 1", {}, Exception("no route"))

    resolver._lookup = _boom
    with pytest.raises(GrantStoreUnavailable):
        resolver.resolve(_identity())


def test_an_outage_is_never_cached_as_a_denial():
    """The bug that made it sticky: a cached ``None`` outlives the outage."""
    resolver = _resolver()
    calls = []

    def _flaky(identity):
        calls.append(1)
        if len(calls) == 1:
            raise _pool_timeout()
        return "principal"

    resolver._lookup = _flaky
    with pytest.raises(GrantStoreUnavailable):
        resolver.resolve(_identity())
    # Immediately afterwards, inside the cache TTL, the recovered store wins.
    assert resolver.resolve(_identity()) == "principal"


def test_a_non_database_bug_still_fails_closed():
    """An unexpected error is not evidence the store is down; deny."""
    resolver = _resolver()

    def _boom(identity):
        raise ValueError("a bug in the lookup")

    resolver._lookup = _boom
    assert resolver.resolve(_identity()) is None


# ── What the client actually receives ─────────────────────────────────


class _Recorder:
    """Collects the ASGI messages a response sends."""

    def __init__(self):
        self.messages = []

    async def __call__(self, message):
        self.messages.append(message)

    @property
    def status(self):
        return next(
            m["status"] for m in self.messages if m["type"] == "http.response.start"
        )

    @property
    def headers(self):
        start = next(m for m in self.messages if m["type"] == "http.response.start")
        return {k.decode(): v.decode() for k, v in start["headers"]}

    @property
    def body(self):
        return b"".join(
            m.get("body", b"")
            for m in self.messages
            if m["type"] == "http.response.body"
        ).decode()


def _scope():
    return {
        "type": "http",
        "path": "/mcp",
        "method": "POST",
        "headers": [(b"authorization", b"Bearer sometoken")],
        "query_string": b"",
    }


async def _receive():
    return {"type": "http.request", "body": b"", "more_body": False}


async def _drive(resolve_side_effect):
    """Run the middleware with a valid token and the given grant behaviour."""
    inner_called = []

    async def _inner(scope, receive, send):
        inner_called.append(True)

    mw = oauth_middleware.OAuthLadderMiddleware(_inner, mcp_path="/mcp")
    send = _Recorder()

    with (
        patch.object(oauth_middleware, "_verify_token", lambda token: _identity()),
        patch.object(oauth_middleware, "_resolve_grant", resolve_side_effect),
    ):
        await mw(_scope(), _receive, send)
    return send, inner_called


@pytest.mark.asyncio
async def test_the_client_gets_503_and_retry_after_when_the_store_is_down():
    def _boom(identity):
        raise GrantStoreUnavailable("pool exhausted")

    send, inner = await _drive(_boom)

    assert send.status == 503, f"got {send.status}, body={send.body}"
    assert "retry-after" in send.headers
    assert "not_connected" not in send.body
    assert not inner, "the request must not reach the tool layer"


@pytest.mark.asyncio
async def test_the_client_still_gets_403_not_connected_for_a_missing_grant():
    """Positive control: the 403 path is unchanged."""
    send, inner = await _drive(lambda identity: None)

    assert send.status == 403
    assert "not_connected" in send.body
    assert not inner
