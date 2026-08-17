"""Tests for the Vex Dashboard API WebSocket service.

The tenancy assertions here are adversarial on purpose. The suite this
replaced asserted the opposite contract -- ``test_broadcast_sends_to_all`` and
``test_websocket_multiple_clients`` both checked that every connected socket
received every message, which is exactly the leak. Any future change that
restores an all-orgs fan-out, or accepts a socket before checking its
credential, has to make one of these fail first.
"""

import asyncio
import time
from unittest.mock import AsyncMock, patch

import pytest
from app.config import WS_TOKEN_SECRET_ENV
from app.main import create_app
from app.websocket import ConnectionManager, manager
from app.ws_auth import (
    WS_SUBPROTOCOL,
    WsAuthError,
    authenticate,
    mint_ws_token,
    verify_ws_token,
)
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from .ws_auth_fixtures import TEST_VECTOR

SECRET = "unit-test-signing-secret"
ORG_A = "org-aaaa-1111"
ORG_B = "org-bbbb-2222"


def token_for(org_id: str, *, ttl_s: int = 120, secret: str = SECRET) -> str:
    """Mint a live token for ``org_id``."""
    return mint_ws_token(org_id, secret, expires_at=int(time.time()) + ttl_s)


@pytest.fixture(autouse=True)
def signing_secret(monkeypatch):
    """Configure the shared secret for every test in this module."""
    monkeypatch.setenv(WS_TOKEN_SECRET_ENV, SECRET)


@pytest.fixture(autouse=True)
def clean_manager():
    """Leave the module-level manager empty between tests."""
    manager.active_connections = {}
    yield
    manager.active_connections = {}


@pytest.fixture
def app():
    """Create a test app with the stream listener disabled."""
    with patch("app.main.stream_updates", new_callable=AsyncMock):
        test_app = create_app()
        yield test_app


@pytest.fixture
def client(app):
    return TestClient(app)


def connect_as(client, org_id: str, **kwargs):
    """Open an authenticated ``/ws`` connection for ``org_id``."""
    return client.websocket_connect(
        "/ws",
        subprotocols=[WS_SUBPROTOCOL, token_for(org_id, **kwargs)],
    )


def broadcast_now(org_id: str, message: dict) -> None:
    """Run a broadcast from inside the TestClient's synchronous context.

    An explicit short-lived loop rather than ``asyncio.get_event_loop()``:
    that helper stopped auto-creating a loop in the main thread as of Python
    3.12 and raises there, which is why the suite this replaced could only run
    on 3.11.
    """
    loop = asyncio.new_event_loop()

    try:
        loop.run_until_complete(manager.broadcast(org_id, message))
    finally:
        loop.close()


def test_health_check(client):
    """GET /health returns healthy status."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


# ---------------------------------------------------------------------------
# Handshake authentication
# ---------------------------------------------------------------------------


class TestHandshakeAuthentication:
    """A socket must prove an org before it is accepted."""

    def test_unauthenticated_connection_is_rejected_without_being_accepted(
        self, client
    ):
        """No credential at all: the handshake never completes.

        The old endpoint called ``accept()`` unconditionally. If that ever
        comes back this raises nothing and the test fails.
        """
        with pytest.raises(WebSocketDisconnect):
            with client.websocket_connect("/ws"):
                pass

        assert manager.active_connections == {}

    def test_forged_signature_is_rejected(self, client):
        """A token signed with the wrong secret buys nothing."""
        forged = mint_ws_token(
            ORG_A, "not-the-real-secret", expires_at=int(time.time()) + 120
        )

        with pytest.raises(WebSocketDisconnect):
            with client.websocket_connect("/ws", subprotocols=[WS_SUBPROTOCOL, forged]):
                pass

        assert manager.active_connections == {}

    def test_expired_token_is_rejected(self, client):
        """An old token is not a standing invitation."""
        with pytest.raises(WebSocketDisconnect):
            with client.websocket_connect(
                "/ws", subprotocols=[WS_SUBPROTOCOL, token_for(ORG_A, ttl_s=-600)]
            ):
                pass

        assert manager.active_connections == {}

    def test_missing_server_secret_rejects_valid_looking_tokens(
        self, client, monkeypatch
    ):
        """Unconfigured must mean "nobody", never "everybody"."""
        good = token_for(ORG_A)
        monkeypatch.delenv(WS_TOKEN_SECRET_ENV, raising=False)

        with pytest.raises(WebSocketDisconnect):
            with client.websocket_connect("/ws", subprotocols=[WS_SUBPROTOCOL, good]):
                pass

        assert manager.active_connections == {}

    def test_valid_token_connects_and_negotiates_the_subprotocol(self, client):
        """A good token is accepted, and the server echoes only the
        subprotocol name -- never the credential."""
        with connect_as(client, ORG_A) as ws:
            assert ws.accepted_subprotocol == WS_SUBPROTOCOL
            assert set(manager.active_connections) == {ORG_A}

            broadcast_now(ORG_A, {"type": "execution.new", "data": {"org_id": ORG_A}})
            assert ws.receive_json()["data"]["org_id"] == ORG_A

    def test_query_string_token_is_accepted_as_a_fallback(self, client):
        """Non-browser clients that cannot negotiate a subprotocol still work."""
        with client.websocket_connect(f"/ws?token={token_for(ORG_A)}") as ws:
            assert set(manager.active_connections) == {ORG_A}

            broadcast_now(ORG_A, {"type": "execution.new", "data": {"n": 1}})
            assert ws.receive_json()["data"]["n"] == 1


# ---------------------------------------------------------------------------
# Tenancy
# ---------------------------------------------------------------------------


class TestOrgIsolation:
    """The property the old suite asserted the negation of."""

    def test_org_a_never_receives_an_org_b_payload(self, client):
        """Two authenticated clients, different orgs, one broadcast each."""
        with connect_as(client, ORG_A) as ws_a:
            with connect_as(client, ORG_B) as ws_b:
                assert set(manager.active_connections) == {ORG_A, ORG_B}

                broadcast_now(
                    ORG_B,
                    {"type": "execution.new", "data": {"execution_id": "b-only"}},
                )

                # B gets it.
                assert ws_b.receive_json()["data"]["execution_id"] == "b-only"

                # A must not. Prove it by sending A its own message afterwards:
                # if B's payload had reached A it would be first in the queue.
                broadcast_now(
                    ORG_A,
                    {"type": "execution.new", "data": {"execution_id": "a-only"}},
                )
                assert ws_a.receive_json()["data"]["execution_id"] == "a-only"

    def test_two_clients_in_the_same_org_both_receive(self, client):
        """Scoping must not break the legitimate multi-tab case.

        This is what ``test_websocket_multiple_clients`` should always have
        asserted: same tenant, not same process.
        """
        with connect_as(client, ORG_A) as ws1:
            with connect_as(client, ORG_A) as ws2:
                broadcast_now(
                    ORG_A,
                    {"type": "execution.new", "data": {"execution_id": "multi-456"}},
                )

                assert ws1.receive_json()["data"]["execution_id"] == "multi-456"
                assert ws2.receive_json()["data"]["execution_id"] == "multi-456"

    def test_disconnect_cleans_up_the_bucket(self, client):
        """The org's entry disappears once its last socket goes."""
        with connect_as(client, ORG_A):
            with connect_as(client, ORG_B):
                assert set(manager.active_connections) == {ORG_A, ORG_B}

            # B's only socket is gone; the key must go with it, or the mapping
            # grows one dead entry per org forever.
            assert ORG_B not in manager.active_connections

        assert manager.active_connections == {}


# ---------------------------------------------------------------------------
# Token format
# ---------------------------------------------------------------------------


class TestWsToken:
    """Unit tests for mint/verify, including the cross-language vector."""

    def test_round_trip(self):
        token = mint_ws_token(ORG_A, SECRET, expires_at=int(time.time()) + 60)
        assert verify_ws_token(token, SECRET) == ORG_A

    def test_shared_vector_matches_the_typescript_minter(self):
        """Pinned byte-for-byte against the Dashboard repo's ws-token test.

        If this fails, the two implementations have diverged and tokens minted
        by the web tier will stop verifying here.
        """
        minted = mint_ws_token(
            TEST_VECTOR["org_id"],
            TEST_VECTOR["secret"],
            expires_at=TEST_VECTOR["exp"],
        )
        assert minted == TEST_VECTOR["token"]

        assert (
            verify_ws_token(
                TEST_VECTOR["token"],
                TEST_VECTOR["secret"],
                now=TEST_VECTOR["exp"] - 10,
            )
            == TEST_VECTOR["org_id"]
        )

    def test_org_id_is_not_attacker_editable(self):
        """Rewriting the org field invalidates the signature."""
        token = mint_ws_token(ORG_A, SECRET, expires_at=int(time.time()) + 60)
        version, _org, exp, sig = token.split(".")
        tampered = ".".join([version, "b3JnLWJiYmItMjIyMg", exp, sig])

        with pytest.raises(WsAuthError):
            verify_ws_token(tampered, SECRET)

    def test_expiry_cannot_be_extended(self):
        token = mint_ws_token(ORG_A, SECRET, expires_at=int(time.time()) + 60)
        version, org, _exp, sig = token.split(".")

        with pytest.raises(WsAuthError):
            verify_ws_token(".".join([version, org, "99999999999", sig]), SECRET)

    def test_absurd_lifetime_is_refused_even_when_correctly_signed(self):
        """A ceiling on lifetime bounds the damage of a careless minter."""
        far_future = int(time.time()) + 365 * 24 * 3600
        token = mint_ws_token(ORG_A, SECRET, expires_at=far_future)

        with pytest.raises(WsAuthError):
            verify_ws_token(token, SECRET)

    def test_empty_secret_verifies_nothing(self):
        token = mint_ws_token(ORG_A, SECRET, expires_at=int(time.time()) + 60)

        with pytest.raises(WsAuthError):
            verify_ws_token(token, "")

    @pytest.mark.parametrize(
        "token",
        ["", "garbage", "v1.a.b", "v2.abc.123.sig", "v1.abc.notanint.sig"],
    )
    def test_malformed_tokens_are_refused(self, token):
        with pytest.raises(WsAuthError):
            verify_ws_token(token, SECRET)

    def test_authenticate_prefers_the_subprotocol_over_the_query_string(self):
        """Query strings land in access logs; the header wins when both exist."""
        good = mint_ws_token(ORG_A, SECRET, expires_at=int(time.time()) + 60)

        class FakeWebSocket:
            headers = {"sec-websocket-protocol": f"{WS_SUBPROTOCOL}, {good}"}
            query_params = {"token": "garbage"}

        assert authenticate(FakeWebSocket(), SECRET) == ORG_A


# ---------------------------------------------------------------------------
# ConnectionManager
# ---------------------------------------------------------------------------


class TestConnectionManager:
    """Unit tests for the ConnectionManager class."""

    def test_initial_state(self):
        mgr = ConnectionManager()
        assert mgr.active_connections == {}
        assert mgr.connection_count == 0

    @pytest.mark.asyncio
    async def test_connect_files_socket_under_its_org(self):
        mgr = ConnectionManager()
        mock_ws = AsyncMock()
        await mgr.connect(mock_ws, ORG_A)

        assert mgr.active_connections == {ORG_A: {mock_ws}}
        mock_ws.accept.assert_called_once_with(subprotocol=WS_SUBPROTOCOL)

    @pytest.mark.asyncio
    async def test_disconnect_removes_socket_and_prunes_empty_bucket(self):
        mgr = ConnectionManager()
        mock_ws = AsyncMock()
        await mgr.connect(mock_ws, ORG_A)
        mgr.disconnect(mock_ws, ORG_A)

        assert mgr.active_connections == {}

    @pytest.mark.asyncio
    async def test_disconnect_keeps_the_bucket_while_siblings_remain(self):
        mgr = ConnectionManager()
        ws1, ws2 = AsyncMock(), AsyncMock()
        await mgr.connect(ws1, ORG_A)
        await mgr.connect(ws2, ORG_A)
        mgr.disconnect(ws1, ORG_A)

        assert mgr.active_connections == {ORG_A: {ws2}}

    @pytest.mark.asyncio
    async def test_disconnect_is_idempotent(self):
        """The endpoint may disconnect a socket a failed broadcast already
        dropped. That must not raise."""
        mgr = ConnectionManager()
        mock_ws = AsyncMock()
        await mgr.connect(mock_ws, ORG_A)
        mgr.disconnect(mock_ws, ORG_A)
        mgr.disconnect(mock_ws, ORG_A)

        assert mgr.active_connections == {}

    @pytest.mark.asyncio
    async def test_broadcast_reaches_only_the_named_org(self):
        mgr = ConnectionManager()
        ws_a, ws_b = AsyncMock(), AsyncMock()
        await mgr.connect(ws_a, ORG_A)
        await mgr.connect(ws_b, ORG_B)

        message = {"type": "test", "data": "for-a"}
        await mgr.broadcast(ORG_A, message)

        ws_a.send_json.assert_called_once_with(message)
        ws_b.send_json.assert_not_called()

    @pytest.mark.asyncio
    async def test_broadcast_reaches_every_socket_in_that_org(self):
        mgr = ConnectionManager()
        ws1, ws2 = AsyncMock(), AsyncMock()
        await mgr.connect(ws1, ORG_A)
        await mgr.connect(ws2, ORG_A)

        message = {"type": "test", "data": "hello"}
        await mgr.broadcast(ORG_A, message)

        ws1.send_json.assert_called_once_with(message)
        ws2.send_json.assert_called_once_with(message)

    @pytest.mark.asyncio
    async def test_broadcast_cleans_up_failed_connections(self):
        mgr = ConnectionManager()
        good_ws, bad_ws = AsyncMock(), AsyncMock()
        bad_ws.send_json.side_effect = Exception("connection lost")

        await mgr.connect(good_ws, ORG_A)
        await mgr.connect(bad_ws, ORG_A)
        assert mgr.connection_count == 2

        await mgr.broadcast(ORG_A, {"type": "test"})

        assert mgr.active_connections == {ORG_A: {good_ws}}

    @pytest.mark.asyncio
    async def test_broadcast_to_an_org_with_no_connections(self):
        mgr = ConnectionManager()
        # Should not raise, and must not create an empty bucket.
        await mgr.broadcast(ORG_A, {"type": "test"})
        assert mgr.active_connections == {}


# ---------------------------------------------------------------------------
# stream_updates
# ---------------------------------------------------------------------------


def _redis_returning(entries):
    """Build a mock Redis whose first xread yields ``entries``, then cancels."""
    mock_redis = AsyncMock()
    call_count = 0

    async def mock_xread(**kwargs):
        nonlocal call_count
        call_count += 1

        if call_count == 1:
            return [("executions.stored", entries)]

        raise asyncio.CancelledError()

    mock_redis.xread = mock_xread
    mock_redis.aclose = AsyncMock()
    return mock_redis


class TestStreamUpdates:
    """Unit tests for the stream_updates background task."""

    @pytest.mark.asyncio
    async def test_routes_a_message_to_its_own_org_only(self):
        """The end-to-end shape of the fix: stream payload in, one org out."""
        import json

        from app.websocket import stream_updates

        mock_redis = _redis_returning(
            [
                (
                    "1234-0",
                    {
                        "data": json.dumps(
                            {
                                "org_id": ORG_A,
                                "execution_id": "e1",
                                "action": "pass",
                            }
                        )
                    },
                )
            ]
        )

        ws_a, ws_b = AsyncMock(), AsyncMock()
        manager.active_connections = {ORG_A: {ws_a}, ORG_B: {ws_b}}

        with patch("redis.asyncio.from_url", return_value=mock_redis):
            await stream_updates("redis://localhost:6379")

        ws_a.send_json.assert_called_once()
        sent = ws_a.send_json.call_args[0][0]
        assert sent["type"] == "execution.new"
        assert sent["data"]["execution_id"] == "e1"

        ws_b.send_json.assert_not_called()

    @pytest.mark.asyncio
    async def test_payload_without_org_id_is_dropped(self):
        """Unattributable is not a licence to fan out.

        This is the failure mode that made the old code a leak: an
        org-less payload went to every socket.
        """
        import json

        from app.websocket import stream_updates

        mock_redis = _redis_returning(
            [("1234-0", {"data": json.dumps({"execution_id": "orphan"})})]
        )

        ws_a, ws_b = AsyncMock(), AsyncMock()
        manager.active_connections = {ORG_A: {ws_a}, ORG_B: {ws_b}}

        with patch("redis.asyncio.from_url", return_value=mock_redis):
            await stream_updates("redis://localhost:6379")

        ws_a.send_json.assert_not_called()
        ws_b.send_json.assert_not_called()

    @pytest.mark.asyncio
    async def test_payload_with_blank_org_id_is_dropped(self):
        """An empty string must not become a bucket key."""
        import json

        from app.websocket import stream_updates

        mock_redis = _redis_returning(
            [("1234-0", {"data": json.dumps({"org_id": "", "execution_id": "x"})})]
        )

        ws_a = AsyncMock()
        manager.active_connections = {ORG_A: {ws_a}}

        with patch("redis.asyncio.from_url", return_value=mock_redis):
            await stream_updates("redis://localhost:6379")

        ws_a.send_json.assert_not_called()
        assert "" not in manager.active_connections

    @pytest.mark.asyncio
    async def test_skips_malformed_json(self):
        """Malformed JSON in stream messages should be skipped without crashing."""
        from app.websocket import stream_updates

        mock_redis = _redis_returning([("1234-0", {"data": "not valid json{{"})])

        ws_a = AsyncMock()
        manager.active_connections = {ORG_A: {ws_a}}

        with patch("redis.asyncio.from_url", return_value=mock_redis):
            await stream_updates("redis://localhost:6379")

        ws_a.send_json.assert_not_called()

    @pytest.mark.asyncio
    async def test_handles_connection_error(self):
        """Redis connection errors in stream_updates should not crash the task."""
        from app.websocket import stream_updates

        mock_redis = AsyncMock()
        call_count = 0

        async def mock_xread(**kwargs):
            nonlocal call_count
            call_count += 1

            if call_count == 1:
                raise ConnectionError("Redis down")

            raise asyncio.CancelledError()

        mock_redis.xread = mock_xread
        mock_redis.aclose = AsyncMock()

        with patch("redis.asyncio.from_url", return_value=mock_redis):
            with patch("asyncio.sleep", new_callable=AsyncMock):
                await stream_updates("redis://localhost:6379")
