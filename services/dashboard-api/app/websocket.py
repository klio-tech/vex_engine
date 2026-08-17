"""WebSocket connection manager and Redis Stream consumer for real-time
dashboard updates.

Subscribes to the ``executions.stored`` Redis Stream and delivers each
execution event to the connections belonging to that event's org -- and to
nobody else.

TENANCY
-------
Connections are bucketed by ``org_id``, resolved at handshake time from a
signed token (see ``app.ws_auth``). ``broadcast`` takes an ``org_id`` and can
only reach that bucket; there is no call shape that fans a message out to
every socket. ``stream_updates`` routes on the ``org_id`` inside the payload,
and a payload without one is dropped rather than sent anywhere -- an
unattributable event is not a licence to send it to everyone.

This replaces a single flat ``list[WebSocket]`` that every stream message was
broadcast to verbatim, which meant any connected client saw every
organisation's execution metadata.
"""

import asyncio
import json
import logging

from fastapi import WebSocket, WebSocketDisconnect

from app.ws_auth import WS_SUBPROTOCOL, WsAuthError, authenticate

logger = logging.getLogger("agentguard.dashboard-api.websocket")

#: Close code sent on a failed handshake. RFC 6455 1008 "policy violation" is
#: the code the WebSocket spec reserves for an authorisation refusal.
POLICY_VIOLATION = 1008


class ConnectionManager:
    """Holds active WebSocket connections, partitioned by organisation.

    ``active_connections`` maps ``org_id`` to the set of sockets authorised
    for that org. Sets, not lists: a socket appears at most once, and removal
    on disconnect is O(1) rather than a scan. Empty buckets are deleted so the
    mapping tracks live orgs instead of every org that has ever connected.
    """

    def __init__(self) -> None:
        self.active_connections: dict[str, set[WebSocket]] = {}

    @property
    def connection_count(self) -> int:
        """Total sockets held across every org bucket."""
        return sum(len(sockets) for sockets in self.active_connections.values())

    def connections_for(self, org_id: str) -> set[WebSocket]:
        """Return a copy of ``org_id``'s bucket (empty if the org has none)."""
        return set(self.active_connections.get(org_id, ()))

    async def connect(self, websocket: WebSocket, org_id: str) -> None:
        """Accept ``websocket`` and file it under ``org_id``.

        The caller MUST have authenticated first: reaching this method is what
        accepts the socket, so an unauthenticated call is the vulnerability
        this class exists to close.

        Args:
            websocket: The pending connection.
            org_id: The organisation the handshake proved, from
                ``ws_auth.authenticate``.
        """
        await websocket.accept(subprotocol=WS_SUBPROTOCOL)
        self.active_connections.setdefault(org_id, set()).add(websocket)
        logger.info(
            "WebSocket client connected for org %s. Org connections: %d, total: %d",
            org_id,
            len(self.active_connections[org_id]),
            self.connection_count,
        )

    def disconnect(self, websocket: WebSocket, org_id: str) -> None:
        """Drop ``websocket`` from ``org_id``'s bucket, pruning it when empty.

        Idempotent: a socket that errored during broadcast and was already
        removed may be disconnected again by the endpoint's cleanup path.
        """
        sockets = self.active_connections.get(org_id)

        if sockets is None:
            return

        sockets.discard(websocket)

        # Prune, or the mapping grows one permanent entry per org that has
        # ever connected and never shrinks.
        if not sockets:
            del self.active_connections[org_id]

        logger.info(
            "WebSocket client disconnected from org %s. Org connections: %d, total: %d",
            org_id,
            len(self.active_connections.get(org_id, ())),
            self.connection_count,
        )

    async def broadcast(self, org_id: str, message: dict) -> None:
        """Send ``message`` to every connection authorised for ``org_id``.

        Reaches exactly one bucket. There is no all-orgs variant on purpose.

        Disconnected or erroring clients are dropped rather than allowed to
        interrupt delivery to the rest of the bucket.

        Args:
            org_id: The organisation the message belongs to.
            message: JSON-serialisable payload.
        """
        sockets = self.active_connections.get(org_id)

        if not sockets:
            return

        failed: list[WebSocket] = []

        # Iterate a snapshot: a send may fail and mutate the bucket underneath.
        for connection in tuple(sockets):
            try:
                await connection.send_json(message)
            except Exception:
                failed.append(connection)

        for connection in failed:
            self.disconnect(connection, org_id)


manager = ConnectionManager()


async def stream_updates(redis_url: str) -> None:
    """Read ``executions.stored`` and route each event to its own org.

    Connects using XREAD (not a consumer group) so multiple dashboard-api
    instances each receive all messages. Uses ``$`` as the initial ID to only
    receive new messages from the point of connection.

    Every payload is routed on its ``org_id``. A payload missing that field is
    logged and dropped: it cannot be attributed, and the previous behaviour --
    send it to everyone -- is precisely the leak.

    Args:
        redis_url: Redis connection URL.
    """
    import redis.asyncio as aioredis

    redis_client = aioredis.from_url(redis_url, decode_responses=True)
    last_id = "$"
    stream_key = "executions.stored"

    logger.info("Starting Redis Stream listener on '%s'", stream_key)

    while True:
        try:
            messages = await redis_client.xread(
                streams={stream_key: last_id},
                count=10,
                block=5000,
            )

            if messages:
                for _stream, entries in messages:
                    for msg_id, data in entries:
                        last_id = msg_id

                        try:
                            payload = json.loads(data.get("data", "{}"))
                        except (json.JSONDecodeError, TypeError):
                            logger.warning(
                                "Skipping malformed message %s",
                                msg_id,
                            )
                            continue

                        org_id = (
                            payload.get("org_id") if isinstance(payload, dict) else None
                        )

                        if not org_id:
                            logger.warning(
                                "Dropping message %s: no org_id, so it cannot "
                                "be attributed to a tenant",
                                msg_id,
                            )
                            continue

                        await manager.broadcast(
                            org_id,
                            {
                                "type": "execution.new",
                                "data": payload,
                            },
                        )
        except asyncio.CancelledError:
            logger.info("Stream listener cancelled, shutting down")
            await redis_client.aclose()
            return
        except Exception as exc:
            logger.error("Stream read error: %s", exc, exc_info=True)
            await asyncio.sleep(1)


async def websocket_endpoint(websocket: WebSocket) -> None:
    """Handle a single WebSocket connection lifecycle.

    Authenticates the handshake BEFORE accepting. A socket that fails is
    closed with 1008 and never accepted, so it is never added to any bucket
    and cannot receive a single event. All failures look identical from
    outside -- same code, no reason string -- so the endpoint reveals nothing
    about why a credential was rejected.

    On success the connection is filed under the token's ``org_id`` and is
    only ever sent that org's events.
    """
    from app.config import ws_token_secret

    try:
        org_id = authenticate(websocket, ws_token_secret())
    except WsAuthError as exc:
        logger.warning("Rejecting WebSocket handshake: %s", exc.reason)
        await websocket.close(code=POLICY_VIOLATION)
        return

    await manager.connect(websocket, org_id)

    try:
        while True:
            # Keep the connection alive by waiting for client messages.
            # Clients may send ping/pong or keepalive messages.
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket, org_id)
    except Exception:
        manager.disconnect(websocket, org_id)
