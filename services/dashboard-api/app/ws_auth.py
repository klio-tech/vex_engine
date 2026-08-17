"""Handshake authentication for the dashboard WebSocket.

The ``/ws`` feed carries execution metadata -- ``org_id``, ``agent_id``,
``session_id``, execution and parent ids, latency, token counts, cost
estimates and the S3 key of the trace payload. Until this module existed the
endpoint accepted every socket that reached it and fanned every org's events
out to all of them.

WHY A SIGNED TOKEN AND NOT ``X-Vex-Key``
----------------------------------------
The obvious move is to reuse ``KeyValidator`` (``shared.auth``) the way
sync-gateway and ingestion-api do, and have the browser present an
``X-Vex-Key``. It does not work here, for two independent reasons:

1. A browser cannot set request headers on a WebSocket handshake. The
   credential would have to travel in the URL or the subprotocol, i.e. it
   would be readable by anything with the page, any browser extension, and
   every proxy access log in front of the service. ``X-Vex-Key`` is a
   long-lived org credential with ``ingest``/``verify``/``read``/``memory``
   scopes. It must never reach a browser.

2. There is no key to send. The dashboard web tier does not hold an
   ``X-Vex-Key``; it reads the engine database directly through its own pool
   and serves many orgs from one deployment. A single key belongs to exactly
   one org, and stored keys are SHA-256 hashes with a show-once plaintext, so
   the web tier cannot obtain the caller's key even in principle.

So the credential is a short-lived, single-purpose, org-scoped bearer token,
minted by the dashboard web tier *after* it has authenticated the session and
asserted account membership (``resolveOrgId`` in the Dashboard repo), and
verified here with a shared HMAC secret. It grants exactly one thing: read
this org's execution feed, for the next few minutes. It is not a key, it
cannot be exchanged for one, and it expires on its own.

Verification is a keyed hash of two fields. It needs no database and no Redis,
which matters: this service's Redis link is currently unreliable, and an auth
path that depends on a flaky backend either fails open (a leak) or takes the
feed down.

TOKEN FORMAT
------------
``v1.<b64url(org_id)>.<exp>.<b64url(hmac_sha256(secret, "v1.<b64url(org_id)>.<exp>"))>``

``exp`` is an integer Unix timestamp. Base64url is unpadded, so every
character of the token is an RFC 7230 ``token`` character and the whole thing
is legal in a ``Sec-WebSocket-Protocol`` header. The Dashboard repo mints the
identical string in TypeScript; both suites pin the same fixed vector
(see ``TEST_VECTOR`` below) so the two implementations cannot drift apart
silently.

TRANSPORT
---------
Preferred: ``Sec-WebSocket-Protocol: vex.ws.v1, <token>`` -- a request header,
so it stays out of access logs, out of ``Referer``, and out of browser
history, while still being settable from browser JavaScript via the
``protocols`` argument of the ``WebSocket`` constructor.

Accepted as a fallback: ``?token=<token>`` in the query string, for clients
that cannot negotiate a subprotocol (curl, load probes). Query strings are
recorded verbatim by Railway's edge and by most reverse proxies, so this path
logs a warning on every use and is the reason the token's lifetime is minutes
rather than hours.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import logging
import time

logger = logging.getLogger("agentguard.dashboard-api.ws_auth")

#: Token version prefix. Bump when the signed payload's shape changes so an
#: old token cannot be reinterpreted under new rules.
TOKEN_VERSION = "v1"

#: The subprotocol the server negotiates. The client offers this *and* the
#: token; the server echoes back only this one, never the credential.
WS_SUBPROTOCOL = "vex.ws.v1"

#: Query parameter checked when no subprotocol credential is present.
TOKEN_QUERY_PARAM = "token"

#: Refuse tokens minted more than this far in the future. Without a ceiling a
#: leaked secret (or a careless minter) could produce a token good for years;
#: with it, the blast radius of any single token is bounded no matter what the
#: minter asked for.
MAX_TOKEN_LIFETIME_S = 15 * 60

#: Tolerance for clock skew between the minting web tier and this service.
CLOCK_SKEW_TOLERANCE_S = 30

#: Fixed vector mirrored by the Dashboard repo's ws-token test. Both sides
#: assert this exact string. If either implementation changes its encoding,
#: padding, separator or field order, one of the two suites goes red before the
#: mismatch can reach production as "the charts stopped updating".
TEST_VECTOR = {
    "secret": "test-secret-do-not-use",
    "org_id": "11111111-2222-3333-4444-555555555555",
    "exp": 1700000000,
    "token": (
        "v1.MTExMTExMTEtMjIyMi0zMzMzLTQ0NDQtNTU1NTU1NTU1NTU1.1700000000"
        ".ixLU_-TiTdxIbjLicOSP5PsoWzB6qPN406MI7yiLlq8"
    ),
}


class WsAuthError(Exception):
    """Raised when a WebSocket handshake credential is absent or invalid.

    The ``reason`` is for our logs only. Nothing derived from it is sent to
    the client: every failure closes with the same code and no body, so the
    endpoint cannot be used to tell "no token" from "wrong org" from "expired".
    """

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


def _b64url_encode(raw: bytes) -> str:
    """Base64url-encode without padding (``=`` would still be legal, but the
    TypeScript minter strips it and the two must agree byte for byte)."""
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def _b64url_decode(value: str) -> bytes:
    """Reverse :func:`_b64url_encode`, restoring the stripped padding."""
    padding = "=" * (-len(value) % 4)
    return base64.urlsafe_b64decode(value + padding)


def _signing_input(encoded_org_id: str, exp: int) -> str:
    return f"{TOKEN_VERSION}.{encoded_org_id}.{exp}"


def _sign(secret: str, signing_input: str) -> str:
    digest = hmac.new(
        secret.encode("utf-8"),
        signing_input.encode("ascii"),
        hashlib.sha256,
    ).digest()
    return _b64url_encode(digest)


def mint_ws_token(org_id: str, secret: str, *, expires_at: int) -> str:
    """Produce a token for ``org_id`` valid until ``expires_at``.

    Production minting happens in the Dashboard web tier, which is the only
    place that knows who the signed-in user is. This exists so the tests here
    can build tokens without reimplementing the format, and so the shared
    vector is generated by the same code that verifies it.

    Args:
        org_id: The organization whose feed the token unlocks.
        secret: The shared HMAC secret.
        expires_at: Integer Unix timestamp after which the token is dead.

    Returns:
        The encoded token string.

    Raises:
        ValueError: If ``org_id`` or ``secret`` is empty.
    """
    if not org_id:
        raise ValueError("org_id must not be empty")

    if not secret:
        raise ValueError("secret must not be empty")

    encoded_org_id = _b64url_encode(org_id.encode("utf-8"))
    signing_input = _signing_input(encoded_org_id, expires_at)

    return f"{signing_input}.{_sign(secret, signing_input)}"


def verify_ws_token(token: str, secret: str, *, now: float | None = None) -> str:
    """Validate ``token`` and return the ``org_id`` it authorises.

    Args:
        token: The credential taken from the handshake.
        secret: The shared HMAC secret.
        now: Current Unix time; defaults to :func:`time.time`. Injectable so
            expiry behaviour is testable without sleeping.

    Returns:
        The ``org_id`` carried by the token. This is the *only* org whose
        events the resulting connection may ever receive.

    Raises:
        WsAuthError: On any failure -- malformed, wrong version, bad
            signature, expired, or minted with an unreasonable lifetime.
    """
    if not secret:
        # Never treat "no secret configured" as "everything is valid".
        raise WsAuthError("no signing secret configured")

    if not token:
        raise WsAuthError("empty token")

    parts = token.split(".")

    if len(parts) != 4:
        raise WsAuthError("malformed token")

    version, encoded_org_id, raw_exp, signature = parts

    if version != TOKEN_VERSION:
        raise WsAuthError(f"unsupported token version {version!r}")

    try:
        exp = int(raw_exp)
    except ValueError as exc:
        raise WsAuthError("non-integer expiry") from exc

    expected = _sign(secret, _signing_input(encoded_org_id, exp))

    # Constant-time: a byte-by-byte comparison leaks the signature one
    # character at a time to anyone who can time the handshake.
    if not hmac.compare_digest(expected, signature):
        raise WsAuthError("signature mismatch")

    current = time.time() if now is None else now

    if exp < current - CLOCK_SKEW_TOLERANCE_S:
        raise WsAuthError("token expired")

    if exp > current + MAX_TOKEN_LIFETIME_S + CLOCK_SKEW_TOLERANCE_S:
        raise WsAuthError("token lifetime exceeds the permitted maximum")

    try:
        org_id = _b64url_decode(encoded_org_id).decode("utf-8")
    except (ValueError, UnicodeDecodeError) as exc:
        # Unreachable for a token we signed, but the signature is over the
        # *encoded* form, so a decode failure must still fail closed.
        raise WsAuthError("undecodable org_id") from exc

    if not org_id:
        raise WsAuthError("empty org_id")

    return org_id


def _credential_from_subprotocols(header_value: str | None) -> str | None:
    """Pull the token out of a ``Sec-WebSocket-Protocol`` header.

    The client offers ``vex.ws.v1, <token>``. Anything that is not the
    negotiated subprotocol name is treated as the candidate credential.
    """
    if not header_value:
        return None

    for offered in header_value.split(","):
        candidate = offered.strip()

        if candidate and candidate != WS_SUBPROTOCOL:
            return candidate

    return None


def extract_credential(websocket) -> str:
    """Read the handshake credential off ``websocket``, preferring the header.

    Args:
        websocket: The Starlette/FastAPI ``WebSocket``, not yet accepted.

    Returns:
        The raw token string.

    Raises:
        WsAuthError: If no credential is present in either location.
    """
    from_header = _credential_from_subprotocols(
        websocket.headers.get("sec-websocket-protocol")
    )

    if from_header:
        return from_header

    from_query = websocket.query_params.get(TOKEN_QUERY_PARAM)

    if from_query:
        logger.warning(
            "WebSocket credential arrived in the query string; it will appear "
            "in proxy access logs. Offer it as a Sec-WebSocket-Protocol value "
            "instead."
        )
        return from_query

    raise WsAuthError("no credential in subprotocol or query string")


def authenticate(websocket, secret: str, *, now: float | None = None) -> str:
    """Resolve the ``org_id`` a pending WebSocket is allowed to read.

    Args:
        websocket: The ``WebSocket``, before ``accept()``.
        secret: The shared HMAC secret.
        now: Current Unix time, for tests.

    Returns:
        The authorised ``org_id``.

    Raises:
        WsAuthError: On any failure. The caller must close *without* accepting.
    """
    return verify_ws_token(extract_credential(websocket), secret, now=now)
