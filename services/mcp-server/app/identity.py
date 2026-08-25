"""Resolve the human behind a request.

This is the ONLY place the engine decides who a caller is. Phase 2's recall
path treats the result as a tenancy boundary equal to ``org_id``, so the rule
lives in one file with one test suite rather than being re-derived at each
call site.

Two trustworthy sources, in order:

1. **OAuth** — ``request.state.oauth_principal.user_id``, a verified session.
   Preferred when both are present because it identifies the human directly
   rather than the human who once minted a token.
2. **API key** — ``KeyInfo.user_id``, copied from the key entry's
   ``created_by``. Keys are personal by policy.

And one source that is never used: ``X-Vex-Agent``. The client declares it,
so treating it as identity would let any caller read another person's private
memories by setting a header. It is a label for WHICH TOOL, never WHICH
PERSON.

``None`` is a legal result. It means "this request cannot be attributed to a
person", and the caller's contract is to write org-scoped and log — not to
fail. Refusing would break every legacy key; inventing an owner would leak one
person's memories into another's private scope.

**``None`` is only legal when it is TRUE.** It used to be produced by a bare
``except: return None`` around the key lookup, which meant a rate-limited
caller — a caller whose key is perfectly valid — was reported as
unattributable and their write silently widened from ``private``/``project`` to
``org``. A failure to establish identity is not the same fact as an identity
that does not exist. A ``429`` therefore propagates rather than becoming
``None``; a bad credential still returns ``None``, but loudly, because the
caller's own principal resolution has already rejected that request and no
write follows it.

**One validation per request.** ``KeyValidator.validate`` spends a
sliding-window rate-limit token on every call — its entry cache short-circuits
the database lookup, not the limiter — so a tool body that resolved the
principal and then the user consumed two tokens and got half its configured
rpm. :func:`resolve_principal_plan` and :func:`resolve_request_user_id` both
route through :func:`request_key_info`, which validates at most once per
request and memoizes the result (success or failure) on ``request.state``.

That is why ``resolve_principal_plan`` lives here as well as in
:mod:`app.auth`: the two questions share one answer, and the only way to spend
one token is to ask once. ``app.auth``'s version remains for callers this
module does not front (``app.onboarding``); it is byte-compatible, and moving
those callers over is a follow-up.
"""

from __future__ import annotations

import logging

from app.headers import agent_header
from typing import Any, Optional

from fastapi import HTTPException, Request
from shared.auth import KeyInfo, normalize_agent_id

logger = logging.getLogger(__name__)

#: Where the per-request memo lives. Leading underscore because it is this
#: module's private business, not part of any request contract.
_KEY_INFO_ATTR = "_klio_key_info"
_KEY_ERROR_ATTR = "_klio_key_error"


def resolve_user_id(
    request: Request,
    key_info: Optional[Any] = None,
) -> Optional[str]:
    """Return the caller's user id, or ``None`` if they cannot be attributed.

    Args:
        request: The inbound request. Only ``request.state.oauth_principal``
            is consulted; headers are deliberately ignored.
        key_info: The validated :class:`~shared.auth.KeyInfo` for key-path
            callers, when the route has one.

    Returns:
        The user's UUID as a string, or ``None``.
    """
    principal = _oauth_principal(request)
    if principal is not None:
        user_id = getattr(principal, "user_id", None)
        if user_id:
            return str(user_id)

    if key_info is not None:
        user_id = getattr(key_info, "user_id", None)
        if user_id:
            return str(user_id)

    return None


def request_key_info(request: Request) -> KeyInfo:
    """The request's validated :class:`~shared.auth.KeyInfo`, resolved once.

    The MCP tool bodies are not FastAPI routes, so they have no
    ``Depends(verify_api_key)`` and must validate inline. Doing that twice —
    once for the tenant, once for the human — spent two rate-limit tokens per
    tool call. This memoizes the outcome on ``request.state``, so the second
    and later questions are answered from the first answer.

    The FAILURE is memoized too, and re-raised. Retrying a rejected key inside
    one request cannot succeed and would spend another token pushing a limited
    caller further past their limit.

    Raises:
        HTTPException: whatever ``verify_api_key`` raises — 401, 403 or 429.
            Deliberately propagated: a caller who cannot be authenticated is
            not the same thing as a caller who is nobody.
    """
    from app.auth import verify_api_key  # noqa: PLC0415

    state = getattr(request, "state", None)

    cached = getattr(state, _KEY_INFO_ATTR, None)
    # Only an actual KeyInfo counts, mirroring _oauth_principal: a MagicMock
    # request answers every attribute with a truthy mock, and treating one as a
    # validated credential would be the worst possible false positive.
    if isinstance(cached, KeyInfo):
        return cached
    cached_error = getattr(state, _KEY_ERROR_ATTR, None)
    if isinstance(cached_error, Exception):
        raise cached_error

    try:
        key_info = verify_api_key(request)
    except Exception as exc:
        _memoize(state, _KEY_ERROR_ATTR, exc)
        raise

    _memoize(state, _KEY_INFO_ATTR, key_info)
    return key_info


def _memoize(state: Any, attr: str, value: Any) -> None:
    """Best-effort per-request memo.

    A request object with no writable ``state`` (a bare stub in a test) simply
    goes uncached: validating twice is a performance and rate-limit fault, not
    a correctness one, so it must never turn into an exception of its own.
    """
    if state is None:
        return
    try:
        setattr(state, attr, value)
    except Exception:  # pragma: no cover - exotic request stubs only
        logger.debug("could not memoize %s on request.state", attr, exc_info=True)


def resolve_request_user_id(request: Request) -> Optional[str]:
    """:func:`resolve_user_id` for callers that do not already hold a KeyInfo.

    Walks the same two-rung ladder and fetches the KeyInfo itself, through
    :func:`request_key_info` so the request pays for at most one validation
    however many times it asks.

    Raises:
        HTTPException: on a 429. A rate-limited caller holds a VALID key: they
            are a known person who is asking too fast. Reporting them as
            unattributable — which is what returning ``None`` here did — landed
            their private or project write at ``org`` scope, readable by the
            whole tenant, with nothing in the response mentioning either the
            rate limit or the widening. There is no reading of ``None`` under
            which that is true, so this one propagates. The tool bodies convert
            it into a clean ``unauthorized`` result carrying the rate-limit
            detail.

    A 401/403 is logged and returns ``None`` instead. Those mean the credential
    itself is bad, and the caller's own principal resolution — same request,
    same key, now the same memoized answer — has already rejected the request
    before reaching here; returning ``None`` cannot widen a write that is not
    going to happen. It is kept non-fatal because this function is also called
    on paths that never validated a key at all, where an unattributed write is
    the correct, documented outcome.
    """
    principal = _oauth_principal(request)
    if principal is not None:
        user_id = getattr(principal, "user_id", None)
        if user_id:
            return str(user_id)
        # An OAuth principal that names no user is still a resolved identity;
        # there is no key to fall back to on this rung.
        return None

    try:
        key_info = request_key_info(request)
    except HTTPException as exc:
        if exc.status_code == 429:
            raise
        logger.warning(
            "identity could not be resolved from the API key (%s); this "
            "request writes unattributed",
            exc.status_code,
        )
        return None
    except Exception:
        # Not an auth verdict — a fault while reaching one (a DB blip, a
        # request object with no key at all). Unattributed is the documented
        # outcome, but it is logged rather than discarded: this branch used to
        # be a bare ``except: return None`` and was where the swallowed 429
        # hid.
        logger.warning(
            "key validation failed unexpectedly; this request writes unattributed",
            exc_info=True,
        )
        return None
    return resolve_user_id(request, key_info)


def resolve_principal(request: Request) -> tuple[str, str]:
    """``(org_id, agent_id)``, sharing this request's KeyInfo.

    The narrower half of :func:`resolve_principal_plan`, for the tools that
    need no plan. It exists here for the same reason: a tool body that resolved
    its principal through ``app.auth`` and its user through this module was
    still validating twice, so fronting only ``resolve_principal_plan`` would
    have left the double count in place on ``forget``, ``share``,
    ``project_*`` and ``explore``.
    """
    org_id, agent_id, _plan, _overrides = resolve_principal_plan(request)
    return org_id, agent_id


def resolve_principal_plan(request: Request) -> tuple[str, str, str, Optional[dict]]:
    """``(org_id, agent_id, plan, plan_overrides)``, sharing this request's KeyInfo.

    Behaviourally identical to :func:`app.auth.resolve_principal_plan` — the
    same two-rung ladder, the same ``X-Vex-Agent`` precedence, the same tuple —
    differing only in that the key path goes through :func:`request_key_info`.
    That is the entire point: the tool body's later call to
    :func:`resolve_request_user_id` then costs nothing, and the configured rpm
    is the rpm the caller actually gets.

    The OAuth branch delegates to ``app.auth`` so the derived agent id has one
    definition rather than a copy that can drift.
    """
    from app.auth import (  # noqa: PLC0415
        DEFAULT_AGENT_ID,
        _oauth_agent_id,
        _oauth_principal_from,
    )

    principal = _oauth_principal_from(request)
    if principal is not None:
        # normalize_agent_id, not `or DEFAULT`: agents.agent_id is VARCHAR(128)
        # and a primary key, the header is client-declared, and the derived
        # oauth:<user8>/<slug> id is built from the client-declared
        # clientInfo.name -- so both arms are caller-influenced and unbounded
        # without it. A blank header also has to fall through to the default,
        # which `or` cannot do: "   " is truthy and would become the principal
        # that dedup and agent-scoped recall then key on.
        agent_id = normalize_agent_id(
            agent_header(request) or _oauth_agent_id(request, principal),
            default=DEFAULT_AGENT_ID,
        )
        return principal.org_id, agent_id, principal.plan, principal.plan_overrides

    key_info = request_key_info(request)
    agent_id = normalize_agent_id(agent_header(request), default=DEFAULT_AGENT_ID)
    return key_info.org_id, agent_id, key_info.plan, key_info.plan_overrides


def _oauth_principal(request: Request) -> Optional[Any]:
    """Return the middleware-attached OAuthPrincipal, or ``None``.

    Mirrors ``app.auth._oauth_principal_from``: tolerates a ``state`` with no
    such attribute, and only accepts an actual ``OAuthPrincipal`` so a
    MagicMock-patched request in a pre-existing test cannot be mistaken for a
    real identity. Never raises.
    """
    candidate = getattr(getattr(request, "state", None), "oauth_principal", None)
    if candidate is None:
        return None
    try:
        from app.oauth import OAuthPrincipal  # noqa: PLC0415
    except Exception:
        # Unchanged outcome: an unimportable ``app.oauth`` means no principal
        # can be recognised, and unattributed is the right answer. But this was
        # the last swallow in the module that left no trace, and its symptom —
        # every OAuth caller silently degrading to unattributed — is otherwise
        # indistinguishable from nobody using OAuth. DEBUG rather than WARNING
        # because on the healthy path this branch is never reached at all.
        logger.debug(
            "app.oauth could not be imported; treating the request as having "
            "no OAuth principal",
            exc_info=True,
        )
        return None
    return candidate if isinstance(candidate, OAuthPrincipal) else None
