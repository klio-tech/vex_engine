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
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from fastapi import Request

logger = logging.getLogger(__name__)


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


def resolve_request_user_id(request: Request) -> Optional[str]:
    """:func:`resolve_user_id` for callers that do not already hold a KeyInfo.

    The MCP tool bodies are not FastAPI routes, so they have no
    ``Depends(verify_api_key)`` and never see a :class:`~shared.auth.KeyInfo` —
    ``resolve_principal_plan`` validates the key and returns only the fields it
    needs. Rather than widen that signature (it is called from a dozen places
    and asserted on in several tests), this walks the same two-rung ladder and
    fetches the KeyInfo itself. The validator caches by key hash, so the second
    lookup in a request is an in-memory hit, not a query.

    Never raises. A key that fails validation here would already have failed in
    the caller's own ``resolve_principal_plan``; if it somehow does not, an
    unattributed write is the correct outcome and an exception is not.
    """
    principal = _oauth_principal(request)
    if principal is not None:
        user_id = getattr(principal, "user_id", None)
        if user_id:
            return str(user_id)

    try:
        from app.auth import verify_api_key  # noqa: PLC0415

        key_info = verify_api_key(request)
    except Exception:
        return None
    return resolve_user_id(request, key_info)


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
    except Exception:  # pragma: no cover - import guard only
        return None
    return candidate if isinstance(candidate, OAuthPrincipal) else None
