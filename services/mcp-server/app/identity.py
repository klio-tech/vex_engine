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
