"""Resolve a teammate's EMAIL to a user id, within the caller's org.

``project_grant(project, user)`` takes an email, not a UUID, because an agent
does not know its teammates' Supabase user ids and never will. Something has to
turn ``"priya@oppla.ai"`` into a principal, and that something is the org's own
membership records — not a header, not a lookup table the caller controls.

**This is the module that converts a prompt injection into data access**, so
every path through it fails closed:

* No ``SUPABASE_DATABASE_URL`` configured → ``None``. An unresolvable directory
  is not an excuse to grant to a raw UUID the caller supplied.
* The email is not a member of the caller's account → ``None``. Membership in
  the org is a precondition, checked in SQL as a join rather than afterwards.
* Any error at all → ``None``, logged. Never an exception that a caller could
  read as "the check did not run".

Identity lives in a different database from memory (Supabase vs TimescaleDB),
which is why this reaches for its own engine rather than the request's session.
The org → account mapping comes from ``organizations.account_slug`` in
TimescaleDB, the same join the key path already performs.

.. warning::

   The email column's exact location depends on the Supabase project's schema.
   Two shapes are tried, in order: ``auth.users.email`` (the canonical Supabase
   identity table) and a personal-account row in ``public.accounts``. BOTH are
   gated by the same ``accounts_memberships`` join, so neither can widen access
   — the fallback changes where the email is READ, never who is authorised.
"""

import logging
import os
import threading
from typing import Any, Optional

from shared.engine_bounds import engine_bounds
from sqlalchemy import create_engine, text

from app.config import DATABASE_URL

logger = logging.getLogger("agentguard.mcp.directory")

SUPABASE_DATABASE_URL = os.environ.get("SUPABASE_DATABASE_URL")

_engine_lock = threading.Lock()
_supabase_engine: Optional[Any] = None
_tsdb_engine: Optional[Any] = None


def _engines() -> tuple[Optional[Any], Optional[Any]]:
    """Lazily build the two engines this module needs, or ``(None, None)``."""
    global _supabase_engine, _tsdb_engine
    if not SUPABASE_DATABASE_URL:
        return None, None
    with _engine_lock:
        # Bounded via shared.engine_bounds: these are long-lived and on the
        # request path, so an unreachable host must fail rather than hang.
        if _supabase_engine is None:
            _supabase_engine = create_engine(
                SUPABASE_DATABASE_URL,
                pool_pre_ping=True,
                pool_size=2,
                max_overflow=2,
                **engine_bounds(),
            )
        if _tsdb_engine is None:
            _tsdb_engine = create_engine(
                DATABASE_URL,
                pool_pre_ping=True,
                pool_size=2,
                max_overflow=2,
                **engine_bounds(),
            )
    return _supabase_engine, _tsdb_engine


def supabase_engine() -> Optional[Any]:
    """The Supabase engine, or ``None`` when no directory is configured.

    Membership resolution (``shared.membership``) needs a Supabase connection
    and this module already owns the only bounded, pooled one in the process.
    A second ``create_engine`` for the same URL elsewhere would double the
    connection footprint and give the boundary check its own, differently
    configured timeouts.

    ``None`` is not "skip the check": every caller must fail closed on it.
    """
    supabase, _tsdb = _engines()
    return supabase


def shutdown_directory() -> None:
    """Dispose both engines. Called at app shutdown."""
    global _supabase_engine, _tsdb_engine
    with _engine_lock:
        for engine in (_supabase_engine, _tsdb_engine):
            if engine is not None:
                engine.dispose()
        _supabase_engine = None
        _tsdb_engine = None


# Both queries END in the same membership join. The SELECT differs; the
# authorisation does not.
_BY_AUTH_USERS = """
    SELECT m.user_id
    FROM public.accounts_memberships m
    JOIN public.accounts a ON a.id = m.account_id
    JOIN auth.users u ON u.id = m.user_id
    WHERE a.slug = :slug
      AND lower(u.email) = lower(:email)
    LIMIT 1
"""

_BY_PERSONAL_ACCOUNT = """
    SELECT m.user_id
    FROM public.accounts_memberships m
    JOIN public.accounts a ON a.id = m.account_id
    JOIN public.accounts p
      ON p.primary_owner_user_id = m.user_id
     AND p.is_personal_account
    WHERE a.slug = :slug
      AND lower(p.email) = lower(:email)
    LIMIT 1
"""


def resolve_org_member_by_email(org_id: str, email: str) -> Optional[str]:
    """Return the user id for ``email`` IF they are a member of ``org_id``'s account.

    Args:
        org_id: The caller's tenant, from the validated key or OAuth principal —
            never from caller input.
        email: The teammate's email address, as the agent typed it. Matched
            case-insensitively; addresses are not case-sensitive in the part
            that matters and a case mismatch would read as "not in the org",
            which is a confusing refusal rather than a safe one.

    Returns:
        The user's UUID as a string, or ``None`` if they are not a member, the
        directory is not configured, or anything failed.
    """
    normalized = (email or "").strip()
    if not normalized or "@" not in normalized:
        return None

    supabase, tsdb = _engines()
    if supabase is None or tsdb is None:
        logger.warning(
            "directory: SUPABASE_DATABASE_URL is not configured; refusing to "
            "resolve %s for org %s (fail closed)",
            normalized,
            org_id,
        )
        return None

    try:
        with tsdb.connect() as conn:
            row = conn.execute(
                text(
                    "SELECT account_slug FROM organizations "
                    "WHERE org_id = :org_id LIMIT 1"
                ),
                {"org_id": org_id},
            ).fetchone()
    except Exception:
        logger.warning(
            "directory: org->account lookup failed for %s (fail closed)",
            org_id,
            exc_info=True,
        )
        return None
    if row is None or not row[0]:
        logger.info("directory: org %s has no account_slug; cannot resolve", org_id)
        return None
    slug = row[0]

    params = {"slug": slug, "email": normalized}
    for query in (_BY_AUTH_USERS, _BY_PERSONAL_ACCOUNT):
        try:
            with supabase.connect() as conn:
                found = conn.execute(text(query), params).fetchone()
        except Exception:
            # A missing table / denied schema is expected for whichever shape
            # this deployment does not use; debug, not warning.
            logger.debug(
                "directory: email lookup shape unavailable for org %s",
                org_id,
                exc_info=True,
            )
            continue
        if found is not None and found[0]:
            return str(found[0])

    logger.info(
        "directory: %s is not a member of org %s (or is not resolvable); refusing",
        normalized,
        org_id,
    )
    return None
