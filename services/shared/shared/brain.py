"""Shared Klio brain module: org-scoped get-or-create helpers.

Provides idempotent ``ensure_space`` / ``ensure_project`` helpers over the
``spaces`` and ``projects`` tables introduced in migration 020. All DB access
uses raw SQL via ``sqlalchemy.text()`` and the synchronous session interface
(``db_session.execute(...)`` / ``db_session.commit()``), matching the house
style of :mod:`shared.memory`.

Unlike :mod:`shared.memory`, these functions are plain ``def`` (synchronous):
they make no LLM/embedding calls, so there is nothing to ``await``.

Tenant-scoping invariant: every query is filtered by ``org_id``. A helper will
never read or return a row belonging to a different org.
"""

import logging
import uuid
from typing import Any, Optional

from sqlalchemy import text

from shared.memory import find_similar_memory, write_memory

logger = logging.getLogger("agentguard.shared.brain")


def ensure_space(
    db_session: Any,
    *,
    org_id: str,
    name: str,
    slug: str,
    embedding_model: Optional[str] = None,
) -> str:
    """Get-or-create a space for an org, deduped by ``(org_id, slug)``.

    Implemented as a single race-safe ``INSERT ... ON CONFLICT ... RETURNING``
    upsert against the ``uq_spaces_org_slug`` unique constraint
    (``(org_id, slug)``). On a fresh ``(org_id, slug)`` a new row is inserted;
    on conflict a no-op ``DO UPDATE`` fires so ``RETURNING`` still yields the
    existing row's id (``DO NOTHING`` would suppress ``RETURNING``). This
    converges correctly under concurrent callers instead of raising an
    unhandled ``IntegrityError``.

    Threading contract:
        This function is synchronous (blocking psycopg2 I/O). When called from
        an async handler, invoke via ``anyio.to_thread.run_sync(...)`` (or a
        FastAPI dependency / plain-``def`` route, which FastAPI runs in a
        threadpool) — never call it inline from an async coroutine, or you
        block the event loop. See :meth:`shared.auth.ApiKeyValidator.validate`
        for the analogous house mitigation.

    Args:
        db_session: A synchronous SQLAlchemy session/connection exposing
            ``.execute()`` and ``.commit()``.
        org_id: Owning org id (tenant scope).
        name: Human-readable space name (used only on insert).
        slug: URL/identifier slug; the dedup key within an org.
        embedding_model: Optional embedding model name (used only on insert).

    Returns:
        The space id as a string.
    """
    row = db_session.execute(
        text(
            """
            INSERT INTO spaces (id, org_id, name, slug, embedding_model)
            VALUES (:id, :org_id, :name, :slug, :embedding_model)
            ON CONFLICT (org_id, slug)
              DO UPDATE SET slug = EXCLUDED.slug
            RETURNING id
            """
        ),
        {
            "id": str(uuid.uuid4()),
            "org_id": org_id,
            "name": name,
            "slug": slug,
            "embedding_model": embedding_model,
        },
    ).one()
    db_session.commit()
    return str(row.id)


def ensure_project(
    db_session: Any,
    *,
    org_id: str,
    display_name: str,
    git_remote: Optional[str] = None,
    repo_root_path: Optional[str] = None,
    space_id: Optional[str] = None,
) -> str:
    """Get-or-create a project for an org, bumping ``last_seen_at`` on a hit.

    Dedup key: ``git_remote`` when present, otherwise ``repo_root_path``. This
    mirrors the partial-unique indexes ``ix_projects_org_remote`` (on
    ``(org_id, git_remote)`` where ``git_remote IS NOT NULL``) and
    ``ix_projects_org_path`` (on ``(org_id, repo_root_path)`` where
    ``git_remote IS NULL AND repo_root_path IS NOT NULL``).

    Implemented as a single race-safe ``INSERT ... ON CONFLICT ... RETURNING``
    upsert. The branch is chosen by ``git_remote`` presence so the partial
    index predicate in the ``ON CONFLICT ... WHERE`` clause matches the
    arbiter index exactly (Postgres requires the predicate to match for it to
    infer a partial unique index as the conflict arbiter). On conflict the
    ``DO UPDATE SET last_seen_at = NOW()`` folds the hit-path ``last_seen_at``
    bump into the same atomic statement, eliminating both the insert race and
    the read-modify-write race; on a miss a new row is inserted. Either way the
    transaction is committed.

    Threading contract:
        This function is synchronous (blocking psycopg2 I/O). When called from
        an async handler, invoke via ``anyio.to_thread.run_sync(...)`` (or a
        FastAPI dependency / plain-``def`` route, which FastAPI runs in a
        threadpool) — never call it inline from an async coroutine, or you
        block the event loop. See :meth:`shared.auth.ApiKeyValidator.validate`
        for the analogous house mitigation.

    Tenancy note:
        When a ``space_id`` is supplied it is verified to belong to ``org_id``
        before the INSERT. The FK (migration 020) only enforces that the space
        *exists*, not that it is owned by this org, so without this guard a
        caller could file a project under another tenant's space. A cross-org
        ``space_id`` is rejected with ``ValueError``.

    Args:
        db_session: A synchronous SQLAlchemy session/connection exposing
            ``.execute()`` and ``.commit()``.
        org_id: Owning org id (tenant scope).
        display_name: Human-readable project name (used only on insert).
        git_remote: Canonical git remote URL; primary dedup key when present.
        repo_root_path: Filesystem repo root; dedup key when no remote.
        space_id: Optional space the project belongs to (used only on insert).
            When provided, must belong to ``org_id``.

    Returns:
        The project id as a string.

    Raises:
        ValueError: If both ``git_remote`` and ``repo_root_path`` are ``None``
            (mirrors the DB ``ck_projects_has_identity`` check; an identity-less
            project is unrepresentable), or if a supplied ``space_id`` does not
            belong to ``org_id`` (cross-tenant association is rejected).
    """
    if git_remote is None and repo_root_path is None:
        raise ValueError("project requires git_remote or repo_root_path")

    # Defense in depth: the projects.space_id FK only enforces that the space
    # EXISTS, not that it belongs to org_id (migration 020). Verify ownership
    # before binding a caller-supplied space_id so a project can never be filed
    # under another tenant's space. Skipped on the common unfiled-write path
    # (space_id is None), so it adds no query when no space is supplied.
    if space_id is not None:
        owned = db_session.execute(
            text(
                """
                SELECT 1 FROM spaces
                WHERE id = :space_id AND org_id = :org_id
                LIMIT 1
                """
            ),
            {"space_id": space_id, "org_id": org_id},
        ).fetchone()
        if owned is None:
            raise ValueError(
                "space_id does not belong to org_id "
                "(cross-tenant space association rejected)"
            )

    if git_remote is not None:
        row = db_session.execute(
            text(
                """
                INSERT INTO projects (
                    id, org_id, space_id, git_remote, repo_root_path,
                    display_name
                ) VALUES (
                    :id, :org_id, :space_id, :git_remote, :repo_root_path,
                    :display_name
                )
                ON CONFLICT (org_id, git_remote) WHERE git_remote IS NOT NULL
                  DO UPDATE SET last_seen_at = NOW()
                RETURNING id
                """
            ),
            {
                "id": str(uuid.uuid4()),
                "org_id": org_id,
                "space_id": space_id,
                "git_remote": git_remote,
                "repo_root_path": repo_root_path,
                "display_name": display_name,
            },
        ).one()
    else:
        row = db_session.execute(
            text(
                """
                INSERT INTO projects (
                    id, org_id, space_id, git_remote, repo_root_path,
                    display_name
                ) VALUES (
                    :id, :org_id, :space_id, NULL, :repo_root_path,
                    :display_name
                )
                ON CONFLICT (org_id, repo_root_path)
                  WHERE git_remote IS NULL AND repo_root_path IS NOT NULL
                  DO UPDATE SET last_seen_at = NOW()
                RETURNING id
                """
            ),
            {
                "id": str(uuid.uuid4()),
                "org_id": org_id,
                "space_id": space_id,
                "repo_root_path": repo_root_path,
                "display_name": display_name,
            },
        ).one()

    db_session.commit()
    return str(row.id)


def find_space_id(db_session: Any, *, org_id: str, slug: str) -> Optional[str]:
    """Return the space id for ``(org_id, slug)``, or ``None`` if no such space.

    Read-only counterpart to :func:`ensure_space`: it resolves a user-facing
    slug to its id on the recall (read) path and NEVER creates a row. An unknown
    slug returns ``None`` so the caller can surface "unknown space" rather than
    silently widening the search. Tenant-scoped: only an ``org_id``-owned row
    can match, so a slug collision across orgs never leaks another org's space.

    Threading contract:
        Synchronous (blocking psycopg2 I/O). When called from an async handler,
        invoke via ``anyio.to_thread.run_sync(...)`` (or a plain-``def`` FastAPI
        route, which FastAPI runs in a threadpool) — never inline from a
        coroutine, or you block the event loop.

    Args:
        db_session: A synchronous SQLAlchemy session/connection exposing
            ``.execute()``.
        org_id: Owning org id (tenant scope).
        slug: Space slug; the dedup key within an org.

    Returns:
        The space id as a string, or ``None`` if no matching space exists.
    """
    row = db_session.execute(
        text(
            """
            SELECT id FROM spaces
            WHERE org_id = :org_id AND slug = :slug
            LIMIT 1
            """
        ),
        {"org_id": org_id, "slug": slug},
    ).fetchone()
    if row is None:
        return None
    return str(row[0])


def find_project_id(db_session: Any, *, org_id: str, identifier: str) -> Optional[str]:
    """Resolve a project identifier to its id within an org, or ``None``.

    Read-only counterpart to :func:`ensure_project`: it resolves a user-facing
    identifier to its id on the recall (read) path and NEVER creates a row.

    Resolution order (all tenant-scoped by ``org_id``):

        1. If ``identifier`` parses as a UUID, look up by ``id`` first.
        2. Otherwise (or if the id lookup misses), match by ``git_remote`` OR
           ``display_name``.

    Returns ``None`` when nothing matches so the caller can surface "unknown
    project" rather than silently widening the search.

    Threading contract:
        Synchronous (blocking psycopg2 I/O). When called from an async handler,
        invoke via ``anyio.to_thread.run_sync(...)`` (or a plain-``def`` FastAPI
        route, which FastAPI runs in a threadpool) — never inline from a
        coroutine, or you block the event loop.

    Args:
        db_session: A synchronous SQLAlchemy session/connection exposing
            ``.execute()``.
        org_id: Owning org id (tenant scope).
        identifier: A project id (UUID), git remote URL, or display name.

    Returns:
        The project id as a string, or ``None`` if no matching project exists.
    """
    # Prefer an exact id match when the identifier is a well-formed UUID. A
    # raw-string compare against the UUID column would be brittle across
    # formats, so parse first and only attempt the id query when it parses.
    try:
        parsed = uuid.UUID(identifier)
    except (ValueError, AttributeError, TypeError):
        parsed = None

    if parsed is not None:
        row = db_session.execute(
            text(
                """
                SELECT id FROM projects
                WHERE org_id = :org_id AND id = :id
                LIMIT 1
                """
            ),
            {"org_id": org_id, "id": str(parsed)},
        ).fetchone()
        if row is not None:
            return str(row[0])
        # Fall through: a UUID-shaped identifier that is not a known id may
        # still legitimately match a git_remote / display_name.

    row = db_session.execute(
        text(
            """
            SELECT id FROM projects
            WHERE org_id = :org_id
              AND (git_remote = :ident OR display_name = :ident)
            LIMIT 1
            """
        ),
        {"org_id": org_id, "ident": identifier},
    ).fetchone()
    if row is None:
        return None
    return str(row[0])


def list_spaces(db_session: Any, *, org_id: str) -> list[dict]:
    """Return all spaces for an org, newest first. Read-only.

    Enumerates the org's spaces on the management (read) path for the ``space``
    MCP tool's ``list`` action. NEVER creates a row — it only ``SELECT``s and
    projects each row to a dict. Tenant-scoped: only ``org_id``-owned rows are
    returned, so a slug/name collision across orgs never leaks another org's
    space.

    Each returned dict has: ``id`` (str — ``spaces.id`` is a UUID column, so it
    is stringified here for caller convenience), ``name`` (str), ``slug`` (str),
    ``embedding_model`` (str|None), and ``created_at`` (a ``datetime`` — left
    as-is so the JSON-facing tool layer controls its serialization, mirroring
    the recall path where the tool isoformats timestamps).

    Threading contract:
        Synchronous (blocking psycopg2 I/O). When called from an async handler,
        invoke via ``anyio.to_thread.run_sync(...)`` (or a plain-``def`` FastAPI
        route, which FastAPI runs in a threadpool) — never inline from a
        coroutine, or you block the event loop.

    Args:
        db_session: A synchronous SQLAlchemy session/connection exposing
            ``.execute()``.
        org_id: Owning org id (tenant scope).

    Returns:
        A list of space dicts (possibly empty), ordered by ``created_at``
        descending (newest first).
    """
    rows = db_session.execute(
        text(
            """
            SELECT id, name, slug, embedding_model, created_at
            FROM spaces
            WHERE org_id = :org_id
            ORDER BY created_at DESC
            """
        ),
        {"org_id": org_id},
    ).fetchall()
    return [
        {
            "id": str(row.id),
            "name": row.name,
            "slug": row.slug,
            "embedding_model": row.embedding_model,
            "created_at": row.created_at,
        }
        for row in rows
    ]


def get_space(db_session: Any, *, org_id: str, slug: str) -> Optional[dict]:
    """Return a single space by ``(org_id, slug)``, or ``None``. Read-only.

    Look-up counterpart used by the ``space`` MCP tool's ``info`` / ``switch``
    actions. Unlike :func:`find_space_id` (which returns only the id), this
    returns the full space dict so the tool can echo human-facing fields. NEVER
    creates a row — it only ``SELECT``s. Tenant-scoped: only an ``org_id``-owned
    row can match.

    Returns the same dict shape as :func:`list_spaces`: ``id`` (str — the UUID
    column is stringified), ``name``, ``slug``, ``embedding_model`` (str|None),
    and ``created_at`` (a ``datetime``, left unserialized for the tool layer).

    Threading contract:
        Synchronous (blocking psycopg2 I/O). When called from an async handler,
        invoke via ``anyio.to_thread.run_sync(...)`` (or a plain-``def`` FastAPI
        route, which FastAPI runs in a threadpool) — never inline from a
        coroutine, or you block the event loop.

    Args:
        db_session: A synchronous SQLAlchemy session/connection exposing
            ``.execute()``.
        org_id: Owning org id (tenant scope).
        slug: Space slug; the dedup key within an org.

    Returns:
        The space dict, or ``None`` if no matching space exists.
    """
    row = db_session.execute(
        text(
            """
            SELECT id, name, slug, embedding_model, created_at
            FROM spaces
            WHERE org_id = :org_id AND slug = :slug
            LIMIT 1
            """
        ),
        {"org_id": org_id, "slug": slug},
    ).fetchone()
    if row is None:
        return None
    return {
        "id": str(row.id),
        "name": row.name,
        "slug": row.slug,
        "embedding_model": row.embedding_model,
        "created_at": row.created_at,
    }


async def write_scoped_memory(
    db_session: Any,
    *,
    org_id: str,
    agent_id: str,
    content: str,
    memory_type: str,
    session_id: str,
    space_id: Optional[str] = None,
    project_id: Optional[str] = None,
    metadata: Optional[dict] = None,
    dedup: bool = True,
    scope: str = "org",
) -> dict:
    """Write a memory into the Klio brain at org or agent scope (with dedup).

    This is the brain-scoped write path layered on top of :mod:`shared.memory`.
    It reuses the existing primitives verbatim — :func:`find_similar_memory`
    for deduplication and :func:`write_memory` for the insert — and then
    *promotes* the freshly inserted row from the session scope to the brain
    (org) scope.

    Why the write-then-UPDATE: :func:`shared.memory.write_memory` hardcodes
    ``scope='session'`` in its INSERT and does not accept ``space_id`` /
    ``project_id``. That function is shared with the verification path and is
    intentionally left untouched. Rather than fork or edit it, we let it insert
    the row at ``scope='session'`` and immediately issue a follow-up UPDATE that
    sets the final scope (``'org'`` or ``'agent'``) and binds the brain scope
    columns. This keeps the primitive single-purpose while the brain composes
    the extra behaviour it needs.

    Scope:
        * ``scope='org'`` (default) — the row is promoted to the shared team
          brain (``scope='org'``). Visible to every agent in the org on recall.
          Dedup is session-scoped (historical behavior). Backward compatible:
          callers that omit ``scope`` are unaffected.
        * ``scope='agent'`` — the row is promoted to ``scope='agent'`` and stays
          private to ``agent_id``. :func:`retrieve_memories_scoped` with
          ``scope='agent'`` returns it to that agent (plus org rows) but never to
          another agent. Dedup is agent-scoped (only against this agent's own
          private rows), so one agent's write never dedups against the org brain
          or another agent's row.

    Async note: this mirrors the house pattern of :func:`write_memory` /
    :func:`find_similar_memory` — an ``async def`` performing inline synchronous
    psycopg2 DB I/O. This is intentional and consistent across the memory
    substrate; do not wrap the DB calls in threads here.

    Args:
        db_session: A synchronous SQLAlchemy session/connection exposing
            ``.execute()`` and ``.commit()``.
        org_id: Owning org id (tenant scope). Always enforced.
        agent_id: Authoring agent principal.
        content: The memory text to store.
        memory_type: One of the brain memory types (e.g. ``memory``,
            ``observation``, ``plan``, ``decision``, ``note``).
        session_id: Stable per-connection brain session id. Has no FK, so any
            string is acceptable; it scopes the dedup lookup.
        space_id: Optional space the memory belongs to.
        project_id: Optional project the memory belongs to.
        metadata: Optional metadata dict persisted on the row.
        dedup: When ``True`` (default), suppress the write if a sufficiently
            similar active memory already exists in the dedup window (see Scope).
        scope: ``'org'`` (default) or ``'agent'``. Determines the final scope the
            row is promoted to and the dedup window.

    Returns:
        A result dict describing the outcome (``scope`` echoes the requested
        scope):

        * dedup hit →
          ``{"stored": False, "deduplicated": True, "id": ..., "memory_type":
          ..., "scope": <scope>}``
        * embedding failure →
          ``{"stored": False, "error": "embedding_failed"}``
        * success →
          ``{"stored": True, "deduplicated": False, "id": ..., "memory_type":
          ..., "scope": <scope>, "space_id": ..., "project_id": ...}``

    Raises:
        ValueError: If ``scope`` is neither ``'org'`` nor ``'agent'``.
    """
    if scope not in ("org", "agent"):
        raise ValueError(
            f"unsupported scope: {scope!r} (write_scoped_memory accepts 'org' or 'agent')"
        )

    if dedup:
        existing = await find_similar_memory(
            db_session,
            session_id,
            org_id,
            content,
            # Agent-private writes dedup only against the same agent's private
            # rows; org writes keep the session-scoped dedup window.
            scope="agent" if scope == "agent" else None,
            agent_id=agent_id if scope == "agent" else None,
        )
        if existing is not None:
            return {
                "stored": False,
                "deduplicated": True,
                "id": str(existing.id),
                "memory_type": existing.memory_type,
                "scope": scope,
            }

    # write_memory's 5th positional arg is execution_id; the column is nullable
    # and brain writes are not tied to a verification execution, so pass None.
    record = await write_memory(
        db_session,
        session_id,
        org_id,
        agent_id,
        None,
        memory_type,
        content,
        metadata=metadata,
    )
    if record is None:
        # write_memory returns None when embedding generation fails.
        return {"stored": False, "error": "embedding_failed"}

    # Promote the just-inserted session-scoped row to the final brain scope
    # ('org' or 'agent') and bind the scope columns (see the write-then-UPDATE
    # rationale above). The row already carries agent_id from write_memory, so
    # an 'agent'-scoped row is private to that agent on recall.
    db_session.execute(
        text(
            """
            UPDATE session_memories
            SET scope = :scope,
                space_id = :space_id,
                project_id = :project_id,
                updated_at = NOW()
            WHERE id = :id
            """
        ),
        {
            "scope": scope,
            "space_id": space_id,
            "project_id": project_id,
            "id": str(record.id),
        },
    )
    db_session.commit()

    return {
        "stored": True,
        "deduplicated": False,
        "id": str(record.id),
        "memory_type": memory_type,
        "scope": scope,
        "space_id": space_id,
        "project_id": project_id,
    }
