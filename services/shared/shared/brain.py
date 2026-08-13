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
import os
import uuid
from typing import Any, Optional

from sqlalchemy import text

from shared.memory import find_similar_memory, write_memory
from shared.redaction import redact

logger = logging.getLogger("agentguard.shared.brain")

REDACTED_METADATA_KEYS = frozenset({"rationale"})
"""Metadata keys whose values are prose and must be screened for secrets.

An explicit set rather than "screen everything", because most metadata is
structural — checksums, storage URIs, ids, counts — and those are high-entropy
by construction. Running them through the entropy gate would corrupt real data
to fix a leak that only free text can carry, which is a worse trade than the
bug. A new prose-bearing field opts in here.
"""


def screen_metadata(
    metadata: dict | None,
) -> tuple[dict | None, list[str]]:
    """Redact the prose-bearing values in *metadata*.

    Returns a new dict (the caller's is never mutated) and the sorted kinds
    that were found, so they can join the row's ``redactions`` record.

    Exceptions are deliberately NOT caught here. The caller treats a redactor
    failure as a rejected write — unscreened text is never stored — and
    swallowing it at this level would silently reopen exactly that hole.
    """
    if not metadata:
        return metadata, []

    cleaned = dict(metadata)
    kinds: set[str] = set()
    for key in REDACTED_METADATA_KEYS:
        value = cleaned.get(key)
        if not isinstance(value, str):
            # Not prose. The redactor takes text, and a non-string here is a
            # structural value that must survive untouched.
            continue
        result = redact(value)
        if result.redacted:
            cleaned[key] = result.clean_text
            kinds.update(f.kind for f in result.findings)

    return cleaned, sorted(kinds)


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
    created_by: Optional[str] = None,
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

    Membership note (migration 039 — "projects are visible only to their
    members"):
        ``created_by`` is written on INSERT only, first-writer-wins — the same
        shape ``rename_project``'s docstring describes for ``display_name``,
        for the same reason. The ``ON CONFLICT ... DO UPDATE`` touches only
        ``last_seen_at``, so a teammate who later writes into the same repo
        never reassigns ownership merely by working there.

        When this call is the one that actually inserts the row (detected via
        Postgres's ``xmax = 0`` idiom — true for a freshly inserted row,
        false when ``DO UPDATE`` fired on a conflict) AND ``created_by`` is
        supplied, the creator is also seated as the project's first
        ``project_members`` admin, in the same transaction. Without this,
        every auto-created project would go on having zero members — exactly
        the state that made org-wide listing look "necessary" in the first
        place, since a members-only listing would otherwise hide a project
        from the very person who just created it. An unattributed write
        (``created_by=None``, e.g. an API key with no resolvable user) creates
        no membership; there is nobody to seat.

    Args:
        db_session: A synchronous SQLAlchemy session/connection exposing
            ``.execute()`` and ``.commit()``.
        org_id: Owning org id (tenant scope).
        display_name: Human-readable project name (used only on insert).
        git_remote: Canonical git remote URL; primary dedup key when present.
        repo_root_path: Filesystem repo root; dedup key when no remote.
        space_id: Optional space the project belongs to (used only on insert).
            When provided, must belong to ``org_id``.
        created_by: The resolved user id of the caller whose write is causing
            this project to exist, if known. Used only on INSERT (first
            writer wins) and only when the project is actually new.

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
                    display_name, created_by
                ) VALUES (
                    :id, :org_id, :space_id, :git_remote, :repo_root_path,
                    :display_name, :created_by
                )
                ON CONFLICT (org_id, git_remote) WHERE git_remote IS NOT NULL
                  DO UPDATE SET last_seen_at = NOW()
                RETURNING id, (xmax = 0) AS inserted
                """
            ),
            {
                "id": str(uuid.uuid4()),
                "org_id": org_id,
                "space_id": space_id,
                "git_remote": git_remote,
                "repo_root_path": repo_root_path,
                "display_name": display_name,
                "created_by": created_by,
            },
        ).one()
    else:
        row = db_session.execute(
            text(
                """
                INSERT INTO projects (
                    id, org_id, space_id, git_remote, repo_root_path,
                    display_name, created_by
                ) VALUES (
                    :id, :org_id, :space_id, NULL, :repo_root_path,
                    :display_name, :created_by
                )
                ON CONFLICT (org_id, repo_root_path)
                  WHERE git_remote IS NULL AND repo_root_path IS NOT NULL
                  DO UPDATE SET last_seen_at = NOW()
                RETURNING id, (xmax = 0) AS inserted
                """
            ),
            {
                "id": str(uuid.uuid4()),
                "org_id": org_id,
                "space_id": space_id,
                "repo_root_path": repo_root_path,
                "display_name": display_name,
                "created_by": created_by,
            },
        ).one()

    project_id = str(row.id)

    # Seat the creator as the project's first admin — but only for the write
    # that actually brought the row into existence. ``row.inserted`` is false
    # on every hit-path call (the ON CONFLICT DO UPDATE branch), so a second
    # writer in the same repo never becomes a member merely by writing there;
    # only the insert itself does, and only once.
    if row.inserted and created_by:
        db_session.execute(
            text(
                """
                INSERT INTO project_members (project_id, user_id, role, granted_by)
                VALUES (:project_id, :user_id, 'admin', :user_id)
                ON CONFLICT (project_id, user_id) DO NOTHING
                """
            ),
            {"project_id": project_id, "user_id": created_by},
        )

    db_session.commit()
    return project_id


def rename_project(
    db_session: Any, *, org_id: str, project_id: str, display_name: str
) -> bool:
    """Set a project's ``display_name``. Returns whether a row was updated.

    :func:`ensure_project` writes ``display_name`` on INSERT only — its
    ``ON CONFLICT DO UPDATE`` deliberately touches nothing but ``last_seen_at``,
    because it runs on the hot write path where the "name" is just whatever the
    current repo directory happens to be called and must never clobber a name a
    human chose. That leaves no way to name a project that already exists,
    which is precisely what ``project_create`` advertises.

    This is that missing operation, kept separate rather than folded into
    ``ensure_project`` so the two callers stay honest about their intent: an
    ordinary write proposes a name, and only an explicit, authorised rename
    imposes one.

    **No authorisation happens here.** A rename changes a project's shared
    identity for everyone in the org, so the caller MUST establish that the
    principal administers the project before calling — see
    ``_perform_project_create``, which does this only after its adoption rule
    has admitted the caller as an admin. Tenant scoping IS enforced here: the
    ``org_id`` predicate means a project id from another tenant matches no row
    and returns ``False`` rather than renaming it.

    Threading contract:
        Synchronous (blocking psycopg2 I/O). When called from an async handler,
        invoke via ``anyio.to_thread.run_sync(...)`` — never inline from a
        coroutine, or you block the event loop.

    Args:
        db_session: A synchronous SQLAlchemy session/connection.
        org_id: Owning org id (tenant scope).
        project_id: The project to rename.
        display_name: The new human-readable name.

    Returns:
        ``True`` if a row was updated, ``False`` if no project in this org has
        that id.
    """
    row = db_session.execute(
        text(
            """
            UPDATE projects
            SET display_name = :display_name
            WHERE id = :project_id AND org_id = :org_id
            RETURNING id
            """
        ),
        {
            "display_name": display_name,
            "project_id": project_id,
            "org_id": org_id,
        },
    ).fetchone()
    db_session.commit()
    return row is not None


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

    Resolution order (all tenant-scoped by ``org_id``), most specific first:

        1. If ``identifier`` parses as a UUID, look up by ``id``.
        2. ``git_remote`` — unique per org via ``ix_projects_org_remote``.
        3. ``repo_root_path`` — unique per org for non-git projects via
           ``ix_projects_org_path``.
        4. ``display_name`` — matched CASE-INSENSITIVELY; NOT unique; see the
           ambiguity note below.

    Why the name match is case-insensitive and the other two are not:
    ``git_remote`` and ``repo_root_path`` are machine-produced identifiers
    where case is meaningful (a path can differ only in case on a
    case-sensitive filesystem), so they stay exact. ``display_name`` is what a
    person types. A project auto-created from a repo is named after its last
    path segment — ``klio`` — so the human who then types ``Klio``, including
    the human who just named it that, got ``unknown_project`` and every tool
    taking a project identifier failed for them. Case is not a distinction any
    caller intends to draw here.

    Why ``repo_root_path`` is here: :func:`ensure_project` dedups on
    ``git_remote`` when present and ``repo_root_path`` otherwise, so a project
    created outside a git repo is keyed by its directory. Omitting the path from
    this read path made those projects unresolvable — the caller fell back to
    org-wide recall — even though the writer had filed memories under them. In
    production two thirds of projects are path-only, so this was the common case,
    not an edge case.

    Ambiguity: ``display_name`` is the last path segment, so one org routinely
    holds several projects named ``web`` or ``app``. The previous query ended in
    a bare ``LIMIT 1`` with no ``ORDER BY``, making the winner arbitrary and
    unstable between calls — silently resolving to the WRONG project, which is
    worse than not resolving at all. Matches are now ranked by specificity, and
    ties within ``display_name`` break by most-recently-seen then by id, so the
    result is deterministic and the likeliest intent wins. Case-insensitivity
    widens that tie bucket — ``web``, ``Web`` and ``WEB`` collapse into one — so
    it is only safe BECAUSE that total ordering is already there; the two
    changes cannot be separated.

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
        identifier: A project id (UUID), git remote URL, repository root path,
            or display name.

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

    # Ranked by specificity so an exact git_remote or repo_root_path always
    # beats a display_name collision, and ordered fully so the winner is
    # deterministic across calls rather than whatever the planner returned first.
    row = db_session.execute(
        text(
            """
            SELECT id FROM projects
            WHERE org_id = :org_id
              AND (
                git_remote = :ident
                OR repo_root_path = :ident
                OR lower(display_name) = lower(:ident)
              )
            ORDER BY
              CASE
                WHEN git_remote = :ident THEN 0
                WHEN repo_root_path = :ident THEN 1
                ELSE 2
              END,
              last_seen_at DESC NULLS LAST,
              id ASC
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
    recall_hidden: bool = False,
    user_id: Optional[str] = None,
    provenance: str = "EXTRACTED",
) -> dict:
    """Write a memory into the Klio brain at org, project, private or agent scope.

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
        * ``scope='private'`` — the row belongs to ``user_id`` and to nobody
          else: not teammates, not admins. ``user_id`` is therefore MANDATORY
          and a missing one raises rather than writing an owner-less private
          row (migration 031's ``ck_session_memories_private_has_owner`` would
          reject it anyway, but failing here says why). Dedup is user-scoped —
          see :func:`shared.memory.find_similar_memory` for why that window is
          a privacy control and not merely a correctness one.
        * ``scope='project'`` — the row is readable by members of
          ``project_id``, which is therefore also MANDATORY. Note the caller,
          not this function, is responsible for having checked that the author
          is entitled to write there; this layer only refuses the
          structurally-impossible case.
        * ``scope='agent'`` — the row is promoted to ``scope='agent'`` and stays
          private to ``agent_id``. :func:`retrieve_memories_scoped` with
          ``scope='agent'`` returns it to that agent (plus org rows) but never to
          another agent. Dedup is agent-scoped (only against this agent's own
          private rows), so one agent's write never dedups against the org brain
          or another agent's row. Deprecated in favour of ``private``: an
          ``agent_id`` comes from the client-declared ``X-Vex-Agent`` header and
          is not an identity.

    ``user_id`` on a SHARED row (``org`` / ``project``) is authorship, not
    visibility. It records who taught the team something and grants that person
    nothing extra; only ``scope='private'`` turns it into an access predicate.

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
        recall_hidden: When ``True`` the row is stored but excluded from agent
            recall results (``retrieve_memories_scoped`` and the recency
            fallback). Raw ``observation`` rows written by passive hooks should
            pass ``True``; deliberate types (``note``, ``plan``, ``decision``,
            ``memory``) should keep the default ``False``.
        user_id: The resolved human. Mandatory for ``scope='private'``,
            recorded as authorship on shared scopes, and ``None`` when the
            request could not be attributed to a person.
        provenance: ``EXTRACTED`` (default — stated or done), ``INFERRED``
            (deduced by the curator or an LLM pass), or ``AMBIGUOUS``.

    Returns:
        A result dict describing the outcome (``scope`` echoes the requested
        scope):

        * redaction failure →
          ``{"stored": False, "error": "redaction_failed"}``
          The redactor raised an unexpected exception; the unscreened content
          was rejected (fail-closed). No DB write was attempted.
        * dedup hit →
          ``{"stored": False, "deduplicated": True, "id": ..., "memory_type":
          ..., "scope": <scope>, "redacted": <bool>, "redaction_kinds": [...]}``
        * embedding failure →
          ``{"stored": False, "error": "embedding_failed"}``
        * success →
          ``{"stored": True, "deduplicated": False, "id": ..., "memory_type":
          ..., "scope": <scope>, "space_id": ..., "project_id": ...,
          "redacted": <bool>, "redaction_kinds": [...]}``

        ``redacted`` is ``True`` when at least one secret/PII pattern was
        detected and masked. ``redaction_kinds`` is a sorted list of the
        distinct kind strings (e.g. ``["stripe_secret_key"]``); it is empty
        when no secrets were found.  When ``MEMORY_REDACTION_DISABLED=1``
        both keys are still present with their no-op values (``False`` / ``[]``).

    Raises:
        ValueError: If ``scope`` is not one of ``'org'`` / ``'project'`` /
            ``'private'`` / ``'agent'``; if ``scope='private'`` without a
            ``user_id``; if ``scope='project'`` without a ``project_id``; or if
            ``provenance`` is not one of the three labels.
    """
    if scope not in ("org", "project", "private", "agent"):
        raise ValueError(
            f"unsupported scope: {scope!r} (write_scoped_memory accepts "
            "'org', 'project', 'private' or 'agent')"
        )
    if scope == "private" and not user_id:
        # An owner-less private row is unreachable by every caller including
        # its author — a bug by construction, not a state to handle.
        raise ValueError("user_id is required when scope='private'")
    if scope == "project" and not project_id:
        raise ValueError("project_id is required when scope='project'")
    if provenance not in ("EXTRACTED", "INFERRED", "AMBIGUOUS"):
        raise ValueError(f"unsupported provenance: {provenance!r}")

    # --- redaction (fail closed) -----------------------------------------
    # Mask secrets/PII BEFORE dedup-embedding and storage so neither the
    # stored text nor any embedding ever contains a detected secret. A
    # redactor crash rejects the write: unscreened content is never stored.
    # MEMORY_REDACTION_DISABLED=1 is a self-hosted escape hatch only.
    redaction_kinds: list[str] = []
    was_redacted = False
    if os.environ.get("MEMORY_REDACTION_DISABLED") != "1":
        try:
            redaction = redact(content)
            # Prose carried in metadata is screened too. `decide` files its
            # caller-supplied rationale there, and that field went to storage
            # unscreened — a secret pasted into a rationale was readable in the
            # clear and fed to the curator's LLM as synthesis fuel.
            metadata, metadata_kinds = screen_metadata(metadata)
        except Exception:
            logger.exception("redaction failed; rejecting write (fail closed)")
            return {"stored": False, "error": "redaction_failed"}
        if redaction.redacted or metadata_kinds:
            if redaction.redacted:
                content = redaction.clean_text
            redaction_kinds = sorted(
                {f.kind for f in redaction.findings} | set(metadata_kinds)
            )
            was_redacted = True
            metadata = dict(metadata or {})
            metadata["redactions"] = redaction_kinds

    if dedup:
        existing = await find_similar_memory(
            db_session,
            session_id,
            org_id,
            content,
            # Private and agent-private writes dedup only against the SAME
            # owner's rows; org/project writes keep the session-scoped window.
            # A private write must never be compared against the org brain or
            # another person's rows: the returned dedup hit carries a row id.
            scope=scope if scope in ("agent", "private") else None,
            agent_id=agent_id if scope == "agent" else None,
            user_id=user_id if scope == "private" else None,
        )
        if existing is not None:
            return {
                "stored": False,
                "deduplicated": True,
                "id": str(existing.id),
                "memory_type": existing.memory_type,
                "scope": scope,
                "redacted": was_redacted,
                "redaction_kinds": redaction_kinds,
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
        recall_hidden=recall_hidden,
        user_id=user_id,
        provenance=provenance,
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
        "user_id": user_id,
        "provenance": provenance,
        "redacted": was_redacted,
        "redaction_kinds": redaction_kinds,
    }
