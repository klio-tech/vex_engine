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
        The caller must guarantee that any supplied ``space_id`` belongs to
        ``org_id``. The FK only enforces that the space *exists*, not that it
        belongs to this org; cross-org leakage is not prevented at the DB layer.

    Args:
        db_session: A synchronous SQLAlchemy session/connection exposing
            ``.execute()`` and ``.commit()``.
        org_id: Owning org id (tenant scope).
        display_name: Human-readable project name (used only on insert).
        git_remote: Canonical git remote URL; primary dedup key when present.
        repo_root_path: Filesystem repo root; dedup key when no remote.
        space_id: Optional space the project belongs to (used only on insert).

    Returns:
        The project id as a string.

    Raises:
        ValueError: If both ``git_remote`` and ``repo_root_path`` are ``None``
            (mirrors the DB ``ck_projects_has_identity`` check; an identity-less
            project is unrepresentable).
    """
    if git_remote is None and repo_root_path is None:
        raise ValueError("project requires git_remote or repo_root_path")

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
