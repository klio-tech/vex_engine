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

    Looks up an existing space by ``(org_id, slug)`` (the
    ``uq_spaces_org_slug`` unique constraint). If found, its id is returned
    unchanged. Otherwise a new row is inserted and committed.

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
    existing = db_session.execute(
        text(
            """
            SELECT id FROM spaces
            WHERE org_id = :org_id AND slug = :slug
            LIMIT 1
            """
        ),
        {"org_id": org_id, "slug": slug},
    ).fetchone()

    if existing is not None:
        return str(existing[0])

    space_id = str(uuid.uuid4())
    db_session.execute(
        text(
            """
            INSERT INTO spaces (id, org_id, name, slug, embedding_model)
            VALUES (:id, :org_id, :name, :slug, :embedding_model)
            """
        ),
        {
            "id": space_id,
            "org_id": org_id,
            "name": name,
            "slug": slug,
            "embedding_model": embedding_model,
        },
    )
    db_session.commit()
    return space_id


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
    ``git_remote IS NULL``).

    On a hit, ``last_seen_at`` is set to ``NOW()`` and the existing id is
    returned. On a miss, a new row is inserted. Either way the transaction is
    committed.

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
        existing = db_session.execute(
            text(
                """
                SELECT id FROM projects
                WHERE org_id = :org_id AND git_remote = :git_remote
                LIMIT 1
                """
            ),
            {"org_id": org_id, "git_remote": git_remote},
        ).fetchone()
    else:
        existing = db_session.execute(
            text(
                """
                SELECT id FROM projects
                WHERE org_id = :org_id
                  AND git_remote IS NULL
                  AND repo_root_path = :repo_root_path
                LIMIT 1
                """
            ),
            {"org_id": org_id, "repo_root_path": repo_root_path},
        ).fetchone()

    if existing is not None:
        project_id = str(existing[0])
        db_session.execute(
            text("UPDATE projects SET last_seen_at = NOW() WHERE id = :id"),
            {"id": project_id},
        )
        db_session.commit()
        return project_id

    project_id = str(uuid.uuid4())
    db_session.execute(
        text(
            """
            INSERT INTO projects (
                id, org_id, space_id, git_remote, repo_root_path, display_name
            ) VALUES (
                :id, :org_id, :space_id, :git_remote, :repo_root_path,
                :display_name
            )
            """
        ),
        {
            "id": project_id,
            "org_id": org_id,
            "space_id": space_id,
            "git_remote": git_remote,
            "repo_root_path": repo_root_path,
            "display_name": display_name,
        },
    )
    db_session.commit()
    return project_id
