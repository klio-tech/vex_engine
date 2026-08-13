"""Project-ownership-transfer sweep — the reconciliation pass that keeps a
departure in Supabase from leaving a Klio project memberless.

Spec: ``docs/superpowers/specs/2026-08-13-project-ownership-transfer-design.md``.
Background: migration 039 seated exactly one owner on 133 of 134 production
projects (``docs/superpowers/specs/2026-08-12-project-listing-visibility-
design.md``), so a single departure makes that project invisible to
everyone — nobody can list it, and its project-scope content becomes
unreachable. Nothing is deleted; the project itself goes dark.

**THE INVARIANT, above everything else in this module: a project must never
have zero members, not even transiently.** Every write this module performs
seats the workspace owner and removes the departed row in the SAME database
transaction (see :func:`_execute_transfer`), ordered seat-then-remove. Since a
Postgres transaction is all-or-nothing, a process death between the two
statements leaves the project in its PRE-transaction state (departed row still
present) rather than a half-applied one — the project is never briefly
memberless either way.

**THE FAILURE MODE THAT MATTERS: a false "this user has departed" reading
would strip memberships across the whole database.** Every departure
judgement in this module comes from exactly one place —
``shared.workspace_owner.fetch_workspace_membership`` — and this module
treats its three possible outcomes as three DIFFERENT branches, never
collapsed into one:

* the query raises :class:`~shared.workspace_owner.WorkspaceLookupError`
  (network blip, bad credentials, Supabase down) -> the org is SKIPPED
  entirely, no rows touched. An error is not evidence that nobody is a
  member.
* the query succeeds but returns an EMPTY set -> the org is SKIPPED, with a
  loud warning. A real workspace always has at least its owner as a member;
  an empty result means something is wrong upstream, not that everyone left
  at once.
* the workspace owner cannot be resolved (``resolve_workspace_owners`` has no
  answer for this org) -> any project that would otherwise be transferred is
  left EXACTLY as it is — the departed row is NOT removed either. A dark
  project is recoverable later; a memberless one after this sweep deleted the
  last row is not.

**Transfer vs. plain cleanup.** A departure only triggers an ownership
transfer (seating the workspace owner) when it would otherwise leave a
project with ZERO current members. A departure from a project that still has
other current members is a plain row removal — no owner is seated, because
nobody's visibility was ever at risk. (Test case 2 in the spec's plan is
exactly this: "no transfer needed, project untouched for the others".)

**The per-run transfer cap.** ``max_transfers`` bounds how many ownership
transfers (owner-seat events) a single sweep pass will perform, counted
whether the run is a dry run or a real one — dry-run output must be a
faithful preview of what a real run would do, including where it would stop.
Once the cap is reached, the WHOLE sweep pass stops immediately (further
orgs and projects are left untouched this run) and :attr:`SweepResult.cap_tripped`
is set. This bounds the blast radius of a bug that produces a
plausible-but-wrong membership set: a cascade trips the cap and surfaces
rather than running to completion. Plain multi-member cleanup is NOT capped —
it cannot violate the zero-member invariant by construction (see above), so
there is nothing for the cap to defend against there.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

from sqlalchemy import text
from sqlalchemy.engine import Engine

from shared.projects import SURFACE_SWEEP, record_grant_audit
from shared.workspace_owner import (
    WorkspaceLookupError,
    fetch_workspace_membership,
    resolve_workspace_owners,
)

logger = logging.getLogger("agentguard.async-worker.ownership_sweep")

#: ``granted_by`` on every audit row this sweep writes. Not a Supabase user
#: UUID (unlike a human-triggered grant) — deliberately a distinct, greppable
#: sentinel so "how did this project's membership change" is answerable
#: without ambiguity months later. No FK or format constraint on the column
#: forbids this (see migration 032/033's docstrings).
SWEEP_ACTOR = "system:ownership-sweep"

ACTION_GRANT = "grant"
ACTION_REVOKE = "revoke"


@dataclass(frozen=True)
class TransferAction:
    """One project whose ownership moved to the workspace owner this pass."""

    org_id: str
    project_id: str
    departed_user_ids: tuple[str, ...]
    new_owner_user_id: str


@dataclass(frozen=True)
class RemovalAction:
    """One project that lost a departed member but needed no transfer —
    other current members remain, so nobody's visibility was at risk."""

    org_id: str
    project_id: str
    departed_user_ids: tuple[str, ...]


@dataclass
class SweepResult:
    """Everything one sweep pass did (or, in dry-run, would have done)."""

    dry_run: bool
    orgs_considered: int = 0
    orgs_skipped_no_supabase_config: bool = False
    orgs_skipped_lookup_failed: list[str] = field(default_factory=list)
    orgs_skipped_empty_membership: list[str] = field(default_factory=list)
    projects_left_alone_owner_unresolved: list[str] = field(default_factory=list)
    transfers: list[TransferAction] = field(default_factory=list)
    removals: list[RemovalAction] = field(default_factory=list)
    cap_tripped: bool = False

    @property
    def transfer_count(self) -> int:
        return len(self.transfers)


def _distinct_orgs_with_projects(conn) -> list[str]:
    rows = conn.execute(
        text(
            """
            SELECT DISTINCT p.org_id
            FROM project_members pm
            JOIN projects p ON p.id = pm.project_id
            ORDER BY p.org_id
            """
        )
    ).fetchall()
    return [row.org_id for row in rows]


def _account_slugs(conn, org_ids: list[str]) -> dict[str, Optional[str]]:
    if not org_ids:
        return {}
    rows = conn.execute(
        text("SELECT org_id, account_slug FROM organizations WHERE org_id = ANY(:ids)"),
        {"ids": org_ids},
    ).fetchall()
    return {row.org_id: row.account_slug for row in rows}


def _project_members_by_org(conn, org_id: str) -> dict[str, list[tuple[str, str]]]:
    """``{project_id: [(user_id, role), ...]}`` for every project this org's
    ``project_members`` rows touch, as of right now."""
    rows = conn.execute(
        text(
            """
            SELECT pm.project_id, pm.user_id, pm.role
            FROM project_members pm
            JOIN projects p ON p.id = pm.project_id
            WHERE p.org_id = :org_id
            ORDER BY pm.project_id, pm.user_id
            """
        ),
        {"org_id": org_id},
    ).fetchall()
    by_project: dict[str, list[tuple[str, str]]] = {}
    for row in rows:
        by_project.setdefault(str(row.project_id), []).append((row.user_id, row.role))
    return by_project


def _execute_transfer(
    engine: Engine,
    *,
    project_id: str,
    org_id: str,
    owner_user_id: str,
    departed: list[tuple[str, str]],
) -> None:
    """Seat the owner, then remove the departed rows — ONE transaction.

    Ordered seat-then-remove per the module's invariant. ``ON CONFLICT DO
    NOTHING`` on the seat is load-bearing exactly as it is in migration 039:
    an existing membership or role for the owner is never clobbered or
    downgraded, whether that row predates this sweep or was seated by an
    earlier pass.
    """
    departed_ids = [user_id for user_id, _role in departed]
    with engine.begin() as tx:
        tx.execute(
            text(
                """
                INSERT INTO project_members (project_id, user_id, role, granted_by)
                VALUES (:project_id, :owner, 'admin', :granted_by)
                ON CONFLICT (project_id, user_id) DO NOTHING
                """
            ),
            {
                "project_id": project_id,
                "owner": owner_user_id,
                "granted_by": SWEEP_ACTOR,
            },
        )
        tx.execute(
            text(
                """
                DELETE FROM project_members
                WHERE project_id = :project_id AND user_id = ANY(:departed)
                """
            ),
            {"project_id": project_id, "departed": departed_ids},
        )

    # Audit, best-effort, after the transfer has committed — record_grant_audit
    # never lets an audit failure undo the change it describes (see its
    # docstring in shared/projects.py).
    with engine.connect() as audit_conn:
        record_grant_audit(
            audit_conn,
            org_id=org_id,
            project_id=project_id,
            granted_to=owner_user_id,
            granted_to_email=None,
            granted_by=SWEEP_ACTOR,
            role="admin",
            surface=SURFACE_SWEEP,
            action=ACTION_GRANT,
        )
        for user_id, role in departed:
            record_grant_audit(
                audit_conn,
                org_id=org_id,
                project_id=project_id,
                granted_to=user_id,
                granted_to_email=None,
                granted_by=SWEEP_ACTOR,
                role=role,
                surface=SURFACE_SWEEP,
                action=ACTION_REVOKE,
            )


def _execute_removal(
    engine: Engine, *, project_id: str, org_id: str, departed: list[tuple[str, str]]
) -> None:
    """Remove departed rows from a project that still has other current
    members — no owner seat, because nobody's visibility was at risk."""
    departed_ids = [user_id for user_id, _role in departed]
    with engine.begin() as tx:
        tx.execute(
            text(
                """
                DELETE FROM project_members
                WHERE project_id = :project_id AND user_id = ANY(:departed)
                """
            ),
            {"project_id": project_id, "departed": departed_ids},
        )

    with engine.connect() as audit_conn:
        for user_id, role in departed:
            record_grant_audit(
                audit_conn,
                org_id=org_id,
                project_id=project_id,
                granted_to=user_id,
                granted_to_email=None,
                granted_by=SWEEP_ACTOR,
                role=role,
                surface=SURFACE_SWEEP,
                action=ACTION_REVOKE,
            )


def run_ownership_sweep(
    engine: Engine,
    *,
    supabase_engine: Optional[Engine],
    dry_run: bool,
    max_transfers: int,
) -> SweepResult:
    """Run one reconciliation pass. Never mutates anything when ``dry_run``.

    ``engine`` is the main Klio database. ``supabase_engine`` is the engine
    built from ``SUPABASE_DATABASE_URL`` — ``None`` means Supabase is not
    configured on this deployment at all, in which case NO org's departure
    can be honestly evaluated and the whole pass is skipped (fails closed,
    same as every other cross-database identity lookup in this codebase).
    """
    result = SweepResult(dry_run=dry_run)

    if supabase_engine is None:
        result.orgs_skipped_no_supabase_config = True
        logger.warning(
            "ownership sweep: no SUPABASE_DATABASE_URL configured; skipping "
            "the whole pass — no org's membership can be honestly evaluated "
            "without it"
        )
        return result

    with engine.connect() as conn:
        org_ids = _distinct_orgs_with_projects(conn)
        slug_by_org = _account_slugs(conn, org_ids)

    result.orgs_considered = len(org_ids)
    owners_cache: dict[str, Optional[str]] = {}

    for org_id in org_ids:
        if result.cap_tripped:
            break

        slug = slug_by_org.get(org_id)
        try:
            membership_set = fetch_workspace_membership(
                supabase_engine, org_id=org_id, account_slug=slug
            )
        except WorkspaceLookupError:
            logger.error(
                "ownership sweep: Supabase membership query failed for "
                "org=%s; skipping this org entirely this pass",
                org_id,
                exc_info=True,
            )
            result.orgs_skipped_lookup_failed.append(org_id)
            continue

        if not membership_set:
            logger.warning(
                "ownership sweep: org=%s resolved to an EMPTY Supabase "
                "membership set; a real workspace always has at least its "
                "owner, so this looks like a bad account/slug resolution, "
                "not a mass departure. Skipping this org entirely this pass.",
                org_id,
            )
            result.orgs_skipped_empty_membership.append(org_id)
            continue

        with engine.connect() as conn:
            members_by_project = _project_members_by_org(conn, org_id)

        for project_id, members in members_by_project.items():
            if result.cap_tripped:
                break

            departed = [(u, r) for u, r in members if u not in membership_set]
            if not departed:
                continue  # every row is a current member: a true no-op.

            current = [(u, r) for u, r in members if u in membership_set]
            if current:
                # Other current members remain — plain cleanup, no transfer.
                result.removals.append(
                    RemovalAction(
                        org_id=org_id,
                        project_id=project_id,
                        departed_user_ids=tuple(u for u, _r in departed),
                    )
                )
                if not dry_run:
                    _execute_removal(
                        engine, project_id=project_id, org_id=org_id, departed=departed
                    )
                continue

            # No current members would remain: this project needs a transfer.
            if org_id not in owners_cache:
                with engine.connect() as conn:
                    owners_cache.update(
                        resolve_workspace_owners(
                            conn, [org_id], supabase_engine=supabase_engine
                        )
                    )
                owners_cache.setdefault(org_id, None)

            owner = owners_cache.get(org_id)
            if not owner:
                logger.error(
                    "ownership sweep: org=%s has an unresolvable workspace "
                    "owner; project=%s would go memberless on transfer, so "
                    "it is left EXACTLY as-is — the departed row is NOT "
                    "removed either",
                    org_id,
                    project_id,
                )
                result.projects_left_alone_owner_unresolved.append(project_id)
                continue

            if result.transfer_count >= max_transfers:
                result.cap_tripped = True
                logger.critical(
                    "ownership sweep: transfer cap (%d) reached; stopping "
                    "this pass immediately rather than cascading further. "
                    "org=%s project=%s was NOT processed this pass.",
                    max_transfers,
                    org_id,
                    project_id,
                )
                break

            # Defense in depth: if the resolved owner themselves shows up in
            # `departed` (e.g. their own project_members row predates a gap
            # in accounts_memberships — primary_owner_user_id set but no
            # current membership row), never let them be seated and then
            # immediately deleted by the same transfer. The seat is the
            # thing that makes them current; the removal list must exclude
            # them regardless of how they were classified above.
            departed_to_remove = [(u, r) for u, r in departed if u != owner]

            result.transfers.append(
                TransferAction(
                    org_id=org_id,
                    project_id=project_id,
                    departed_user_ids=tuple(u for u, _r in departed_to_remove),
                    new_owner_user_id=owner,
                )
            )
            if not dry_run:
                _execute_transfer(
                    engine,
                    project_id=project_id,
                    org_id=org_id,
                    owner_user_id=owner,
                    departed=departed_to_remove,
                )

    return result
