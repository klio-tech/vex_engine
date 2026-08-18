"""Configuration for the async worker, including the ownership-transfer sweep.

Mirrors ``services/mcp-server/app/config.py``'s pattern: environment-driven,
parsed through ``shared.env_config`` so a malformed value falls back to a
safe default with a logged warning instead of crashing the process or
silently disagreeing with the operator.
"""

import os

from shared.env_config import parse_bool, parse_int

DATABASE_URL = os.environ.get(
    "DATABASE_URL",
    "postgresql://agentguard:agentguard_dev@localhost:5432/agentguard",  # pragma: allowlist secret
)

#: Set by ops when a Supabase project backs this deployment. ``None`` means
#: the ownership sweep cannot honestly evaluate any org's membership and
#: skips its entire pass — see ``app.ownership_sweep.run_ownership_sweep``.
SUPABASE_DATABASE_URL = os.environ.get("SUPABASE_DATABASE_URL")

OWNERSHIP_SWEEP_ENABLED = parse_bool(
    os.environ.get("OWNERSHIP_SWEEP_ENABLED"),
    default=True,
    name="OWNERSHIP_SWEEP_ENABLED",
)

# Defaults to ARMED = False. Rollout ("Deploy with dry-run enabled. Inspect
# one production cycle's log. Arm it.") requires that a plain deploy of this
# code cannot mutate project_members until someone deliberately sets
# OWNERSHIP_SWEEP_DRY_RUN=0.
OWNERSHIP_SWEEP_DRY_RUN = parse_bool(
    os.environ.get("OWNERSHIP_SWEEP_DRY_RUN"),
    default=True,
    name="OWNERSHIP_SWEEP_DRY_RUN",
)

OWNERSHIP_SWEEP_INTERVAL_SECONDS = parse_int(
    os.environ.get("OWNERSHIP_SWEEP_INTERVAL_SECONDS"),
    default=3600,
    name="OWNERSHIP_SWEEP_INTERVAL_SECONDS",
    minimum=60,
)

#: A configurable ceiling on ownership transfers (owner-seat events) per
#: sweep pass. Defaults low on purpose (spec: "a cascade should trip the cap
#: and surface, not run to completion") — production today has 133
#: single-owner projects total, so a healthy pass transferring anywhere near
#: this many in one tick is itself a signal something upstream is wrong.
OWNERSHIP_SWEEP_MAX_TRANSFERS_PER_RUN = parse_int(
    os.environ.get("OWNERSHIP_SWEEP_MAX_TRANSFERS_PER_RUN"),
    default=25,
    name="OWNERSHIP_SWEEP_MAX_TRANSFERS_PER_RUN",
    minimum=1,
)

#: The dreamer (``app.dreamer.dream_for_org``) runs for an org once it has
#: accumulated this many newly-graded sessions since its last dream pass.
#: "Newly graded" is counted in ``dream_state.sessions_since`` (migration
#: 046) — PERSISTED, not per-process. The previous in-memory counter reset on
#: every redeploy, which is why this threshold was never reached in
#: production and ``memory_proposals`` held zero rows.
#:
#: WHY 10, NOT 25. The old default of 25 was set against no measurement. At
#: the time this was diagnosed the entire production corpus was 19
#: ``session_traces`` rows and 10 ``session_outcomes`` rows org-wide, so a
#: 25-session threshold was indistinguishable from "never" — it could not be
#: crossed even by grading every session that had ever existed. 10 is chosen
#: to be reachable by a real young org within days of normal use while still
#: requiring a genuine cluster of sessions rather than firing on one. The
#: interval trigger below remains the backstop for an org that grades more
#: slowly than that.
DREAM_MIN_GRADED_SESSIONS = parse_int(
    os.environ.get("DREAM_MIN_GRADED_SESSIONS"),
    default=10,
    name="DREAM_MIN_GRADED_SESSIONS",
    minimum=1,
)

#: ...or this many hours have elapsed since its last dream pass, whichever
#: comes first — so a low-traffic org still gets dreamed for occasionally
#: rather than waiting indefinitely for DREAM_MIN_GRADED_SESSIONS sessions.
#: Persisted in ``dream_state.last_run_at`` (migration 046) and, since the
#: periodic tick below exists, this is now a real timer rather than a
#: condition only evaluated when a grading event happens to arrive.
DREAM_INTERVAL_HOURS = parse_int(
    os.environ.get("DREAM_INTERVAL_HOURS"),
    default=24,
    name="DREAM_INTERVAL_HOURS",
    minimum=1,
)

#: How often the periodic dream tick asks the DATABASE which orgs are past
#: their ``DREAM_INTERVAL_HOURS`` clock. This is not the dream cadence — it
#: is only the polling rate for the timer; ``DREAM_INTERVAL_HOURS`` still
#: decides when an org is actually due. Cheap (one indexed query per tick),
#: so it defaults to a few minutes: the interval trigger should fire close to
#: when it comes due rather than whenever traffic next happens to arrive.
DREAM_TICK_SECONDS = parse_int(
    os.environ.get("DREAM_TICK_SECONDS"),
    default=300,
    name="DREAM_TICK_SECONDS",
    minimum=10,
)

#: A ceiling on how many orgs ONE periodic tick will dream for, matching the
#: ownership sweep's per-pass cap convention: a backlog (e.g. the first tick
#: after this migration lands, when every org's clock is seeded from its
#: oldest trace and many are immediately due) drains over several ticks
#: instead of blocking one worker thread on every org at once.
DREAM_MAX_ORGS_PER_TICK = parse_int(
    os.environ.get("DREAM_MAX_ORGS_PER_TICK"),
    default=5,
    name="DREAM_MAX_ORGS_PER_TICK",
    minimum=1,
)
