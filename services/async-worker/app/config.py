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
