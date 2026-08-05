"""Shared safety rails for the WP-G reconciliation scripts.

Three defects left residue in production. These scripts exist to **measure**
it. Producing counts is the deliverable; changing data is a separate decision
a human makes after reading those counts, so every script in the set is
dry-run by default and executing is deliberately awkward.

Two independent conditions gate a write:

``--execute``
    An explicit, unambiguous flag. Nothing else spells it, and no environment
    variable can supply it — a guard keyed on the environment is one stale
    exported shell variable away from being no guard at all.

``--confirm <token>``
    A token this module cannot produce on its own behalf. It is a digest over
    the exact candidate set the dry-run measured (plus the script name and the
    org), and the dry-run is what prints it. So an operator cannot reach the
    execute path without having first run and read a dry-run — and the token
    stops matching the moment the population moves, which is the point: the
    thing they approved was a set of rows, not a verb.

Execution additionally requires ``--org``. The counting mode may sweep the
fleet; the mutating mode is always scoped to one tenant. ``run_retention.py``
holds the repo's one deliberately global DELETE, documented at length as
such — nothing here copies that shape.

Output hygiene: these scripts read ``scope='private'`` rows and, in the case
of :mod:`reconcile_unredacted_metadata`, rows selected precisely *because*
they contain an unmasked secret. Nothing but identifiers, counts, timestamps
and scope labels may ever reach a log, so :func:`format_sample` refuses to
render anything that is not identifier-shaped rather than trusting its
callers.
"""

import argparse
import hashlib
import re
from typing import Iterable, Optional, Sequence

#: How many ids a dry-run may print. Enough to spot-check a population by
#: hand, far short of exporting it.
SAMPLE_LIMIT = 10

#: The identifier shapes these scripts legitimately emit: a UUID, and an
#: object key (``<org_id>/<artifact_id>/v<n>``). Deliberately strict — the
#: printer is the last checkpoint before a log, and a permissive pattern here
#: is how memory content ends up in one.
_IDENTIFIER_RE = re.compile(r"\A[A-Za-z0-9][A-Za-z0-9_.:/\-]{0,190}\Z")


class ConfirmationError(RuntimeError):
    """Execution was requested but not properly confirmed; nothing was written."""


def build_parser(description: str) -> argparse.ArgumentParser:
    """The argument surface every reconciliation script shares.

    ``--execute`` has no short form and no alias on purpose. There is exactly
    one way to spell "write to the database", and it is not one keystroke away
    from anything else on the command line.
    """
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--org",
        default=None,
        help=("Restrict to one org_id. Optional when counting; REQUIRED to execute."),
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help=(
            "Write changes. Requires --org and a --confirm token printed by a "
            "dry-run of the same population. Omit for a dry run."
        ),
    )
    parser.add_argument(
        "--confirm",
        default=None,
        metavar="TOKEN",
        help="The confirmation token printed by the dry-run you reviewed.",
    )
    parser.add_argument(
        "--sample-limit",
        type=int,
        default=SAMPLE_LIMIT,
        help=f"How many ids to print per population (default {SAMPLE_LIMIT}).",
    )
    return parser


def confirmation_token(
    script: str, org_id: Optional[str], candidate_ids: Iterable[str]
) -> str:
    """Digest identifying one script, one org, and one exact candidate set.

    Order-insensitive, because the candidate set is a set — a different SQL
    plan returning the same rows in another order must not invalidate a token
    an operator is holding. Membership changing must.
    """
    digest = hashlib.sha256()
    digest.update(script.encode("utf-8"))
    digest.update(b"\x00")
    digest.update((org_id or "*ALL*").encode("utf-8"))
    digest.update(b"\x00")
    for candidate in sorted({str(c) for c in candidate_ids}):
        digest.update(candidate.encode("utf-8"))
        digest.update(b"\x00")
    return digest.hexdigest()[:16]


def resolve_execute(
    args: argparse.Namespace,
    *,
    script: str,
    candidate_ids: Sequence[str],
) -> bool:
    """Decide whether this run may write. ``False`` means dry-run.

    Args:
        args: Parsed arguments from :func:`build_parser`.
        script: The script's module name; scopes the token so the three
            scripts' tokens cannot be swapped for one another.
        candidate_ids: The population just measured. The token is checked
            against *this* set, not against whatever was measured earlier.

    Returns:
        ``True`` only when ``--execute``, ``--org`` and a matching
        ``--confirm`` are all present.

    Raises:
        ConfirmationError: When execution was asked for and refused. Always
            raised rather than downgraded to a dry-run: an operator who typed
            ``--execute`` and got a silent no-op would reasonably conclude the
            population was empty.
    """
    if not args.execute:
        return False

    if not args.org:
        raise ConfirmationError(
            "--execute requires --org. Counting may span the fleet; writing is "
            "always scoped to one tenant."
        )

    expected = confirmation_token(script, args.org, candidate_ids)
    if not args.confirm:
        raise ConfirmationError(
            "--execute requires --confirm <token>. Run this script without "
            "--execute, review the counts, and pass the token it prints. "
            f"For the population just measured that token is: {expected}"
        )
    if args.confirm.strip().lower() != expected:
        raise ConfirmationError(
            "the --confirm token no longer matches this population: the rows "
            "in scope have changed since the dry-run that produced it, or the "
            "token belongs to another script or org. Re-run the dry-run and "
            "review the current counts. Nothing was written."
        )
    return True


def format_sample(ids: Sequence[str], limit: int = SAMPLE_LIMIT) -> str:
    """Render a bounded sample of identifiers for a log line.

    Raises:
        ValueError: If any element is not identifier-shaped. These scripts
            select rows *because* they are private or because they contain an
            unmasked secret; a caller that reaches for this function with a
            content column has made a mistake that must stop here rather than
            be printed.
    """
    if not ids:
        return "none"
    for candidate in ids:
        if not _IDENTIFIER_RE.match(str(candidate)):
            raise ValueError(
                "refusing to print a non-identifier value; only ids, counts, "
                "timestamps and scopes may be logged"
            )
    shown = [str(c) for c in ids[: max(limit, 0)]]
    return f"{', '.join(shown)} ({len(shown)} of {len(ids)})"
