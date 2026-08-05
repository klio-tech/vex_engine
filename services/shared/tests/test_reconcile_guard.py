"""The shared execution guard for the WP-G reconciliation scripts.

These three scripts exist to *measure* the residue three defects left behind.
The measurement is the deliverable; changing the data is a separate, human
decision. So the guard is the load-bearing part of the package, and it is
tested harder than anything the scripts actually count.

Two independent conditions must hold before a single row is written:

1. ``--execute`` — an explicit, unambiguous flag. No ``--apply``-adjacent
   spelling, no environment variable that a stale shell could carry in.
2. ``--confirm <token>`` — a token the script **cannot derive on its own**. It
   is a digest of the exact candidate set the dry-run measured, printed by
   that dry-run. You cannot produce it without having run the dry-run, and it
   stops matching the moment the population moves. A stale token from
   yesterday's run authorises nothing.

And execution is additionally refused without ``--org``: the counting mode may
sweep the fleet, the mutating mode may not. ``run_retention.py`` has one
global DELETE and it is documented as deliberate; nothing here copies that
shape.
"""

import importlib
import os
import sys

import pytest


def _load_common():
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "scripts"))
    return importlib.import_module("reconcile_common")


common = _load_common()

SCRIPT = "reconcile_curator_skipped"
IDS = ["11111111-1111-1111-1111-111111111111", "22222222-2222-2222-2222-222222222222"]


def _parse(argv):
    parser = common.build_parser("test")
    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# Default is dry-run
# ---------------------------------------------------------------------------


def test_no_flags_is_dry_run():
    """Absent every flag the mode is dry-run. This is the whole contract."""
    args = _parse([])
    assert args.execute is False
    assert common.resolve_execute(args, script=SCRIPT, candidate_ids=IDS) is False


def test_org_only_is_still_dry_run():
    """Scoping to an org is not consent to write to it."""
    args = _parse(["--org", "org-a"])
    assert common.resolve_execute(args, script=SCRIPT, candidate_ids=IDS) is False


# ---------------------------------------------------------------------------
# --execute alone is not enough
# ---------------------------------------------------------------------------


def test_execute_without_confirm_is_refused():
    args = _parse(["--execute", "--org", "org-a"])
    with pytest.raises(common.ConfirmationError) as exc:
        common.resolve_execute(args, script=SCRIPT, candidate_ids=IDS)
    assert "--confirm" in str(exc.value)


def test_execute_with_wrong_confirm_is_refused():
    args = _parse(["--execute", "--org", "org-a", "--confirm", "deadbeefdeadbeef"])
    with pytest.raises(common.ConfirmationError):
        common.resolve_execute(args, script=SCRIPT, candidate_ids=IDS)


def test_execute_with_a_token_for_a_different_population_is_refused():
    """A token from an earlier dry-run must not authorise a moved population.

    This is why the token is a digest of the candidate set rather than a
    constant string: between the dry-run an operator reviewed and the execute
    they typed, rows can arrive, be retracted, or age out of the TTL. The
    token they hold describes the set they were shown.
    """
    stale = common.confirmation_token(SCRIPT, "org-a", IDS)
    args = _parse(["--execute", "--org", "org-a", "--confirm", stale])
    moved = IDS + ["33333333-3333-3333-3333-333333333333"]
    with pytest.raises(common.ConfirmationError) as exc:
        common.resolve_execute(args, script=SCRIPT, candidate_ids=moved)
    assert "no longer matches" in str(exc.value)


def test_execute_with_a_token_for_a_different_org_is_refused():
    other = common.confirmation_token(SCRIPT, "org-b", IDS)
    args = _parse(["--execute", "--org", "org-a", "--confirm", other])
    with pytest.raises(common.ConfirmationError):
        common.resolve_execute(args, script=SCRIPT, candidate_ids=IDS)


def test_execute_with_a_token_for_a_different_script_is_refused():
    """The three scripts touch three different tables. Tokens do not travel."""
    other = common.confirmation_token("reconcile_orphan_artifacts", "org-a", IDS)
    args = _parse(["--execute", "--org", "org-a", "--confirm", other])
    with pytest.raises(common.ConfirmationError):
        common.resolve_execute(args, script=SCRIPT, candidate_ids=IDS)


def test_execute_without_org_is_refused():
    """Counting may sweep the fleet. Writing may not."""
    token = common.confirmation_token(SCRIPT, None, IDS)
    args = _parse(["--execute", "--confirm", token])
    with pytest.raises(common.ConfirmationError) as exc:
        common.resolve_execute(args, script=SCRIPT, candidate_ids=IDS)
    assert "--org" in str(exc.value)


# ---------------------------------------------------------------------------
# Positive control — the guard is not simply always refusing
# ---------------------------------------------------------------------------


def test_execute_with_the_matching_token_is_allowed():
    """Positive control. Without this, every test above passes on a guard that
    hard-codes ``raise``, and the suite proves nothing."""
    token = common.confirmation_token(SCRIPT, "org-a", IDS)
    args = _parse(["--execute", "--org", "org-a", "--confirm", token])
    assert common.resolve_execute(args, script=SCRIPT, candidate_ids=IDS) is True


def test_token_is_order_insensitive():
    """The candidate set is a set; SQL ordering must not invalidate a token."""
    assert common.confirmation_token(SCRIPT, "org-a", IDS) == (
        common.confirmation_token(SCRIPT, "org-a", list(reversed(IDS)))
    )


# ---------------------------------------------------------------------------
# The environment cannot be the second factor
# ---------------------------------------------------------------------------


def test_environment_variables_cannot_authorise_execution(monkeypatch):
    """A guard keyed on an env var is one stale shell away from being no guard.

    Named for the values an operator might plausibly have exported while
    working on this package.
    """
    for name in (
        "KLIO_RECONCILE_CONFIRM",
        "KLIO_RECONCILE_EXECUTE",
        "RECONCILE_EXECUTE",
        "KLIO_EXECUTE",
    ):
        monkeypatch.setenv(name, "1")
    args = _parse([])
    assert common.resolve_execute(args, script=SCRIPT, candidate_ids=IDS) is False


# ---------------------------------------------------------------------------
# Output hygiene
# ---------------------------------------------------------------------------


def test_format_sample_is_bounded():
    ids = [f"{n:032x}" for n in range(500)]
    rendered = common.format_sample(ids)
    assert rendered.count(",") < common.SAMPLE_LIMIT
    assert "of 500" in rendered


def test_format_sample_of_an_empty_population_says_so():
    assert common.format_sample([]) == "none"


def test_format_sample_rejects_anything_that_is_not_an_identifier():
    """These scripts read private-scope rows. The sample printer is the last
    place a caller could accidentally hand memory content to a log."""
    with pytest.raises(ValueError):
        common.format_sample(["the database password is hunter2"])


def test_format_sample_accepts_uuids_and_object_keys():
    """Positive control for the printer's own guard: the two identifier shapes
    these scripts genuinely emit must pass."""
    rendered = common.format_sample(
        [
            "11111111-1111-1111-1111-111111111111",
            "org-a/11111111-1111-1111-1111-111111111111/v1",
        ]
    )
    assert "org-a/11111111-1111-1111-1111-111111111111/v1" in rendered
