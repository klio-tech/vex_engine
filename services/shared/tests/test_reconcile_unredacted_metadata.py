"""Counting the secrets sitting unmasked in ``session_memories.metadata``.

Two defects put them there, and only one of them is fixed.

WP-B (fixed)
    ``sync-gateway`` and ``async-worker`` called ``write_memory`` directly and
    imported no redactor at all, so LLM summaries and extracted facts — which
    routinely quote the transcript back — were stored verbatim. Their *content*
    is what ``run_redaction_backfill.py`` already sweeps.

The metadata gap (NOT fixed by WP-B, and still live)
    ``shared.brain.write_scoped_memory`` redacts ``content`` and only
    ``content``. The ``decide`` tool puts its caller-supplied ``rationale``
    into ``metadata``, where nothing screens it. A rationale is free prose an
    agent wrote about why it did something — exactly the shape of text a
    credential ends up inside — and no backfill in this repo has ever looked
    at it. That is the population this script measures.

The hardest rule in this package applies most sharply here. This script exists
*because* the rows it selects contain unmasked secrets. Printing the value it
found, or the text around it, would defeat the entire purpose. So the report
is counts by pattern class and key path, and the tests below assert that with
real secret-shaped fixtures.
"""

import importlib
import json
import logging
import os
import sys
import uuid

import pytest
from sqlalchemy import create_engine, text

from shared.testing import database_url, requires_db

pytestmark = requires_db("the unredacted-metadata reconciliation tests")


def _load():
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "scripts"))
    return importlib.import_module("reconcile_unredacted_metadata")


script = _load()
common = importlib.import_module("reconcile_common")

# Documentation-grade fixtures, all published examples, none live.
AWS_KEY = "AKIAIOSFODNN7EXAMPLE"
STRIPE_KEY = "sk_live_4eC39HqLyjWDarjtT1zdp7dcABCDEFGH"
SLACK_TOKEN = "xoxb-1234567890-abcdefghijklmnop"


@pytest.fixture()
def db():
    engine = create_engine(database_url(), pool_pre_ping=True)
    with engine.connect() as conn:
        yield conn
    engine.dispose()


@pytest.fixture()
def org(db):
    org_id = f"meta-{uuid.uuid4().hex[:12]}"
    yield org_id
    db.execute(text("DELETE FROM session_memories WHERE org_id = :o"), {"o": org_id})
    db.commit()


def _row(db, org_id, metadata, *, memory_type="decision", content="a decision"):
    row_id = str(uuid.uuid4())
    db.execute(
        text(
            """
            INSERT INTO session_memories
                (id, session_id, org_id, agent_id, memory_type, content,
                 scope, status, metadata, created_at)
            VALUES (:id, 'sess', :org, 'agent-1', :mtype, :content,
                    'org', 'active', CAST(:meta AS jsonb), now())
            """
        ),
        {
            "id": row_id,
            "org": org_id,
            "mtype": memory_type,
            "content": content,
            "meta": json.dumps(metadata) if metadata is not None else None,
        },
    )
    db.commit()
    return row_id


def _meta(db, row_id):
    return db.execute(
        text("SELECT metadata FROM session_memories WHERE id = :i"), {"i": row_id}
    ).fetchone()[0]


# ---------------------------------------------------------------------------
# What it finds
# ---------------------------------------------------------------------------


def test_a_decide_rationale_carrying_a_secret_is_found(db, org):
    """The headline case: metadata['rationale'] is never screened on write."""
    row_id = _row(
        db,
        org,
        {"source": "mcp", "rationale": f"we rotated to {AWS_KEY} last week"},
    )

    findings = script.scan_metadata(db, org_id=org)
    assert len(findings) == 1
    assert findings[0].row_id == row_id
    assert findings[0].key_path == "rationale"
    assert findings[0].kinds == ("aws_access_key_id",)


def test_a_secret_nested_inside_metadata_is_found(db, org):
    """Metadata is free-form JSONB. A scanner that only checks top-level
    strings would miss anything a future tool nests."""
    _row(
        db,
        org,
        {"source": "mcp", "context": {"steps": [f"exported {STRIPE_KEY}"]}},
    )

    findings = script.scan_metadata(db, org_id=org)
    assert len(findings) == 1
    assert findings[0].key_path == "context.steps[0]"
    assert findings[0].kinds == ("stripe_secret_key",)


def test_several_kinds_in_one_value_are_all_reported(db, org):
    _row(db, org, {"rationale": f"{AWS_KEY} and {SLACK_TOKEN}"})

    findings = script.scan_metadata(db, org_id=org)
    assert findings[0].kinds == ("aws_access_key_id", "slack_token")


def test_counts_are_aggregated_by_pattern_class(db, org):
    _row(db, org, {"rationale": f"key {AWS_KEY}"})
    _row(db, org, {"rationale": f"other key {AWS_KEY}"})
    _row(db, org, {"rationale": f"token {SLACK_TOKEN}"})

    counts = script.counts_by_kind(script.scan_metadata(db, org_id=org))
    assert counts == {"aws_access_key_id": 2, "slack_token": 1}


def test_every_kind_in_one_value_is_counted_not_just_the_first(db, org):
    """One rationale can name two different credentials.

    Counting only the first kind per value would under-report the rarer one —
    and the rare kind is the one an operator most needs to see, because it
    tells them which system was exposed.
    """
    _row(db, org, {"rationale": f"{AWS_KEY} and {SLACK_TOKEN}"})

    counts = script.counts_by_kind(script.scan_metadata(db, org_id=org))
    assert counts == {"aws_access_key_id": 1, "slack_token": 1}


# ---------------------------------------------------------------------------
# What it must NOT find — positive controls against a vacuous scanner
# ---------------------------------------------------------------------------


def test_ordinary_metadata_produces_nothing(db, org):
    """Positive control. A scanner that flagged everything would satisfy every
    test above and be worthless."""
    _row(db, org, {"source": "mcp", "rationale": "cheaper and simpler to run"})
    _row(db, org, {"source": "curator", "input_count": 12})

    assert script.scan_metadata(db, org_id=org) == []


def test_a_null_metadata_row_is_skipped(db, org):
    _row(db, org, None)
    assert script.scan_metadata(db, org_id=org) == []


def test_already_redacted_placeholders_are_not_re_reported(db, org):
    """The redactor is idempotent and placeholders never re-match. A second
    run of this script over its own output must report a clean database, or
    nobody can ever tell whether the remediation worked."""
    _row(db, org, {"rationale": "we rotated to [REDACTED:aws_access_key_id:1a2b]"})
    assert script.scan_metadata(db, org_id=org) == []


def test_the_content_column_is_not_scanned(db, org):
    """content is run_redaction_backfill.py's job — it re-embeds, which this
    script cannot do. Two scripts silently fighting over one column is worse
    than a documented division of labour."""
    _row(db, org, {"source": "mcp"}, content=f"the key is {AWS_KEY}")
    assert script.scan_metadata(db, org_id=org) == []


# ---------------------------------------------------------------------------
# Tenancy
# ---------------------------------------------------------------------------


def test_another_orgs_metadata_is_never_scanned(db, org):
    other = f"meta-other-{uuid.uuid4().hex[:8]}"
    _row(db, other, {"rationale": f"key {AWS_KEY}"})
    mine = _row(db, org, {"rationale": f"key {STRIPE_KEY}"})

    findings = script.scan_metadata(db, org_id=org)
    assert [f.row_id for f in findings] == [mine]

    db.execute(text("DELETE FROM session_memories WHERE org_id = :o"), {"o": other})
    db.commit()


# ---------------------------------------------------------------------------
# Output hygiene — the point of this script
# ---------------------------------------------------------------------------


def test_the_report_never_prints_the_secret_it_found(db, org, caplog):
    """The whole reason this script exists is that these rows hold unmasked
    credentials. Printing one to a log would move it somewhere with worse
    access control than the database it came from.

    Asserted via caplog, not capsys: the reporter uses ``logging``, whose
    handler holds the real stderr from import time, so a capsys-based version
    of this assertion captures nothing and passes unconditionally.
    """
    row_id = _row(db, org, {"rationale": f"rotated to {AWS_KEY} after the leak"})

    parser = common.build_parser("t")
    with caplog.at_level(logging.INFO, logger="reconcile.unredacted_metadata"):
        script.run(db, parser.parse_args(["--org", org]))

    # Positive control: the log was captured, so the absences below mean
    # something.
    assert row_id in caplog.text
    assert "aws_access_key_id" in caplog.text
    # The secret, any prefix of it long enough to matter, and the surrounding
    # words must all be absent.
    assert AWS_KEY not in caplog.text
    assert "AKIA" not in caplog.text
    assert "after the leak" not in caplog.text


def test_the_report_never_prints_the_metadata_value_even_when_clean(db, org, caplog):
    """A scanner that echoes the value on a near-miss leaks just as hard."""
    row_id = _row(db, org, {"rationale": f"see {STRIPE_KEY} in the vault"})

    parser = common.build_parser("t")
    with caplog.at_level(logging.INFO, logger="reconcile.unredacted_metadata"):
        script.run(db, parser.parse_args(["--org", org]))

    assert row_id in caplog.text
    assert "in the vault" not in caplog.text
    assert STRIPE_KEY not in caplog.text


# ---------------------------------------------------------------------------
# The guard
# ---------------------------------------------------------------------------


def test_dry_run_leaves_the_metadata_byte_identical(db, org):
    original = {"source": "mcp", "rationale": f"rotated to {AWS_KEY}"}
    row_id = _row(db, org, original)

    parser = common.build_parser("t")
    script.run(db, parser.parse_args(["--org", org]))

    assert _meta(db, row_id) == original


def test_execute_without_confirm_writes_nothing(db, org):
    original = {"rationale": f"rotated to {AWS_KEY}"}
    row_id = _row(db, org, original)

    parser = common.build_parser("t")
    with pytest.raises(common.ConfirmationError):
        script.run(db, parser.parse_args(["--org", org, "--execute"]))

    assert _meta(db, row_id) == original


# ---------------------------------------------------------------------------
# The execute path, against fixtures only
# ---------------------------------------------------------------------------


def _confirmed(db, org, parser):
    findings = script.scan_metadata(db, org_id=org)
    token = common.confirmation_token(
        script.SCRIPT_NAME, org, script.candidate_ids(findings)
    )
    script.run(db, parser.parse_args(["--org", org, "--execute", "--confirm", token]))


def test_confirmed_execute_masks_the_secret_and_keeps_everything_else(db, org):
    """Positive control for the guard, and for the rewrite being surgical.

    Metadata is not embedded, so unlike the content backfill this rewrite
    needs no re-embedding and cannot desynchronise a row from its vector.
    """
    row_id = _row(
        db,
        org,
        {"source": "mcp", "input_count": 3, "rationale": f"rotated to {AWS_KEY}"},
    )

    _confirmed(db, org, common.build_parser("t"))

    after = _meta(db, row_id)
    assert AWS_KEY not in after["rationale"]
    assert "[REDACTED:aws_access_key_id:" in after["rationale"]
    assert after["source"] == "mcp"
    assert after["input_count"] == 3


def test_confirmed_execute_masks_a_nested_value_in_place(db, org):
    row_id = _row(db, org, {"context": {"steps": [f"exported {STRIPE_KEY}", "ok"]}})

    _confirmed(db, org, common.build_parser("t"))

    after = _meta(db, row_id)
    assert STRIPE_KEY not in after["context"]["steps"][0]
    assert after["context"]["steps"][1] == "ok"


def test_confirmed_execute_never_touches_the_content_column(db, org):
    """This script cannot re-embed, so it must never edit an embedded column."""
    content = f"the key is {AWS_KEY}"
    row_id = _row(db, org, {"rationale": f"rotated to {AWS_KEY}"}, content=content)

    _confirmed(db, org, common.build_parser("t"))

    after = db.execute(
        text("SELECT content FROM session_memories WHERE id = :i"), {"i": row_id}
    ).fetchone()
    assert after.content == content


def test_scrub_row_refuses_an_id_that_belongs_to_another_org(db, org):
    """Defence in depth on the write itself, not just on which ids reach it.

    ``scrub_row`` takes a row id from a scan result. Between the scan and the
    write that id can go stale, and nothing about a UUID says which tenant it
    belongs to. So the UPDATE carries its own ``org_id`` predicate, and this
    test drives ``scrub_row`` directly with a mismatched pair — which is the
    only way to exercise it, because the normal path never assembles one.
    """
    other = f"meta-other-{uuid.uuid4().hex[:8]}"
    other_meta = {"rationale": f"rotated to {AWS_KEY}"}
    other_id = _row(db, other, other_meta)

    assert script.scrub_row(db, org_id=org, row_id=other_id) is False
    assert _meta(db, other_id) == other_meta

    db.execute(text("DELETE FROM session_memories WHERE org_id = :o"), {"o": other})
    db.commit()


def test_confirmed_execute_never_touches_another_org(db, org):
    other = f"meta-other-{uuid.uuid4().hex[:8]}"
    other_meta = {"rationale": f"rotated to {AWS_KEY}"}
    other_id = _row(db, other, other_meta)
    _row(db, org, {"rationale": f"rotated to {STRIPE_KEY}"})

    _confirmed(db, org, common.build_parser("t"))

    assert _meta(db, other_id) == other_meta

    db.execute(text("DELETE FROM session_memories WHERE org_id = :o"), {"o": other})
    db.commit()


def test_execute_is_idempotent(db, org):
    """A second run finds nothing, because placeholders never re-match. That
    is what makes it safe to re-run after a partial failure."""
    _row(db, org, {"rationale": f"rotated to {AWS_KEY}"})

    _confirmed(db, org, common.build_parser("t"))
    assert script.scan_metadata(db, org_id=org) == []
