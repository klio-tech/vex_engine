"""Reconciling the artifact bucket against the ``artifacts`` table.

Nobody currently knows whether orphans exist. ``forget`` and ``mark_superseded``
only flip a status; the GC sweep is what actually deletes the bytes, and if it
has ever failed silently — a throttled DELETE, a crashed tick, a row purged by
retention before its object was collected — the deletion promise has not held
and there is no signal anywhere that says so. This script is the only way to
find out.

Because nobody knows, the classification has to be careful in both directions:

* Calling a live object an orphan invites deleting a customer's document.
* Calling an orphan "fine" reports a clean bill of health that is false.

So an object is only an orphan once it is past a grace window, an object whose
key does not parse is never an orphan (it is reported, untouched, as
unrecognised), and the reverse direction — a row claiming bytes that are not
there — is reported too, because half a reconciliation is not a reconciliation.

Everything here runs against a fake S3 client. There are no object-store
credentials in this environment and none were requested; the live run is the
user's to make.
"""

import importlib
import logging
import os
import sys
import uuid
from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy import create_engine, text

from shared.testing import database_url, requires_db

pytestmark = requires_db("the orphan-artifact reconciliation tests")


def _load():
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "scripts"))
    return importlib.import_module("reconcile_orphan_artifacts")


script = _load()
common = importlib.import_module("reconcile_common")


class FakeS3:
    """The three calls this script makes, and a record of what it deleted.

    Deliberately not a MagicMock: the assertion that matters most is "this
    deleted nothing", and a mock that auto-creates ``delete_object`` will
    happily record a call against a bucket name that does not exist.
    """

    def __init__(self, objects=None):
        self.objects = dict(objects or {})
        self.deleted = []
        self.list_calls = []

    def get_paginator(self, operation):
        assert operation == "list_objects_v2"
        return self

    def paginate(self, **kwargs):
        self.list_calls.append(kwargs)
        prefix = kwargs.get("Prefix") or ""
        contents = [
            {"Key": key, "Size": 10, "LastModified": last_modified}
            for key, last_modified in sorted(self.objects.items())
            if key.startswith(prefix)
        ]
        # Two pages, so a single-page implementation is caught.
        yield {"Contents": contents[:1]}
        if contents[1:]:
            yield {"Contents": contents[1:]}

    def delete_object(self, Bucket, Key):  # noqa: N803 - boto3's signature
        assert Bucket
        if Key not in self.objects:
            raise AssertionError(f"deleted a key that is not in the bucket: {Key}")
        self.deleted.append(Key)
        del self.objects[Key]


def _ago(**kwargs):
    return datetime.now(timezone.utc) - timedelta(**kwargs)


@pytest.fixture()
def db():
    engine = create_engine(database_url(), pool_pre_ping=True)
    with engine.connect() as conn:
        yield conn
    engine.dispose()


@pytest.fixture()
def org(db):
    org_id = f"orph-{uuid.uuid4().hex[:12]}"
    yield org_id
    db.execute(text("DELETE FROM artifacts WHERE org_id = :o"), {"o": org_id})
    db.commit()


def _artifact(db, org_id, *, status="active", version=1, retracted_hours=None):
    """Insert one artifact row and return its id."""
    artifact_id = str(uuid.uuid4())
    inline = "body" if status == "active" else None
    db.execute(
        text(
            """
            INSERT INTO artifacts
                (id, org_id, agent_id, title, summary, kind, mime_type,
                 size_bytes, checksum, inline_text, version, status,
                 retracted_at, created_at)
            VALUES (:id, :org, 'agent-1', 't', 's', 'document', 'text/plain',
                    4, :sum, :inline, :version, :status, :retracted,
                    now() - interval '30 days')
            """
        ),
        {
            "id": artifact_id,
            "org": org_id,
            "sum": uuid.uuid4().hex,
            "inline": inline,
            "version": version,
            "status": status,
            "retracted": (
                _ago(hours=retracted_hours) if retracted_hours is not None else None
            ),
        },
    )
    db.commit()
    return artifact_id


def _key(org_id, artifact_id, version=1):
    return f"{org_id}/{artifact_id}/v{version}"


# ---------------------------------------------------------------------------
# The orphan classes
# ---------------------------------------------------------------------------


def test_an_object_with_no_row_at_all_is_an_orphan(db, org):
    """The deletion promise broken, or an upload whose row never landed."""
    key = _key(org, str(uuid.uuid4()))
    s3 = FakeS3({key: _ago(days=30)})

    result = script.reconcile(db, s3, org_id=org)
    assert result.orphan_keys == [key]


def test_an_object_whose_row_is_active_is_not_an_orphan(db, org):
    """Positive control. A reconciler that flagged everything would satisfy
    the test above and delete the entire bucket."""
    artifact_id = _artifact(db, org)
    key = _key(org, artifact_id)
    s3 = FakeS3({key: _ago(days=30)})

    result = script.reconcile(db, s3, org_id=org)
    assert result.orphan_keys == []
    assert result.claimed == 1


def test_a_dead_row_past_the_purge_grace_is_reported_as_unpurged(db, org):
    """The GC should have deleted these bytes and did not.

    Kept apart from the rowless orphans on purpose: this population names a
    specific failing mechanism (the purge sweep), while a rowless object could
    have come from anywhere. Merging them would hide which one is broken.
    """
    artifact_id = _artifact(db, org, status="retracted", retracted_hours=24 * 30)
    key = _key(org, artifact_id)
    s3 = FakeS3({key: _ago(days=30)})

    result = script.reconcile(db, s3, org_id=org)
    assert result.unpurged_keys == [key]
    assert result.orphan_keys == []


def test_a_dead_row_still_inside_the_purge_grace_is_not_reported(db, org):
    """PURGE_GRACE_HOURS exists so an undelete stays possible. An object
    inside that window is doing exactly what it is supposed to do."""
    artifact_id = _artifact(db, org, status="retracted", retracted_hours=1)
    key = _key(org, artifact_id)
    s3 = FakeS3({key: _ago(days=30)})

    result = script.reconcile(db, s3, org_id=org)
    assert result.unpurged_keys == []
    assert result.orphan_keys == []


def test_a_pending_row_is_never_an_orphan(db, org):
    """An upload in flight. Deleting its object races a live presigned PUT."""
    artifact_id = _artifact(db, org, status="pending")
    key = _key(org, artifact_id)
    s3 = FakeS3({key: _ago(minutes=1)})

    result = script.reconcile(db, s3, org_id=org)
    assert result.orphan_keys == []
    assert result.unpurged_keys == []


def test_a_recently_uploaded_rowless_object_is_not_yet_an_orphan(db, org):
    """The grace window, and the reason there is one.

    A presigned PUT can land before this script reads the table, or in the gap
    between the two. Without a window the reconciler would classify a healthy
    in-flight upload as orphaned bytes — the same class of mistake as the
    curator advancing past a row it had not read.
    """
    key = _key(org, str(uuid.uuid4()))
    s3 = FakeS3({key: _ago(minutes=5)})

    result = script.reconcile(db, s3, org_id=org)
    assert result.orphan_keys == []
    assert result.too_recent == 1


def test_a_key_that_does_not_parse_is_reported_but_never_an_orphan(db, org):
    """An unparseable key cannot be matched to a row, so it cannot be shown to
    be unclaimed. Reporting it as an orphan would put a bucket the script does
    not understand on a delete list."""
    s3 = FakeS3({f"{org}/not-a-uuid-shaped-thing": _ago(days=30)})

    result = script.reconcile(db, s3, org_id=org)
    assert result.unrecognised_keys == [f"{org}/not-a-uuid-shaped-thing"]
    assert result.orphan_keys == []


def test_the_wrong_version_of_a_real_artifact_is_an_orphan(db, org):
    """A key is (org, artifact, version). v2 bytes for a v1 row are unclaimed —
    matching on artifact id alone would silently protect them forever."""
    artifact_id = _artifact(db, org, version=1)
    stale = _key(org, artifact_id, version=2)
    s3 = FakeS3({_key(org, artifact_id, 1): _ago(days=30), stale: _ago(days=30)})

    result = script.reconcile(db, s3, org_id=org)
    assert result.orphan_keys == [stale]


# ---------------------------------------------------------------------------
# The other direction
# ---------------------------------------------------------------------------


def test_a_row_whose_object_is_missing_is_reported(db, org):
    """Half a reconciliation is not a reconciliation.

    Nothing can be executed about this — bytes cannot be recreated — so it is
    reported and nothing more. But an active artifact whose object has
    vanished is a read that will fail for a customer, and this is the only
    place that would ever notice.
    """
    artifact_id = _artifact(db, org)
    db.execute(
        text("UPDATE artifacts SET inline_text = NULL, storage_uri = :u WHERE id = :i"),
        {"u": f"s3://bucket/{_key(org, artifact_id)}", "i": artifact_id},
    )
    db.commit()

    result = script.reconcile(db, FakeS3({}), org_id=org)
    assert result.missing_object_ids == [artifact_id]


def test_an_inline_artifact_is_not_reported_as_missing(db, org):
    """Positive control: small artifacts live in inline_text and have no
    object at all. Flagging every one of them would drown the real signal."""
    _artifact(db, org)

    result = script.reconcile(db, FakeS3({}), org_id=org)
    assert result.missing_object_ids == []


# ---------------------------------------------------------------------------
# Tenancy and paging
# ---------------------------------------------------------------------------


def test_scoping_to_an_org_lists_only_that_orgs_prefix(db, org):
    """Object keys are prefixed by the owning org. Listing the whole bucket to
    filter in Python would read every tenant's key names to answer a
    single-tenant question."""
    s3 = FakeS3({})
    script.reconcile(db, s3, org_id=org)
    assert s3.list_calls[0]["Prefix"] == f"{org}/"


def test_another_orgs_object_is_never_classified(db, org):
    other = f"orph-other-{uuid.uuid4().hex[:8]}"
    s3 = FakeS3({_key(other, str(uuid.uuid4())): _ago(days=30)})

    result = script.reconcile(db, s3, org_id=org)
    assert result.orphan_keys == []
    assert result.scanned == 0


def test_a_key_naming_another_orgs_artifact_id_is_not_claimed_by_it(db, org):
    """An object key is a string from the object store, not a capability.

    Here the key says ``<our org>/<another org's artifact id>/v1``. The row
    exists — but not in this org. A claim lookup keyed on the artifact id
    alone would find it, mark the bytes claimed, and let one tenant's row
    vouch for another tenant's object indefinitely.
    """
    other = f"orph-other-{uuid.uuid4().hex[:8]}"
    foreign_id = _artifact(db, other)
    key = _key(org, foreign_id)
    s3 = FakeS3({key: _ago(days=30)})

    result = script.reconcile(db, s3, org_id=org)
    assert result.orphan_keys == [key]
    assert result.claimed == 0

    db.execute(text("DELETE FROM artifacts WHERE org_id = :o"), {"o": other})
    db.commit()


def test_run_re_checks_each_key_against_the_table_before_deleting_it(db, org):
    """The pre-delete re-check must be wired into ``run``, not merely exist.

    ``reconcile`` is stubbed to hand back a result that has gone stale — the
    state that arises when an upload confirms between ``run``'s own scan and
    its delete loop. The token is computed over that same stale result, so the
    token check passes and the ONLY thing standing between this object and a
    DELETE is the re-check.
    """
    artifact_id = _artifact(db, org)
    key = _key(org, artifact_id)
    s3 = FakeS3({key: _ago(days=30)})

    stale = script.Reconciliation(scanned=1, orphan_keys=[key])
    original = script.reconcile
    script.reconcile = lambda *a, **kw: stale
    try:
        token = common.confirmation_token(
            script.SCRIPT_NAME, org, script.candidate_ids(stale)
        )
        parser = common.build_parser("t")
        script.run(
            db, s3, parser.parse_args(["--org", org, "--execute", "--confirm", token])
        )
    finally:
        script.reconcile = original

    assert s3.deleted == []
    assert key in s3.objects


def test_run_deletes_a_stale_key_that_is_genuinely_unclaimed(db, org):
    """Positive control for the test above: with the same stubbed path but no
    row in the table, the delete must actually happen — otherwise that test
    passes against a ``run`` that deletes nothing under any circumstances."""
    key = _key(org, str(uuid.uuid4()))
    s3 = FakeS3({key: _ago(days=30)})

    stale = script.Reconciliation(scanned=1, orphan_keys=[key])
    original = script.reconcile
    script.reconcile = lambda *a, **kw: stale
    try:
        token = common.confirmation_token(
            script.SCRIPT_NAME, org, script.candidate_ids(stale)
        )
        parser = common.build_parser("t")
        script.run(
            db, s3, parser.parse_args(["--org", org, "--execute", "--confirm", token])
        )
    finally:
        script.reconcile = original

    assert s3.deleted == [key]


def test_every_page_of_the_listing_is_read(db, org):
    """A paginator that stops after the first page under-reports silently, and
    a clean bill of health from an under-read bucket is worse than no run."""
    keys = {_key(org, str(uuid.uuid4())): _ago(days=30) for _ in range(5)}
    s3 = FakeS3(keys)

    result = script.reconcile(db, s3, org_id=org)
    assert result.scanned == 5
    assert sorted(result.orphan_keys) == sorted(keys)


# ---------------------------------------------------------------------------
# The guard — this one deletes customer document bytes
# ---------------------------------------------------------------------------


def test_dry_run_deletes_nothing(db, org):
    key = _key(org, str(uuid.uuid4()))
    s3 = FakeS3({key: _ago(days=30)})

    parser = common.build_parser("t")
    script.run(db, s3, parser.parse_args(["--org", org]))

    assert s3.deleted == []
    assert key in s3.objects


def test_execute_without_confirm_deletes_nothing(db, org):
    key = _key(org, str(uuid.uuid4()))
    s3 = FakeS3({key: _ago(days=30)})

    parser = common.build_parser("t")
    with pytest.raises(common.ConfirmationError):
        script.run(db, s3, parser.parse_args(["--org", org, "--execute"]))

    assert s3.deleted == []
    assert key in s3.objects


def test_dry_run_prints_no_credential_and_no_endpoint(db, org, caplog):
    """The script builds an S3 client from environment credentials. Nothing
    about that may reach a log."""
    key = _key(org, str(uuid.uuid4()))
    s3 = FakeS3({key: _ago(days=30)})

    parser = common.build_parser("t")
    with caplog.at_level(logging.INFO, logger="reconcile.orphan_artifacts"):
        script.run(db, s3, parser.parse_args(["--org", org]))

    assert key in caplog.text  # positive control: the log was captured
    for forbidden in ("agentguard_dev", "SECRET", "aws_secret", "http://"):
        assert forbidden not in caplog.text


# ---------------------------------------------------------------------------
# The execute path, against the fake client only
# ---------------------------------------------------------------------------


def _confirmed(db, s3, org, argv_extra=()):
    result = script.reconcile(db, s3, org_id=org)
    token = common.confirmation_token(
        script.SCRIPT_NAME, org, script.candidate_ids(result)
    )
    parser = common.build_parser("t")
    script.run(
        db,
        s3,
        parser.parse_args(["--org", org, "--execute", "--confirm", token, *argv_extra]),
    )


def test_confirmed_execute_deletes_only_the_orphans(db, org):
    """Positive control for the guard, and the blast-radius assertion.

    The live artifact and the in-grace retraction must both survive, or the
    guard passing is worth nothing.
    """
    live = _key(org, _artifact(db, org))
    in_grace = _key(org, _artifact(db, org, status="retracted", retracted_hours=1))
    orphan = _key(org, str(uuid.uuid4()))
    s3 = FakeS3({k: _ago(days=30) for k in (live, in_grace, orphan)})

    _confirmed(db, s3, org)

    assert s3.deleted == [orphan]
    assert live in s3.objects
    assert in_grace in s3.objects


def test_confirmed_execute_never_deletes_an_unrecognised_key(db, org):
    """A key the script could not parse is a key it does not understand.
    Understanding is the precondition for deleting."""
    weird = f"{org}/mystery-object"
    s3 = FakeS3({weird: _ago(days=30)})

    _confirmed(db, s3, org)

    assert s3.deleted == []
    assert weird in s3.objects


def test_confirmed_execute_does_not_delete_unpurged_bytes_by_default(db, org):
    """A dead row past its grace is the GC's job, and the GC is idempotent.

    Doing it here as a side effect of an orphan sweep would delete customer
    document content that a human never separately approved.
    """
    artifact_id = _artifact(db, org, status="retracted", retracted_hours=24 * 30)
    key = _key(org, artifact_id)
    s3 = FakeS3({key: _ago(days=30)})

    _confirmed(db, s3, org)

    assert s3.deleted == []
    assert key in s3.objects


def test_confirmed_execute_never_touches_another_orgs_object(db, org):
    other = f"orph-other-{uuid.uuid4().hex[:8]}"
    other_key = _key(other, str(uuid.uuid4()))
    orphan = _key(org, str(uuid.uuid4()))
    s3 = FakeS3({other_key: _ago(days=30), orphan: _ago(days=30)})

    _confirmed(db, s3, org)

    assert s3.deleted == [orphan]
    assert other_key in s3.objects


def test_still_unclaimed_is_false_once_a_row_claims_the_key(db, org):
    """The last word before a DELETE belongs to the database, not to the scan.

    ``run`` re-reconciles before it deletes, so the token check already
    catches an upload that confirms between the operator's dry-run and their
    execute (the test below proves that). This covers the narrower race the
    token cannot see: a row appearing between ``run``'s own scan and the
    delete loop a few lines later. Driven directly, because the normal path
    cannot assemble that state on purpose.
    """
    artifact_id = str(uuid.uuid4())
    key = _key(org, artifact_id)
    assert script._still_unclaimed(db, key) is True

    db.execute(
        text(
            """
            INSERT INTO artifacts
                (id, org_id, agent_id, title, summary, kind, mime_type,
                 size_bytes, checksum, inline_text, version, status)
            VALUES (:id, :org, 'agent-1', 't', 's', 'document', 'text/plain',
                    4, :sum, 'body', 1, 'active')
            """
        ),
        {"id": artifact_id, "org": org, "sum": uuid.uuid4().hex},
    )
    db.commit()

    assert script._still_unclaimed(db, key) is False


def test_still_unclaimed_refuses_a_key_it_cannot_parse(db, org):
    """An unparseable key can never be shown to be unclaimed, so it is never
    deletable — the same rule as the classifier, enforced again at the gate."""
    assert script._still_unclaimed(db, f"{org}/mystery-object") is False


def test_an_upload_confirmed_after_the_dry_run_invalidates_the_token(db, org):
    """The realistic race, and the guard that actually catches it.

    The operator reviewed a dry-run listing this object as unclaimed. Before
    they ran the execute, a presigned upload confirmed and a row now owns
    those bytes. The token describes the set they approved; the set has
    changed; the run is refused and nothing is deleted.
    """
    artifact_id = uuid.uuid4()
    orphan = _key(org, str(artifact_id))
    s3 = FakeS3({orphan: _ago(days=30)})

    result = script.reconcile(db, s3, org_id=org)
    assert result.orphan_keys == [orphan]
    token = common.confirmation_token(
        script.SCRIPT_NAME, org, script.candidate_ids(result)
    )

    # The upload confirms. The row now claims exactly these bytes.
    db.execute(
        text(
            """
            INSERT INTO artifacts
                (id, org_id, agent_id, title, summary, kind, mime_type,
                 size_bytes, checksum, inline_text, version, status)
            VALUES (:id, :org, 'agent-1', 't', 's', 'document', 'text/plain',
                    4, :sum, 'body', 1, 'active')
            """
        ),
        {"id": str(artifact_id), "org": org, "sum": uuid.uuid4().hex},
    )
    db.commit()

    parser = common.build_parser("t")
    with pytest.raises(common.ConfirmationError):
        script.run(
            db, s3, parser.parse_args(["--org", org, "--execute", "--confirm", token])
        )

    assert s3.deleted == []
    assert orphan in s3.objects
