"""Artifact lifecycle over a fake DB session.

The fake session records executed SQL and its parameters and returns queued
row payloads, so these tests assert on tenancy predicates and on the TS-side
transformation without a live database.
"""



from shared import artifacts as af


class FakeResult:
    def __init__(self, rows):
        self._rows = rows

    def fetchone(self):
        return self._rows[0] if self._rows else None

    def fetchall(self):
        return self._rows

    def scalar(self):
        return self._rows[0][0] if self._rows else None


class FakeSession:
    """Records (sql, params) per execute and returns queued results FIFO."""

    def __init__(self, *results):
        self.calls = []
        self._queue = list(results)

    def execute(self, sql, params=None):
        self.calls.append((str(sql), params or {}))
        return self._queue.pop(0) if self._queue else FakeResult([])

    def commit(self):
        pass


def test_human_size_is_readable():
    assert af.human_size(900) == "900 B"
    assert af.human_size(2048) == "2.0 KB"
    assert af.human_size(5 * 1024 * 1024) == "5.0 MB"
    assert af.human_size(3 * 1024 * 1024 * 1024) == "3.0 GB"


def test_card_content_matches_the_specified_format():
    card = af.card_content("Q3 pricing", "The model finance signed off", "file", 2048)
    assert card == "[artifact] Q3 pricing — The model finance signed off (file, 2.0 KB)"


def test_org_storage_bytes_counts_only_active_rows_for_the_org():
    db = FakeSession(FakeResult([(4096,)]))

    total = af.org_storage_bytes(db, org_id="acme")

    assert total == 4096
    sql, params = db.calls[0]
    assert "org_id = :org_id" in sql
    assert "status = 'active'" in sql
    assert params["org_id"] == "acme"


def test_org_storage_bytes_is_zero_when_nothing_stored():
    db = FakeSession(FakeResult([(None,)]))
    assert af.org_storage_bytes(db, org_id="acme") == 0


def test_get_artifact_is_org_scoped():
    db = FakeSession(FakeResult([]))

    assert af.get_artifact(db, org_id="other-org", artifact_id="a1") is None

    sql, params = db.calls[0]
    assert "org_id = :org_id" in sql
    assert params["org_id"] == "other-org"


def test_find_active_by_checksum_is_org_scoped():
    db = FakeSession(FakeResult([]))
    af.find_active_by_checksum(db, org_id="acme", checksum="deadbeef")
    sql, params = db.calls[0]
    assert "org_id = :org_id" in sql
    assert "status = 'active'" in sql
    assert params["checksum"] == "deadbeef"


def test_create_pending_inserts_without_size_or_checksum():
    db = FakeSession(FakeResult([("new-id",)]))

    artifact_id = af.create_pending(
        db,
        org_id="acme",
        agent_id="agent-1",
        title="t",
        summary="s",
        kind="document",
        mime_type="text/markdown",
    )

    assert artifact_id == "new-id"
    sql, params = db.calls[0]
    assert "'pending'" in sql
    # Measured values are never written at create time.
    assert "size_bytes" not in sql
    assert "checksum" not in sql
    assert params["org_id"] == "acme"


def test_activate_writes_measured_values_and_is_org_scoped():
    db = FakeSession(FakeResult([("a1", 3, "abc")]))

    af.activate(
        db,
        org_id="acme",
        artifact_id="a1",
        size_bytes=3,
        checksum="abc",
        inline_text="hey",
    )

    sql, params = db.calls[0]
    assert "org_id = :org_id" in sql
    assert "'active'" in sql
    assert params["size_bytes"] == 3
    assert params["checksum"] == "abc"
    assert params["inline_text"] == "hey"


def test_retract_is_org_scoped_and_reports_whether_it_hit():
    db = FakeSession(FakeResult([]))
    assert af.retract(db, org_id="other-org", artifact_id="a1") is False
    sql, params = db.calls[0]
    assert "org_id = :org_id" in sql
    assert "'retracted'" in sql


def test_collect_abandoned_selects_only_stale_pending_rows():
    db = FakeSession(FakeResult([("a1", "acme", 1)]))

    rows = af.collect_abandoned(db, older_than_hours=24)

    assert rows == [{"id": "a1", "org_id": "acme", "version": 1}]
    sql, params = db.calls[0]
    assert "'pending'" in sql
    assert params["hours"] == 24


# --- C2: retracted / superseded rows must become collectable -------------


def test_collect_purgeable_selects_retracted_and_superseded_rows():
    """``forget`` and supersede leave bytes behind unless something sweeps them.

    Without this selector there is no code path anywhere that will ever delete
    a retracted artifact's object: the row reads as gone from every API, stops
    counting against the pool, and the bytes stay in the bucket forever. A
    deletion request cannot be honoured and storage grows monotonically while
    quota under-counts it.
    """
    db = FakeSession(FakeResult([("a1", "acme", 1, "retracted")]))

    rows = af.collect_purgeable(db, older_than_hours=af.PURGE_GRACE_HOURS)

    assert rows == [
        {"id": "a1", "org_id": "acme", "version": 1, "status": "retracted"}
    ]
    sql, params = db.calls[0]
    assert "'retracted'" in sql
    assert "'superseded'" in sql
    # Pending rows belong to the other sweep; mixing them would delete an
    # in-flight upload with no grace period of its own.
    assert "'pending'" not in sql
    assert params["hours"] == af.PURGE_GRACE_HOURS


def test_purge_grace_window_is_a_named_nonzero_constant():
    """A grace window is what keeps a future undelete possible."""
    assert isinstance(af.PURGE_GRACE_HOURS, int)
    assert af.PURGE_GRACE_HOURS > 0


# --- I1: a confirm that duplicates existing bytes must finalise cleanly ---


def test_delete_pending_is_org_scoped_and_status_guarded():
    """Used when a confirm dedups: the redundant pending row must not linger
    for 24h as a GC candidate, but it must never delete an active row.
    """
    db = FakeSession(FakeResult([("a1",)]))

    assert af.delete_pending(db, org_id="acme", artifact_id="a1") is True

    sql, params = db.calls[0]
    assert "DELETE FROM artifacts" in sql
    assert "org_id = :org_id" in sql
    assert "status = 'pending'" in sql
    assert params["org_id"] == "acme"


def test_delete_pending_reports_a_miss_for_another_org():
    db = FakeSession(FakeResult([]))
    assert af.delete_pending(db, org_id="other-org", artifact_id="a1") is False


# --- I3: pending rows are an unbilled, unbounded reservation -------------


def test_pending_count_is_org_scoped_to_pending_rows():
    """Each pending row is a presigned PUT nobody counts. Uncapped, an org can
    mint a thousand of them and upload through every one.
    """
    db = FakeSession(FakeResult([(7,)]))

    assert af.pending_count(db, org_id="acme") == 7

    sql, params = db.calls[0]
    assert "org_id = :org_id" in sql
    assert "status = 'pending'" in sql
    assert params["org_id"] == "acme"


def test_max_pending_per_org_is_a_named_positive_constant():
    assert isinstance(af.MAX_PENDING_PER_ORG, int)
    assert af.MAX_PENDING_PER_ORG > 0
