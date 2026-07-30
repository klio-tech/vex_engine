"""Artifact lifecycle over a fake DB session.

The fake session records executed SQL and its parameters and returns queued
row payloads, so these tests assert on tenancy predicates and on the TS-side
transformation without a live database.
"""

from unittest.mock import MagicMock

import pytest

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
