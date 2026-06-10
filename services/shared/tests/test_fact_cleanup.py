"""run_fact_cleanup: dry-run reports, apply retracts + stamps, idempotent."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "scripts"))
from run_fact_cleanup import process_fact_batch  # noqa: E402


class _Row:
    def __init__(self, id, org_id, content):
        self.id = id
        self.org_id = org_id
        self.content = content


class FakeDB:
    def __init__(self):
        self.updates = []

    def execute(self, sql, params=None):
        self.updates.append((str(sql), params))

        class R:
            def fetchall(self_inner):
                return []

        return R()

    def commit(self):
        pass


def test_dry_run_reports_but_writes_nothing():
    db = FakeDB()
    rows = [
        _Row("f1", "org-a", "Tool Bash was used with command: ls"),
        _Row("f2", "org-a", "The team decided to use Stripe."),
        _Row("f3", "org-a", "Used tool Read on file x.py"),
    ]
    report = process_fact_batch(db, rows, apply=False)
    assert report == {"org-a": 2}
    assert db.updates == []


def test_apply_retracts_noise_only():
    db = FakeDB()
    rows = [
        _Row("f1", "org-a", "Tool Bash was used with command: ls"),
        _Row("f2", "org-a", "The user prefers production-ready changes."),
    ]
    report = process_fact_batch(db, rows, apply=True)
    assert report == {"org-a": 1}
    assert len(db.updates) == 1
    sql, params = db.updates[0]
    assert "status = 'retracted'" in sql
    assert params["id"] == "f1"


def test_idempotent_second_pass_noop_on_clean_rows():
    db = FakeDB()
    rows = [_Row("f2", "org-a", "A durable fact.")]
    report = process_fact_batch(db, rows, apply=True)
    assert report == {}
    assert db.updates == []
