"""Backfill: dry-run reports, apply masks + re-embeds, idempotent."""

import sys
import os


sys.path.insert(
    0, os.path.join(os.path.dirname(__file__), "..", "..", "scripts")
)
from run_redaction_backfill import process_memory_batch  # noqa: E402

SECRET = (
    "sk_test_51TcfsTP4cXbUl5jnQJG2Ou9BUcBChSVqEDNj7d65uQjZXtAVzI1XR2"
    "eewh5KDq8GmCfrr1a1xRNcSdB1c7TtWrU00GsK8y8Je"
)


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


async def _fake_embed(text_input):
    return [0.0, 0.1]


async def test_dry_run_reports_but_writes_nothing():
    db = FakeDB()
    rows = [_Row("m1", "org-a", f"the key is {SECRET}"), _Row("m2", "org-a", "clean")]
    report = await process_memory_batch(db, rows, apply=False, embed_fn=_fake_embed)
    assert report == {("org-a", "stripe_secret_key"): 1}
    assert db.updates == []


async def test_apply_masks_and_reembeds():
    db = FakeDB()
    rows = [_Row("m1", "org-a", f"the key is {SECRET}")]
    report = await process_memory_batch(db, rows, apply=True, embed_fn=_fake_embed)
    assert report == {("org-a", "stripe_secret_key"): 1}
    assert len(db.updates) == 1
    sql, params = db.updates[0]
    assert "UPDATE session_memories" in sql
    assert SECRET not in params["content"]
    assert "[REDACTED:stripe_secret_key:" in params["content"]
    assert params["embedding"] == str([0.0, 0.1])


async def test_embed_failure_leaves_row_unmodified():
    async def _fail(_):
        return None

    db = FakeDB()
    rows = [_Row("m1", "org-a", f"the key is {SECRET}")]
    report = await process_memory_batch(db, rows, apply=True, embed_fn=_fail)
    assert report == {("org-a", "stripe_secret_key"): 1}
    assert db.updates == []  # skipped, reported, not corrupted


async def test_idempotent_second_pass_is_noop():
    db = FakeDB()
    masked = "the key is [REDACTED:stripe_secret_key:ab12]"
    rows = [_Row("m1", "org-a", masked)]
    report = await process_memory_batch(db, rows, apply=True, embed_fn=_fake_embed)
    assert report == {}
    assert db.updates == []
