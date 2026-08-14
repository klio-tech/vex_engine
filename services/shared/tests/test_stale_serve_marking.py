from datetime import datetime, timezone
from unittest.mock import MagicMock

from shared import stale


def test_predicate_is_after_superseding_belief_existed():
    db = MagicMock()
    db.execute.return_value.rowcount = 2
    ts = datetime(2026, 8, 1, tzinfo=timezone.utc)
    marked = stale.mark_stale_serves(
        db, org_id="o", old_memory_id="m-old", superseding_created_at=ts
    )
    assert marked == 2
    sql = " ".join(str(db.execute.call_args.args[0]).split())
    params = db.execute.call_args.args[1]
    assert "bre.created_at >= :superseding_created_at" in sql
    assert ":old_memory_id = ANY(bre.memory_ids)" in sql
    assert params["superseding_created_at"] == ts
    # the naive trailing-window predicate must NOT be present
    assert "make_interval" not in sql


def test_never_raises():
    db = MagicMock()
    db.execute.side_effect = RuntimeError("down")
    assert (
        stale.mark_stale_serves(
            db,
            org_id="o",
            old_memory_id="m",
            superseding_created_at=datetime.now(timezone.utc),
        )
        == 0
    )
