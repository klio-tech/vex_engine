"""The maintenance selectors must be bounded, and must starve nothing.

``collect_abandoned`` and ``collect_purgeable`` returned every matching row in
the deployment. ``collect_purgeable`` in particular has no natural cap —
retention or a mass ``forget`` can make every artifact in every org dead at
once — and the sweep then attempts all of them in a single tick, each with a
commit and a boto3 ``delete_object`` (5 s connect / 30 s read / 3 adaptive
retries).

A LIMIT alone would be a bug: without a deterministic order, one tick's window
can repeat forever and the oldest rows never get collected. So the bound is
LIMIT *plus* oldest-first, which makes successive ticks drain the backlog.
"""

import uuid

import pytest
from shared import artifacts
from shared.testing import database_url, requires_db
from sqlalchemy import create_engine, text

pytestmark = requires_db("the artifact maintenance sweep bounds")


@pytest.fixture
def db():
    engine = create_engine(database_url())
    conn = engine.connect()
    try:
        yield conn
    finally:
        conn.close()
        engine.dispose()


def _seed(db, org_id, *, status, count, base_hours):
    """Insert *count* artifacts, each older than the last."""
    ids = []
    for i in range(count):
        artifact_id = str(uuid.uuid4())
        age = base_hours + count - i  # row 0 is the OLDEST
        db.execute(
            text(
                """
                INSERT INTO artifacts (
                    id, org_id, agent_id, title, summary, kind, mime_type,
                    size_bytes, checksum, storage_uri, version, status,
                    created_at, retracted_at
                ) VALUES (
                    :id, :org_id, 'agent-1', :title, 'seed', 'file',
                    'text/plain', 10, :checksum, :uri, 1, :status,
                    NOW() - (:age || ' hours')::interval,
                    CASE WHEN :status = 'pending' THEN NULL
                         ELSE NOW() - (:age || ' hours')::interval END
                )
                """
            ),
            {
                "id": artifact_id,
                "org_id": org_id,
                "title": f"a{i}",
                "checksum": f"c{artifact_id}",
                "uri": f"s3://b/{org_id}/{artifact_id}/v1",
                "status": status,
                "age": age,
            },
        )
        ids.append(artifact_id)
    db.commit()
    return ids


@pytest.fixture
def org_id(db):
    oid = f"org-sweep-{uuid.uuid4()}"
    yield oid
    db.execute(text("DELETE FROM artifacts WHERE org_id = :o"), {"o": oid})
    db.commit()


def test_collect_abandoned_honours_a_limit(db, org_id):
    _seed(db, org_id, status="pending", count=5, base_hours=48)
    rows = artifacts.collect_abandoned(db, limit=2)
    mine = [r for r in rows if r["org_id"] == org_id]
    assert len(rows) <= 2
    assert mine, "positive control: the seeded rows are selectable at all"


def test_collect_purgeable_honours_a_limit(db, org_id):
    _seed(db, org_id, status="retracted", count=5, base_hours=24 * 30)
    rows = artifacts.collect_purgeable(db, limit=2)
    assert len(rows) <= 2
    assert rows, "positive control: dead rows exist"


def test_no_limit_still_returns_everything(db, org_id):
    """Positive control: the bound is opt-in; existing callers are unchanged."""
    _seed(db, org_id, status="pending", count=5, base_hours=48)
    rows = artifacts.collect_abandoned(db)
    mine = [r for r in rows if r["org_id"] == org_id]
    assert len(mine) == 5


def test_the_oldest_rows_are_taken_first(db, org_id):
    """A bound with no order starves the backlog it exists to drain."""
    ids = _seed(db, org_id, status="pending", count=6, base_hours=48)
    # Isolate to this org so other tests' rows cannot mask the ordering.
    rows = artifacts.collect_abandoned(db, limit=None)
    mine = [r["id"] for r in rows if r["org_id"] == org_id]
    assert mine == ids, "rows are not returned oldest-first"


def test_purgeable_is_also_ordered_oldest_first(db, org_id):
    ids = _seed(db, org_id, status="retracted", count=4, base_hours=24 * 30)
    rows = artifacts.collect_purgeable(db, limit=None)
    mine = [r["id"] for r in rows if r["org_id"] == org_id]
    assert mine == ids, "rows are not returned oldest-first"


def test_a_nonsense_limit_is_refused_not_silently_unbounded(db):
    """A zero or negative bound is a caller bug, not a request for 'all rows'."""
    with pytest.raises(ValueError):
        artifacts.collect_abandoned(db, limit=0)
    with pytest.raises(ValueError):
        artifacts.collect_purgeable(db, limit=-1)
