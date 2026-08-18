import sqlalchemy as sa

from shared.testing import requires_db, database_url

pytestmark = requires_db("the dream_state migration tests")


def _insp():
    return sa.inspect(sa.create_engine(database_url()))


def test_dream_state_columns():
    cols = {c["name"]: c for c in _insp().get_columns("dream_state")}
    assert {
        "org_id",
        "sessions_since",
        "last_run_at",
        "last_dreamed_at",
        "last_proposals_written",
        "dream_passes",
        "created_at",
        "updated_at",
    } <= set(cols)
    # The two trigger columns must never be NULL — a NULL counter or a NULL
    # clock would make the claim predicate in app.dream_state.claim_dream_pass
    # silently unsatisfiable for that row.
    assert cols["sessions_since"]["nullable"] is False
    assert cols["last_run_at"]["nullable"] is False
    assert cols["dream_passes"]["nullable"] is False
    # NULL here is load-bearing: it is how "this org has never been dreamed
    # for" is distinguished from "a pass ran and wrote nothing".
    assert cols["last_dreamed_at"]["nullable"] is True
    assert cols["last_proposals_written"]["nullable"] is True


def test_dream_state_org_id_is_the_primary_key():
    """One row per org — this is a cursor, not an audit log. The primary key
    is also what makes ``ON CONFLICT (org_id)`` in
    ``app.dream_state.record_graded_session`` a valid upsert target.
    """
    pk = _insp().get_pk_constraint("dream_state")
    assert pk["constrained_columns"] == ["org_id"]


def test_dream_state_last_run_at_index():
    """Serves the interval-timer scan (``orgs_due_for_dream``), the only query
    that does not already know its org_id.
    """
    indexes = _insp().get_indexes("dream_state")
    ix = [i for i in indexes if i["name"] == "ix_dream_state_last_run_at"]
    assert ix and ix[0]["column_names"] == ["last_run_at"]


def test_dream_state_defaults_let_a_bare_insert_succeed():
    """A row created with org_id alone must be immediately valid: the counter
    starts at 0, the clock starts now, and the pass count starts at 0.
    """
    engine = sa.create_engine(database_url())
    with engine.begin() as c:
        c.execute(
            sa.text("INSERT INTO dream_state (org_id) VALUES ('mig046-defaults')")
        )
        row = (
            c.execute(
                sa.text(
                    "SELECT sessions_since, last_run_at, last_dreamed_at,"
                    " last_proposals_written, dream_passes FROM dream_state"
                    " WHERE org_id = 'mig046-defaults'"
                )
            )
            .mappings()
            .one()
        )
        assert row["sessions_since"] == 0
        assert row["dream_passes"] == 0
        assert row["last_run_at"] is not None
        assert row["last_dreamed_at"] is None
        assert row["last_proposals_written"] is None
        c.execute(sa.text("DELETE FROM dream_state WHERE org_id = 'mig046-defaults'"))
