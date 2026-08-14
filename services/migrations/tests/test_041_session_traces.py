import sqlalchemy as sa

from shared.testing import requires_db, database_url

pytestmark = requires_db("the session_traces migration tests")


def _insp():
    return sa.inspect(sa.create_engine(database_url()))


def test_session_traces_columns():
    cols = {c["name"]: c for c in _insp().get_columns("session_traces")}
    for n in (
        "id",
        "org_id",
        "session_id",
        "transcript_ref",
        "message_count",
        "tool_calls_count",
        "redaction_version",
        "created_at",
        "updated_at",
    ):
        assert n in cols, n
    assert cols["org_id"]["nullable"] is False
    assert cols["message_count"]["nullable"] is False


def test_session_traces_unique_org_session():
    constraints = _insp().get_unique_constraints("session_traces")
    u = [c for c in constraints if c["name"] == "uq_session_traces_org_session"]
    assert u and u[0]["column_names"] == ["org_id", "session_id"]
