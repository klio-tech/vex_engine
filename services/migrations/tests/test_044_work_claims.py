import sqlalchemy as sa

from shared.testing import requires_db, database_url

pytestmark = requires_db("the work_claims migration tests")


def _insp():
    return sa.inspect(sa.create_engine(database_url()))


def test_work_claims_columns():
    cols = {c["name"]: c for c in _insp().get_columns("work_claims")}
    assert {
        "id",
        "org_id",
        "project_id",
        "agent_id",
        "user_id",
        "description",
        "embedding",
        "created_at",
        "expires_at",
        "released_at",
    } <= set(cols)
    assert cols["expires_at"]["nullable"] is False
    assert cols["description"]["nullable"] is False
    assert cols["released_at"]["nullable"] is True


def test_work_claims_org_expires_index():
    indexes = _insp().get_indexes("work_claims")
    ix = [i for i in indexes if i["name"] == "ix_work_claims_org_expires"]
    assert ix and ix[0]["column_names"] == ["org_id", "expires_at"]
