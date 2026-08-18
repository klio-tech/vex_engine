import sqlalchemy as sa

from shared.testing import requires_db, database_url

pytestmark = requires_db("the project default scope migration tests")


def _insp():
    return sa.inspect(sa.create_engine(database_url()))


def test_projects_has_default_scope_not_null_defaulting_private():
    cols = {c["name"]: c for c in _insp().get_columns("projects")}

    assert "default_scope" in cols
    # NOT NULL is load-bearing: a NULL default would make the router's
    # normalize step decide routing for every capture in that project, which
    # is a decision the column exists to hold explicitly.
    assert cols["default_scope"]["nullable"] is False
    assert "private" in str(cols["default_scope"].get("default"))


def test_project_members_has_a_nullable_scope_override():
    cols = {c["name"]: c for c in _insp().get_columns("project_members")}

    assert "scope_override" in cols
    # NULL here is load-bearing: it is how "follow the project" is expressed,
    # distinct from an explicit narrowing.
    assert cols["scope_override"]["nullable"] is True


def test_the_check_constraints_exist():
    engine = sa.create_engine(database_url())
    with engine.connect() as conn:
        names = {
            r[0]
            for r in conn.execute(
                sa.text(
                    "SELECT conname FROM pg_constraint "
                    "WHERE conname IN ('ck_projects_default_scope', "
                    "'ck_project_members_scope_override')"
                )
            )
        }

    assert "ck_projects_default_scope" in names
    assert "ck_project_members_scope_override" in names


def test_an_override_that_widens_is_rejected_by_the_database():
    # The override exists to narrow. A member who could store 'org' here would
    # be publishing into a project whose admins chose privacy.
    engine = sa.create_engine(database_url())
    with engine.begin() as conn:
        conn.execute(sa.text("SAVEPOINT probe"))
        try:
            conn.execute(
                sa.text(
                    "INSERT INTO project_members "
                    "(project_id, user_id, role, scope_override) VALUES "
                    "('aaaaaaaa-0000-0000-0000-0000000000fe','probe','write','org')"
                )
            )
        except Exception:
            return
        finally:
            conn.execute(sa.text("ROLLBACK TO SAVEPOINT probe"))
    raise AssertionError("the CHECK must refuse a widening override")
