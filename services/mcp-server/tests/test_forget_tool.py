"""forget: the pure-Python half of retract_memory's authorization ladder.

Scope, deliberately: ``FakeDB`` can test the branches that are decided in
Python — a malformed id, a private row with the wrong owner, an agent row with
the wrong agent, and that a refusal emits no UPDATE. It CANNOT test the
tenancy boundary, and no fake can. ``retract_memory`` builds its parameter
dict from its own arguments, so ``params`` is identical whether or not the SQL
references ``:org_id``; a params-inspecting fake would see no difference
between the guarded query and an unguarded one, and asserting on the emitted
SQL *text* only proves the string was written, not that the server honoured
it.

The cross-org attack therefore lives on real Postgres, in
``services/shared/tests/test_forget_isolation.py``. Deleting
``org_id = :org_id`` from the SELECT above fails five tests there and none
here — which is the correct division of labour, not a gap in this file.
"""

import uuid


from shared.memory import retract_memory


class _Row:
    def __init__(self, scope, agent_id):
        self.scope = scope
        self.agent_id = agent_id


class FakeDB:
    def __init__(self, row=None):
        self.row = row
        self.updates = []

    def execute(self, sql, params=None):
        sql_s = str(sql)
        self.updates.append((sql_s, params))

        class R:
            def __init__(self, row):
                self._row = row

            def fetchone(self_inner):
                return self.row if "SELECT" in sql_s else None

        return R(self.row)

    def commit(self):
        pass


def test_retract_unknown_id_returns_not_found():
    db = FakeDB(row=None)
    result = retract_memory(
        db, org_id="org-1", agent_id="a1", memory_id=str(uuid.uuid4()), via="mcp"
    )
    assert result == {"forgotten": False, "error": "not_found"}


def test_retract_org_row_any_agent_succeeds():
    """An org-scoped row is retractable by any agent in that org — the ladder
    falls through both the private and the agent check.

    This asserts the ladder's *decision*, not the SQL text. The old version
    also asserted ``"status = 'retracted'" in update_sql``, which passed
    whether or not the UPDATE matched a row and so proved nothing; that the
    retraction actually lands is asserted against a real server in
    ``test_forget_isolation.py::test_the_owning_org_can_retract_it``."""
    db = FakeDB(row=_Row(scope="org", agent_id="someone-else"))
    result = retract_memory(
        db, org_id="org-1", agent_id="a1", memory_id=str(uuid.uuid4()), via="mcp"
    )
    assert result["forgotten"] is True
    assert any("UPDATE" in s for s, _ in db.updates)


def test_retract_agent_row_wrong_agent_forbidden():
    db = FakeDB(row=_Row(scope="agent", agent_id="owner-agent"))
    result = retract_memory(
        db, org_id="org-1", agent_id="intruder", memory_id=str(uuid.uuid4()), via="mcp"
    )
    assert result == {"forgotten": False, "error": "forbidden"}
    assert not any("UPDATE" in s for s, _ in db.updates)


def test_retract_rejects_malformed_id():
    db = FakeDB()
    result = retract_memory(
        db, org_id="org-1", agent_id="a1", memory_id="not-a-uuid", via="mcp"
    )
    assert result == {"forgotten": False, "error": "invalid_id"}


def test_forget_is_registered():
    """Phase 2/3 added share + the four project tools, so the count moved from
    10 to 15. Compression added klio_retrieve and then compress, taking it to
    17. The knowledge graph added explore, taking it to 18. Onboarding added
    init, taking it to 19. The names are asserted rather than only the count,
    because a count alone would pass if a tool were swapped rather than
    added."""
    from app.mcp_app import EXPECTED_TOOL_NAMES

    assert "forget" in EXPECTED_TOOL_NAMES
    assert {
        "share",
        "project_create",
        "project_list",
        "project_members",
        "project_grant",
    } <= set(EXPECTED_TOOL_NAMES)
    assert {"klio_retrieve", "compress"} <= set(EXPECTED_TOOL_NAMES)
    assert "explore" in EXPECTED_TOOL_NAMES
    assert "init" in EXPECTED_TOOL_NAMES
    assert len(EXPECTED_TOOL_NAMES) == 19
