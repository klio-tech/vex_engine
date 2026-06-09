"""forget: retract a memory by id with org/agent authorization."""

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
    db = FakeDB(row=_Row(scope="org", agent_id="someone-else"))
    result = retract_memory(
        db, org_id="org-1", agent_id="a1", memory_id=str(uuid.uuid4()), via="mcp"
    )
    assert result["forgotten"] is True
    update_sql = [s for s, _ in db.updates if "UPDATE" in s][0]
    assert "status = 'retracted'" in update_sql


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


def test_forget_is_registered_as_eighth_tool():
    from app.mcp_app import EXPECTED_TOOL_NAMES

    assert "forget" in EXPECTED_TOOL_NAMES
    assert len(EXPECTED_TOOL_NAMES) == 8
