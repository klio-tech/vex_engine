from unittest.mock import MagicMock, patch

from shared import memory as memory_mod


def test_stale_base_revision_returns_conflict_payload():
    db = MagicMock()
    guard = MagicMock(rowcount=0)  # CAS guard missed
    current = MagicMock()
    current.mappings.return_value.first.return_value = {
        "id": "m1", "content": "newer text", "revision": "r2", "user_id": "u2"}
    db.execute.side_effect = [guard, current]
    result = memory_mod.supersede_with_revision(
        db, org_id="o", memory_id="m1", base_revision="r1", new_content="mine")
    assert result["conflict"] is True
    assert result["current"]["revision"] == "r2"
    sql = " ".join(str(db.execute.call_args_list[0].args[0]).split())
    assert "revision = :base_revision" in sql and "org_id = :org_id" in sql


def test_current_base_revision_proceeds_to_supersede():
    db = MagicMock()
    db.execute.return_value.rowcount = 1
    with patch.object(memory_mod, "_apply_supersession_for_cas",
                      return_value={"new_id": "m2", "new_revision": "r9"}) as ap:
        result = memory_mod.supersede_with_revision(
            db, org_id="o", memory_id="m1", base_revision="r1",
            new_content="mine")
    assert result == {"conflict": False, "new_id": "m2", "new_revision": "r9"}
    assert ap.called
