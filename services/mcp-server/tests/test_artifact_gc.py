"""Abandoned-upload collection.

An artifact_put that hands out an upload URL and is never confirmed leaves a
pending row and possibly orphaned bytes. Neither is reachable — no card was
written — so both are collected after 24 hours.
"""

from unittest.mock import MagicMock, patch

import pytest

from app.scheduler import collect_artifacts_once


@pytest.mark.asyncio
async def test_collects_pending_rows_and_deletes_their_objects():
    db = MagicMock()
    with (
        patch("app.scheduler.artifacts") as af,
        patch("app.scheduler.artifact_storage") as st,
    ):
        af.collect_abandoned.return_value = [
            {"id": "a1", "org_id": "acme", "version": 1},
            {"id": "a2", "org_id": "acme", "version": 1},
        ]
        st.object_key.side_effect = lambda o, a, v: f"{o}/{a}/v{v}"
        st.get_artifact_s3_client.return_value = MagicMock()

        collected = await collect_artifacts_once(lambda: db)

    assert collected == 2
    assert st.delete.call_count == 2


@pytest.mark.asyncio
async def test_a_missing_object_does_not_stop_the_sweep():
    db = MagicMock()
    with (
        patch("app.scheduler.artifacts") as af,
        patch("app.scheduler.artifact_storage") as st,
    ):
        af.collect_abandoned.return_value = [
            {"id": "a1", "org_id": "acme", "version": 1},
            {"id": "a2", "org_id": "acme", "version": 1},
        ]
        st.get_artifact_s3_client.return_value = MagicMock()
        st.delete.side_effect = [Exception("gone"), None]

        collected = await collect_artifacts_once(lambda: db)

    # The row is deleted even when its object was never uploaded — that is the
    # common case for an abandoned upload.
    assert collected == 2


@pytest.mark.asyncio
async def test_nothing_to_collect_is_not_an_error():
    db = MagicMock()
    with (
        patch("app.scheduler.artifacts") as af,
        patch("app.scheduler.artifact_storage"),
    ):
        af.collect_abandoned.return_value = []
        assert await collect_artifacts_once(lambda: db) == 0


@pytest.mark.asyncio
async def test_a_row_confirmed_before_delete_keeps_its_object():
    """TOCTOU: activate() can flip a row from 'pending' to 'active' in the
    window between collect_abandoned's SELECT and this sweep's DELETE. The
    DELETE itself must carry a ``status = 'pending'`` guard so it only
    removes rows that are still pending — and the object for a row the
    DELETE declined to remove must be left untouched, never destroyed.

    a1 simulates the race (confirmed in the window, DELETE returns no row);
    a2 simulates the ordinary case (still pending, DELETE removes it).
    """
    db = MagicMock()
    with (
        patch("app.scheduler.artifacts") as af,
        patch("app.scheduler.artifact_storage") as st,
    ):
        af.collect_abandoned.return_value = [
            {"id": "a1", "org_id": "acme", "version": 1},
            {"id": "a2", "org_id": "acme", "version": 1},
        ]
        st.get_artifact_s3_client.return_value = MagicMock()
        # First DELETE (a1) finds it no longer pending -> no row returned.
        # Second DELETE (a2) still pending -> row returned.
        db.execute.return_value.fetchone.side_effect = [None, ("a2",)]

        collected = await collect_artifacts_once(lambda: db)

    assert collected == 1
    # The object for the row that was NOT deleted (a1) must never be touched.
    st.object_key.assert_called_once_with("acme", "a2", 1)
    assert st.delete.call_count == 1

    # Every DELETE issued must be guarded by status = 'pending' — this is
    # what prevents the sweep from ever removing (or orphaning the bytes of)
    # a row that was confirmed in the race window.
    delete_calls = db.execute.call_args_list
    assert delete_calls, "expected at least one DELETE to have been issued"
    for call in delete_calls:
        statement = str(call.args[0])
        assert "DELETE FROM artifacts" in statement
        assert "status = 'pending'" in statement


@pytest.mark.asyncio
async def test_disabled_gc_does_not_start_the_loop():
    """ARTIFACT_GC_ENABLED=False (or enabled=False) must prevent the loop
    from starting at all — the only remedy on-call has for a misbehaving
    sweep short of a code change and redeploy.
    """
    from types import SimpleNamespace

    from app.scheduler import start_artifact_gc

    app = SimpleNamespace(state=SimpleNamespace())
    task = start_artifact_gc(app, lambda: MagicMock(), enabled=False)

    assert task is None
    assert app.state.artifact_gc_task is None
