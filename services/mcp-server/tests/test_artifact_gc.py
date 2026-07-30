"""Abandoned-upload collection, and purging of dead artifacts' bytes.

An artifact_put that hands out an upload URL and is never confirmed leaves a
pending row and possibly orphaned bytes. Neither is reachable — no card was
written — so both are collected after 24 hours.

A retracted (``forget``) or superseded artifact is the other half: its row and
card make it look gone from every API, but nothing else in the system ever
removes the object it points at. The same sweep purges those after a grace
window.
"""

from unittest.mock import MagicMock, patch

import pytest

from app.scheduler import collect_artifacts_once


@pytest.fixture
def gc_seams():
    """Patch the sweep's two collaborators with both selectors defaulting empty."""
    with (
        patch("app.scheduler.artifacts") as af,
        patch("app.scheduler.artifact_storage") as st,
    ):
        af.collect_abandoned.return_value = []
        af.collect_purgeable.return_value = []
        st.object_key.side_effect = lambda o, a, v: f"{o}/{a}/v{v}"
        st.get_artifact_s3_client.return_value = MagicMock()
        yield af, st


@pytest.mark.asyncio
async def test_collects_pending_rows_and_deletes_their_objects():
    db = MagicMock()
    with (
        patch("app.scheduler.artifacts") as af,
        patch("app.scheduler.artifact_storage") as st,
    ):
        af.collect_purgeable.return_value = []
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
        af.collect_purgeable.return_value = []
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
        af.collect_purgeable.return_value = []
        assert await collect_artifacts_once(lambda: db) == 0


@pytest.mark.asyncio
async def test_an_idle_tick_never_touches_s3():
    """An empty sweep must be a quiet no-op, including when S3 is misconfigured.

    ``get_artifact_s3_client`` reads endpoint and credentials from the
    environment, so a bad or missing S3 config makes it raise. Building the
    client unconditionally turns every hourly tick of an otherwise idle
    deployment into a logged exception — noise that trains on-call to ignore
    this sweep's alerts, on a job whose real failures delete data. Nothing to
    collect means nothing to connect to.
    """
    db = MagicMock()
    with (
        patch("app.scheduler.artifacts") as af,
        patch("app.scheduler.artifact_storage") as st,
    ):
        af.collect_abandoned.return_value = []
        af.collect_purgeable.return_value = []
        st.get_artifact_s3_client.side_effect = RuntimeError("S3_ENDPOINT unset")

        assert await collect_artifacts_once(lambda: db) == 0

    st.get_artifact_s3_client.assert_not_called()


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
        af.collect_purgeable.return_value = []
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


# ---------------------------------------------------------------------------
# C2: forget / supersede must actually destroy the bytes.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_purges_retracted_and_superseded_rows_and_their_objects(gc_seams):
    """The deletion path that did not ship.

    ``retract`` sets status='retracted' and nothing else. Without this pass a
    customer who calls ``forget`` on a 900MB PDF gets ``{"forgotten": true}``
    while the object stays in the bucket forever — a deletion request that
    cannot be honoured, and a pool that under-counts what is stored.
    """
    af, st = gc_seams
    db = MagicMock()
    af.collect_purgeable.return_value = [
        {"id": "r1", "org_id": "acme", "version": 1, "status": "retracted"},
        {"id": "s1", "org_id": "acme", "version": 2, "status": "superseded"},
    ]

    collected = await collect_artifacts_once(lambda: db)

    assert collected == 2
    deleted_keys = {call.args[1] for call in st.delete.call_args_list}
    assert deleted_keys == {"acme/r1/v1", "acme/s1/v2"}


@pytest.mark.asyncio
async def test_purge_deletes_the_card_so_it_can_never_resurface(gc_seams):
    """The card carries the customer's own title and summary text.

    Hiding it (``recall_hidden``) is what ``forget`` does; purging must remove
    it, or a deletion request leaves the description of the deleted thing
    behind indefinitely.
    """
    af, _st = gc_seams
    db = MagicMock()
    af.collect_purgeable.return_value = [
        {"id": "r1", "org_id": "acme", "version": 1, "status": "retracted"}
    ]

    await collect_artifacts_once(lambda: db)

    card_deletes = [
        call
        for call in db.execute.call_args_list
        if "DELETE FROM session_memories" in str(call.args[0])
    ]
    assert card_deletes, "the artifact's card must be deleted with its row"
    params = card_deletes[0].args[1]
    assert params["org_id"] == "acme"
    assert params["artifact_id"] == "r1"


@pytest.mark.asyncio
async def test_purge_delete_is_guarded_on_a_dead_status(gc_seams):
    """Same ordering discipline as the pending sweep.

    ``retract`` is not final: an undelete (the reason the grace window exists)
    could flip a row back to 'active' between the SELECT and the DELETE.
    Bytes must never be destroyed for a row this sweep did not also remove.
    """
    af, st = gc_seams
    db = MagicMock()
    af.collect_purgeable.return_value = [
        {"id": "r1", "org_id": "acme", "version": 1, "status": "retracted"},
        {"id": "r2", "org_id": "acme", "version": 1, "status": "superseded"},
    ]
    # r1 is no longer dead when the DELETE runs; r2 still is.
    db.execute.return_value.fetchone.side_effect = [None, ("r2",)]

    collected = await collect_artifacts_once(lambda: db)

    assert collected == 1
    # r1's bytes survive, because r1's row did.
    assert st.delete.call_count == 1
    assert st.delete.call_args.args[1] == "acme/r2/v1"

    row_deletes = [
        call
        for call in db.execute.call_args_list
        if "DELETE FROM artifacts" in str(call.args[0])
    ]
    assert row_deletes
    for call in row_deletes:
        statement = str(call.args[0])
        assert "'retracted'" in statement and "'superseded'" in statement


@pytest.mark.asyncio
async def test_one_failed_object_delete_does_not_abort_the_purge_batch(gc_seams):
    af, st = gc_seams
    db = MagicMock()
    af.collect_purgeable.return_value = [
        {"id": "r1", "org_id": "acme", "version": 1, "status": "retracted"},
        {"id": "r2", "org_id": "beta", "version": 1, "status": "superseded"},
    ]
    st.delete.side_effect = [Exception("s3 down"), None]

    assert await collect_artifacts_once(lambda: db) == 2


@pytest.mark.asyncio
async def test_purge_uses_each_rows_own_org_id_for_its_key(gc_seams):
    """The sweep is cross-org by design; every key must come from its own row."""
    af, st = gc_seams
    db = MagicMock()
    af.collect_purgeable.return_value = [
        {"id": "r1", "org_id": "acme", "version": 1, "status": "retracted"},
        {"id": "r2", "org_id": "beta", "version": 3, "status": "superseded"},
    ]

    await collect_artifacts_once(lambda: db)

    assert {call.args[1] for call in st.delete.call_args_list} == {
        "acme/r1/v1",
        "beta/r2/v3",
    }


@pytest.mark.asyncio
async def test_purge_deletes_the_object_even_for_an_inline_artifact(gc_seams):
    """M1: ``artifact_get(mode='url')`` on an INLINE artifact materialises an
    object at the deterministic key while leaving the row inline
    (``storage_uri`` NULL). Keying the purge off ``storage_uri`` would leave
    that object uncollected forever, so the sweep deletes at the key derived
    from (org_id, id, version) unconditionally — a no-op when nothing is there.
    """
    af, st = gc_seams
    db = MagicMock()
    af.collect_purgeable.return_value = [
        # An inline artifact: no storage_uri anywhere in the row.
        {"id": "inline1", "org_id": "acme", "version": 1, "status": "retracted"}
    ]

    await collect_artifacts_once(lambda: db)

    st.delete.assert_called_once()
    assert st.delete.call_args.args[1] == "acme/inline1/v1"


@pytest.mark.asyncio
async def test_both_passes_run_in_one_sweep(gc_seams):
    af, st = gc_seams
    db = MagicMock()
    af.collect_abandoned.return_value = [{"id": "p1", "org_id": "acme", "version": 1}]
    af.collect_purgeable.return_value = [
        {"id": "r1", "org_id": "acme", "version": 1, "status": "retracted"}
    ]

    assert await collect_artifacts_once(lambda: db) == 2
    assert st.delete.call_count == 2


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
