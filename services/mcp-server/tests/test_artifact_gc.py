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
