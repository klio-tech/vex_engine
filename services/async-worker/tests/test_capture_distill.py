"""Transcript distillation, moved off the request path.

The behaviour under test is what the old inline path in
``mcp-server/app/capture.py`` did, plus the two things it could not do:
distinguish "nothing durable in this turn" from "the model failed", and
refuse to run the same transcript twice after a stream redelivery.

The scope assertions are the load-bearing ones. Distillation now happens in a
DIFFERENT PROCESS from the request that authorised it, so every boundary value
-- scope, user_id, project_id -- has to survive the queue intact. A job that
came in ``private`` and gets written ``org`` is a cross-tenant leak, and it
would be invisible: the memory would simply appear in somebody else's recall.
"""

from unittest.mock import AsyncMock, MagicMock, patch


from app import capture_distill
from shared.taxonomy import ExtractedMemory

JOB_ID = "33333333-3333-3333-3333-333333333333"


def _job(**over):
    job = {
        "id": JOB_ID,
        "org_id": "org-1",
        "agent_id": "agent-1",
        "session_id": "sess-1",
        "user_id": "user-1",
        "scope": "private",
        "project_id": None,
        "branch_id": None,
        "work_item_id": None,
        "transcript": "user: we decided to drop redis\nassistant: noted",
        "metadata": {"source": "hook-transcript"},
        "over_cap": False,
        "attempts": 1,
    }
    job.update(over)
    return job


def _patches(*, job, summary="a summary", memories=None, write=None):
    """Patch the seams around distill_job: the claim, the two LLM calls, and
    the write primitive."""
    memories = [] if memories is None else memories
    return (
        patch.object(capture_distill, "claim_job", MagicMock(return_value=job)),
        patch.object(capture_distill, "finish_job", MagicMock()),
        patch.object(capture_distill, "fail_job", MagicMock()),
        patch.object(
            capture_distill, "generate_summary", AsyncMock(return_value=summary)
        ),
        patch.object(
            capture_distill,
            "extract_typed_memories",
            AsyncMock(return_value=memories),
        ),
        patch.object(
            capture_distill,
            "write_scoped_memory",
            write or AsyncMock(return_value={"stored": True}),
        ),
        patch.object(capture_distill, "record_usage", MagicMock()),
    )


async def _run(job, **kw):
    write = AsyncMock(return_value={"stored": True})
    ps = _patches(job=job, write=write, **kw)
    for p in ps:
        p.start()
    try:
        outcome = await capture_distill.distill_job(MagicMock(), JOB_ID)
    finally:
        for p in ps:
            p.stop()
    return outcome, write


async def test_writes_the_summary_and_each_typed_memory():
    memories = [
        ExtractedMemory(content="Redis was dropped.", memory_type="decision"),
        ExtractedMemory(
            content="Team prefers Postgres streams.", memory_type="preference"
        ),
    ]
    outcome, write = await _run(_job(), memories=memories)
    assert outcome == "done"
    kinds = [c.kwargs["memory_type"] for c in write.await_args_list]
    assert kinds == ["summary", "decision", "preference"]


async def test_every_boundary_value_survives_the_queue():
    """scope / user_id / project_id must reach the write exactly as issued."""
    job = _job(scope="private", user_id="user-1", project_id="proj-9")
    _, write = await _run(
        job, memories=[ExtractedMemory(content="X.", memory_type="fact")]
    )
    for call in write.await_args_list:
        assert call.kwargs["scope"] == "private"
        assert call.kwargs["user_id"] == "user-1"
        assert call.kwargs["project_id"] == "proj-9"
        assert call.kwargs["org_id"] == "org-1"


async def test_a_turn_with_nothing_durable_is_done_not_failed():
    """The distinction the inline path could not make."""
    ps = _patches(job=_job(), summary=None, memories=[])
    for p in ps:
        p.start()
    try:
        outcome = await capture_distill.distill_job(MagicMock(), JOB_ID)
        finish = capture_distill.finish_job
        assert outcome == "done"
        assert finish.call_args.kwargs == {
            "summary_written": False,
            "memories_written": 0,
        }
        capture_distill.fail_job.assert_not_called()
    finally:
        for p in ps:
            p.stop()


async def test_a_failed_summary_does_not_block_extraction():
    """Independent tolerance, as the original endpoint documented."""
    outcome, write = await _run(
        _job(),
        summary=None,
        memories=[ExtractedMemory(content="Still kept.", memory_type="fact")],
    )
    assert outcome == "done"
    assert [c.kwargs["memory_type"] for c in write.await_args_list] == ["fact"]


async def test_an_unclaimable_job_does_no_model_work():
    """Redis redelivers; a second delivery must not re-spend two LLM calls."""
    with (
        patch.object(capture_distill, "claim_job", MagicMock(return_value=None)),
        patch.object(capture_distill, "generate_summary", AsyncMock()) as gs,
        patch.object(capture_distill, "extract_typed_memories", AsyncMock()) as ex,
    ):
        assert await capture_distill.distill_job(MagicMock(), JOB_ID) == "skipped"
        gs.assert_not_awaited()
        ex.assert_not_awaited()


async def test_an_exception_marks_the_job_failed_rather_than_crashing():
    with (
        patch.object(capture_distill, "claim_job", MagicMock(return_value=_job())),
        patch.object(
            capture_distill,
            "generate_summary",
            AsyncMock(side_effect=RuntimeError("gateway on fire")),
        ),
        patch.object(
            capture_distill, "extract_typed_memories", AsyncMock(return_value=[])
        ),
        patch.object(capture_distill, "write_scoped_memory", AsyncMock()),
        patch.object(capture_distill, "record_usage", MagicMock()),
        patch.object(capture_distill, "finish_job", MagicMock()),
        patch.object(capture_distill, "fail_job", MagicMock()) as fail,
    ):
        assert await capture_distill.distill_job(MagicMock(), JOB_ID) == "failed"
        assert "gateway on fire" in fail.call_args.kwargs["error"]


async def test_usage_is_recorded_once_per_stored_memory_only():
    """A deduped write is not a new memory and must not be billed as one."""
    write = AsyncMock(
        side_effect=[{"stored": True}, {"stored": False}, {"stored": True}]
    )
    ps = _patches(
        job=_job(),
        write=write,
        memories=[
            ExtractedMemory(content="A.", memory_type="fact"),
            ExtractedMemory(content="B.", memory_type="fact"),
        ],
    )
    for p in ps:
        p.start()
    try:
        await capture_distill.distill_job(MagicMock(), JOB_ID)
        assert capture_distill.record_usage.call_count == 2
        assert capture_distill.finish_job.call_args.kwargs["memories_written"] == 1
    finally:
        for p in ps:
            p.stop()


async def test_an_empty_extracted_item_is_skipped():
    outcome, write = await _run(
        _job(),
        memories=[
            ExtractedMemory(content="", memory_type="fact"),
            ExtractedMemory(content="Real.", memory_type="fact"),
        ],
    )
    assert outcome == "done"
    assert [c.kwargs["content"] for c in write.await_args_list] == [
        "a summary",
        "Real.",
    ]
