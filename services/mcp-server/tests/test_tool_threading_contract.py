"""The MCP tool bodies must honour the threading contract ``capture.py`` follows.

``shared/projects.py`` states it:

    Synchronous (blocking psycopg2 I/O), matching the house pattern in
    :mod:`shared.brain`. Call from async handlers via ``anyio.to_thread.run_sync``.

``app/capture.py`` honours it at 14 sites, each with a comment citing it.
``app/mcp_app.py`` used ``to_thread`` only for S3 and never for the database,
so every ``remember`` and every ``recall`` executed its quota check, project
resolution, space resolution, scope decision, metering and label lookup
directly on the event loop.

Measured with 6 statements at a 10 ms round trip plus one 100 ms embed::

    inline      concurrency= 1  rps=  5.5  worst_loop_lag=   75.7ms
    inline      concurrency=20  rps= 11.5  worst_loop_lag= 1626.4ms
    inline      concurrency=50  rps= 11.9  worst_loop_lag= 4092.2ms
    to_thread   concurrency= 1  rps=  5.9  worst_loop_lag=    1.3ms
    to_thread   concurrency=20  rps= 80.5  worst_loop_lag=    2.0ms
    to_thread   concurrency=50  rps=126.3  worst_loop_lag=    1.4ms

Throughput plateaued at ~12 rps however much concurrency was offered: the
event loop, not the connection pool, was the first ceiling.

These tests assert the mechanism — the thread each blocking helper is called
on is not the thread running the event loop — rather than a wall-clock number,
which would be flaky.
"""

import threading
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import app.mcp_app as mcp_app
import pytest
from app.mcp_app import build_mcp_server


def _fake_ctx(headers: dict | None = None):
    request = SimpleNamespace(headers=headers if headers is not None else {})
    return SimpleNamespace(request_context=SimpleNamespace(request=request))


def _get_tool_fn(name: str):
    server = build_mcp_server()
    tool = server._tool_manager.get_tool(name)
    assert tool is not None, f"tool {name!r} not registered"
    return tool.fn


class _ThreadRecorder:
    """A callable that records the thread it was invoked on."""

    def __init__(self, return_value=None):
        self.threads: list[int] = []
        self.return_value = return_value

    def __call__(self, *args, **kwargs):
        self.threads.append(threading.get_ident())
        return self.return_value

    def off_loop(self, loop_ident: int) -> bool:
        return bool(self.threads) and all(t != loop_ident for t in self.threads)


@pytest.mark.asyncio
async def test_remember_does_not_run_its_db_work_on_the_event_loop():
    loop_ident = threading.get_ident()
    fn = _get_tool_fn("remember")

    quota = _ThreadRecorder(return_value=False)
    project = _ThreadRecorder(return_value=None)
    space = _ThreadRecorder(return_value="space-1")
    usage = _ThreadRecorder(return_value=None)

    write_mock = AsyncMock(
        return_value={
            "stored": True,
            "deduplicated": False,
            "id": "00000000-0000-0000-0000-000000000001",
            "memory_type": "memory",
            "scope": "org",
        }
    )

    with (
        patch(
            "app.mcp_app.resolve_principal_plan",
            return_value=("org-x", "agent-y", "free", None),
        ),
        patch("app.mcp_app.SessionLocal", MagicMock(return_value=MagicMock())),
        patch("app.mcp_app.check_memory_quota", quota),
        patch("app.mcp_app.resolve_project_from_request", project),
        patch("app.mcp_app.record_usage", usage),
        patch.object(mcp_app.brain, "ensure_space", space),
        patch.object(mcp_app.brain, "write_scoped_memory", write_mock),
    ):
        result = await fn(_fake_ctx({"mcp-session-id": "conn-1"}), "content", space="s")

    assert result["stored"] is True, "positive control: the write path completed"
    for name, rec in [
        ("check_memory_quota", quota),
        ("resolve_project_from_request", project),
        ("brain.ensure_space", space),
        ("record_usage", usage),
    ]:
        assert rec.threads, f"positive control: {name} was never called"
        assert rec.off_loop(loop_ident), (
            f"{name} ran on the event loop thread ({rec.threads})"
        )


@pytest.mark.asyncio
async def test_recall_does_not_run_its_db_work_on_the_event_loop():
    loop_ident = threading.get_ident()
    fn = _get_tool_fn("recall")

    quota = _ThreadRecorder(return_value=False)
    find_space = _ThreadRecorder(return_value="space-1")
    usage = _ThreadRecorder(return_value=None)
    log = _ThreadRecorder(return_value=None)

    retrieve = AsyncMock(return_value=[])

    with (
        patch(
            "app.mcp_app.resolve_principal_plan",
            return_value=("org-x", "agent-y", "free", None),
        ),
        patch("app.mcp_app.SessionLocal", MagicMock(return_value=MagicMock())),
        patch("app.mcp_app.check_memory_quota", quota),
        patch("app.mcp_app.record_usage", usage),
        patch("app.mcp_app.log_recall_event", log),
        patch("app.mcp_app.retrieve_memories_scoped", retrieve),
        patch.object(mcp_app.brain, "find_space_id", find_space),
    ):
        result = await fn(_fake_ctx({"mcp-session-id": "conn-1"}), "q", space="s")

    assert "error" not in result, f"positive control: recall failed: {result}"
    for name, rec in [
        ("check_memory_quota", quota),
        ("brain.find_space_id", find_space),
        ("log_recall_event", log),
        ("record_usage", usage),
    ]:
        assert rec.threads, f"positive control: {name} was never called"
        assert rec.off_loop(loop_ident), (
            f"{name} ran on the event loop thread ({rec.threads})"
        )
