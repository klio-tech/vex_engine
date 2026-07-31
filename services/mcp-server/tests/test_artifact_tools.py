"""Artifact tool bodies, exercised through the FastMCP registry.

Same seams as test_write_tools.py: resolve_principal_plan, SessionLocal, and
brain.write_scoped_memory are patched at the app.mcp_app boundary, so no DB,
no embeddings and no S3 are touched.
"""

import threading
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app import mcp_app
from app.mcp_app import _is_text_mime, build_mcp_server

MB = 1024 * 1024


def _ctx():
    request = SimpleNamespace(headers={"x-vex-agent": "agent-1"})
    return SimpleNamespace(request_context=SimpleNamespace(request=request))


def _tool(name):
    return build_mcp_server()._tool_manager.get_tool(name).fn


@pytest.fixture
def seams():
    """Patch every external dependency of the artifact tools."""
    plan = SimpleNamespace(
        artifact_max_file_bytes=10 * MB,
        artifact_storage_bytes=100 * MB,
    )
    with (
        patch(
            "app.mcp_app.resolve_principal_plan",
            return_value=("acme", "agent-1", "free", None),
        ),
        # ``forget`` resolves the principal via the plain ``resolve_principal``
        # (shared with the older single-memory path), not ``resolve_principal_plan``
        # used by the rest of the artifact tools — patch both so ``_ctx()``
        # (which carries no ``X-Vex-Key``) authenticates for every tool.
        patch("app.mcp_app.resolve_principal", return_value=("acme", "agent-1")),
        patch("app.mcp_app.get_plan_config", return_value=plan),
        patch("app.mcp_app.SessionLocal", MagicMock()) as session_local,
        patch("app.mcp_app.artifacts") as af,
        patch("app.mcp_app.brain") as brain,
        patch("app.mcp_app.resolve_project_from_request", return_value=None),
    ):
        af.INLINE_MAX_BYTES = 65_536
        af.human_size.return_value = "3 B"
        af.card_content.return_value = "[artifact] t — s (document, 3 B)"
        af.org_storage_bytes.return_value = 0
        # I3: the request path now consults both of these before handing out a
        # presigned PUT.
        af.MAX_PENDING_PER_ORG = 32
        af.pending_count.return_value = 0
        af.find_active_by_checksum.return_value = None
        af.create_pending.return_value = "new-artifact-id"
        af.activate.return_value = {
            "id": "new-artifact-id",
            "size_bytes": 3,
            "checksum": "abc",
        }
        brain.write_scoped_memory = AsyncMock(return_value={"stored": True})
        brain.ensure_space.return_value = None
        # ``SessionLocal()`` (called once per tool invocation) always yields
        # this same fake session, so tests can inspect raw ``db.execute`` calls
        # (e.g. the supersede/hide-card path) without a real DB.
        yield SimpleNamespace(artifacts=af, brain=brain, db=session_local.return_value)


@pytest.mark.asyncio
async def test_inline_put_activates_and_writes_a_card(seams):
    result = await _tool("artifact_put")(
        _ctx(), title="t", summary="s", kind="document", content="hey"
    )

    assert result["stored"] is True
    assert result["artifact_id"] == "new-artifact-id"
    assert result["status"] == "active"

    # The card is the discovery mechanism; without it the artifact is invisible.
    seams.brain.write_scoped_memory.assert_awaited_once()
    kwargs = seams.brain.write_scoped_memory.await_args.kwargs
    assert kwargs["memory_type"] == "artifact"
    assert kwargs["metadata"]["artifact_id"] == "new-artifact-id"
    assert kwargs["content"].startswith("[artifact] ")


@pytest.mark.asyncio
async def test_inline_put_stores_text_inline_not_in_object_storage(seams):
    await _tool("artifact_put")(
        _ctx(), title="t", summary="s", kind="document", content="hey"
    )

    kwargs = seams.artifacts.activate.call_args.kwargs
    assert kwargs["inline_text"] == "hey"
    assert kwargs["storage_uri"] is None
    # Measured server-side, never taken from the caller.
    assert kwargs["size_bytes"] == 3


@pytest.mark.asyncio
async def test_content_over_the_inline_cap_is_rejected_with_a_pointer(seams):
    result = await _tool("artifact_put")(
        _ctx(), title="t", summary="s", kind="document", content="x" * 65_537
    )

    assert result["stored"] is False
    assert result["error"] == "too_large_for_inline"
    seams.brain.write_scoped_memory.assert_not_awaited()


@pytest.mark.asyncio
async def test_over_pool_write_is_rejected_before_any_work(seams):
    seams.artifacts.org_storage_bytes.return_value = 100 * MB

    result = await _tool("artifact_put")(
        _ctx(), title="t", summary="s", kind="document", content="hey"
    )

    assert result["stored"] is False
    assert result["error"] == "storage_quota_exceeded"
    seams.artifacts.create_pending.assert_not_called()
    seams.brain.write_scoped_memory.assert_not_awaited()


@pytest.mark.asyncio
async def test_duplicate_bytes_return_the_existing_artifact(seams):
    seams.artifacts.find_active_by_checksum.return_value = {"id": "old", "version": 1}

    result = await _tool("artifact_put")(
        _ctx(), title="t", summary="s", kind="document", content="hey"
    )

    assert result["artifact_id"] == "old"
    assert result["deduplicated"] is True
    seams.artifacts.create_pending.assert_not_called()


@pytest.mark.asyncio
async def test_auth_failure_returns_a_clean_error_not_an_exception():
    from fastapi import HTTPException

    with patch(
        "app.mcp_app.resolve_principal_plan",
        side_effect=HTTPException(status_code=401, detail="bad key"),
    ):
        result = await _tool("artifact_put")(
            _ctx(), title="t", summary="s", kind="document", content="hey"
        )

    assert result["stored"] is False
    assert result["error"] == "unauthorized"


@pytest.mark.asyncio
async def test_supersedes_marks_old_artifact_superseded_and_hides_its_card(seams):
    result = await _tool("artifact_put")(
        _ctx(),
        title="t",
        summary="s",
        kind="document",
        content="hey",
        supersedes="old-id",
    )

    # The new artifact is still created and carded normally in this call.
    assert result["stored"] is True
    assert result["artifact_id"] == "new-artifact-id"
    seams.brain.write_scoped_memory.assert_awaited_once()

    # The old artifact is superseded using the resolved org_id — never a
    # caller-supplied one.
    seams.artifacts.mark_superseded.assert_called_once()
    supersede_kwargs = seams.artifacts.mark_superseded.call_args.kwargs
    assert supersede_kwargs["org_id"] == "acme"
    assert supersede_kwargs["artifact_id"] == "old-id"

    # Its card is hidden from recall (raw UPDATE against session_memories,
    # since there is no shared.artifacts helper for hiding a card).
    hide_call = next(
        call
        for call in seams.db.execute.call_args_list
        if "recall_hidden" in str(call.args[0])
    )
    params = hide_call.args[1]
    assert params["org_id"] == "acme"
    assert params["artifact_id"] == "old-id"


@pytest.mark.asyncio
async def test_card_write_failure_retracts_the_artifact_and_returns_an_error(seams):
    """A card-write failure must not leave a paid-for, invisible artifact.

    ``artifacts.create_pending`` / ``activate`` each commit independently and
    only then does the card get written, so a failure here (e.g. an embedding
    service hiccup) would otherwise leave the artifact ``status='active'`` with
    no discovery card: billed, listed by any future artifacts UI, and
    permanently invisible to recall. The fix compensates by retracting the
    just-activated artifact and surfacing an error rather than a false
    success.
    """
    seams.brain.write_scoped_memory.side_effect = RuntimeError("embedding service down")

    result = await _tool("artifact_put")(
        _ctx(), title="t", summary="s", kind="document", content="hey"
    )

    assert result["stored"] is False
    assert result["error"] == "card_write_failed"

    seams.artifacts.retract.assert_called_once()
    retract_kwargs = seams.artifacts.retract.call_args.kwargs
    assert retract_kwargs["org_id"] == "acme"
    assert retract_kwargs["artifact_id"] == "new-artifact-id"


@pytest.fixture
def storage(seams):
    """Add a patched artifact_storage to the existing seams."""
    with patch("app.mcp_app.artifact_storage") as st:
        st.MULTIPART_THRESHOLD_BYTES = 100 * MB
        st.PRESIGN_TTL_SECONDS = 900
        st.object_key.return_value = "acme/new-artifact-id/v1"
        st.storage_uri.return_value = "s3://klio-artifacts/acme/new-artifact-id/v1"
        st.presign_put.return_value = "https://example/upload"
        st.head.return_value = {"size_bytes": 2048, "checksum": "abc"}
        # C1: size and checksum are now measured in two separate steps so the
        # limit can be enforced before the bytes are streamed.
        st.head_size.return_value = 2048
        st.checksum.return_value = "abc"
        st.get_artifact_s3_client.return_value = MagicMock()
        yield SimpleNamespace(
            storage=st,
            artifacts=seams.artifacts,
            brain=seams.brain,
            db=seams.db,
        )


def _pending_row(**overrides):
    row = {
        "id": "new-artifact-id",
        "status": "pending",
        "title": "t",
        "summary": "s",
        "kind": "file",
        "version": 1,
        "space_id": "space-1",
        "project_id": "project-1",
    }
    row.update(overrides)
    return row


@pytest.mark.asyncio
async def test_request_returns_an_upload_url_and_writes_no_card(storage):
    result = await _tool("artifact_put")(
        _ctx(),
        title="t",
        summary="s",
        kind="file",
        mime_type="application/pdf",
        size_bytes=2048,
    )

    assert result["artifact_id"] == "new-artifact-id"
    assert result["upload_url"] == "https://example/upload"
    assert result["expires_in"] == 900
    # No card until the bytes actually exist — an abandoned upload must never
    # appear in recall.
    storage.brain.write_scoped_memory.assert_not_awaited()


@pytest.mark.asyncio
async def test_the_upload_url_is_signed_with_the_declared_size(storage):
    """``MAX_PENDING_PER_ORG``'s bound is only real if each URL is capped.

    ``presign_put`` signed ``Bucket`` and ``Key`` only, so every outstanding
    reservation accepted an object of unbounded size: the cap bounded the
    number of concurrent objects and not one byte. Binding the declared size
    into the signature is what turns 32 reservations into a real ceiling of
    32 x the plan's per-file limit.
    """
    await _tool("artifact_put")(
        _ctx(),
        title="t",
        summary="s",
        kind="file",
        mime_type="application/pdf",
        size_bytes=2048,
    )

    assert storage.storage.presign_put.call_args.kwargs["content_length"] == 2048


@pytest.mark.asyncio
async def test_a_request_without_a_declared_size_is_refused(storage):
    """No declared size means no cap to sign, i.e. an unbounded URL.

    The old ``declared = size_bytes or 0`` also let this bypass the pool check
    entirely: zero always fits.
    """
    result = await _tool("artifact_put")(
        _ctx(), title="t", summary="s", kind="file", mime_type="application/pdf"
    )

    assert result["stored"] is False
    assert result["error"] == "size_bytes_required"
    storage.storage.presign_put.assert_not_called()
    storage.artifacts.create_pending.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("bad", [0, -1])
async def test_a_request_with_a_non_positive_declared_size_is_refused(storage, bad):
    result = await _tool("artifact_put")(
        _ctx(),
        title="t",
        summary="s",
        kind="file",
        mime_type="application/pdf",
        size_bytes=bad,
    )

    assert result["error"] == "size_bytes_required"
    storage.storage.presign_put.assert_not_called()


# --- an unknown project must not leave an orphan space behind -------------


@pytest.mark.asyncio
async def test_unknown_project_on_the_inline_path_creates_no_space(seams):
    """``ensure_space`` CREATES a space row. Running it before the project is
    resolved means a rejected write still mints tenant rows: retry a typo'd
    project name in a loop and the org's space list fills with junk it never
    asked for and cannot see the origin of.
    """
    seams.brain.find_project_id.return_value = None

    result = await _tool("artifact_put")(
        _ctx(),
        title="t",
        summary="s",
        kind="document",
        content="hey",
        space="new-space",
        project="does-not-exist",
    )

    assert result["error"] == "unknown_project"
    seams.brain.ensure_space.assert_not_called()


@pytest.mark.asyncio
async def test_unknown_project_on_the_request_path_creates_no_space(storage):
    storage.brain.find_project_id.return_value = None

    result = await _tool("artifact_put")(
        _ctx(),
        title="t",
        summary="s",
        kind="file",
        mime_type="application/pdf",
        size_bytes=2048,
        space="new-space",
        project="does-not-exist",
    )

    assert result["error"] == "unknown_project"
    storage.brain.ensure_space.assert_not_called()
    storage.artifacts.create_pending.assert_not_called()


@pytest.mark.asyncio
async def test_confirm_measures_the_object_and_writes_the_card(storage):
    storage.artifacts.get_artifact.return_value = {
        "id": "new-artifact-id",
        "status": "pending",
        "title": "t",
        "summary": "s",
        "kind": "file",
        "version": 1,
        "space_id": "space-1",
        "project_id": "project-1",
    }

    result = await _tool("artifact_put")(
        _ctx(), title="t", summary="s", kind="file", artifact_id="new-artifact-id"
    )

    assert result["stored"] is True
    assert result["status"] == "active"
    # The measured size wins, not anything the client declared.
    assert storage.artifacts.activate.call_args.kwargs["size_bytes"] == 2048
    storage.brain.write_scoped_memory.assert_awaited_once()
    # The card must be filed into the same space/project the pending row was
    # created with (Task 5) — not left unscoped. Using non-None values here
    # means a regression back to `row.get(...)` returning None would fail
    # this assertion.
    card_kwargs = storage.brain.write_scoped_memory.await_args.kwargs
    assert card_kwargs["space_id"] == "space-1"
    assert card_kwargs["project_id"] == "project-1"


@pytest.mark.asyncio
async def test_confirm_rejects_bytes_that_exceed_the_per_file_limit(storage):
    storage.storage.head_size.return_value = 50 * MB
    storage.artifacts.get_artifact.return_value = {
        "id": "new-artifact-id",
        "status": "pending",
        "title": "t",
        "summary": "s",
        "kind": "file",
        "version": 1,
        "space_id": "space-1",
        "project_id": "project-1",
    }

    # Declared 1KB, actually uploaded 50MB against a 10MB plan limit.
    result = await _tool("artifact_put")(
        _ctx(), title="t", summary="s", kind="file", artifact_id="new-artifact-id"
    )

    assert result["stored"] is False
    assert result["error"] == "file_too_large"
    storage.artifacts.activate.assert_not_called()
    storage.brain.write_scoped_memory.assert_not_awaited()


@pytest.mark.asyncio
async def test_confirm_without_uploaded_bytes_is_rejected(storage):
    storage.storage.head_size.return_value = None
    storage.artifacts.get_artifact.return_value = {
        "id": "new-artifact-id",
        "status": "pending",
        "title": "t",
        "summary": "s",
        "kind": "file",
        "version": 1,
        "space_id": "space-1",
        "project_id": "project-1",
    }

    result = await _tool("artifact_put")(
        _ctx(), title="t", summary="s", kind="file", artifact_id="new-artifact-id"
    )

    assert result["stored"] is False
    assert result["error"] == "no_bytes_uploaded"


@pytest.mark.asyncio
async def test_confirming_another_orgs_artifact_reads_as_not_found(storage):
    storage.artifacts.get_artifact.return_value = None

    result = await _tool("artifact_put")(
        _ctx(), title="t", summary="s", kind="file", artifact_id="someone-elses"
    )

    assert result["stored"] is False
    assert result["error"] == "not_found"


@pytest.mark.asyncio
async def test_confirm_card_write_failure_retracts_the_artifact(storage):
    """Same compensating-retract pattern as the inline path (Task 6 review):
    a card-write failure on confirm must not leave a paid-for, invisible
    artifact billed against the org's storage pool.
    """
    storage.artifacts.get_artifact.return_value = {
        "id": "new-artifact-id",
        "status": "pending",
        "title": "t",
        "summary": "s",
        "kind": "file",
        "version": 1,
        "space_id": "space-1",
        "project_id": "project-1",
    }
    storage.brain.write_scoped_memory.side_effect = RuntimeError(
        "embedding service down"
    )

    result = await _tool("artifact_put")(
        _ctx(), title="t", summary="s", kind="file", artifact_id="new-artifact-id"
    )

    assert result["stored"] is False
    assert result["error"] == "card_write_failed"

    storage.artifacts.retract.assert_called_once()
    retract_kwargs = storage.artifacts.retract.call_args.kwargs
    assert retract_kwargs["org_id"] == "acme"
    assert retract_kwargs["artifact_id"] == "new-artifact-id"


@pytest.mark.asyncio
async def test_get_returns_inline_text(storage):
    storage.artifacts.TEXT_PAGE_CHARS = 51_200
    storage.artifacts.get_artifact.return_value = {
        "id": "a1",
        "status": "active",
        "title": "t",
        "kind": "document",
        "mime_type": "text/markdown",
        "inline_text": "hello",
        "storage_uri": None,
        "version": 1,
        "size_bytes": 5,
    }

    result = await _tool("artifact_get")(_ctx(), artifact_id="a1")

    assert result["text"] == "hello"
    assert result["has_more"] is False


@pytest.mark.asyncio
async def test_get_pages_long_text_and_reports_more(storage):
    storage.artifacts.TEXT_PAGE_CHARS = 10
    storage.artifacts.get_artifact.return_value = {
        "id": "a1",
        "status": "active",
        "title": "t",
        "kind": "document",
        "mime_type": "text/plain",
        "inline_text": "abcdefghijKLMNO",
        "storage_uri": None,
        "version": 1,
        "size_bytes": 15,
    }

    first = await _tool("artifact_get")(_ctx(), artifact_id="a1")
    assert first["text"] == "abcdefghij"
    assert first["has_more"] is True

    second = await _tool("artifact_get")(_ctx(), artifact_id="a1", offset=10)
    assert second["text"] == "KLMNO"
    assert second["has_more"] is False


@pytest.mark.asyncio
async def test_get_text_on_a_binary_points_at_url_mode(storage):
    storage.artifacts.get_artifact.return_value = {
        "id": "a1",
        "status": "active",
        "title": "t",
        "kind": "file",
        "mime_type": "application/pdf",
        "inline_text": None,
        "storage_uri": "s3://klio-artifacts/acme/a1/v1",
        "version": 1,
        "size_bytes": 2048,
    }

    result = await _tool("artifact_get")(_ctx(), artifact_id="a1", mode="text")

    # v1 does no extraction; saying so beats returning mojibake.
    assert result["error"] == "not_text"
    assert "url" in result["detail"]


@pytest.mark.asyncio
async def test_get_url_returns_a_short_lived_signed_url(storage):
    storage.storage.presign_get.return_value = "https://example/download"
    storage.artifacts.get_artifact.return_value = {
        "id": "a1",
        "status": "active",
        "title": "t",
        "kind": "file",
        "mime_type": "application/pdf",
        "inline_text": None,
        "storage_uri": "s3://klio-artifacts/acme/a1/v1",
        "version": 1,
        "size_bytes": 2048,
    }

    result = await _tool("artifact_get")(_ctx(), artifact_id="a1", mode="url")

    assert result["url"] == "https://example/download"
    assert result["expires_in"] == 900


@pytest.mark.asyncio
async def test_get_on_another_orgs_artifact_reads_as_not_found(storage):
    storage.artifacts.get_artifact.return_value = None
    result = await _tool("artifact_get")(_ctx(), artifact_id="someone-elses")
    assert result["error"] == "not_found"


@pytest.mark.asyncio
async def test_get_on_a_retracted_artifact_reads_as_not_found(storage):
    storage.artifacts.get_artifact.return_value = {
        "id": "a1",
        "status": "retracted",
        "title": "t",
        "kind": "document",
        "mime_type": "text/plain",
        "inline_text": "gone",
        "storage_uri": None,
        "version": 1,
        "size_bytes": 4,
    }
    result = await _tool("artifact_get")(_ctx(), artifact_id="a1")
    assert result["error"] == "not_found"


# --- _is_text_mime: parameterized / cased MIME types (code-review finding) ---


def test_is_text_mime_accepts_json_with_charset_parameter():
    # "application/json" alone hits the exact-match set, but a real-world
    # Content-Type nearly always carries a "; charset=..." parameter — the
    # non-text/* exact-match types are exactly where structured `record`
    # artifacts live, so this must not be misclassified as binary.
    assert _is_text_mime("application/json; charset=utf-8") is True


def test_is_text_mime_is_case_insensitive():
    assert _is_text_mime("TEXT/PLAIN") is True


def test_is_text_mime_still_rejects_binary_types():
    assert _is_text_mime("application/pdf") is False


# --- mode='text' reading a non-inline (S3-backed) text artifact ---


def _s3_text_row(**overrides):
    row = {
        "id": "a1",
        "status": "active",
        "title": "t",
        "kind": "record",
        "mime_type": "application/json",
        "inline_text": None,
        "storage_uri": "s3://klio-artifacts/acme/a1/v1",
        "version": 1,
        "size_bytes": 13,
    }
    row.update(overrides)
    return row


@pytest.mark.asyncio
async def test_get_text_reads_a_text_artifact_stored_in_s3(storage):
    storage.artifacts.TEXT_PAGE_CHARS = 51_200
    storage.artifacts.get_artifact.return_value = _s3_text_row()
    storage.storage.get_text_page.return_value = ('{"ok": true}', False)

    result = await _tool("artifact_get")(_ctx(), artifact_id="a1")

    storage.storage.object_key.assert_called_with("acme", "a1", 1)
    call = storage.storage.get_text_page.call_args
    assert call.args == (
        storage.storage.get_artifact_s3_client.return_value,
        storage.storage.object_key.return_value,
    )
    assert result["text"] == '{"ok": true}'
    assert result["has_more"] is False


@pytest.mark.asyncio
async def test_get_text_asks_s3_for_only_one_page(storage):
    """The Critical finding: reading a page must not read the object.

    ``get_text(s3, key)`` was ``get_object(...)["Body"].read().decode("utf-8")``
    — unbounded, synchronous, and with no size gate anywhere on this path. On a
    plan allowing 1GB files, serving 51,200 characters of a 900MB
    ``text/plain`` artifact buffered all 900MB plus its decode into RAM. The
    page window must be pushed down into the storage call so the transfer is
    bounded by what is returned.
    """
    storage.artifacts.TEXT_PAGE_CHARS = 51_200
    storage.artifacts.get_artifact.return_value = _s3_text_row(size_bytes=900 * MB)
    storage.storage.get_text_page.return_value = ("x" * 51_200, True)

    result = await _tool("artifact_get")(_ctx(), artifact_id="a1", offset=51_200)

    kwargs = storage.storage.get_text_page.call_args.kwargs
    assert kwargs["offset"] == 51_200
    assert kwargs["limit"] == 51_200
    assert result["offset"] == 51_200
    assert result["has_more"] is True


@pytest.mark.asyncio
async def test_get_text_past_the_end_answers_without_touching_s3(storage):
    """``offset`` is the one client input that scales the ranged read.

    A character is at least one byte, so an offset at or beyond the object's
    server-measured size is certainly past its last character — answerable from
    the row, with no request to S3 at all.
    """
    storage.artifacts.TEXT_PAGE_CHARS = 51_200
    storage.artifacts.get_artifact.return_value = _s3_text_row(size_bytes=13)

    result = await _tool("artifact_get")(_ctx(), artifact_id="a1", offset=999_999_999)

    assert result["text"] == ""
    assert result["has_more"] is False
    storage.storage.get_text_page.assert_not_called()


@pytest.mark.asyncio
async def test_get_text_from_s3_runs_off_the_event_loop(storage):
    """Blocking boto3 I/O on the loop stalls every other MCP request.

    Even bounded, a page read is a network round trip. The confirm path already
    runs its S3 calls through ``anyio.to_thread.run_sync``; the read path was
    left calling boto3 inline. Asserting on the *thread the call actually runs
    on* is what makes this falsifiable — a mock-call assertion would pass
    either way.
    """
    storage.artifacts.TEXT_PAGE_CHARS = 51_200
    storage.artifacts.get_artifact.return_value = _s3_text_row()
    threads = []

    def _record(*_a, **_k):
        threads.append(threading.current_thread())
        return ("body", False)

    storage.storage.get_text_page.side_effect = _record

    await _tool("artifact_get")(_ctx(), artifact_id="a1")

    assert threads and threading.main_thread() not in threads


@pytest.mark.asyncio
async def test_get_url_runs_its_s3_calls_off_the_event_loop(storage):
    """``put_text`` (inline materialisation) and ``presign_get`` are bounded or
    local, but they are still blocking network calls and belong off the loop
    for the same reason as the read above."""
    storage.artifacts.get_artifact.return_value = {
        "id": "a1",
        "status": "active",
        "title": "t",
        "kind": "document",
        "mime_type": "text/markdown",
        "inline_text": "hello",
        "storage_uri": None,
        "version": 1,
        "size_bytes": 5,
    }
    threads = []

    def _record(result):
        def _inner(*_a, **_k):
            threads.append(threading.current_thread())
            return result

        return _inner

    storage.storage.put_text.side_effect = _record({"size_bytes": 5, "checksum": "c"})
    storage.storage.presign_get.side_effect = _record("https://example/download")

    result = await _tool("artifact_get")(_ctx(), artifact_id="a1", mode="url")

    assert result["url"] == "https://example/download"
    assert len(threads) == 2
    assert threading.main_thread() not in threads


@pytest.mark.asyncio
async def test_get_text_on_s3_read_failure_returns_internal_error_not_a_crash(storage):
    storage.artifacts.get_artifact.return_value = {
        "id": "a1",
        "status": "active",
        "title": "t",
        "kind": "document",
        "mime_type": "text/plain",
        "inline_text": None,
        "storage_uri": "s3://klio-artifacts/acme/a1/v1",
        "version": 1,
        "size_bytes": 100,
    }
    storage.storage.get_text_page.side_effect = RuntimeError("s3 unavailable")

    result = await _tool("artifact_get")(_ctx(), artifact_id="a1")

    assert result == {"error": "internal_error"}


# --- forget: retract an artifact (bytes + card) by id ---


@pytest.mark.asyncio
async def test_forget_retracts_the_artifact_and_hides_its_card(storage):
    storage.artifacts.retract.return_value = True

    result = await _tool("forget")(_ctx(), memory_id="a1")

    assert result["forgotten"] is True
    storage.artifacts.retract.assert_called_once()
    assert storage.artifacts.retract.call_args.kwargs["org_id"] == "acme"


@pytest.mark.asyncio
async def test_forget_with_a_non_uuid_id_rolls_back_and_tries_memory(storage):
    # Memory ids are not always UUIDs. The ::uuid cast raises, and without a
    # rollback Postgres leaves the transaction aborted — breaking the memory
    # path that should have handled this id.
    storage.artifacts.retract.side_effect = Exception("invalid input for uuid")
    db = mcp_app.SessionLocal.return_value

    result = await _tool("forget")(_ctx(), memory_id="not-a-uuid")

    db.rollback.assert_called_once()
    assert result.get("kind") != "artifact"


# ---------------------------------------------------------------------------
# C1: the confirm path must size-check before it moves any bytes.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_confirm_rejects_an_oversized_object_without_hashing_it(storage):
    """An over-limit upload must be rejected on the strength of HEAD alone.

    Free plan, 10MB limit, 5GB uploaded through the presigned PUT. If the
    server computes a checksum before checking the size, it streams all 5GB on
    the *rejection* path — an authenticated remote DoS from the cheapest tier
    that quota cannot stop, because the request is being refused anyway.
    """
    storage.storage.head_size.return_value = 5 * 1024**3
    storage.artifacts.get_artifact.return_value = _pending_row()

    result = await _tool("artifact_put")(
        _ctx(), title="t", summary="s", kind="file", artifact_id="new-artifact-id"
    )

    assert result["stored"] is False
    assert result["error"] == "file_too_large"
    storage.storage.checksum.assert_not_called()
    # And the whole-object read must not be reachable from this path at all.
    storage.storage.head.assert_not_called()
    storage.artifacts.activate.assert_not_called()


@pytest.mark.asyncio
async def test_confirm_rejects_an_over_pool_object_without_hashing_it(storage):
    """The pool check is the other pre-transfer gate."""
    storage.storage.head_size.return_value = 5 * MB
    storage.artifacts.org_storage_bytes.return_value = 98 * MB
    storage.artifacts.get_artifact.return_value = _pending_row()

    result = await _tool("artifact_put")(
        _ctx(), title="t", summary="s", kind="file", artifact_id="new-artifact-id"
    )

    assert result["stored"] is False
    assert result["error"] == "storage_quota_exceeded"
    storage.storage.checksum.assert_not_called()


@pytest.mark.asyncio
async def test_confirm_absent_object_is_detected_by_size_alone(storage):
    storage.storage.head_size.return_value = None
    storage.artifacts.get_artifact.return_value = _pending_row()

    result = await _tool("artifact_put")(
        _ctx(), title="t", summary="s", kind="file", artifact_id="new-artifact-id"
    )

    assert result["error"] == "no_bytes_uploaded"
    storage.storage.checksum.assert_not_called()


@pytest.mark.asyncio
async def test_confirm_runs_blocking_s3_work_off_the_event_loop(storage):
    """boto3 is blocking and ``_perform_artifact_put`` is a coroutine.

    Hashing a multi-GB object inline would stall the event loop for the whole
    transfer, starving every other MCP request on the replica. The S3 calls
    must go through the codebase's ``anyio.to_thread.run_sync`` seam.
    """
    storage.artifacts.get_artifact.return_value = _pending_row()

    with patch(
        "app.mcp_app.anyio.to_thread.run_sync", wraps=mcp_app.anyio.to_thread.run_sync
    ) as run_sync:
        result = await _tool("artifact_put")(
            _ctx(), title="t", summary="s", kind="file", artifact_id="new-artifact-id"
        )

    assert result["stored"] is True, result
    offloaded = {call.args[0] for call in run_sync.call_args_list}
    assert storage.storage.head_size in offloaded
    assert storage.storage.checksum in offloaded


# ---------------------------------------------------------------------------
# I1: duplicate bytes on the presigned path must not wedge the upload.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_confirm_of_duplicate_bytes_returns_the_existing_artifact(storage):
    """Without dedup here, ``activate``'s UPDATE violates
    ``ux_artifacts_org_checksum_active`` and the generic handler reports
    ``internal_error``. The row stays pending forever and the agent cannot
    distinguish that from a server bug.
    """
    storage.artifacts.get_artifact.return_value = _pending_row()
    storage.artifacts.find_active_by_checksum.return_value = {
        "id": "already-here",
        "version": 2,
    }

    result = await _tool("artifact_put")(
        _ctx(), title="t", summary="s", kind="file", artifact_id="new-artifact-id"
    )

    assert result["stored"] is True
    assert result["artifact_id"] == "already-here"
    assert result["version"] == 2
    assert result["deduplicated"] is True
    storage.artifacts.activate.assert_not_called()
    # The redundant upload's bytes and its pending reservation are cleaned up.
    storage.storage.delete.assert_called_once()
    storage.artifacts.delete_pending.assert_called_once()
    assert (
        storage.artifacts.delete_pending.call_args.kwargs["artifact_id"]
        == "new-artifact-id"
    )


@pytest.mark.asyncio
async def test_confirm_dedup_lookup_uses_the_measured_checksum(storage):
    storage.artifacts.get_artifact.return_value = _pending_row()
    storage.storage.checksum.return_value = "measured-sha"

    await _tool("artifact_put")(
        _ctx(), title="t", summary="s", kind="file", artifact_id="new-artifact-id"
    )

    assert (
        storage.artifacts.find_active_by_checksum.call_args.kwargs["checksum"]
        == "measured-sha"
    )


@pytest.mark.asyncio
async def test_confirm_integrity_error_falls_back_to_dedup(storage):
    """The concurrent-race case: two confirms of identical bytes at once.

    One wins the partial unique index; the loser's UPDATE raises
    IntegrityError. That must resolve to the winning artifact, not
    ``internal_error``.
    """
    from sqlalchemy.exc import IntegrityError

    storage.artifacts.get_artifact.return_value = _pending_row()
    storage.artifacts.activate.side_effect = IntegrityError(
        "UPDATE artifacts", {}, Exception("ux_artifacts_org_checksum_active")
    )
    # Nothing on the first (pre-activate) lookup; the winner is visible on the
    # retry after the constraint fires.
    storage.artifacts.find_active_by_checksum.side_effect = [
        None,
        {"id": "winner", "version": 1},
    ]

    result = await _tool("artifact_put")(
        _ctx(), title="t", summary="s", kind="file", artifact_id="new-artifact-id"
    )

    assert result["stored"] is True
    assert result["artifact_id"] == "winner"
    assert result["deduplicated"] is True


@pytest.mark.asyncio
async def test_confirm_integrity_error_with_no_winner_is_not_a_false_success(storage):
    """An IntegrityError that is NOT a dedup collision must still be an error."""
    from sqlalchemy.exc import IntegrityError

    storage.artifacts.get_artifact.return_value = _pending_row()
    storage.artifacts.activate.side_effect = IntegrityError(
        "UPDATE artifacts", {}, Exception("some other constraint")
    )
    storage.artifacts.find_active_by_checksum.side_effect = [None, None]

    result = await _tool("artifact_put")(
        _ctx(), title="t", summary="s", kind="file", artifact_id="new-artifact-id"
    )

    assert result["stored"] is False
    assert result["error"] == "internal_error"


# ---------------------------------------------------------------------------
# I2: the advertised `project` argument must actually file the artifact.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_inline_put_honours_an_explicit_project_argument(seams):
    """``project`` is in the tool schema and was silently discarded.

    An agent that passes project="apollo" with no X-Vex-Project header got
    project_id=None, so a later recall(project="apollo") never returned the
    card — with no error anywhere.
    """
    seams.brain.find_project_id.return_value = "project-apollo"

    result = await _tool("artifact_put")(
        _ctx(),
        title="t",
        summary="s",
        kind="document",
        content="hey",
        project="apollo",
    )

    assert result["stored"] is True
    assert seams.artifacts.create_pending.call_args.kwargs["project_id"] == (
        "project-apollo"
    )
    assert seams.brain.write_scoped_memory.await_args.kwargs["project_id"] == (
        "project-apollo"
    )
    assert seams.brain.find_project_id.call_args.kwargs["org_id"] == "acme"
    assert seams.brain.find_project_id.call_args.kwargs["identifier"] == "apollo"


@pytest.mark.asyncio
async def test_explicit_project_wins_over_the_header(seams):
    """Same precedence rule ``recall`` already uses: explicit arg beats header."""
    seams.brain.find_project_id.return_value = "project-apollo"

    with patch(
        "app.mcp_app.resolve_project_from_request", return_value="project-from-header"
    ) as header_resolve:
        await _tool("artifact_put")(
            _ctx(),
            title="t",
            summary="s",
            kind="document",
            content="hey",
            project="apollo",
        )

    assert seams.artifacts.create_pending.call_args.kwargs["project_id"] == (
        "project-apollo"
    )
    header_resolve.assert_not_called()


@pytest.mark.asyncio
async def test_no_project_argument_still_falls_back_to_the_header(seams):
    with patch(
        "app.mcp_app.resolve_project_from_request", return_value="project-from-header"
    ):
        await _tool("artifact_put")(
            _ctx(), title="t", summary="s", kind="document", content="hey"
        )

    assert seams.artifacts.create_pending.call_args.kwargs["project_id"] == (
        "project-from-header"
    )


@pytest.mark.asyncio
async def test_an_unknown_project_argument_is_surfaced_not_swallowed(seams):
    """``recall`` warns rather than silently widening; a write must not file the
    artifact somewhere the caller did not ask for either.
    """
    seams.brain.find_project_id.return_value = None

    result = await _tool("artifact_put")(
        _ctx(),
        title="t",
        summary="s",
        kind="document",
        content="hey",
        project="nope",
    )

    assert result["stored"] is False
    assert result["error"] == "unknown_project"
    seams.artifacts.create_pending.assert_not_called()


@pytest.mark.asyncio
async def test_request_path_honours_an_explicit_project_argument(storage):
    storage.brain.find_project_id.return_value = "project-apollo"

    await _tool("artifact_put")(
        _ctx(),
        title="t",
        summary="s",
        kind="file",
        mime_type="application/pdf",
        size_bytes=2048,
        project="apollo",
    )

    assert storage.artifacts.create_pending.call_args.kwargs["project_id"] == (
        "project-apollo"
    )


# ---------------------------------------------------------------------------
# I3: the presigned-request path must not bypass the pool.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_request_path_checks_the_declared_size_against_the_pool(storage):
    """A Free org with 95MB of a 100MB pool used cannot reserve another 9MB.

    Before this check the request branch looked only at the per-file limit, so
    an org could loop presigned PUTs and upload unbounded unbilled bytes.
    """
    storage.artifacts.org_storage_bytes.return_value = 95 * MB

    result = await _tool("artifact_put")(
        _ctx(),
        title="t",
        summary="s",
        kind="file",
        mime_type="application/pdf",
        size_bytes=9 * MB,
    )

    assert result["stored"] is False
    assert result["error"] == "storage_quota_exceeded"
    storage.artifacts.create_pending.assert_not_called()
    storage.storage.presign_put.assert_not_called()


@pytest.mark.asyncio
async def test_request_path_caps_concurrent_pending_reservations(storage):
    """``size_bytes=1`` a thousand times is a thousand uncounted upload URLs."""
    from shared import artifacts as real_artifacts

    storage.artifacts.MAX_PENDING_PER_ORG = real_artifacts.MAX_PENDING_PER_ORG
    storage.artifacts.pending_count.return_value = real_artifacts.MAX_PENDING_PER_ORG

    result = await _tool("artifact_put")(
        _ctx(),
        title="t",
        summary="s",
        kind="file",
        mime_type="application/pdf",
        size_bytes=1,
    )

    assert result["stored"] is False
    assert result["error"] == "too_many_pending_uploads"
    storage.artifacts.create_pending.assert_not_called()
    storage.storage.presign_put.assert_not_called()


@pytest.mark.asyncio
async def test_request_path_still_succeeds_under_both_limits(storage):
    from shared import artifacts as real_artifacts

    storage.artifacts.MAX_PENDING_PER_ORG = real_artifacts.MAX_PENDING_PER_ORG
    storage.artifacts.pending_count.return_value = 1
    storage.artifacts.org_storage_bytes.return_value = 1 * MB

    result = await _tool("artifact_put")(
        _ctx(),
        title="t",
        summary="s",
        kind="file",
        mime_type="application/pdf",
        size_bytes=2048,
    )

    assert result["upload_url"] == "https://example/upload"


def test_perform_artifact_put_docstring_does_not_overclaim_the_pool_check():
    """The docstring asserted the pool check runs before ANY write work.

    That was false for the request branch, and a false invariant in a
    docstring is worse than none — the next reader trusts it.
    """
    doc = mcp_app._perform_artifact_put.__doc__ or ""
    assert "pool check runs before any write work" not in doc.lower()


@pytest.mark.asyncio
async def test_recall_description_tells_agents_about_artifact_get():
    tool = build_mcp_server()._tool_manager.get_tool("recall")
    # Without this, an agent receives an artifact card and has no idea a
    # follow-up tool exists.
    assert "artifact_get" in tool.description


@pytest.mark.asyncio
async def test_request_path_refuses_oversized_without_reserving_anything(storage):
    """Multipart is not implemented, so refuse before creating a pending row.

    This branch used to sit AFTER create_pending and told the caller to "use
    the Klio CLI, which performs the multipart upload". No such command
    exists, so the caller received an artifact_id it could never fulfil — and
    the reservation it burned counted against MAX_PENDING_PER_ORG until GC
    collected it 24 hours later.
    """
    storage.artifacts.MAX_PENDING_PER_ORG = 32
    storage.artifacts.pending_count.return_value = 0
    storage.artifacts.org_storage_bytes.return_value = 0
    storage.storage.MULTIPART_THRESHOLD_BYTES = 100 * MB

    # Only reachable on a plan whose per-file limit exceeds the multipart
    # threshold — i.e. Max (1GB per file). Every smaller plan is stopped by
    # file_too_large first, so this dead end only ever reached the customers
    # paying the most.
    max_plan = SimpleNamespace(
        artifact_max_file_bytes=1024 * MB,
        artifact_storage_bytes=1024 * 1024 * MB,
    )

    with patch("app.mcp_app.get_plan_config", return_value=max_plan):
        result = await _tool("artifact_put")(
            _ctx(),
            title="t",
            summary="s",
            kind="file",
            mime_type="application/octet-stream",
            size_bytes=200 * MB,
        )

    assert result["stored"] is False
    assert result["error"] == "too_large_for_single_upload"
    # The old copy pointed at a CLI command that does not exist.
    assert "CLI" not in result["detail"]
    # Nothing may be reserved, signed, or filed for a request we refuse.
    storage.artifacts.create_pending.assert_not_called()
    storage.storage.presign_put.assert_not_called()
    storage.brain.ensure_space.assert_not_called()
