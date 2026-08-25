# Context Network — Track D (Chat v1, backend + capture) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:subagent-driven-development or superpowers:executing-plans, task-by-task.

**Goal:** Channels on workspace and project nodes in the Plane fork, whose conversation becomes the node's memory through the engine's existing extraction pipeline. Backend and capture bridge in this plan; the web UI is deliberately its own follow-up (D3) so the data model is settled and tested before pixels.

**Architecture:** A new `chat` domain inside Plane's Django API following its own conventions (`WorkspaceBaseModel`, DRF viewsets, `plane/bgtasks` Celery tasks). Access rides node membership — a workspace channel is readable by workspace members, a project channel by project members; **no per-channel membership in v1** (a channel belongs to a node, the node gates). Capture is a Celery task that POSTs batched messages to the engine's `/capture/event` with the workspace's engine credentials, landing at the channel's node scope. Attach-a-node-to-a-channel is recorded and announced in v1 (the read-grant enforcement lives engine-side and follows with agent participants in D4).

**Scope decisions locked by the spec/conversation:** chat only inside orgs (no personal chat); no DMs; messages write to the channel's own node only; reuse the engine extractor (the bridge sends raw content, the engine distils).

## Global Constraints

- **Plane conventions, exactly.** Models in `apps/api/plane/db/models/chat.py` re-exported from `models/__init__.py`; UUID PKs via `BaseModel`/`WorkspaceBaseModel`; `db_table` names set explicitly; soft-delete inherited; DRF serializers/views/urls in the `app` module mirroring an existing small feature (`intake` is a good template).
- **Node gating, not channel gating.** Workspace channel → `WorkspaceMember(is_active=True)`. Project channel → Plane project membership. A non-member gets 404-shaped absence, never 403.
- **Capture is fire-and-forget** (Celery, retries, never blocks message send) and **idempotent** (a retried batch must not double-store: send a stable `session_id` per channel+day so the engine's dedup window applies).
- **Engine credentials are instance configuration** (env → `ENGINE_BASE_URL`, `ENGINE_CAPTURE_KEY`), absent ⇒ capture silently skips (chat works without the engine; the memory feature lights up when configured). Log at WARNING once, not per message.
- **Tests run in the `klio-api-1` container** (docker cp workflow, pytest, `plane.settings.test`) — the P1 discipline. Migration must roll back.
- **Plane full suite baseline: 521 passed** (excluding the factory_boy-missing collection issue in `bg_tasks/test_cleanup_task.py`). Do not regress it.

---

### Task D1 (Plane): models, API, membership gating

**Files:**
- Create: `apps/api/plane/db/models/chat.py`; modify `apps/api/plane/db/models/__init__.py`
- Create migration `0124_chat.py` (via makemigrations in the container)
- Create: `apps/api/plane/app/serializers/chat.py`, `apps/api/plane/app/views/chat/base.py`, url wiring following how `intake`/`workspace` routes register
- Create: `apps/api/plane/tests/unit/chat/test_channels.py`, `test_messages.py`

**Models:**
```python
class Channel(WorkspaceBaseModel):
    # WorkspaceBaseModel provides workspace FK + project FK (nullable) — a
    # project-less row is a workspace (org-node) channel, a project row is a
    # project-node channel. Exactly the two placements the spec allows; there
    # is deliberately no personal placement.
    name = models.CharField(max_length=80)
    description = models.TextField(blank=True)
    created_by_member = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.SET_NULL,
                                          null=True, related_name="created_channels")
    is_default = models.BooleanField(default=False)  # the node's #general

    class Meta:
        db_table = "channels"
        constraints = [
            models.UniqueConstraint(
                fields=["workspace", "project", "name"],
                condition=models.Q(deleted_at__isnull=True),
                name="uq_channel_node_name",
            )
        ]

class ChannelMessage(WorkspaceBaseModel):
    channel = models.ForeignKey(Channel, on_delete=models.CASCADE, related_name="messages")
    actor = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.SET_NULL, null=True,
                              related_name="channel_messages")
    content = models.TextField()  # v1: plain text; rich content is a UI-era concern
    captured_at = models.DateTimeField(null=True)  # set when the engine batch включ it

    class Meta:
        db_table = "channel_messages"
        ordering = ["created_at"]

class ChannelContextAttachment(WorkspaceBaseModel):
    # "Attach a node to the conversation": v1 records + announces; the engine-
    # side read grant follows with agent participants. Only a member of the
    # attached project may attach it (you cannot grant what you do not have).
    channel = models.ForeignKey(Channel, on_delete=models.CASCADE, related_name="attachments")
    attached_project = models.ForeignKey("db.Project", on_delete=models.CASCADE,
                                         related_name="channel_attachments")
    attached_by = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.SET_NULL, null=True)
    detached_at = models.DateTimeField(null=True)

    class Meta:
        db_table = "channel_context_attachments"
```
(If `WorkspaceBaseModel` does not provide the project FK, follow whatever it actually provides — read it first — and add an explicit nullable `project` FK on `Channel`. The invariant that matters: project-channel rows carry the project, workspace-channel rows do not. `включ` above is a typo to remove — write `included`.)

**API (DRF, following `intake`'s viewset shape):**
- `GET/POST /api/workspaces/<slug>/channels/` — list channels the caller can see (workspace channels + project channels where they are a member — one queryset with the membership `EXISTS`), create (any active workspace member; creating a project channel requires project membership).
- `GET /api/workspaces/<slug>/channels/<pk>/messages/?cursor=` and `POST .../messages/` — paginated (Plane has a cursor paginator — reuse it), member-gated by the channel's node.
- `POST .../channels/<pk>/attachments/` + `DELETE .../attachments/<pk>/` — attach/detach a project; attaching posts a system message into the channel (`actor=None`, content `"@<user> attached <project> to this conversation"`) so the grant is announced in-band, per the spec.
- Non-member of a project channel: the channel does not exist for them (404), and it is absent from their list. Both directions tested.

**Tests (RED first, in-container):** creation + uniqueness per node; workspace member sees workspace channels; project channel invisible to non-members (list AND detail AND messages); message post + ordering + cursor; attach requires membership in the attached project; attach announces; soft-deleted channel vanishes.

Commit: `feat(chat): channels and messages on workspace and project nodes`

### Task D2 (Plane): the capture bridge

**Files:**
- Create: `apps/api/plane/bgtasks/chat_capture_task.py`
- Modify: the message-create view to enqueue (post-commit — `transaction.on_commit`, matching how other bgtasks are enqueued)
- Create: `apps/api/plane/tests/unit/chat/test_capture_bridge.py`

Behaviour:
- Task takes `channel_id`, batches uncaptured messages (`captured_at IS NULL`, cap 50), formats `"<display_name>: <content>"` lines joined, POSTs to `{ENGINE_BASE_URL}/capture/event` with `X-Vex-Key: {ENGINE_CAPTURE_KEY}` and, for a project channel, the engine project addressed via `git_remote`-less identity — send `X-Vex-Project: <plane project id>` **only if** the engine can resolve it; it cannot yet (engine `find_project_id` matches git_remote/repo_root/display_name — `plane_project_id` linking is Track B leftover). So v1: workspace channels capture at org scope with no project header; project channels send `X-Vex-Project: <project display_name>` (the engine resolves display_name today, fail-open to org — acceptable and documented in the task's docstring as the interim until `plane_project_id` resolution lands engine-side).
- `session_id`: `chat:<channel_id>:<YYYY-MM-DD>` — stable per channel-day, so engine session-window dedup absorbs retries.
- Success marks `captured_at`; failure leaves rows for the next batch (retry with backoff, max 3, then leave — the next message re-enqueues).
- Unset config ⇒ skip, one WARNING per process.
- Tests: batch marks captured; retry does not double-send `captured_at IS NULL` filtering proven; unset config skips; the system attach-message is captured like any other (it is context!); payload shape asserted against a recorded request (mock `requests`/`httpx` at the boundary).

Commit: `feat(chat): channel messages feed the node's memory through the engine`

### Task D3 (deferred, separate plan): web UI

Channel list + message pane in the web app, riding the existing app shell. Deliberately not in this plan: the model/API surface above must settle first, and UI carries its own design pass.

### Task D4 (deferred): agent participants + enforced attach-grants engine-side.

## Out of scope
DMs, threads, files, search, notification fan-out, mobile, presence (v1 ships without presence — it needs the realtime lane, which arrives with the UI task), riding the `live` websocket (same), per-channel membership.
