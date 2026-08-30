"""The tools an agent may use, bound to one turn's scope.

Every tool is built per turn and closes over that turn's
:class:`~shared.audience_scope.AudienceScope`, so a tool physically cannot
be called with a wider scope than the room allows. The alternative — a
generic tool that takes a scope argument — puts the security decision in
the model's hands, and the model is the one component here that is
explicitly not trusted.

Reads go straight to the memory layer in this process. Writes — both the
memory write and the workspace writes — go through the SAME scope
boundary: `remember` is pinned to `scope="org"` for exactly the reason
`_recall` is pinned to `scope="org"` below, and workspace actions go
through the workspace's external API with the AGENT's own credential, so
anything an agent changes there is attributable to it and constrained by
its membership. Independent limits: a bug in one does not open the door
on another.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any

import anyio
import requests

logger = logging.getLogger("agentguard.async-worker.agent_tools")

WORKSPACE_API_URL = os.environ.get("WORKSPACE_API_URL", "").strip().rstrip("/")
HTTP_TIMEOUT = 30

#: Cap on what one tool hands back to the model. A tool that returns a
#: whole page turns one turn into an expensive one, and the model cannot
#: use what it cannot read.
MAX_RESULT_CHARS = 4000

#: How many work items one `list_work_items` call will show. Was 25 with no
#: signal that anything had been cut, which is how an agent confidently
#: described "the full picture" of a project after seeing a quarter of it.
#: 100 matches the workspace API's own per-page ceiling (see
#: `app.workspace_tools`, which requests the same cap) — raising it further
#: would just move the same silent cut one page later.
MAX_WORK_ITEMS = 100


TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "recall",
            "description": (
                "Search what this team already knows. Use this before "
                "answering anything factual about the team, its projects, "
                "or past decisions."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "What to look for."}
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_work_items",
            "description": "List work items in a project.",
            "parameters": {
                "type": "object",
                "properties": {
                    "project_id": {"type": "string"},
                },
                "required": ["project_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_projects",
            "description": "List the projects in this workspace.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "remember",
            "description": (
                "Record a durable fact or decision this team would want "
                "recalled later — something that will still be true and "
                "still useful next week, not this message's chatter, not "
                "your own reasoning, and not a recap of something you just "
                "read with `recall` or `list_work_items`. Use it when the "
                "conversation states a fact, makes a decision, or settles "
                "something worth remembering — not on every turn."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "content": {
                        "type": "string",
                        "description": "The fact or decision to store, in your own words.",
                    }
                },
                "required": ["content"],
            },
        },
    },
]


def _truncate(value: str) -> str:
    if len(value) <= MAX_RESULT_CHARS:
        return value
    return value[:MAX_RESULT_CHARS] + f"\n…(truncated at {MAX_RESULT_CHARS} characters)"


def build_tools(scope, org_id, token, slug, agent_id=None):
    """``(schemas, run_tool)`` for one turn.

    Args:
        scope: The audience scope. Closed over, never a tool argument.
        org_id: Tenant whose memories may be read and written. ``None``
            disables recall and remember rather than widening either.
        token: The agent's credential for workspace writes.
        slug: The workspace being acted in.
        agent_id: The agent principal, for attributing memories it writes.
            ``None`` disables `remember` rather than writing an unowned row.
    """
    from app.db import SessionLocal  # noqa: PLC0415
    from shared.brain import write_scoped_memory  # noqa: PLC0415
    from shared.memory import retrieve_memories_scoped  # noqa: PLC0415

    def _recall(arguments: dict[str, Any]) -> str:
        query = str(arguments.get("query") or "").strip()
        if not query:
            return "recall needs a query"
        if not org_id:
            return "no memory is connected to this conversation"

        db = SessionLocal()
        try:
            # `retrieve_memories_scoped` is async and the turn runs in a
            # worker thread, so it gets its own loop rather than borrowing
            # one that is not there.
            records = anyio.run(_retrieve, db, retrieve_memories_scoped, query)
        except Exception as exc:
            logger.warning("agent recall failed: %s", exc, exc_info=True)
            return f"memory search failed: {exc}"
        finally:
            db.close()

        if not records:
            return (
                "nothing in this conversation's memory matches that. Say so "
                "plainly rather than guessing."
            )
        lines = [f"- {record.content}" for record in records]
        return _truncate("\n".join(lines))

    async def _retrieve(db, retrieve, query):
        # The scope is applied HERE and nowhere else. `user_id` is set only
        # when the audience is one person, which is what keeps a private
        # tier out of a shared room.
        return await retrieve(
            db,
            org_id,
            query,
            scope="org",
            user_id=scope.user_id,
            project_id=scope.project_id,
            work_item_id=scope.work_item_id,
            limit=8,
        )

    def _remember(arguments: dict[str, Any]) -> str:
        content = str(arguments.get("content") or "").strip()
        if not content:
            return "remember needs content"
        if not org_id or not agent_id:
            # Same failure shape as `_recall` with no org: say so plainly
            # rather than writing an unattributed or unscoped row.
            return "no memory is connected to this conversation"

        db = SessionLocal()
        try:
            # Same off-loop reasoning as `_recall`: write_scoped_memory is
            # async, the turn runs in a worker thread, so it gets its own
            # loop.
            result = anyio.run(_write, db, write_scoped_memory, content)
        except Exception as exc:
            logger.warning("agent remember failed: %s", exc, exc_info=True)
            return f"could not store that: {exc}"
        finally:
            db.close()

        if not result.get("stored"):
            if result.get("deduplicated"):
                return "the team already knows this — nothing new was stored"
            return f"could not store that: {result.get('error', 'unknown error')}"
        return "stored"

    async def _write(db, write, content):
        # Pinned to scope="org" for the SAME reason `_retrieve` above is
        # pinned to scope="org": this is the exact scope `_recall` reads,
        # narrowed by the exact same `project_id` / `work_item_id`. Nothing
        # here can widen to another org (org_id is closed over, never a tool
        # argument) or land in a user's private tier (scope is never
        # "private" — user_id below is authorship on a shared row, not a
        # visibility predicate; see `write_scoped_memory`'s docstring).
        #
        # `agent_id` makes the row attributable to the agent that asserted
        # it. `metadata.source` is what lets a human reading the brain tell
        # this apart from a hook-captured, curator-extracted row: those
        # carry `source: "capture"` / go through the curator, this one is
        # the agent's own stated claim, made in the room, right now.
        return await write(
            db,
            org_id=org_id,
            agent_id=agent_id,
            content=content,
            memory_type="memory",
            session_id=f"agent-turn:{org_id}:{agent_id}",
            project_id=scope.project_id,
            work_item_id=scope.work_item_id if scope.project_id else None,
            metadata={"source": "agent_chat"},
            scope="org",
            user_id=scope.user_id,
        )

    def _list_projects(_arguments: dict[str, Any]) -> str:
        response = requests.get(
            f"{WORKSPACE_API_URL}/api/v1/workspaces/{slug}/projects/",
            headers={"X-Api-Key": token},
            timeout=HTTP_TIMEOUT,
        )
        if response.status_code != 200:
            return f"could not list projects (HTTP {response.status_code})"
        rows = (response.json() or {}).get("results") or []
        return _truncate(
            json.dumps([{"id": r.get("id"), "name": r.get("name")} for r in rows])
        )

    def _state_labels(project_id: str) -> dict[str, str]:
        """Work item ``state`` id -> human-readable name, for this project.

        The issues list returns `state` as a raw id (a UUID) — the
        workspace's own `states` endpoint is the only place that maps it to
        something like "In Progress". Best-effort: a failure here degrades
        to showing the raw id rather than losing the work item list over it.
        """
        try:
            response = requests.get(
                f"{WORKSPACE_API_URL}/api/v1/workspaces/{slug}/projects/{project_id}/states/",
                headers={"X-Api-Key": token},
                timeout=HTTP_TIMEOUT,
            )
        except requests.RequestException:
            return {}
        if response.status_code != 200:
            return {}
        payload = response.json()
        rows = payload.get("results", payload) if isinstance(payload, dict) else payload
        if not isinstance(rows, list):
            return {}
        return {
            row["id"]: row["name"]
            for row in rows
            if isinstance(row, dict) and row.get("id") and row.get("name")
        }

    def _list_work_items(arguments: dict[str, Any]) -> str:
        project_id = str(arguments.get("project_id") or "").strip()
        if not project_id:
            return "list_work_items needs a project_id"
        response = requests.get(
            f"{WORKSPACE_API_URL}/api/v1/workspaces/{slug}/projects/{project_id}/issues/",
            headers={"X-Api-Key": token},
            # Ask for a full page ourselves rather than accepting whatever
            # the API defaults to — a silent server-side default is the
            # same bug this fix removes, one layer further out.
            params={"per_page": MAX_WORK_ITEMS},
            timeout=HTTP_TIMEOUT,
        )
        if response.status_code != 200:
            return f"could not list work items (HTTP {response.status_code})"
        payload = response.json() or {}
        rows = payload.get("results") or []
        state_names = _state_labels(project_id)

        shown = rows[:MAX_WORK_ITEMS]
        items = []
        for r in shown:
            item = {"id": r.get("id"), "name": r.get("name")}
            state = r.get("state")
            if state:
                item["state"] = state_names.get(state, state)
            priority = r.get("priority")
            if priority:
                item["priority"] = priority
            items.append(item)
        text = json.dumps(items)

        # Truncation must be VISIBLE to the model — a cut list with no
        # marker is how an agent ends up confidently describing "the full
        # picture" from a fraction of it. `total_count` is the workspace
        # API's own paginated-response field; when it is absent we can only
        # say what this page returned, which is still better than silence.
        total = payload.get("total_count")
        if isinstance(total, int) and total > len(shown):
            text += f"\n…showing {len(shown)} of {total} work items in this project."
        elif len(rows) > len(shown):
            text += f"\n…showing {len(shown)} of {len(rows)} returned; more may exist."

        return _truncate(text)

    handlers = {
        "recall": _recall,
        "list_projects": _list_projects,
        "list_work_items": _list_work_items,
        "remember": _remember,
    }

    def run_tool(name: str, arguments: dict[str, Any]) -> str:
        handler = handlers.get(name)
        if handler is None:
            # Named rather than raised: the model can pick another tool.
            return f"no such tool: {name}"
        return handler(arguments or {})

    return TOOL_SCHEMAS, run_tool
