"""The tools an agent may use, bound to one turn's scope.

Every tool is built per turn and closes over that turn's
:class:`~shared.audience_scope.AudienceScope`, so a tool physically cannot
be called with a wider scope than the room allows. The alternative — a
generic tool that takes a scope argument — puts the security decision in
the model's hands, and the model is the one component here that is
explicitly not trusted.

Reads go straight to the memory layer in this process. Writes go through
the workspace's external API with the AGENT's own credential, so anything
an agent changes is attributable to it and constrained by its membership.
Two independent limits: a bug in one does not open the door.
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
]


def _truncate(value: str) -> str:
    if len(value) <= MAX_RESULT_CHARS:
        return value
    return value[:MAX_RESULT_CHARS] + f"\n…(truncated at {MAX_RESULT_CHARS} characters)"


def build_tools(scope, org_id, token, slug):
    """``(schemas, run_tool)`` for one turn.

    Args:
        scope: The audience scope. Closed over, never a tool argument.
        org_id: Tenant whose memories may be read. ``None`` disables recall
            rather than widening it.
        token: The agent's credential for workspace writes.
        slug: The workspace being acted in.
    """
    from app.db import SessionLocal  # noqa: PLC0415
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
    }

    def run_tool(name: str, arguments: dict[str, Any]) -> str:
        handler = handlers.get(name)
        if handler is None:
            # Named rather than raised: the model can pick another tool.
            return f"no such tool: {name}"
        return handler(arguments or {})

    return TOOL_SCHEMAS, run_tool
