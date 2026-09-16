"""Workspace tools: one MCP that both acts and remembers.

Until now an agent could recall and record Klio's memory but could not
*do* anything in the workspace — no project, no work item, no status
change. These tools close that gap on the SAME endpoint and the SAME
credential, because the workspace's personal access token
(``klio_pat_…``; legacy ``plane_api_…`` still honoured) is already valid
for both: the engine validates it against ``api_tokens.token_hash`` (see
:mod:`shared.auth` — the raw value is hashed on lookup, never stored), and
the workspace REST API validates the identical string as ``X-Api-Key``.
There is no second server to run and no second key to issue.

Design notes:

* **Two tools, not thirty.** Every tool definition is context an agent
  pays for on every request, so the actions are consolidated behind an
  ``action`` parameter. (Plane's own MCP server learned this and
  collapsed 177 per-operation tools into 30.)
* **The caller's own token is forwarded** — never a service credential.
  Actions therefore run as that person, with exactly the project
  permissions they already have; the workspace stays the authority on
  authorization and this module never second-guesses it.
* **The workspace is resolved from the caller**, not passed in blind: a
  token belongs to a workspace, and a caller who belongs to several may
  name one with ``workspace``.
"""

import asyncio
import html
import logging
import os
import re
import uuid as uuid_module
from typing import Any, Optional

import httpx

from app.headers import key_header
from app.identity import resolve_request_user_id
from app.workspace_db import workspace_engine
from shared.membership import member_workspaces

logger = logging.getLogger("agentguard.mcp.workspace_tools")

#: Where the workspace app lives. Unset ⇒ the tools report themselves as
#: unconfigured rather than guessing a host.
WORKSPACE_API_URL_ENV = "WORKSPACE_API_URL"

REQUEST_TIMEOUT_SECONDS = 20

#: Work-item fields worth returning; the raw API rows are large and most of
#: it is presentation state an agent has no use for.
_ITEM_FIELDS = (
    "id",
    "name",
    "sequence_id",
    "priority",
    "state",
    "description_stripped",
    "created_at",
    "updated_at",
    "target_date",
)

_PROJECT_FIELDS = ("id", "name", "identifier", "description", "created_at")

_PAGE_FIELDS = ("id", "name", "access", "is_locked", "created_at", "updated_at")

#: Extra fields worth returning when ONE item is read rather than listed.
_ITEM_DETAIL_FIELDS = _ITEM_FIELDS + (
    "project",
    "project_id",
    "start_date",
    "completed_at",
    "sub_issues_count",
    "assignees",
    "created_by",
)

_COMMENT_FIELDS = ("id", "created_at", "updated_at", "created_by")

#: Activity fields whose values are ids rather than words, and so need the
#: project's own lookups before a person can read them. `state` stores names
#: already; `priority` stores ids. Found by reading production rows.
_ID_VALUED_ACTIVITY_FIELDS = ("priority",)

#: Activity whose values are markup or noise: the entry is worth reporting,
#: the diff is not.
_OPAQUE_ACTIVITY_FIELDS = ("description", "description_binary")

#: The key a person reads off the board, e.g. ``YUMMY-8``. Accepted wherever
#: a work item id is: nobody carries the uuid of the item they are looking at.
_WORK_ITEM_KEY = re.compile(r"^[A-Za-z][A-Za-z0-9_]*-\d+$")

_TAG = re.compile(r"<[^>]+>")
_INLINE_SPACE = re.compile(r"[ \t]+")
#: Block tags whose boundary is a line break once the markup is gone.
_BLOCK_BREAK = re.compile(r"</(p|div|li|h[1-6]|tr|blockquote|pre)>|<br\s*/?>", re.I)


def _base_url() -> Optional[str]:
    url = os.environ.get(WORKSPACE_API_URL_ENV, "").strip().rstrip("/")
    return url or None


def _unconfigured() -> dict:
    return {
        "error": "not_configured",
        "detail": (
            "This deployment has no workspace API configured, so work items "
            "are unavailable. Memory tools are unaffected."
        ),
    }


def _slim(row: Any, fields: tuple[str, ...]) -> dict:
    if not isinstance(row, dict):
        return {"value": row}
    out = {}
    for field in fields:
        if field not in row:
            continue
        value = row[field]
        if value in (None, "", []):
            continue
        out[field] = value
    return out


def _rows(payload: Any) -> list:
    """The workspace API paginates; a bare list comes back from some routes."""
    if isinstance(payload, dict):
        return payload.get("results", [])
    return payload if isinstance(payload, list) else []


def _is_uuid(value: Any) -> bool:
    """Whether ``value`` is a uuid, and so safe to put in a URL path.

    Ids reach these tools from an agent, not from the workspace, and every
    one of them is interpolated into a path on an AUTHENTICATED client: a
    ref carrying ``/`` or ``..`` would address a different resource
    entirely. Parse it rather than pattern-match it.
    """
    if not isinstance(value, str):
        return False
    try:
        uuid_module.UUID(value.strip())
    except (ValueError, AttributeError, TypeError):
        return False
    return True


def _is_work_item_key(ref: Any) -> bool:
    """Whether ``ref`` is a board key (``YUMMY-8``) rather than a uuid."""
    return bool(isinstance(ref, str) and _WORK_ITEM_KEY.match(ref.strip()))


def _plain_text(markup: Any) -> str:
    """Readable text from the workspace's stored HTML.

    An agent asked to act on a work item needs the words, not the markup,
    and the API only fills ``description_stripped`` for items edited through
    surfaces that maintain it.
    """
    if not isinstance(markup, str) or not markup:
        return ""
    text = html.unescape(_TAG.sub("", _BLOCK_BREAK.sub("\n", markup)))
    lines = [_INLINE_SPACE.sub(" ", line).strip() for line in text.splitlines()]
    return "\n".join(line for line in lines if line)


def _names_by_id(payload: Any) -> dict:
    """``{id: name}`` for states and priorities, so a read returns neither
    raw uuid nor a second round trip for the caller to make."""
    out = {}
    for row in _rows(payload):
        if isinstance(row, dict) and row.get("id") and row.get("name"):
            out[str(row["id"])] = row["name"]
    return out


def _member_names(payload: Any) -> dict:
    """``{user id: display name}``, for attributing comments to people."""
    out = {}
    for row in _rows(payload):
        if not isinstance(row, dict):
            continue
        member = row.get("member")
        member_dict = member if isinstance(member, dict) else {}
        user_id = member_dict.get("id") or (member if isinstance(member, str) else None)
        name = (
            row.get("display_name")
            or member_dict.get("display_name")
            or member_dict.get("email")
        )
        if user_id and name:
            out[str(user_id)] = name
    return out


def _shape_comment(row: Any, author_names: dict) -> dict:
    if not isinstance(row, dict):
        return {"value": row}
    out = _slim(row, _COMMENT_FIELDS)
    body = row.get("comment_stripped") or _plain_text(row.get("comment_html"))
    author = out.pop("created_by", None)
    if author:
        out["author"] = author_names.get(str(author), str(author))
    if body:
        out["comment"] = body
    return out


def _shape_work_item(
    row: Any, *, states: dict, priorities: dict, people: Optional[dict] = None
) -> dict:
    """One work item as something to act on: the body as text, and state,
    priority, assignees and author as the words on the board rather than
    their ids."""
    if not isinstance(row, dict):
        return {"value": row}
    people = people or {}
    out = _slim(row, _ITEM_DETAIL_FIELDS)
    description = row.get("description_stripped") or _plain_text(
        row.get("description_html")
    )
    out.pop("description_stripped", None)
    if description:
        out["description"] = description
    state = out.get("state")
    if state:
        out["state"] = states.get(str(state), state)
    priority = out.get("priority")
    if priority:
        out["priority"] = priorities.get(str(priority), priority)
    assignees = out.get("assignees")
    if isinstance(assignees, list):
        out["assignees"] = [people.get(str(one), str(one)) for one in assignees]
    author = out.get("created_by")
    if author:
        out["created_by"] = people.get(str(author), str(author))
    return out


def resolve_workspace_slug(
    request: Any, requested: Optional[str]
) -> tuple[Optional[str], Optional[dict]]:
    """The workspace to act in: the one named, else the caller's only one.

    Returns ``(slug, error)``. An unattributable caller gets an error: a
    work item must be filed by a person, never anonymously.
    """
    user_id = resolve_request_user_id(request)
    if not user_id:
        return None, {
            "error": "unauthorized",
            "detail": "Work items need an attributable caller; use a personal access token.",
        }

    engine = workspace_engine()
    if engine is None:
        return None, _unconfigured()

    with engine.connect() as connection:
        workspaces = member_workspaces(connection, user_id=user_id)

    if not workspaces:
        return None, {
            "error": "not_found",
            "detail": "This token's user is not an active member of any workspace.",
        }

    if requested:
        for workspace in workspaces:
            if workspace["slug"] == requested:
                return requested, None
        return None, {
            "error": "not_found",
            "detail": f"No workspace '{requested}' that you are a member of.",
        }

    organizations = [w for w in workspaces if not w["is_personal"]]
    candidates = organizations or workspaces
    if len(candidates) == 1:
        return candidates[0]["slug"], None
    return None, {
        "error": "ambiguous_workspace",
        "detail": "You belong to several workspaces; pass `workspace`.",
        "workspaces": [w["slug"] for w in candidates],
    }


#: Prefixes the WORKSPACE API can authenticate. An engine key (``ag_live_…``)
#: is a valid Klio credential for memory and meaningless here: the two
#: credential systems are separate, and only these reach ``api_tokens``.
_WORKSPACE_KEY_PREFIXES = ("klio_pat_", "klio_bot_", "plane_api_")


def _is_workspace_key(token: str) -> bool:
    """Whether ``token`` is the kind of key the workspace API can validate."""
    return token.startswith(_WORKSPACE_KEY_PREFIXES)


def _not_a_workspace_key(token: str) -> dict:
    """The error for a credential the workspace cannot authenticate.

    Names the fix, because the caller cannot see their own MCP config from
    here and the previous wording ("may not do that in this workspace")
    described a permission problem that was never the cause.
    """
    kind = "an engine key" if token.startswith("ag_live_") else "not a workspace token"
    return {
        "error": "unauthorized",
        "detail": (
            f"This key is {kind}, so the workspace cannot authenticate it. "
            "Workspace actions need a personal access token (klio_pat_…) "
            "minted in Klio under Settings; set it as X-Klio-Key on the MCP "
            "server. Memory tools keep working with either key."
        ),
    }


def _not_an_id(value: Any) -> dict:
    """Refuse an id that is not one, before it reaches a URL."""
    return {
        "error": "invalid_argument",
        "detail": (
            f"{value!r} is not a work item id. Pass the item's uuid, or the "
            "key from the board (e.g. YUMMY-8)."
        ),
    }


async def _call(
    request: Any,
    method: str,
    path: str,
    *,
    params: Optional[dict] = None,
    json_body: Optional[dict] = None,
) -> tuple[Optional[Any], Optional[dict]]:
    """One authenticated call to the workspace API as the caller."""
    base = _base_url()
    if base is None:
        return None, _unconfigured()
    token = key_header(request)
    if not token:
        return None, {"error": "unauthorized", "detail": "No API key on the request."}
    # Fail before the round trip when the key CANNOT work here. An engine
    # key authenticates to the memory API perfectly well and is unknown to
    # the workspace, so forwarding it returns an indistinguishable 403 and
    # the caller goes looking for a permissions problem that does not
    # exist. This cost a full debugging session on 2026-09-03.
    if not _is_workspace_key(token):
        return None, _not_a_workspace_key(token)

    try:
        async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT_SECONDS) as client:
            response = await client.request(
                method,
                f"{base}{path}",
                params=params,
                json=json_body,
                headers={"X-Api-Key": token},
            )
    except httpx.HTTPError as exc:
        logger.warning("workspace API unreachable: %s", exc)
        return None, {
            "error": "unreachable",
            "detail": "The workspace API did not respond.",
        }

    if response.status_code == 401:
        return None, _not_a_workspace_key(token)
    if response.status_code == 403:
        # A credential the workspace RECOGNISED and then refused: a real
        # permission answer, and the workspace stays the authority on it.
        return None, {
            "error": "forbidden",
            "detail": "This token may not do that in this workspace.",
        }
    if response.status_code == 404:
        return None, {"error": "not_found", "detail": "No such project or work item."}
    if response.status_code >= 400:
        # The workspace's own validation message is the most useful thing
        # an agent can get here, so pass it through rather than flatten it.
        try:
            detail = response.json()
        except ValueError:
            detail = response.text[:400]
        return None, {
            "error": "rejected",
            "status": response.status_code,
            "detail": detail,
        }

    if response.status_code == 204 or not response.content:
        return {}, None
    try:
        return response.json(), None
    except ValueError:
        return None, {
            "error": "bad_response",
            "detail": "The workspace API returned non-JSON.",
        }


def _describe_activity(row: dict, *, names: dict, people: dict) -> str:
    """One activity row as a sentence.

    The board is where the conversation happens on a team that changes
    state and assignees rather than commenting: on YUMMY-10, four activity
    rows and no comments. The raw row is ids and half-sentences
    ("updated the priority to", old_value=<uuid>), so it is assembled here
    rather than handed over as it is.
    """
    field = row.get("field")
    verb = (row.get("verb") or "").strip()
    old_value = (row.get("old_value") or "").strip()
    new_value = (row.get("new_value") or "").strip()

    if not field:
        return f"{verb} the work item".strip() if verb else "changed the work item"

    if field in _OPAQUE_ACTIVITY_FIELDS:
        return "updated the description"

    if field in _ID_VALUED_ACTIVITY_FIELDS:
        old_value = names.get(old_value, old_value)
        new_value = names.get(new_value, new_value)

    if field == "assignees":
        if new_value:
            return f"added {people.get(new_value, new_value)} as assignee"
        if old_value:
            return f"removed {people.get(old_value, old_value)} as assignee"
        return "changed the assignees"

    label = field.replace("_", " ")
    if old_value and new_value:
        return f"changed {label} from {old_value} to {new_value}"
    if new_value:
        return f"set {label} to {new_value}"
    if old_value:
        return f"cleared {label} (was {old_value})"
    return f"{verb} {label}".strip()


def _shape_activity(row: Any, *, names: dict, people: dict) -> Optional[dict]:
    if not isinstance(row, dict):
        return None
    actor = row.get("actor")
    shaped = {
        "at": row.get("created_at"),
        "who": people.get(str(actor), str(actor)) if actor else None,
        "field": row.get("field") or "work item",
        "change": _describe_activity(row, names=names, people=people),
    }
    return {k: v for k, v in shaped.items() if v not in (None, "")}


async def read_work_item(
    request: Any,
    *,
    slug: str,
    project_id: Optional[str],
    work_item_id: Optional[str],
    limit: int = 50,
) -> dict:
    """One work item, its body and its comment thread.

    Everything else here writes; without this an agent could comment on an
    item it had never read. `list` returns titles only, so "read this work
    item and act on it" meant leaving the MCP for the REST API.

    Accepts the board key (``YUMMY-8``) as well as the uuid — the key is
    what a person has in front of them, and it needs no project id.
    """
    # Normalised once, here: `_is_uuid` parses a stripped value, so a padded
    # id would validate and then be interpolated into the path with its
    # spaces intact.
    ref = (work_item_id or "").strip()
    project_id = (project_id or "").strip() or None
    if not ref:
        return {
            "error": "missing_argument",
            "detail": (
                "read needs work_item_id: the key from the board (e.g. YUMMY-8) "
                "or the item's uuid."
            ),
        }

    if _is_work_item_key(ref):
        payload, error = await _call(
            request, "GET", f"/api/v1/workspaces/{slug}/work-items/{ref.upper()}/"
        )
        if error:
            return error
        item = payload if isinstance(payload, dict) else {}
        from_payload = item.get("project_id") or item.get("project")
        if isinstance(from_payload, str) and from_payload.strip():
            project_id = from_payload.strip()
    else:
        if not project_id:
            return {
                "error": "missing_argument",
                "detail": (
                    "Reading by uuid needs project_id. Pass the board key "
                    "(e.g. YUMMY-8) to read without one."
                ),
            }
        if not _is_uuid(ref) or not _is_uuid(project_id):
            return _not_an_id(ref if not _is_uuid(ref) else project_id)
        payload, error = await _call(
            request,
            "GET",
            f"/api/v1/workspaces/{slug}/projects/{project_id}/issues/{ref}/",
        )
        if error:
            return error
        item = payload if isinstance(payload, dict) else {}

    item_id = item.get("id")
    item_id = item_id.strip() if isinstance(item_id, str) else item_id
    if not item_id or not project_id:
        return {"error": "not_found", "detail": f"No work item {ref} in {slug}."}

    # The project came back from the workspace on the key path; it still
    # goes into a path, so it is checked on both.
    if not _is_uuid(item_id) or not _is_uuid(project_id):
        return _not_an_id(item_id if not _is_uuid(item_id) else project_id)

    root = f"/api/v1/workspaces/{slug}/projects/{project_id}"

    # Five independent lookups, so they go together rather than one after
    # another: a read is meant to be the cheap thing an agent does before
    # acting. Ids on the row and in the activity feed are resolved from
    # these, and every one of them is best effort — an unreadable lookup
    # must not cost the caller the read.
    per_page = max(1, min(int(limit or 50), 100))
    (
        (states_payload, _states_error),
        (priorities_payload, _priorities_error),
        (members_payload, _members_error),
        (comments_payload, comments_error),
        (activity_payload, activity_error),
    ) = await asyncio.gather(
        _call(request, "GET", f"{root}/states/"),
        _call(request, "GET", f"{root}/priorities/"),
        _call(request, "GET", f"{root}/members/"),
        _call(
            request,
            "GET",
            f"{root}/issues/{item_id}/comments/",
            params={"per_page": per_page},
        ),
        _call(
            request,
            "GET",
            f"{root}/issues/{item_id}/activities/",
            params={"per_page": per_page},
        ),
    )

    states = _names_by_id(states_payload)
    priorities = _names_by_id(priorities_payload)
    people = _member_names(members_payload)

    result: dict = {
        "workspace": slug,
        "work_item": _shape_work_item(
            item,
            states=states,
            priorities=priorities,
            people=people,
        ),
    }

    # What happened to this item, oldest first. On a board where people
    # move state and reassign rather than comment, this IS the discussion:
    # the item that prompted it had four activity rows and no comments.
    if activity_error:
        result["activity_error"] = activity_error
    else:
        activity = [
            shaped
            for shaped in (
                _shape_activity(row, names=priorities, people=people)
                for row in _rows(activity_payload)
            )
            if shaped
        ]
        activity.sort(key=lambda row: str(row.get("at") or ""))
        result["activity"] = activity

    if comments_error:
        # Say so rather than implying the thread is empty.
        result["comments_error"] = comments_error
        return result

    comments = [_shape_comment(row, people) for row in _rows(comments_payload)]
    # Oldest first: a thread reads as a conversation, not as a stack.
    comments.sort(key=lambda row: str(row.get("created_at") or ""))
    result["comments"] = comments
    return result


def register(mcp: Any) -> None:
    """Add the workspace tools to the Klio MCP server."""
    from typing import Annotated, Literal  # noqa: PLC0415 -- local to the schema

    from mcp.server.fastmcp import Context  # noqa: PLC0415
    from pydantic import Field  # noqa: PLC0415

    @mcp.tool(
        name="work_projects",
        description=(
            "The WORKSPACE projects you can see — the boards work items live "
            "on, not the memory projects that `project_list` reports. "
            "Args: action ('list' default | 'create' | 'members'); for create: "
            "name and identifier (a short uppercase key, e.g. LAUNCH); "
            "optional description, workspace. "
            "Acts as you, with your own permissions."
        ),
    )
    async def work_projects(
        ctx: Context,
        action: Annotated[
            Literal["list", "create", "members"],
            Field(description="What to do."),
        ] = "list",
        name: Annotated[
            Optional[str], Field(description="Project name (create).")
        ] = None,
        identifier: Annotated[
            Optional[str],
            Field(description="Short uppercase key, e.g. LAUNCH (create)."),
        ] = None,
        description: Annotated[
            Optional[str], Field(description="Project description.")
        ] = None,
        project_id: Annotated[
            Optional[str], Field(description="Project id (members).")
        ] = None,
        workspace: Annotated[
            Optional[str],
            Field(description="Workspace slug, when you belong to several."),
        ] = None,
    ) -> dict:
        request = ctx.request_context.request
        if request is None:
            return {"error": "unauthorized", "detail": "no request context"}

        slug, error = resolve_workspace_slug(request, workspace)
        if error:
            return error

        if action == "list":
            payload, error = await _call(
                request, "GET", f"/api/v1/workspaces/{slug}/projects/"
            )
            if error:
                return error
            return {
                "workspace": slug,
                "projects": [_slim(row, _PROJECT_FIELDS) for row in _rows(payload)],
            }

        if action == "members":
            if not project_id:
                return {
                    "error": "missing_argument",
                    "detail": "members needs project_id.",
                }
            payload, error = await _call(
                request,
                "GET",
                f"/api/v1/workspaces/{slug}/projects/{project_id}/members/",
            )
            if error:
                return error
            return {
                "workspace": slug,
                "members": [
                    _slim(row, ("id", "member", "role", "display_name"))
                    for row in _rows(payload)
                ],
            }

        if not name or not identifier:
            return {
                "error": "missing_argument",
                "detail": "create needs both name and identifier.",
            }
        body = {"name": name, "identifier": identifier.upper()}
        if description:
            body["description"] = description
        payload, error = await _call(
            request, "POST", f"/api/v1/workspaces/{slug}/projects/", json_body=body
        )
        if error:
            return error
        return {"workspace": slug, "created": _slim(payload, _PROJECT_FIELDS)}

    @mcp.tool(
        name="work_items",
        description=(
            "Work items (issues) on a workspace project — read them, create "
            "them, list them, move them, comment on them. This is how you DO "
            "the work you remember. "
            "Args: action ('list' default | 'read' | 'create' | 'update' | "
            "'comment' | 'states'); project_id (required for everything "
            "except reading by key); for read: work_item_id, which accepts "
            "the key from the board (e.g. YUMMY-8, no project_id needed) or "
            "the item's uuid — it returns the body, the comment thread and what has happened to the item, and `list` returns none of that; "
            "and `list` returns neither; for create: name, optional description, "
            "priority ('urgent'|'high'|'medium'|'low'|'none'), state_id; for "
            "update: work_item_id plus any of name/description/priority/"
            "state_id; for comment: work_item_id and comment. "
            "Use 'states' first to learn a project's state ids before moving "
            "an item. Acts as you, with your own permissions."
        ),
    )
    async def work_items(
        ctx: Context,
        action: Annotated[
            Literal["list", "read", "create", "update", "comment", "states"],
            Field(description="What to do."),
        ] = "list",
        project_id: Annotated[
            Optional[str], Field(description="Workspace project id.")
        ] = None,
        work_item_id: Annotated[
            Optional[str],
            Field(
                description=(
                    "Work item id (read, update, comment). For read this may "
                    "also be the board key, e.g. YUMMY-8."
                )
            ),
        ] = None,
        name: Annotated[
            Optional[str], Field(description="Title (create/update).")
        ] = None,
        description: Annotated[
            Optional[str], Field(description="Plain-text body (create/update).")
        ] = None,
        body_html: Annotated[
            Optional[str],
            Field(
                description=(
                    "Rich body as HTML, used instead of `description` when given. "
                    "Supports <h2>, <p>, <ul><li>, <pre><code>, <strong>, <em>, <a>. "
                    "Prefer this when the item needs real context: what is wrong, the "
                    "evidence, where the code lives, and what done means."
                )
            ),
        ] = None,
        priority: Annotated[
            Optional[Literal["urgent", "high", "medium", "low", "none"]],
            Field(description="Priority."),
        ] = None,
        state_id: Annotated[
            Optional[str], Field(description="State id to move to.")
        ] = None,
        comment: Annotated[Optional[str], Field(description="Comment body.")] = None,
        limit: Annotated[
            int, Field(description="Max rows for list (default 50).")
        ] = 50,
        workspace: Annotated[
            Optional[str],
            Field(description="Workspace slug, when you belong to several."),
        ] = None,
    ) -> dict:
        request = ctx.request_context.request
        if request is None:
            return {"error": "unauthorized", "detail": "no request context"}

        slug, error = resolve_workspace_slug(request, workspace)
        if error:
            return error

        # Reading by board key needs no project: the key names it.
        if action == "read":
            return await read_work_item(
                request,
                slug=slug,
                project_id=project_id,
                work_item_id=work_item_id,
                limit=limit,
            )

        if not project_id:
            return {
                "error": "missing_argument",
                "detail": "project_id is required; call work_projects(action='list') first.",
            }

        root = f"/api/v1/workspaces/{slug}/projects/{project_id}"

        if action == "states":
            payload, error = await _call(request, "GET", f"{root}/states/")
            if error:
                return error
            return {
                "states": [
                    _slim(row, ("id", "name", "group")) for row in _rows(payload)
                ]
            }

        if action == "list":
            payload, error = await _call(
                request,
                "GET",
                f"{root}/issues/",
                params={"per_page": max(1, min(int(limit or 50), 100))},
            )
            if error:
                return error
            return {"work_items": [_slim(row, _ITEM_FIELDS) for row in _rows(payload)]}

        if action == "create":
            if not name:
                return {"error": "missing_argument", "detail": "create needs a name."}
            body: dict = {"name": name}
            # Rich HTML wins when supplied; plain text stays a convenience.
            if body_html:
                body["description_html"] = body_html
            elif description:
                body["description_html"] = f"<p>{description}</p>"
            if priority:
                body["priority"] = priority
            if state_id:
                body["state"] = state_id
            payload, error = await _call(
                request, "POST", f"{root}/issues/", json_body=body
            )
            if error:
                return error
            return {"created": _slim(payload, _ITEM_FIELDS)}

        if action == "update":
            if not work_item_id:
                return {
                    "error": "missing_argument",
                    "detail": "update needs work_item_id.",
                }
            body = {}
            if name:
                body["name"] = name
            if body_html:
                body["description_html"] = body_html
            elif description:
                body["description_html"] = f"<p>{description}</p>"
            if priority:
                body["priority"] = priority
            if state_id:
                body["state"] = state_id
            if not body:
                return {"error": "missing_argument", "detail": "Nothing to update."}
            payload, error = await _call(
                request, "PATCH", f"{root}/issues/{work_item_id}/", json_body=body
            )
            if error:
                return error
            return {"updated": _slim(payload, _ITEM_FIELDS)}

        if not work_item_id or not comment:
            return {
                "error": "missing_argument",
                "detail": "comment needs work_item_id and comment.",
            }
        payload, error = await _call(
            request,
            "POST",
            f"{root}/issues/{work_item_id}/comments/",
            json_body={"comment_html": f"<p>{comment}</p>"},
        )
        if error:
            return error
        return {"commented": {"id": (payload or {}).get("id")}}

    @mcp.tool(
        name="work_pages",
        description=(
            "Documentation pages — read the architecture before you change "
            "it, and write down what you learned after. "
            "OMIT project_id for WORKSPACE pages: durable docs that outlive "
            "any one project (architecture, conventions, runbooks) — that is "
            "where the Klio architecture pages live. Pass project_id only for "
            "notes that belong to one project and should disappear with it. "
            "Args: action ('list' default | 'read' | 'create' | 'update'); "
            "optional project_id; for create: title and body_html; for "
            "read/update: page_id; for update: title and/or body_html. "
            "`body_html` is HTML — use <h2>, <p>, <ul><li>, <pre><code> and "
            "<strong>. Pages you write are visible to the whole project, so "
            "write them for the next person, not as notes to yourself."
        ),
    )
    async def work_pages(
        ctx: Context,
        action: Annotated[
            Literal["list", "read", "create", "update"],
            Field(description="What to do."),
        ] = "list",
        project_id: Annotated[
            Optional[str], Field(description="Workspace project id.")
        ] = None,
        page_id: Annotated[
            Optional[str], Field(description="Page id (read, update).")
        ] = None,
        title: Annotated[Optional[str], Field(description="Page title.")] = None,
        body_html: Annotated[
            Optional[str], Field(description="Page body as HTML.")
        ] = None,
        workspace: Annotated[
            Optional[str],
            Field(description="Workspace slug, when you belong to several."),
        ] = None,
    ) -> dict:
        request = ctx.request_context.request
        if request is None:
            return {"error": "unauthorized", "detail": "no request context"}

        slug, error = resolve_workspace_slug(request, workspace)
        if error:
            return error
        # No project_id means workspace level — the default home for
        # documentation, not an error.
        root = (
            f"/api/v1/workspaces/{slug}/projects/{project_id}/pages"
            if project_id
            else f"/api/v1/workspaces/{slug}/pages"
        )

        if action == "list":
            payload, error = await _call(request, "GET", f"{root}/")
            if error:
                return error
            return {"pages": [_slim(row, _PAGE_FIELDS) for row in _rows(payload)]}

        if action == "read":
            if not page_id:
                return {"error": "missing_argument", "detail": "read needs page_id."}
            payload, error = await _call(request, "GET", f"{root}/{page_id}/")
            if error:
                return error
            page = _slim(payload, _PAGE_FIELDS)
            # The body is the point of reading a page, so it is returned in
            # full rather than trimmed like the list rows.
            page["body_html"] = (payload or {}).get("description_html", "")
            return {"page": page}

        if action == "create":
            if not title:
                return {"error": "missing_argument", "detail": "create needs a title."}
            payload, error = await _call(
                request,
                "POST",
                f"{root}/",
                json_body={"name": title, "description_html": body_html or "<p></p>"},
            )
            if error:
                return error
            return {"created": _slim(payload, _PAGE_FIELDS)}

        if not page_id:
            return {"error": "missing_argument", "detail": "update needs page_id."}
        body: dict = {}
        if title:
            body["name"] = title
        if body_html:
            body["description_html"] = body_html
        if not body:
            return {"error": "missing_argument", "detail": "Nothing to update."}
        payload, error = await _call(
            request, "PATCH", f"{root}/{page_id}/", json_body=body
        )
        if error:
            return error
        return {"updated": _slim(payload, _PAGE_FIELDS)}
