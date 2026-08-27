"""Execute one agent turn: load it, run it, post the reply.

This is the integration layer. The judgement lives elsewhere on purpose —
:mod:`shared.audience_scope` decides what may be read, and
:mod:`shared.agent_runtime` decides how the loop terminates — so both can
be tested without a database, a model, or a network. What is here is the
wiring those two need to touch the real world.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Optional

import requests
from sqlalchemy import text

from shared.agent_runtime import TurnContext, run_turn
from shared.audience_scope import resolve_audience_scope

logger = logging.getLogger("agentguard.async-worker.agent")

WORKSPACE_API_URL = os.environ.get("WORKSPACE_API_URL", "").strip().rstrip("/")
LITELLM_API_URL = os.environ.get("LITELLM_API_URL", "").strip().rstrip("/")
LITELLM_API_KEY = os.environ.get("LITELLM_API_KEY", "").strip()
DEFAULT_MODEL = os.environ.get("AGENT_DEFAULT_MODEL", "").strip()

HTTP_TIMEOUT = 30

#: How much of the thread the agent is shown. Enough to follow a short
#: exchange; not so much that an old channel becomes an expensive prompt.
HISTORY_LIMIT = 12


_AGENT_SQL = text(
    """
    SELECT a.id, a.name, a.instructions, a.model, a.is_enabled,
           a.user_id AS bot_user_id,
           t.token AS agent_token
    FROM agents a
    LEFT JOIN api_tokens t
           ON t.user_id = a.user_id AND t.is_active = true
    WHERE a.id = CAST(:agent_id AS uuid)
      AND a.deleted_at IS NULL
    LIMIT 1
    """
)

_MESSAGE_SQL = text(
    """
    SELECT m.id, m.content, m.actor_id, m.parent_id,
           c.id AS channel_id, c.kind, c.is_private, c.project_id,
           c.workspace_id, w.slug AS workspace_slug
    FROM channel_messages m
    JOIN channels c ON c.id = m.channel_id
    JOIN workspaces w ON w.id = c.workspace_id
    WHERE m.id = CAST(:message_id AS uuid)
      AND m.removed_at IS NULL
    LIMIT 1
    """
)

_PARTICIPANTS_SQL = text(
    """
    SELECT member_id FROM channel_participants
    WHERE channel_id = CAST(:channel_id AS uuid)
    """
)

_HISTORY_SQL = text(
    """
    SELECT m.content, u.display_name
    FROM channel_messages m
    LEFT JOIN users u ON u.id = m.actor_id
    WHERE m.channel_id = CAST(:channel_id AS uuid)
      AND m.removed_at IS NULL
      AND m.created_at <= (SELECT created_at FROM channel_messages
                           WHERE id = CAST(:message_id AS uuid))
    ORDER BY m.created_at DESC
    LIMIT :limit
    """
)

_LINKED_PROJECT_SQL = text(
    """
    SELECT id, org_id FROM projects
    WHERE plane_project_id = CAST(:plane_project_id AS uuid)
    LIMIT 1
    """
)

#: The org behind a workspace. Resolved even when no project is linked,
#: because org-scoped memory is the floor an agent always has — without
#: this an agent in an unlinked channel would have no memory at all, which
#: reads as "the agent is broken" rather than "this project is not linked".
_ORG_BY_WORKSPACE_SQL = text(
    """
    SELECT org_id FROM organizations
    WHERE workspace_id = CAST(:workspace_id AS uuid)
    LIMIT 1
    """
)


def _post_message(
    slug: str, channel_id: str, token: str, content: str, parent: str
) -> Optional[str]:
    """Post as the agent through the external API, returning the message id.

    The external API because it takes an API key; the app's chat endpoints
    are session-only, and giving an agent a session would be a far larger
    change than giving the public API a chat write.
    """
    response = requests.post(
        f"{WORKSPACE_API_URL}/api/v1/workspaces/{slug}/channels/{channel_id}/messages/",
        json={"content": content, "parent": parent},
        headers={"X-Api-Key": token},
        timeout=HTTP_TIMEOUT,
    )
    if response.status_code != 201:
        logger.warning(
            "agent could not post: HTTP %s %s",
            response.status_code,
            response.text[:200],
        )
        return None
    return (response.json() or {}).get("id")


def _edit_message(
    slug: str, channel_id: str, token: str, message_id: str, content: str
) -> None:
    try:
        requests.patch(
            f"{WORKSPACE_API_URL}/api/v1/workspaces/{slug}/channels/{channel_id}/messages/{message_id}/",
            json={"content": content},
            headers={"X-Api-Key": token},
            timeout=HTTP_TIMEOUT,
        )
    except requests.RequestException:
        # Progress is a courtesy; losing it must not lose the turn.
        logger.info("agent progress edit failed", exc_info=True)


def _call_litellm(model: str):
    """A ``call_model`` bound to the gateway.

    Returns the narrow shape :mod:`shared.agent_runtime` expects — a tool
    request or text — so the loop never learns an API's response format.
    """

    def call(
        messages: list[dict[str, Any]], tools: list[dict[str, Any]]
    ) -> dict[str, Any]:
        body: dict[str, Any] = {"model": model, "messages": messages}
        if tools:
            body["tools"] = tools
            body["tool_choice"] = "auto"
        response = requests.post(
            f"{LITELLM_API_URL}/chat/completions",
            json=body,
            headers={"Authorization": f"Bearer {LITELLM_API_KEY}"},
            timeout=HTTP_TIMEOUT,
        )
        response.raise_for_status()
        choice = (response.json().get("choices") or [{}])[0]
        message = choice.get("message") or {}
        calls = message.get("tool_calls") or []
        if calls:
            first = calls[0].get("function") or {}
            try:
                arguments = json.loads(first.get("arguments") or "{}")
            except (TypeError, ValueError):
                arguments = {}
            return {"tool": first.get("name"), "arguments": arguments}
        return {"text": (message.get("content") or "").strip()}

    return call


def load_turn(
    workspace_conn, engine_conn, *, agent_id: str, message_id: str
) -> Optional[dict]:
    """Everything one turn needs, or ``None`` if it should not run.

    Returns ``None`` — rather than raising — for every ordinary reason a
    turn is moot: the agent was disabled or deleted, the message was
    removed, the agent has no credential, or it would be answering itself.
    A turn that should not happen is not an error.
    """
    agent = workspace_conn.execute(_AGENT_SQL, {"agent_id": agent_id}).fetchone()
    if agent is None or not agent.is_enabled:
        return None
    if not agent.agent_token:
        logger.warning("agent %s has no active token; cannot act", agent_id)
        return None

    message = workspace_conn.execute(
        _MESSAGE_SQL, {"message_id": message_id}
    ).fetchone()
    if message is None:
        return None
    if str(message.actor_id) == str(agent.bot_user_id):
        # An agent must never answer itself.
        return None

    participants = [
        str(row.member_id)
        for row in workspace_conn.execute(
            _PARTICIPANTS_SQL, {"channel_id": str(message.channel_id)}
        ).fetchall()
    ]

    # The org is the floor: every channel has a workspace, and a workspace
    # maps to an org whose memories are always in scope.
    org_row = engine_conn.execute(
        _ORG_BY_WORKSPACE_SQL, {"workspace_id": str(message.workspace_id)}
    ).fetchone()
    org_id = org_row.org_id if org_row is not None else None

    # A linked project narrows further, when there is one.
    project_id = None
    if message.project_id:
        linked = engine_conn.execute(
            _LINKED_PROJECT_SQL, {"plane_project_id": str(message.project_id)}
        ).fetchone()
        if linked is not None:
            project_id = str(linked.id)
            # A linked project's org wins: it is the org that actually owns
            # the memories being read.
            org_id = linked.org_id

    scope = resolve_audience_scope(
        channel_kind=message.kind,
        is_private=bool(message.is_private),
        project_id=project_id,
        participant_user_ids=participants,
        asker_user_id=str(message.actor_id) if message.actor_id else None,
    )

    history = [
        (row.display_name or "someone", row.content)
        for row in reversed(
            workspace_conn.execute(
                _HISTORY_SQL,
                {
                    "channel_id": str(message.channel_id),
                    "message_id": message_id,
                    "limit": HISTORY_LIMIT,
                },
            ).fetchall()
        )
    ]

    return {
        "agent": agent,
        "message": message,
        "scope": scope,
        "org_id": org_id,
        "history": history[:-1] if history else [],
        # A reply goes in a thread off the message that summoned the agent;
        # if that message is itself a reply, threads are one level, so the
        # root is where it belongs.
        "thread_root": str(message.parent_id or message.id),
    }


def execute_turn(loaded: dict, *, build_tools) -> str:
    """Run the loop and put the result in the thread.

    Args:
        loaded: The output of :func:`load_turn`.
        build_tools: ``(scope, org_id, token) -> (schemas, run_tool)``.
            Injected so the tool surface can grow — including, later, tools
            from a connected MCP server — without this function changing.

    Returns:
        A short outcome string for the log.
    """
    agent = loaded["agent"]
    message = loaded["message"]
    slug = message.workspace_slug
    channel_id = str(message.channel_id)
    token = agent.agent_token

    # Posted BEFORE the model is called, so the room can see it was heard.
    # Silence and thinking look identical, and that ambiguity is what makes
    # people stop trusting an agent.
    placeholder = _post_message(
        slug, channel_id, token, "…thinking", loaded["thread_root"]
    )
    if placeholder is None:
        return "could not post"

    schemas, run_tool = build_tools(loaded["scope"], loaded["org_id"], token, slug)

    def on_progress(line: str) -> None:
        _edit_message(slug, channel_id, token, placeholder, f"…{line}")

    result = run_turn(
        TurnContext(
            system_prompt=agent.instructions or "You are a helpful teammate.",
            user_message=message.content,
            history=loaded["history"],
            call_model=_call_litellm(agent.model or DEFAULT_MODEL),
            run_tool=run_tool,
            tools=schemas,
            on_progress=on_progress,
        )
    )

    if result.answered:
        final = result.text
    elif result.reason == "max_steps":
        final = "I stopped after too many steps — ask me something narrower and I'll try again."
    elif result.reason == "deadline":
        final = "That took too long, so I stopped. Try narrowing it down?"
    else:
        final = "I couldn't finish that one. Nothing was changed."

    _edit_message(slug, channel_id, token, placeholder, final)
    return f"{result.reason or 'answered'} in {result.steps} step(s)"
