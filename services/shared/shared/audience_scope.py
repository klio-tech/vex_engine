"""What an agent may read, given where it is about to speak.

The rule, and the only one this module exists to enforce:

    An agent reads at the scope of the AUDIENCE it is replying to.

Anything it posts into a room must be something every member of that room
could already have read. That is checkable in one sentence — *could every
member of this channel have read this?* — which is what makes it survive
new features.

The alternative, acting with the ASKER's permissions, was considered and
rejected during design. It stops an agent reading beyond the person who
summoned it, but not publishing that person's private context into a
shared room. The leak is not to the asker; it is through them. A failure
that looks safe is worse than the problem it solves.

The DM row below is not an exception. When the audience is one person,
that person's own context is exactly what is safe to use.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class AudienceScope:
    """The read window for one reply.

    Attributes:
        project_id: Engine project whose memories are in scope, if any.
        work_item_id: Work item to rank first, if the thread is about one.
        user_id: Whose PRIVATE memories may be read — set only when the
            audience is that single person (a 1:1 DM with the agent).
            ``None`` everywhere else, which is what keeps a private tier
            out of a shared room.
    """

    project_id: Optional[str] = None
    work_item_id: Optional[str] = None
    user_id: Optional[str] = None

    @property
    def includes_private(self) -> bool:
        return self.user_id is not None


def resolve_audience_scope(
    *,
    channel_kind: str,
    is_private: bool,
    project_id: Optional[str],
    participant_user_ids: Optional[list[str]],
    asker_user_id: Optional[str],
    work_item_id: Optional[str] = None,
) -> AudienceScope:
    """The scope for a reply into this channel.

    Args:
        channel_kind: ``"direct"`` or ``"plot"``.
        is_private: Whether a plot is private.
        project_id: The ENGINE project linked to this channel's project,
            or ``None`` for a workspace-wide channel.
        participant_user_ids: Members of a direct channel. Used only to
            decide whether the audience is exactly one person besides the
            agent.
        asker_user_id: Who summoned the agent.
        work_item_id: The work item this thread concerns, if any.

    Returns:
        An :class:`AudienceScope`. Private memories are included ONLY for a
        one-to-one direct conversation, and then only the asker's own.
    """
    if channel_kind == "direct":
        # "Direct" is not automatically one-to-one: a group DM has several
        # people in it, and one member's private notes are not shared
        # context just because the room is small. Private reads need an
        # audience of exactly the agent and the asker.
        others = [
            user_id
            for user_id in (participant_user_ids or [])
            if user_id and user_id != asker_user_id
        ]
        is_one_to_one = len(others) <= 1
        return AudienceScope(
            project_id=project_id,
            work_item_id=work_item_id,
            user_id=asker_user_id if (is_one_to_one and asker_user_id) else None,
        )

    # Every plot, private or not. A private plot narrows WHO is in the room,
    # not what those people have individually kept private: being in a
    # closed room together does not pool the members' private memories.
    del is_private  # named for the caller's clarity; deliberately unused
    return AudienceScope(project_id=project_id, work_item_id=work_item_id, user_id=None)
