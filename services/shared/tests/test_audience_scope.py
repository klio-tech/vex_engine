"""The rule an agent's reads obey, and the one that must never quietly widen.

Every test here is a statement about what an agent may publish into a room.
The failure this guards against is not "the agent saw too much" — it is
"the agent said too much, to people who should not have heard it". That is
why the private tier is the subject of most of these cases.
"""

import pytest

from shared.audience_scope import AudienceScope, resolve_audience_scope

ASKER = "user-asking"
OTHER = "user-other"


def plot(**kw):
    base = dict(
        channel_kind="plot",
        is_private=False,
        project_id="proj-1",
        participant_user_ids=None,
        asker_user_id=ASKER,
    )
    base.update(kw)
    return resolve_audience_scope(**base)


def dm(participants, **kw):
    base = dict(
        channel_kind="direct",
        is_private=False,
        project_id=None,
        participant_user_ids=participants,
        asker_user_id=ASKER,
    )
    base.update(kw)
    return resolve_audience_scope(**base)


class TestPlots:
    def test_a_public_plot_never_reads_a_private_tier(self):
        scope = plot()
        assert scope.user_id is None
        assert scope.includes_private is False

    def test_a_public_plot_reads_its_project(self):
        assert plot().project_id == "proj-1"

    def test_a_private_plot_still_never_reads_a_private_tier(self):
        # The room being closed narrows WHO is in it, not what each person
        # kept to themselves. Pooling private memories because a plot is
        # private would be the single worst bug in this design.
        scope = plot(is_private=True)
        assert scope.user_id is None
        assert scope.includes_private is False

    def test_a_workspace_plot_has_no_project(self):
        assert plot(project_id=None).project_id is None

    def test_a_thread_about_a_work_item_carries_it(self):
        assert plot(work_item_id="wi-9").work_item_id == "wi-9"


class TestDirectMessages:
    def test_a_one_to_one_dm_may_read_the_askers_private_memories(self):
        # The audience is one person, so their own context is safe to use.
        scope = dm([ASKER, "agent-bot"])
        assert scope.user_id == ASKER
        assert scope.includes_private is True

    def test_a_dm_with_only_the_asker_still_counts_as_one_to_one(self):
        assert dm([ASKER]).user_id == ASKER

    def test_a_group_dm_does_not_read_anyones_private_memories(self):
        # Three people in a room is a room. One member's private notes are
        # not shared context just because the group is small.
        scope = dm([ASKER, OTHER, "agent-bot"])
        assert scope.user_id is None
        assert scope.includes_private is False

    def test_an_unattributable_asker_gets_no_private_tier(self):
        # Nobody to attribute means nobody whose privacy we could honour.
        scope = dm([ASKER, "agent-bot"], asker_user_id=None)
        assert scope.user_id is None


class TestTheInvariant:
    @pytest.mark.parametrize(
        "kind,private,participants",
        [
            ("plot", False, None),
            ("plot", True, None),
            ("direct", False, [ASKER, OTHER, "agent-bot"]),
        ],
    )
    def test_private_memories_are_reachable_only_in_a_one_to_one(
        self, kind, private, participants
    ):
        """The single claim the whole design rests on, stated once.

        If this ever passes with `includes_private` True for a shared
        audience, an agent can publish one person's private context to a
        room — which is the failure the design exists to prevent.
        """
        scope = resolve_audience_scope(
            channel_kind=kind,
            is_private=private,
            project_id="proj-1",
            participant_user_ids=participants,
            asker_user_id=ASKER,
        )
        assert scope.includes_private is False

    def test_the_scope_is_immutable(self):
        # A scope handed to the recall layer must not be widened en route.
        scope = AudienceScope(project_id="p", user_id=None)
        with pytest.raises(Exception):
            scope.user_id = ASKER  # type: ignore[misc]
