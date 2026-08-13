"""Project role policy — assign, touch, write — without a live database.

The helpers under test are pure. Storage (is_member / can_write against
Postgres) stays in the silo / grant integration tests.
"""

import pytest

from shared.projects import (
    ROLE_ADMIN,
    ROLE_MANAGE,
    ROLE_READ,
    ROLE_WRITE,
    can_assign_role,
    can_touch_member,
    normalize_role,
)


def test_member_is_a_synonym_for_write():
    assert normalize_role("member") == ROLE_WRITE
    assert normalize_role("read") == ROLE_READ


@pytest.mark.parametrize(
    ("actor", "new_role", "allowed"),
    [
        (ROLE_ADMIN, ROLE_READ, True),
        (ROLE_ADMIN, ROLE_WRITE, True),
        (ROLE_ADMIN, ROLE_MANAGE, True),
        (ROLE_ADMIN, ROLE_ADMIN, True),
        (ROLE_MANAGE, ROLE_READ, True),
        (ROLE_MANAGE, ROLE_WRITE, True),
        (ROLE_MANAGE, ROLE_MANAGE, False),
        (ROLE_MANAGE, ROLE_ADMIN, False),
        (ROLE_WRITE, ROLE_READ, False),
        (ROLE_READ, ROLE_WRITE, False),
        (None, ROLE_WRITE, False),
        (ROLE_ADMIN, "owner", False),
    ],
)
def test_can_assign_role(actor, new_role, allowed):
    assert can_assign_role(actor, new_role) is allowed


@pytest.mark.parametrize(
    ("actor", "target", "allowed"),
    [
        (ROLE_ADMIN, ROLE_ADMIN, True),
        (ROLE_ADMIN, ROLE_MANAGE, True),
        (ROLE_MANAGE, ROLE_WRITE, True),
        (ROLE_MANAGE, ROLE_READ, True),
        (ROLE_MANAGE, ROLE_MANAGE, False),
        (ROLE_MANAGE, ROLE_ADMIN, False),
        (ROLE_WRITE, ROLE_READ, False),
        (None, ROLE_READ, False),
    ],
)
def test_can_touch_member(actor, target, allowed):
    assert can_touch_member(actor, target) is allowed


def test_legacy_member_is_treated_as_write_for_policy():
    assert can_assign_role(ROLE_MANAGE, "member") is True
    assert can_touch_member(ROLE_MANAGE, "member") is True
    assert can_touch_member(ROLE_MANAGE, "admin") is False
