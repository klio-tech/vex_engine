"""Key-owner membership: workspace-linked orgs answer from the workspace DB.

The ownership guard on the key path (auth.py step 4.5) asked Supabase's
``accounts_memberships`` — pre-handshake truth. After the Plane⇄engine
handshake, an org with a non-NULL ``workspace_id`` has its membership truth
in ``workspace_members``, and a key whose ``created_by`` is a WORKSPACE user
uuid does not exist in Supabase at all: the guard answered "not a member"
for every workspace-identity key on a linked org, which is a 403 on a valid
credential.

Found live: the desktop key minted for klio-internal (workspace-linked) was
refused with "no longer a member of this workspace" although its owner is
the workspace's admin.
"""

import hashlib
import threading
import json
import uuid
from unittest.mock import MagicMock

import pytest

from shared.auth import AuthError, KeyValidator

WS_ID = "875c194c-78b7-4678-88ad-6a084e826783"
OWNER = "d665eedc-7935-4343-be25-96e9ceafd2f8"


def _entry_and_key():
    raw = "ag_live_test_" + uuid.uuid4().hex
    return {
        "id": str(uuid.uuid4()),
        "key_hash": hashlib.sha256(raw.encode()).hexdigest(),
        "scopes": ["memory"],
        "created_by": OWNER,
        "revoked": False,
    }, raw


class _Result:
    def __init__(self, row):
        self._row = row

    def fetchone(self):
        return self._row


class _Conn:
    def __init__(self, rows, calls):
        self._rows = rows
        self._calls = calls

    def execute(self, sql, params=None):
        self._calls.append((str(sql), params))
        for needle, row in self._rows:
            if needle in str(sql):
                return _Result(row)
        return _Result(None)

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


class _Engine:
    def __init__(self, rows):
        self.calls = []
        self._rows = rows

    def connect(self):
        return _Conn(self._rows, self.calls)

    def dispose(self):
        pass


def _validator(entry, *, workspace_row, workspace_engine_rows=None):
    v = KeyValidator.__new__(KeyValidator)
    v._required_scope = "memory"
    v._cache = {}
    v._cache_ttl_s = 0
    v._cache_lock = threading.Lock()
    v._warn_lock = threading.Lock()
    v._warned_supabase_unconfigured = False
    v._warned_workspace_unconfigured = False
    v._engine = _Engine(
        [
            (
                "FROM organizations",
                ("be3b0e93-org", json.dumps([entry]), "klio-internal", workspace_row),
            )
        ]
    )
    v._supabase_engine = None
    v._workspace_engine = (
        _Engine(workspace_engine_rows) if workspace_engine_rows is not None else None
    )
    # neutralize the enforcement stages that need infra
    v._check_rate_limit = MagicMock()
    v._check_quota = MagicMock()
    v._check_agent_limit = MagicMock()
    v._track_usage = MagicMock()
    v._usage_lock = MagicMock()
    return v


def test_workspace_member_passes_even_though_supabase_knows_nothing():
    entry, raw = _entry_and_key()
    v = _validator(
        entry,
        workspace_row=WS_ID,
        workspace_engine_rows=[("workspace_members", (True,))],
    )
    info = v.validate(raw)
    assert info.org_id == "be3b0e93-org"


def test_workspace_non_member_is_refused():
    entry, raw = _entry_and_key()
    v = _validator(
        entry,
        workspace_row=WS_ID,
        workspace_engine_rows=[("workspace_members", (False,))],
    )
    with pytest.raises(AuthError) as e:
        v.validate(raw)
    assert e.value.status_code == 403


def test_unlinked_org_never_touches_the_workspace_engine():
    entry, raw = _entry_and_key()
    rows = [("workspace_members", (False,))]  # would refuse if consulted
    v = _validator(entry, workspace_row=None, workspace_engine_rows=rows)
    info = v.validate(raw)
    assert info.org_id == "be3b0e93-org"
    assert v._workspace_engine.calls == []


def test_linked_org_without_a_workspace_engine_skips_not_refuses():
    # The Railway dark spare has no WORKSPACE_DATABASE_URL; a linked org's
    # key must not brick there. Skip (None) mirrors the Supabase-unconfigured
    # contract.
    entry, raw = _entry_and_key()
    v = _validator(entry, workspace_row=WS_ID, workspace_engine_rows=None)
    info = v.validate(raw)
    assert info.org_id == "be3b0e93-org"


def test_workspace_truth_overrides_whatever_supabase_would_say():
    # The defect shape: Supabase says non-member (workspace uuid unknown
    # there), the workspace says member. Workspace wins on linked orgs.
    entry, raw = _entry_and_key()
    v = _validator(
        entry,
        workspace_row=WS_ID,
        workspace_engine_rows=[("workspace_members", (True,))],
    )
    v._query_plan_from_supabase = MagicMock(return_value=("free", None, False))
    info = v.validate(raw)
    assert info.org_id == "be3b0e93-org"
