"""X-Klio-* headers are first-class; X-Vex-* remain accepted forever.

The product is Klio; the wire headers said Vex. Every published connect
command was teaching new users a legacy brand in their first minute
(2026-08-11). The fix is an alias, not a rename: existing integrations carry
X-Vex-Key in configs we do not control, so that header can never break.

Precedence when both are present: the Klio header wins. A caller setting both
is migrating, and the newer name is the one they mean.
"""

from types import SimpleNamespace

from app.headers import agent_header, key_header, project_header


def _req(**headers):
    lowered = {k.lower(): v for k, v in headers.items()}

    class H:
        def get(self, name, default=None):
            return lowered.get(name.lower(), default)

    return SimpleNamespace(headers=H())


class TestKeyHeader:
    def test_klio_key_is_accepted(self):
        assert key_header(_req(**{"X-Klio-Key": "k1"})) == "k1"

    def test_vex_key_still_works(self):
        """The compatibility guarantee. Existing configs never break."""
        assert key_header(_req(**{"X-Vex-Key": "v1"})) == "v1"

    def test_agentguard_key_still_works(self):
        """The older alias keeps working too."""
        assert key_header(_req(**{"X-AgentGuard-Key": "a1"})) == "a1"

    def test_klio_wins_when_both_present(self):
        req = _req(**{"X-Klio-Key": "k1", "X-Vex-Key": "v1"})
        assert key_header(req) == "k1"

    def test_absent_is_none(self):
        assert key_header(_req()) is None


class TestAgentHeader:
    def test_klio_agent_is_accepted(self):
        assert agent_header(_req(**{"X-Klio-Agent": "claude-code"})) == "claude-code"

    def test_vex_agent_still_works(self):
        assert agent_header(_req(**{"X-Vex-Agent": "cursor"})) == "cursor"

    def test_klio_wins_when_both_present(self):
        req = _req(**{"X-Klio-Agent": "new", "X-Vex-Agent": "old"})
        assert agent_header(req) == "new"


class TestProjectHeader:
    def test_klio_project_is_accepted(self):
        assert project_header(_req(**{"X-Klio-Project": "p1"})) == "p1"

    def test_vex_project_still_works(self):
        assert project_header(_req(**{"X-Vex-Project": "p2"})) == "p2"


def test_every_request_header_read_goes_through_the_helpers():
    """No stray `headers.get("X-Vex-...")` outside app/headers.py.

    The alias only holds if every read site honours it. Asserted over the
    source so a future call site cannot quietly accept only the legacy name.
    """
    import pathlib

    app_dir = pathlib.Path(__file__).parents[1] / "app"
    offenders = []
    for f in app_dir.glob("*.py"):
        if f.name == "headers.py":
            continue
        text = f.read_text()
        if 'headers.get("X-Vex-' in text or 'headers.get("X-Klio-' in text:
            offenders.append(f.name)
    assert not offenders, (
        f"{offenders} read identity headers directly; route them through "
        "app.headers so X-Klio-* and X-Vex-* stay interchangeable"
    )
