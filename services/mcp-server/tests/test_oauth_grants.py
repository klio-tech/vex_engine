"""GrantResolver: (user, client) -> (org_id, account_slug, plan)."""

from app.oauth import GrantResolver, OAuthIdentity, OAuthPrincipal


class FakeConn:
    def __init__(self, rows):
        self._rows = rows

    def execute(self, sql, params=None):
        sql_s = str(sql)

        class R:
            def __init__(self, row):
                self._row = row

            def fetchone(self_inner):
                return self_inner._row

        for needle, row in self._rows:
            if needle in sql_s:
                return R(row)
        return R(None)

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


class FakeEngine:
    def __init__(self, rows):
        self._rows = rows

    def connect(self):
        return FakeConn(self._rows)


def _resolver(grant_slug="acme", org_id="org-acme", plan=("pro", None)):
    grant_row = (grant_slug,) if grant_slug else None
    org_row = (org_id,) if org_id else None
    resolver = GrantResolver(
        supabase_engine=FakeEngine([("oauth_grants", grant_row), ("accounts", plan)]),
        tsdb_engine=FakeEngine([("organizations", org_row)]),
    )
    return resolver


def test_active_grant_resolves_principal():
    principal = _resolver().resolve(OAuthIdentity(user_id="u-1", client_id="c-1"))
    assert isinstance(principal, OAuthPrincipal)
    assert principal.org_id == "org-acme"
    assert principal.account_slug == "acme"
    assert principal.user_id == "u-1"
    assert principal.client_id == "c-1"
    assert principal.plan == "pro"


def test_no_grant_returns_none():
    assert (
        _resolver(grant_slug=None).resolve(
            OAuthIdentity(user_id="u-1", client_id="c-1")
        )
        is None
    )


def test_grant_with_unknown_account_returns_none():
    """Spec: a grant whose account_slug no longer resolves = revoked."""
    assert (
        _resolver(org_id=None).resolve(OAuthIdentity(user_id="u-1", client_id="c-1"))
        is None
    )


def test_lookup_error_fails_closed():
    class BoomEngine:
        def connect(self):
            raise RuntimeError("db down")

    resolver = GrantResolver(supabase_engine=BoomEngine(), tsdb_engine=BoomEngine())
    assert resolver.resolve(OAuthIdentity(user_id="u", client_id="c")) is None


def test_resolution_is_cached_for_ttl(monkeypatch):
    resolver = _resolver()
    calls = {"n": 0}
    real = resolver._lookup

    def counting(identity):
        calls["n"] += 1
        return real(identity)

    monkeypatch.setattr(resolver, "_lookup", counting)
    identity = OAuthIdentity(user_id="u-1", client_id="c-1")
    resolver.resolve(identity)
    resolver.resolve(identity)
    assert calls["n"] == 1  # second hit served from cache


def test_negative_result_is_cached(monkeypatch):
    """A no-grant miss is served from cache within the negative TTL."""
    resolver = _resolver(grant_slug=None)
    calls = {"n": 0}
    real = resolver._lookup

    def counting(identity):
        calls["n"] += 1
        return real(identity)

    monkeypatch.setattr(resolver, "_lookup", counting)
    identity = OAuthIdentity(user_id="u-1", client_id="c-1")
    assert resolver.resolve(identity) is None
    assert resolver.resolve(identity) is None
    assert calls["n"] == 1


def test_negative_cache_expires_faster_than_positive(monkeypatch):
    """Negative entries expire after 15s (not 60s) so a fresh consent
    becomes usable quickly."""
    import app.oauth as oauth_mod

    resolver = _resolver(grant_slug=None)
    calls = {"n": 0}
    real = resolver._lookup

    def counting(identity):
        calls["n"] += 1
        return real(identity)

    monkeypatch.setattr(resolver, "_lookup", counting)
    clock = {"now": 1000.0}
    monkeypatch.setattr(oauth_mod.time, "monotonic", lambda: clock["now"])
    identity = OAuthIdentity(user_id="u-1", client_id="c-1")
    resolver.resolve(identity)
    clock["now"] += 16.0  # past the 15s negative TTL, well inside 60s
    resolver.resolve(identity)
    assert calls["n"] == 2
