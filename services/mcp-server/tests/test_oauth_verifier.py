"""SupabaseTokenVerifier: local JWT validation against a JWKS fixture."""

import time
import uuid

import jwt as pyjwt
import pytest
from cryptography.hazmat.primitives.asymmetric import rsa

from app.oauth import OAuthIdentity, SupabaseTokenVerifier

ISSUER = "https://proj.supabase.co/auth/v1"
KID = "test-key-1"


@pytest.fixture(scope="module")
def rsa_key():
    return rsa.generate_private_key(public_exponent=65537, key_size=2048)


@pytest.fixture
def verifier(rsa_key, monkeypatch):
    v = SupabaseTokenVerifier(issuer=ISSUER)
    # Bypass the network: seed the JWKS cache from the fixture key.
    public_jwk = pyjwt.algorithms.RSAAlgorithm.to_jwk(
        rsa_key.public_key(), as_dict=True
    )
    public_jwk["kid"] = KID
    public_jwk["alg"] = "RS256"
    monkeypatch.setattr(v, "_fetch_jwks", lambda: {"keys": [public_jwk]})
    return v


def _mint(rsa_key, **overrides) -> str:
    now = int(time.time())
    claims = {
        "iss": ISSUER,
        "sub": str(uuid.uuid4()),
        "client_id": "client-abc",
        "aud": "authenticated",
        "iat": now,
        "exp": now + 3600,
    }
    claims.update(overrides)
    return pyjwt.encode(claims, rsa_key, algorithm="RS256", headers={"kid": KID})


async def test_valid_token_yields_identity(verifier, rsa_key):
    token = _mint(rsa_key, sub="11111111-2222-3333-4444-555555555555")
    identity = await verifier.verify_token(token)
    assert isinstance(identity, OAuthIdentity)
    assert identity.user_id == "11111111-2222-3333-4444-555555555555"
    assert identity.client_id == "client-abc"


async def test_expired_token_rejected(verifier, rsa_key):
    # Use an offset well beyond the 30s leeway so the token is truly expired.
    token = _mint(rsa_key, exp=int(time.time()) - 60)
    assert await verifier.verify_token(token) is None


async def test_wrong_issuer_rejected(verifier, rsa_key):
    token = _mint(rsa_key, iss="https://evil.example.com/auth/v1")
    assert await verifier.verify_token(token) is None


async def test_garbage_token_rejected(verifier):
    assert await verifier.verify_token("not-a-jwt") is None


async def test_token_without_client_id_rejected(verifier, rsa_key):
    """A plain dashboard session JWT (no OAuth client) must NOT authenticate
    the MCP endpoint — only tokens minted through the OAuth client flow do."""
    # Re-mint without client_id (and without azp).
    import time as _t

    now = int(_t.time())
    claims = {
        "iss": ISSUER,
        "sub": str(uuid.uuid4()),
        "aud": "authenticated",
        "iat": now,
        "exp": now + 3600,
    }
    token = pyjwt.encode(claims, rsa_key, algorithm="RS256", headers={"kid": KID})
    assert await verifier.verify_token(token) is None


async def test_unknown_kid_triggers_refetch_then_rejects(
    verifier, rsa_key, monkeypatch
):
    calls = {"n": 0}
    real = verifier._fetch_jwks

    def counting():
        calls["n"] += 1
        return real()

    monkeypatch.setattr(verifier, "_fetch_jwks", counting)
    token = pyjwt.encode(
        {
            "iss": ISSUER,
            "sub": "u",
            "client_id": "c",
            "iat": int(time.time()),
            "exp": int(time.time()) + 60,
        },
        rsa_key,
        algorithm="RS256",
        headers={"kid": "unknown-kid"},
    )
    assert await verifier.verify_token(token) is None
    assert calls["n"] >= 1  # refreshed on miss


async def test_jwks_fetch_failure_fails_closed(rsa_key, monkeypatch):
    v = SupabaseTokenVerifier(issuer=ISSUER)

    def boom():
        raise RuntimeError("network down")

    monkeypatch.setattr(v, "_fetch_jwks", boom)
    token = _mint(rsa_key)
    assert await v.verify_token(token) is None


async def test_unknown_kid_flood_is_backoff_limited(verifier, rsa_key, monkeypatch):
    """A flood of bad-kid tokens must not force a JWKS fetch per request."""
    calls = {"n": 0}
    real = verifier._fetch_jwks

    def counting():
        calls["n"] += 1
        return real()

    monkeypatch.setattr(verifier, "_fetch_jwks", counting)

    for i in range(20):
        token = pyjwt.encode(
            {
                "iss": ISSUER,
                "sub": "u",
                "client_id": "c",
                "iat": int(time.time()),
                "exp": int(time.time()) + 60,
            },
            rsa_key,
            algorithm="RS256",
            headers={"kid": f"bad-{i}"},
        )
        assert await verifier.verify_token(token) is None

    # cold load + at most one forced refresh within the backoff window
    assert calls["n"] <= 2


async def test_valid_token_still_verifies_after_backoff_engaged(verifier, rsa_key):
    """A bad-kid token engaging the backoff must not break subsequent valid tokens."""
    bad = pyjwt.encode(
        {
            "iss": ISSUER,
            "sub": "u",
            "client_id": "c",
            "iat": int(time.time()),
            "exp": int(time.time()) + 60,
        },
        rsa_key,
        algorithm="RS256",
        headers={"kid": "bad-x"},
    )
    assert await verifier.verify_token(bad) is None

    good = _mint(rsa_key)
    identity = await verifier.verify_token(good)
    assert identity is not None
