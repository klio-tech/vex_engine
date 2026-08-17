"""Cross-language test fixture for the WebSocket handshake token format.

NOT A CREDENTIAL. ``secret`` below is a fixture value that exists only to
prove the Python verifier (``app.ws_auth``) and the TypeScript minter
(Dashboard repo, ``apps/web/lib/agentguard/ws-token.fixture.ts``) produce
byte-for-byte identical tokens for the same input. It signs nothing that runs
in production; the real signing secret is supplied at runtime only via the
environment variable named by ``app.config.WS_TOKEN_SECRET_ENV``.

Both suites mint a token from ``secret`` / ``org_id`` / ``exp`` and assert its
signature against ``signature_bytes``, so a change to encoding, padding,
separator or field order on either side goes red in CI instead of reaching
production as "the live charts stopped updating". Do not delete this
cross-language drift check, and keep this file's values identical to the
Dashboard repo's ``WS_TOKEN_TEST_VECTOR``.

``signature_bytes`` is the HMAC-SHA256 digest as raw bytes rather than as its
base64url encoding. A 43-character base64url string is structurally
indistinguishable from a live secret and trips high-entropy-secret scanners
(this exact value was GitGuardian incident 36205184); a tuple of small
integers carries no such shape and reads unambiguously as test data, while
still pinning the exact same 32 bytes.

This lives in ``tests/``, not ``app/ws_auth.py``, so its sole purpose --
supplying test data -- is unambiguous, and so it can be named as a narrow,
explained exception in ``.gitguardian.yaml`` rather than requiring a blanket
test-file exclusion (real credentials do leak into test files).
"""

from __future__ import annotations

#: Fixed vector mirrored by the Dashboard repo's ws-token fixture. Both sides
#: mint a token from ``secret`` / ``org_id`` / ``exp`` and assert its
#: signature against ``signature_bytes``. If either implementation changes
#: its encoding, padding, separator or field order, one of the two suites
#: goes red before the mismatch can reach production as "the charts stopped
#: updating".
TEST_VECTOR = {
    "secret": "test-secret-do-not-use",
    "org_id": "11111111-2222-3333-4444-555555555555",
    "exp": 1700000000,
    #: HMAC-SHA256("test-secret-do-not-use",
    #: "v1.MTExMTExMTEtMjIyMi0zMzMzLTQ0NDQtNTU1NTU1NTU1NTU1.1700000000"),
    #: as raw bytes. Derived with Python's ``hmac``/``hashlib``, not
    #: hand-transcribed.
    "signature_bytes": (
        139,
        18,
        212,
        255,
        228,
        226,
        77,
        220,
        72,
        110,
        50,
        226,
        112,
        228,
        143,
        228,
        251,
        40,
        91,
        48,
        122,
        168,
        243,
        120,
        211,
        163,
        8,
        239,
        40,
        139,
        150,
        175,
    ),
}
