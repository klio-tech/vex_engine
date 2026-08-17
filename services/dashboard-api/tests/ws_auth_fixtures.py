"""Cross-language test fixture for the WebSocket handshake token format.

NOT A CREDENTIAL. ``secret`` below is a fixture value that exists only to
prove the Python verifier (``app.ws_auth``) and the TypeScript minter
(Dashboard repo, ``apps/web/lib/agentguard/ws-token.fixture.ts``) produce
byte-for-byte identical tokens for the same input. It signs nothing that runs
in production; the real signing secret is supplied at runtime only via the
environment variable named by ``app.config.WS_TOKEN_SECRET_ENV``.

Both suites assert this exact ``token`` string, so a change to encoding,
padding, separator or field order on either side goes red in CI instead of
reaching production as "the live charts stopped updating". Do not delete this
cross-language drift check, and keep this file's values identical to the
Dashboard repo's ``WS_TOKEN_TEST_VECTOR``.

This lives in ``tests/``, not ``app/ws_auth.py``, so its sole purpose --
supplying test data -- is unambiguous, and so it can be named as a narrow,
explained exception in ``.gitguardian.yaml`` rather than requiring a blanket
test-file exclusion (real credentials do leak into test files).
"""

from __future__ import annotations

#: Fixed vector mirrored by the Dashboard repo's ws-token fixture. Both sides
#: assert this exact string. If either implementation changes its encoding,
#: padding, separator or field order, one of the two suites goes red before
#: the mismatch can reach production as "the charts stopped updating".
TEST_VECTOR = {
    "secret": "test-secret-do-not-use",
    "org_id": "11111111-2222-3333-4444-555555555555",
    "exp": 1700000000,
    "token": (
        "v1.MTExMTExMTEtMjIyMi0zMzMzLTQ0NDQtNTU1NTU1NTU1NTU1.1700000000"
        ".ixLU_-TiTdxIbjLicOSP5PsoWzB6qPN406MI7yiLlq8"
    ),
}
