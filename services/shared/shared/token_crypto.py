"""Workspace API-token hashing and the bot-token cipher — the engine's half.

The workspace (Klio's Django backend) stopped storing API tokens in
plaintext on 2026-09-02. ``api_tokens.token_hash`` now holds the SHA-256 of
the raw token, and only BOT tokens keep a reversible copy
(``token_encrypted``, Fernet) so this engine can present an agent's own
credential to the workspace REST API on the agent's behalf.

Both halves are a CONTRACT with ``apps/api/plane/utils/api_token.py`` in
the workspace repo:

- :func:`hash_token` must produce exactly what the workspace stores, or
  every PAT lookup silently 401s.
- :func:`derive_key` must derive exactly the workspace's Fernet key from
  the shared secret, or every agent turn fails to decrypt its token.

Each side's test suite pins the same fixed vectors (``KDF_VECTOR``,
``SHA_VECTOR``) so an edit to one implementation fails the other's tests
rather than production.

The secret is a DEDICATED one, ``BOT_TOKEN_CIPHER_KEY``, set to the same
value in both services' environments. It is deliberately not one of the
existing internal-call secrets: ``INTERNAL_API_SECRET`` (engine) and
``LIVE_SERVER_SECRET_KEY`` (workspace) were assumed to be one value and
turned out, in production, to be two — reusing either would have derived
two different keys and broken every agent turn silently.
"""

from __future__ import annotations

import base64
import hashlib
import os
from functools import lru_cache

# ``cryptography`` is imported lazily, inside the decrypt path only.
# ``shared.auth`` imports this module for :func:`hash_token` in EVERY
# service, but only the async-worker ever decrypts a bot token — an
# import-time dependency here would turn a missing wheel in any other
# service's image into a crash at startup rather than a clear error at the
# one call site that actually needs it. ``InvalidToken`` is re-exported
# lazily for the same reason.
try:
    from cryptography.fernet import Fernet, InvalidToken
except ImportError:  # pragma: no cover - exercised only in images without the wheel
    Fernet = None  # type: ignore[assignment]

    class InvalidToken(Exception):  # type: ignore[no-redef]
        """Stand-in so callers can still ``except InvalidToken`` — the decrypt
        path itself raises ``RuntimeError`` naming the missing dependency."""


_KDF_SALT = b"klio-bot-token-v1"
_KDF_ITERATIONS = 200_000
SECRET_ENV = "BOT_TOKEN_CIPHER_KEY"

__all__ = [
    "InvalidToken",
    "SECRET_ENV",
    "decrypt_bot_token",
    "derive_key",
    "hash_token",
]


def hash_token(raw: str) -> str:
    """SHA-256 hex of a raw token — the workspace's stored form."""
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


@lru_cache(maxsize=8)
def derive_key(secret: str) -> bytes:
    """Fernet key from the shared secret. Byte-identical to the workspace's.

    Cached: 200k PBKDF2 rounds cost ~100ms, and an agent turn would
    otherwise pay that on every decrypt. Keyed by the secret, so a rotated
    secret derives fresh.
    """
    if not secret:
        raise ValueError("bot-token cipher secret is empty")
    dk = hashlib.pbkdf2_hmac(
        "sha256", secret.encode("utf-8"), _KDF_SALT, _KDF_ITERATIONS
    )
    return base64.urlsafe_b64encode(dk)


def decrypt_bot_token(encrypted: str, *, secret: str | None = None) -> str:
    """Recover a bot's raw token from ``api_tokens.token_encrypted``.

    Raises ``InvalidToken`` for a wrong secret or tampered ciphertext and
    ``ValueError`` for a missing secret — never returns a guess, because a
    guessed credential would be sent to the workspace as ``X-Api-Key``.
    """
    if Fernet is None:
        raise RuntimeError(
            "decrypting a bot token requires the 'cryptography' package; "
            "install services/shared with its declared dependencies"
        )
    key = derive_key(os.environ.get(SECRET_ENV, "") if secret is None else secret)
    return Fernet(key).decrypt(encrypted.encode("ascii")).decode("utf-8")
