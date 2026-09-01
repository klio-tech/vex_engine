"""Pins the cross-repo contract with the workspace's ``plane/utils/api_token.py``.

The two vectors below are the SAME literals the workspace's
``test_api_token.py`` asserts. If either implementation changes its KDF
salt, iteration count, or hash, one of the two suites goes red — which is
the whole point: drift must fail a test, not production auth.
"""

import pytest
from cryptography.fernet import Fernet

from shared.token_crypto import InvalidToken, decrypt_bot_token, derive_key, hash_token

KDF_VECTOR = "HYrhy7Rl4SRGw6S0q1iSE5An4SNXyE5nFOAlLs5uxBc="
SHA_VECTOR = "4e00342c5266cb198840876e3c8c83ee70ecb863693e24797c746f5dd977a5ee"


def test_hash_matches_the_workspace_stored_form():
    assert hash_token("klio_pat_" + "0" * 32) == SHA_VECTOR


def test_kdf_vector_matches_the_workspace():
    assert derive_key("klio-test-secret").decode() == KDF_VECTOR


def test_empty_secret_refuses():
    with pytest.raises(ValueError):
        derive_key("")


def test_decrypts_what_the_workspace_encrypted():
    # Encrypt exactly as the workspace does (Fernet under the derived key),
    # then decrypt through the engine path with the explicit secret.
    raw = "klio_bot_" + "ab" * 16
    sealed = Fernet(derive_key("klio-test-secret")).encrypt(raw.encode()).decode()
    assert decrypt_bot_token(sealed, secret="klio-test-secret") == raw


def test_wrong_secret_fails_closed_rather_than_guessing():
    sealed = Fernet(derive_key("klio-test-secret")).encrypt(b"klio_bot_x").decode()
    with pytest.raises(InvalidToken):
        decrypt_bot_token(sealed, secret="not-the-secret")


def test_reads_secret_from_env_when_not_given(monkeypatch):
    monkeypatch.setenv("BOT_TOKEN_CIPHER_KEY", "klio-test-secret")
    sealed = Fernet(derive_key("klio-test-secret")).encrypt(b"klio_bot_env").decode()
    assert decrypt_bot_token(sealed) == "klio_bot_env"
