"""Embeddings resolve their endpoint separately from chat.

This split exists because the two failure modes are not symmetric. A chat
model can be swapped between gateways freely. An embedding model cannot:
every vector already stored was produced by one specific model, and
pointing embeddings at a gateway that does not serve that model does not
fail loudly — it returns confidently wrong neighbours against vectors that
are no longer comparable.

The tests below are what stops someone "simplifying" the two back into one
variable later.
"""

import pytest

from shared.memory import _embed_endpoint


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for name in (
        "MEMORY_EMBEDDING_API_URL",
        "MEMORY_EMBEDDING_API_KEY",
        "LITELLM_API_URL",
        "LITELLM_API_KEY",
        "OPENAI_API_BASE",
        "OPENAI_API_KEY",
    ):
        monkeypatch.delenv(name, raising=False)


def test_it_falls_back_to_the_shared_gateway(monkeypatch):
    """An installation that never split them behaves exactly as before."""
    monkeypatch.setenv("LITELLM_API_URL", "https://shared.example/v1")
    monkeypatch.setenv("LITELLM_API_KEY", "shared-key")
    assert _embed_endpoint() == ("https://shared.example/v1", "shared-key")


def test_a_dedicated_embedding_endpoint_wins(monkeypatch):
    """The whole point: chat can move gateways while embeddings stay put."""
    monkeypatch.setenv("LITELLM_API_URL", "https://chat.example/v1")
    monkeypatch.setenv("LITELLM_API_KEY", "chat-key")
    monkeypatch.setenv("MEMORY_EMBEDDING_API_URL", "https://embed.example/v1")
    monkeypatch.setenv("MEMORY_EMBEDDING_API_KEY", "embed-key")
    assert _embed_endpoint() == ("https://embed.example/v1", "embed-key")


def test_the_url_and_key_are_independent(monkeypatch):
    """A split URL with no split key still uses the shared key.

    Not a hypothetical: the two gateways may share a credential, and
    forcing both to be set would make the safe change harder than the
    unsafe one.
    """
    monkeypatch.setenv("LITELLM_API_URL", "https://chat.example/v1")
    monkeypatch.setenv("LITELLM_API_KEY", "shared-key")
    monkeypatch.setenv("MEMORY_EMBEDDING_API_URL", "https://embed.example/v1")
    assert _embed_endpoint() == ("https://embed.example/v1", "shared-key")


def test_openai_variables_remain_the_last_resort(monkeypatch):
    monkeypatch.setenv("OPENAI_API_BASE", "https://api.openai.com/v1")
    monkeypatch.setenv("OPENAI_API_KEY", "openai-key")
    assert _embed_endpoint() == ("https://api.openai.com/v1", "openai-key")


def test_nothing_configured_is_not_an_error(monkeypatch):
    # litellm has its own defaults; this layer must not invent a URL.
    assert _embed_endpoint() == (None, None)
