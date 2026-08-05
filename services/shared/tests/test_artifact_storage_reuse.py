"""The S3 client is reused, and every streamed body is closed.

Two small availability costs found while verifying the larger findings.

**The client was never cached.** ``get_artifact_s3_client()`` was called fresh
at ``mcp_app.py:697, 976, 1067, 1116`` and ``scheduler.py:267``, so every
artifact operation built a new botocore client and paid a fresh TLS handshake
to the object store. botocore clients are safe to reuse and are the intended
unit of connection pooling.

**``checksum`` leaked its stream.** ``get_text_page`` closes its
``StreamingBody`` in a ``finally`` with a comment explaining why — "abandoning
a ranged read must release the connection rather than leave it to the garbage
collector to reap under load". ``checksum`` streams a whole object and did not,
so under load its connections were released only when the GC got round to them.
"""

import threading
from unittest.mock import MagicMock, patch

import pytest
from shared import artifact_storage


@pytest.fixture(autouse=True)
def _reset_client_cache():
    artifact_storage.reset_artifact_s3_client()
    yield
    artifact_storage.reset_artifact_s3_client()


# ── Client reuse ──────────────────────────────────────────────────────


def test_the_client_is_built_once_and_reused():
    with patch.object(artifact_storage.boto3, "client") as factory:
        factory.side_effect = lambda *a, **k: MagicMock()
        first = artifact_storage.get_artifact_s3_client()
        second = artifact_storage.get_artifact_s3_client()

    assert first is second
    assert factory.call_count == 1, f"the client was rebuilt {factory.call_count} times"


def test_the_client_is_still_configured_the_same_way():
    """Positive control: caching must not drop endpoint, creds or timeouts."""
    with patch.object(artifact_storage.boto3, "client") as factory:
        factory.return_value = MagicMock()
        artifact_storage.get_artifact_s3_client()

    args, kwargs = factory.call_args
    assert args[0] == "s3"
    assert "endpoint_url" in kwargs
    assert kwargs["aws_access_key_id"]
    assert kwargs["aws_secret_access_key"]
    config = kwargs["config"]
    assert config.connect_timeout == 5
    assert config.read_timeout == 30
    assert config.retries == {"max_attempts": 3, "mode": "adaptive"}


def test_concurrent_callers_share_one_client():
    """The cache must not race into two clients under a burst."""
    built = []

    def _slow_factory(*a, **k):
        import time

        time.sleep(0.02)
        client = MagicMock()
        built.append(client)
        return client

    results = []
    with patch.object(artifact_storage.boto3, "client", _slow_factory):
        threads = [
            threading.Thread(
                target=lambda: results.append(artifact_storage.get_artifact_s3_client())
            )
            for _ in range(8)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

    assert len(built) == 1, f"{len(built)} clients were built"
    assert len({id(r) for r in results}) == 1


def test_a_failed_build_is_not_cached():
    """An idle deployment with no credentials must retry, not cache the error."""
    calls = []

    def _factory(*a, **k):
        calls.append(1)
        if len(calls) == 1:
            raise RuntimeError("no credentials")
        return MagicMock()

    with patch.object(artifact_storage.boto3, "client", _factory):
        with pytest.raises(RuntimeError):
            artifact_storage.get_artifact_s3_client()
        assert artifact_storage.get_artifact_s3_client() is not None
    assert len(calls) == 2


# ── Stream closing ────────────────────────────────────────────────────


class _Body:
    """A StreamingBody stand-in that records whether it was closed."""

    def __init__(self, chunks):
        self._chunks = chunks
        self.closed = False

    def iter_chunks(self, chunk_size=None):
        yield from self._chunks

    def close(self):
        self.closed = True


def test_checksum_closes_its_stream():
    body = _Body([b"hello ", b"world"])
    s3 = MagicMock()
    s3.get_object.return_value = {"Body": body}

    digest = artifact_storage.checksum(s3, "org/a/v1")

    import hashlib

    assert digest == hashlib.sha256(b"hello world").hexdigest(), (
        "positive control: the checksum is still correct"
    )
    assert body.closed, "the StreamingBody was left for the garbage collector"


def test_checksum_closes_its_stream_even_when_hashing_raises():
    class _Exploding(_Body):
        def iter_chunks(self, chunk_size=None):
            yield b"partial"
            raise OSError("connection reset mid-stream")

    body = _Exploding([])
    s3 = MagicMock()
    s3.get_object.return_value = {"Body": body}

    with pytest.raises(OSError):
        artifact_storage.checksum(s3, "org/a/v1")
    assert body.closed, "an aborted stream must still release its connection"


def test_checksum_tolerates_a_body_with_no_close():
    """Same defensive shape ``get_text_page`` uses for non-botocore fakes."""

    class _NoClose:
        def iter_chunks(self, chunk_size=None):
            yield b"x"

    s3 = MagicMock()
    s3.get_object.return_value = {"Body": _NoClose()}
    assert artifact_storage.checksum(s3, "org/a/v1")
