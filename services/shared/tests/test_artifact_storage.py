"""Object-storage helpers for artifacts, exercised against a fake S3 client.

The fake records calls and returns canned payloads — no network, no moto. This
matches how storage-worker tests fake their client.
"""

import hashlib
import urllib.parse
from unittest.mock import MagicMock

import pytest
from botocore.exceptions import ClientError

from shared import artifact_storage as st


def _fake_s3():
    return MagicMock()


def test_object_key_is_org_scoped_and_versioned():
    key = st.object_key("acme", "abc-123", 2)
    assert key == "acme/abc-123/v2"


def test_storage_uri_points_at_the_artifact_bucket():
    uri = st.storage_uri("acme", "abc-123", 1)
    assert uri == "s3://klio-artifacts/acme/abc-123/v1"


def test_presign_put_uses_the_short_ttl():
    s3 = _fake_s3()
    s3.generate_presigned_url.return_value = "https://example/put"

    url = st.presign_put(s3, "acme/a/v1", content_length=2048)

    assert url == "https://example/put"
    kwargs = s3.generate_presigned_url.call_args.kwargs
    assert kwargs["ExpiresIn"] == 900
    assert kwargs["Params"]["Bucket"] == "klio-artifacts"
    assert kwargs["Params"]["Key"] == "acme/a/v1"


def test_presign_put_binds_the_declared_size_into_the_signature():
    """The signed URL must cap what can be uploaded through it.

    ``MAX_PENDING_PER_ORG`` claims to bound an org's un-billed exposure to
    ``N x the plan's per-file limit``. That claim is only true if each
    presigned PUT is itself capped. Signing ``Bucket``/``Key`` alone caps
    nothing: the URL accepts an object of any size. Signing ``ContentLength``
    puts ``content-length`` into ``SignedHeaders``, so a PUT declaring any
    other length fails signature verification at the storage layer.
    """
    s3 = _fake_s3()
    s3.generate_presigned_url.return_value = "https://example/put"

    st.presign_put(s3, "acme/a/v1", content_length=2048)

    params = s3.generate_presigned_url.call_args.kwargs["Params"]
    assert params["ContentLength"] == 2048


@pytest.mark.parametrize("bad", [0, -1])
def test_presign_put_refuses_a_non_positive_content_length(bad):
    """An uncapped URL must be unrepresentable, not merely discouraged.

    ``ContentLength=0`` would sign a URL that only accepts an empty body, and a
    negative value is nonsense; both mean the caller has no real declared size,
    which is precisely the case that must not reach a signature.
    """
    with pytest.raises(ValueError):
        st.presign_put(_fake_s3(), "acme/a/v1", content_length=bad)


def test_artifact_client_presigns_with_sigv4():
    """SigV2 presigned URLs are rejected outright by MinIO (and by S3 in every
    region created after 2014), and SigV2 query auth has no ``SignedHeaders``
    concept at all — so a ``ContentLength`` bound to a SigV2 URL is silently
    ignored and the cap above becomes a no-op.

    Asserting on ``config.signature_version`` is NOT enough: botocore reports a
    resolved ``s3v4`` there while still emitting a SigV2 *presign* unless the
    version is set explicitly on the client's ``Config``. This asserts on the
    generated URL, which is the thing that actually has to be V4.
    """
    url = st.presign_put(st.get_artifact_s3_client(), "acme/a/v1", content_length=8)
    query = urllib.parse.parse_qs(urllib.parse.urlparse(url).query)

    assert "X-Amz-Signature" in query, f"not a SigV4 URL: {sorted(query)}"
    # ...and the size bound survives into the signature as a signed header.
    assert "content-length" in query["X-Amz-SignedHeaders"][0]


def test_presign_get_uses_the_short_ttl():
    s3 = _fake_s3()
    s3.generate_presigned_url.return_value = "https://example/get"

    assert st.presign_get(s3, "acme/a/v1") == "https://example/get"
    assert s3.generate_presigned_url.call_args.kwargs["ExpiresIn"] == 900


def test_presign_get_includes_response_content_headers():
    """Security test: presigned GET must force attachment disposition and octet-stream type."""
    s3 = _fake_s3()
    s3.generate_presigned_url.return_value = "https://example/get"

    st.presign_get(s3, "acme/a/v1")

    kwargs = s3.generate_presigned_url.call_args.kwargs
    params = kwargs["Params"]
    assert params["ResponseContentDisposition"] == "attachment"
    assert params["ResponseContentType"] == "application/octet-stream"


def _streaming_body(body: bytes, chunk_size: int = 4):
    """A stand-in for botocore's StreamingBody.

    ``read`` is deliberately made to blow up: any code path that pulls a whole
    object into RAM is a remote DoS (see ``head_size``'s docstring), so the
    fake refuses to serve one rather than quietly letting a regression pass.
    """

    def _no_read(*_args, **_kwargs):
        raise AssertionError(
            "the whole object was read into memory; stream it in chunks instead"
        )

    return MagicMock(
        read=_no_read,
        iter_chunks=lambda chunk_size=chunk_size: (
            body[i : i + chunk_size] for i in range(0, len(body), chunk_size)
        ),
    )


def test_head_returns_measured_size_and_checksum():
    body = b"hello world"
    s3 = _fake_s3()
    s3.head_object.return_value = {"ContentLength": len(body)}
    s3.get_object.return_value = {"Body": _streaming_body(body)}

    meta = st.head(s3, "acme/a/v1")

    assert meta == {
        "size_bytes": len(body),
        "checksum": hashlib.sha256(body).hexdigest(),
    }


# --- C1: size must be knowable WITHOUT downloading the object -------------


def test_head_size_never_fetches_the_body():
    """The per-file limit has to be enforceable before any transfer.

    A Free-plan org can request a presigned PUT, upload 5GB, and then confirm.
    If measuring the object requires downloading it, the server pulls all 5GB
    into memory *on the rejection path* — an authenticated remote DoS from the
    cheapest tier that quota cannot stop. ``head_size`` must answer from
    ``ContentLength`` alone.
    """
    s3 = _fake_s3()
    s3.head_object.return_value = {"ContentLength": 5 * 1024**3}

    assert st.head_size(s3, "acme/a/v1") == 5 * 1024**3
    s3.get_object.assert_not_called()


def test_head_size_returns_none_when_the_object_is_absent():
    s3 = _fake_s3()
    s3.head_object.side_effect = ClientError(
        {"Error": {"Code": "404", "Message": "Not Found"}}, "HeadObject"
    )

    assert st.head_size(s3, "acme/a/v1") is None


def test_head_size_raises_non_404_errors():
    s3 = _fake_s3()
    s3.head_object.side_effect = ClientError(
        {"Error": {"Code": "AccessDenied", "Message": "Access Denied"}}, "HeadObject"
    )

    with pytest.raises(ClientError):
        st.head_size(s3, "acme/a/v1")


def test_checksum_streams_the_body_in_chunks():
    """The checksum is computed incrementally, never over a buffered body."""
    body = b"hello world, streamed in pieces"
    s3 = _fake_s3()
    s3.get_object.return_value = {"Body": _streaming_body(body, chunk_size=3)}

    assert st.checksum(s3, "acme/a/v1") == hashlib.sha256(body).hexdigest()


def test_head_uses_the_streaming_checksum_path():
    """``head`` keeps its size+checksum contract, but must not buffer either.

    The fake's ``read`` raises, so this fails if ``head`` regresses to
    ``get_object(...)["Body"].read()``.
    """
    body = b"x" * 1024
    s3 = _fake_s3()
    s3.head_object.return_value = {"ContentLength": len(body)}
    s3.get_object.return_value = {"Body": _streaming_body(body, chunk_size=7)}

    assert st.head(s3, "acme/a/v1") == {
        "size_bytes": len(body),
        "checksum": hashlib.sha256(body).hexdigest(),
    }


def test_head_returns_none_when_the_object_is_absent():
    s3 = _fake_s3()
    s3.head_object.side_effect = ClientError(
        {"Error": {"Code": "404", "Message": "Not Found"}}, "HeadObject"
    )

    assert st.head(s3, "acme/a/v1") is None


def test_head_raises_non_404_errors():
    """Transient S3 failures must surface as real failures, not silent upload-absent."""
    s3 = _fake_s3()
    s3.head_object.side_effect = ClientError(
        {"Error": {"Code": "AccessDenied", "Message": "Access Denied"}}, "HeadObject"
    )

    with pytest.raises(ClientError):
        st.head(s3, "acme/a/v1")


def test_put_text_stores_and_returns_measured_metadata():
    s3 = _fake_s3()
    meta = st.put_text(s3, "acme/a/v1", "hey", "text/plain")

    assert meta == {
        "size_bytes": 3,
        "checksum": hashlib.sha256(b"hey").hexdigest(),
    }
    kwargs = s3.put_object.call_args.kwargs
    assert kwargs["Bucket"] == "klio-artifacts"
    assert kwargs["Key"] == "acme/a/v1"
    assert kwargs["Body"] == b"hey"
    assert kwargs["ContentType"] == "text/plain"


# --- C3: mode='text' must transfer roughly what it returns ----------------


class _RangedS3:
    """A fake S3 that honours ``Range`` and records what was actually asked for.

    ``iter_chunks`` yields only the bytes inside the requested range, so a test
    can assert on ``transferred`` — the number of bytes the server would really
    have pulled over the wire — rather than on the code's own bookkeeping.
    """

    def __init__(self, body: bytes, chunk_size: int = 8):
        self.body = body
        self.chunk_size = chunk_size
        self.ranges: list = []
        self.transferred = 0
        self.closed = False

    def get_object(self, *, Bucket, Key, Range=None, **_kwargs):
        self.ranges.append(Range)
        start, end = 0, len(self.body) - 1
        if Range is not None:
            assert Range.startswith("bytes="), Range
            raw_start, raw_end = Range[len("bytes=") :].split("-")
            start = int(raw_start)
            end = min(int(raw_end), len(self.body) - 1)
        window = self.body[start : end + 1]

        outer = self

        def _no_read(*_a, **_k):
            raise AssertionError(
                "the whole object was read into memory; stream it in chunks instead"
            )

        def _iter_chunks(chunk_size=outer.chunk_size):
            for i in range(0, len(window), chunk_size):
                piece = window[i : i + chunk_size]
                outer.transferred += len(piece)
                yield piece

        return {
            "Body": MagicMock(
                read=_no_read,
                iter_chunks=_iter_chunks,
                close=lambda: setattr(outer, "closed", True),
            )
        }


def test_get_text_page_returns_the_first_page_and_flags_more():
    s3 = _RangedS3(b"abcdefghijKLMNO")

    page, has_more = st.get_text_page(s3, "acme/a/v1", offset=0, limit=10)

    assert page == "abcdefghij"
    assert has_more is True


def test_get_text_page_returns_a_later_page_and_flags_the_end():
    s3 = _RangedS3(b"abcdefghijKLMNO")

    page, has_more = st.get_text_page(s3, "acme/a/v1", offset=10, limit=10)

    assert page == "KLMNO"
    assert has_more is False


def test_get_text_page_transfers_only_the_window_not_the_whole_object():
    """The finding this test exists for.

    ``get_text(...)`` did ``get_object(...)["Body"].read().decode("utf-8")``:
    an unbounded, synchronous whole-object read with no size gate anywhere.
    Top plans allow 1GB per file, so serving one 51,200-character page of a
    900MB ``text/plain`` artifact buffered all 900MB (plus a UTF-8 decode)
    into the server's RAM. The read must be bounded by the page window.
    """
    body = b"x" * (8 * 1024 * 1024)  # 8MB "artifact"
    s3 = _RangedS3(body, chunk_size=64 * 1024)

    page, has_more = st.get_text_page(s3, "acme/a/v1", offset=0, limit=1024)

    assert page == "x" * 1024
    assert has_more is True
    # A range was requested at all...
    assert s3.ranges and s3.ranges[0].startswith("bytes=0-")
    # ...and the bytes actually moved are on the order of the page, not the
    # object. Generous bound: 4x the page (worst-case UTF-8 expansion) plus one
    # chunk of read-ahead.
    assert s3.transferred <= 4 * 1024 + 64 * 1024
    assert s3.transferred < len(body) / 10


def test_get_text_page_never_buffers_the_whole_body():
    """``read()`` on the fake raises: a regression to ``.read()`` fails here."""
    s3 = _RangedS3(b"hello world", chunk_size=3)

    assert st.get_text_page(s3, "acme/a/v1", offset=0, limit=5) == ("hello", True)


def test_get_text_page_closes_the_stream_it_abandons():
    """Stopping early must release the connection, not leak it to GC."""
    s3 = _RangedS3(b"a" * 4096, chunk_size=16)

    st.get_text_page(s3, "acme/a/v1", offset=0, limit=8)

    assert s3.closed is True


def test_get_text_page_handles_multibyte_characters_at_both_edges():
    """The page unit is CHARACTERS; ``Range`` is BYTES.

    A byte range can slice a multi-byte character at either edge, and a naive
    ``bytes[offset:offset+limit].decode()`` would either raise or return
    mojibake. Every character here is 3 bytes, so byte and character offsets
    diverge immediately and a byte-indexed implementation returns the wrong
    window rather than merely a ragged one.
    """
    text = "日本語テキストのアーティファクト" * 40
    s3 = _RangedS3(text.encode("utf-8"), chunk_size=7)

    first, more_1 = st.get_text_page(s3, "acme/a/v1", offset=0, limit=25)
    second, more_2 = st.get_text_page(s3, "acme/a/v1", offset=25, limit=25)

    assert first == text[:25]
    assert more_1 is True
    assert second == text[25:50]
    assert more_2 is True
    # And the pages chain seamlessly — no dropped or duplicated character.
    assert first + second == text[:50]


def test_get_text_page_walks_a_multibyte_document_to_completion():
    """Paging to the end must terminate and reconstruct the document exactly."""
    text = "café ☕ 日本語 " * 500
    s3 = _RangedS3(text.encode("utf-8"), chunk_size=13)

    collected = ""
    offset = 0
    for _ in range(200):  # generous guard against a non-terminating loop
        page, has_more = st.get_text_page(s3, "acme/a/v1", offset=offset, limit=64)
        collected += page
        offset += len(page)
        if not has_more:
            break
    else:  # pragma: no cover - only reached if paging fails to terminate
        raise AssertionError("paging did not terminate")

    assert collected == text


def test_get_text_page_on_an_emoji_only_document():
    """4-byte characters are the worst case for the byte/character ratio."""
    text = "🙂🙃🚀🛰️" * 100
    s3 = _RangedS3(text.encode("utf-8"), chunk_size=5)

    page, has_more = st.get_text_page(s3, "acme/a/v1", offset=7, limit=11)

    assert page == text[7:18]
    assert has_more is True


def test_get_text_page_past_the_end_returns_empty_and_no_more():
    s3 = _RangedS3(b"short")

    assert st.get_text_page(s3, "acme/a/v1", offset=99, limit=10) == ("", False)


def test_get_text_page_treats_an_unsatisfiable_range_as_an_empty_page():
    """A zero-length object cannot satisfy ``bytes=0-N``.

    An empty artifact is an empty page, not an ``internal_error``.
    """
    s3 = MagicMock()
    s3.get_object.side_effect = ClientError(
        {"Error": {"Code": "InvalidRange", "Message": "Range Not Satisfiable"}},
        "GetObject",
    )

    assert st.get_text_page(s3, "acme/a/v1", offset=0, limit=10) == ("", False)


def test_get_text_page_still_raises_other_s3_errors():
    """AccessDenied or throttling must surface, not read as an empty artifact."""
    s3 = MagicMock()
    s3.get_object.side_effect = ClientError(
        {"Error": {"Code": "AccessDenied", "Message": "Access Denied"}}, "GetObject"
    )

    with pytest.raises(ClientError):
        st.get_text_page(s3, "acme/a/v1", offset=0, limit=10)


def test_get_text_page_rejects_a_negative_offset_or_empty_limit():
    s3 = _RangedS3(b"abc")
    with pytest.raises(ValueError):
        st.get_text_page(s3, "acme/a/v1", offset=-1, limit=10)
    with pytest.raises(ValueError):
        st.get_text_page(s3, "acme/a/v1", offset=0, limit=0)


def test_unbounded_whole_object_text_read_is_not_available_at_all():
    """The footgun is removed, not merely unused.

    A helper that pulls an entire artifact into RAM has no safe caller on a
    multi-tenant request path; leaving it in the module is how it comes back.
    """
    assert not hasattr(st, "get_text")


def test_delete_is_scoped_to_the_bucket():
    s3 = _fake_s3()
    st.delete(s3, "acme/a/v1")
    s3.delete_object.assert_called_once_with(Bucket="klio-artifacts", Key="acme/a/v1")


# ---------------------------------------------------------------------------
# Credential resolution
#
# Artifacts and Vex traces live in different buckets with different blast
# radii. storage-worker reads the shared S3_* names for traces, so the artifact
# client must be able to take a DIFFERENT credential — otherwise the two share
# one key by construction and bucket-scoping the tokens buys nothing.
# ---------------------------------------------------------------------------

_S3_VARS = (
    "KLIO_ARTIFACT_S3_ENDPOINT",
    "KLIO_ARTIFACT_S3_ACCESS_KEY",
    "KLIO_ARTIFACT_S3_SECRET_KEY",
    "KLIO_ARTIFACT_S3_REGION",
    "S3_ENDPOINT",
    "S3_ACCESS_KEY",
    "S3_SECRET_KEY",
    "S3_REGION",
)


@pytest.fixture
def clean_s3_env(monkeypatch):
    """Remove every S3 variable so each test states its own environment."""
    for var in _S3_VARS:
        monkeypatch.delenv(var, raising=False)
    return monkeypatch


def _client_settings(client):
    return {
        "endpoint": client.meta.endpoint_url,
        "region": client.meta.region_name,
    }


def test_artifact_vars_win_over_the_shared_ones(clean_s3_env):
    """A dedicated artifact credential must not be shadowed by the traces one."""
    clean_s3_env.setenv("S3_ENDPOINT", "https://traces.example")
    clean_s3_env.setenv("S3_ACCESS_KEY", "traces-key")
    clean_s3_env.setenv("S3_SECRET_KEY", "traces-secret")
    clean_s3_env.setenv("KLIO_ARTIFACT_S3_ENDPOINT", "https://artifacts.example")
    clean_s3_env.setenv("KLIO_ARTIFACT_S3_ACCESS_KEY", "artifact-key")
    clean_s3_env.setenv("KLIO_ARTIFACT_S3_SECRET_KEY", "artifact-secret")

    assert (
        st._artifact_s3_setting("ENDPOINT", "fallback") == "https://artifacts.example"
    )
    assert st._artifact_s3_setting("ACCESS_KEY", "fallback") == "artifact-key"
    assert st._artifact_s3_setting("SECRET_KEY", "fallback") == "artifact-secret"

    client = st.get_artifact_s3_client()
    assert _client_settings(client)["endpoint"] == "https://artifacts.example"


def test_falls_back_to_the_shared_vars_when_no_artifact_var_is_set(clean_s3_env):
    """Existing single-bucket deployments must keep working with no new config."""
    clean_s3_env.setenv("S3_ENDPOINT", "https://shared.example")
    clean_s3_env.setenv("S3_ACCESS_KEY", "shared-key")
    clean_s3_env.setenv("S3_SECRET_KEY", "shared-secret")

    assert st._artifact_s3_setting("ENDPOINT", "fallback") == "https://shared.example"
    assert st._artifact_s3_setting("ACCESS_KEY", "fallback") == "shared-key"
    assert st.get_artifact_s3_client().meta.endpoint_url == "https://shared.example"


def test_falls_back_to_the_local_dev_defaults_when_nothing_is_set(clean_s3_env):
    """docker-compose MinIO keeps working with an empty environment."""
    assert st._artifact_s3_setting("ENDPOINT", "http://localhost:9000") == (
        "http://localhost:9000"
    )
    assert st.get_artifact_s3_client().meta.endpoint_url == "http://localhost:9000"


def test_an_empty_artifact_var_does_not_shadow_the_shared_one(clean_s3_env):
    """An unset-but-present variable is a deploy footgun; treat blank as absent."""
    clean_s3_env.setenv("KLIO_ARTIFACT_S3_ENDPOINT", "")
    clean_s3_env.setenv("S3_ENDPOINT", "https://shared.example")

    assert st._artifact_s3_setting("ENDPOINT", "fallback") == "https://shared.example"


def test_region_prefers_the_artifact_var(clean_s3_env):
    """R2 needs region 'auto'; AWS-shaped default must be overridable per client."""
    import importlib

    clean_s3_env.setenv("S3_REGION", "us-east-1")
    clean_s3_env.setenv("KLIO_ARTIFACT_S3_REGION", "auto")
    reloaded = importlib.reload(st)
    try:
        assert reloaded.ARTIFACT_S3_REGION == "auto"
        assert reloaded.get_artifact_s3_client().meta.region_name == "auto"
    finally:
        for var in _S3_VARS:
            clean_s3_env.delenv(var, raising=False)
        importlib.reload(st)
