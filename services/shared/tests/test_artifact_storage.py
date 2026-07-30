"""Object-storage helpers for artifacts, exercised against a fake S3 client.

The fake records calls and returns canned payloads — no network, no moto. This
matches how storage-worker tests fake their client.
"""

import hashlib
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

    url = st.presign_put(s3, "acme/a/v1")

    assert url == "https://example/put"
    kwargs = s3.generate_presigned_url.call_args.kwargs
    assert kwargs["ExpiresIn"] == 900
    assert kwargs["Params"]["Bucket"] == "klio-artifacts"
    assert kwargs["Params"]["Key"] == "acme/a/v1"


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


def test_get_text_decodes_utf8():
    s3 = _fake_s3()
    s3.get_object.return_value = {"Body": MagicMock(read=lambda: b"caf\xc3\xa9")}

    assert st.get_text(s3, "acme/a/v1") == "café"


def test_delete_is_scoped_to_the_bucket():
    s3 = _fake_s3()
    st.delete(s3, "acme/a/v1")
    s3.delete_object.assert_called_once_with(
        Bucket="klio-artifacts", Key="acme/a/v1"
    )
