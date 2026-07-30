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


def test_head_returns_measured_size_and_checksum():
    body = b"hello world"
    s3 = _fake_s3()
    s3.head_object.return_value = {"ContentLength": len(body)}
    s3.get_object.return_value = {"Body": MagicMock(read=lambda: body)}

    meta = st.head(s3, "acme/a/v1")

    assert meta == {
        "size_bytes": len(body),
        "checksum": hashlib.sha256(body).hexdigest(),
    }


def test_head_returns_none_when_the_object_is_absent():
    s3 = _fake_s3()
    s3.head_object.side_effect = ClientError(
        {"Error": {"Code": "404", "Message": "Not Found"}}, "HeadObject"
    )

    assert st.head(s3, "acme/a/v1") is None


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
