"""Object storage for Klio artifacts.

The only artifact module that imports boto3. Kept separate from
:mod:`shared.artifacts` (which owns the rows) so the lifecycle tests never
need an S3 fake, mirroring how ``storage-worker`` separates ``s3_client`` from
``worker``.

The artifact bucket is deliberately NOT ``agentguard-traces``: traces are
internal telemetry, artifacts are customer data with different retention and a
different blast radius.
"""

import hashlib
import os
from typing import Any, Optional

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError

ARTIFACT_BUCKET = os.environ.get("KLIO_ARTIFACT_BUCKET", "klio-artifacts")

# Presigned URLs are bearer credentials: anyone holding one can read or write
# that single object. Short TTL is the mitigation. Never log them.
PRESIGN_TTL_SECONDS = 900

# Above this declared size a single PUT is unreliable, so the caller is handed
# multipart part URLs instead.
MULTIPART_THRESHOLD_BYTES = 100 * 1024 * 1024


def get_artifact_s3_client() -> Any:
    """Return a boto3 S3 client configured for the artifact bucket."""
    return boto3.client(
        "s3",
        endpoint_url=os.environ.get("S3_ENDPOINT", "http://localhost:9000"),
        aws_access_key_id=os.environ.get("S3_ACCESS_KEY", "agentguard"),
        aws_secret_access_key=os.environ.get("S3_SECRET_KEY", "agentguard_dev"),
        config=Config(
            retries={"max_attempts": 3, "mode": "adaptive"},
            connect_timeout=5,
            read_timeout=30,
        ),
    )


def object_key(org_id: str, artifact_id: str, version: int) -> str:
    """Bytes live under the owning org, so a key is never ambiguous."""
    return f"{org_id}/{artifact_id}/v{version}"


def storage_uri(org_id: str, artifact_id: str, version: int) -> str:
    return f"s3://{ARTIFACT_BUCKET}/{object_key(org_id, artifact_id, version)}"


def presign_put(s3: Any, key: str) -> str:
    return s3.generate_presigned_url(
        "put_object",
        Params={"Bucket": ARTIFACT_BUCKET, "Key": key},
        ExpiresIn=PRESIGN_TTL_SECONDS,
    )


def presign_get(s3: Any, key: str) -> str:
    """Presigned download.

    The response headers are forced on the URL itself: Klio never executes an
    artifact, and a browser following this link must not be talked into
    rendering someone's uploaded HTML as a page on our origin.
    """
    return s3.generate_presigned_url(
        "get_object",
        Params={
            "Bucket": ARTIFACT_BUCKET,
            "Key": key,
            "ResponseContentDisposition": "attachment",
            "ResponseContentType": "application/octet-stream",
        },
        ExpiresIn=PRESIGN_TTL_SECONDS,
    )


def _measure(body: bytes) -> dict:
    return {"size_bytes": len(body), "checksum": hashlib.sha256(body).hexdigest()}


def head(s3: Any, key: str) -> Optional[dict]:
    """Server-side measurement of a stored object.

    Returns ``None`` when the object does not exist, which is how an
    unconfirmed upload is detected. The body is read to compute the checksum:
    a client-supplied one could not be trusted, and dedup depends on it.

    Raises ClientError if the error is not a missing-object (404/NoSuchKey):
    transient failures (AccessDenied, throttling, internal errors) must surface
    as real failures, not silently tell users their upload failed.
    """
    try:
        s3.head_object(Bucket=ARTIFACT_BUCKET, Key=key)
    except ClientError as e:
        error_code = e.response.get("Error", {}).get("Code") if e.response else None
        if error_code in ("404", "NoSuchKey"):
            return None
        raise

    body = s3.get_object(Bucket=ARTIFACT_BUCKET, Key=key)["Body"].read()
    return _measure(body)


def put_text(s3: Any, key: str, text: str, mime_type: str) -> dict:
    body = text.encode("utf-8")
    s3.put_object(
        Bucket=ARTIFACT_BUCKET,
        Key=key,
        Body=body,
        ContentType=mime_type,
        ContentDisposition="attachment",
    )
    return _measure(body)


def get_text(s3: Any, key: str) -> str:
    return s3.get_object(Bucket=ARTIFACT_BUCKET, Key=key)["Body"].read().decode("utf-8")


def delete(s3: Any, key: str) -> None:
    s3.delete_object(Bucket=ARTIFACT_BUCKET, Key=key)
