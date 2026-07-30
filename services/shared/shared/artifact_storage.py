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


# Bytes pulled per iteration when hashing an object. Large enough that a
# multi-GB object does not cost a million round trips through the generator,
# small enough that peak resident memory per concurrent confirm stays flat
# regardless of object size.
CHECKSUM_CHUNK_BYTES = 1024 * 1024


def head_size(s3: Any, key: str) -> Optional[int]:
    """Object size from ``ContentLength`` alone — no body transfer.

    This is the size-only half of :func:`head`, and it exists so a caller can
    enforce a size limit BEFORE paying to move the bytes. The confirm path
    hands out a presigned PUT with no server-side cap on what is actually
    uploaded, so a Free-plan org (10MB limit) can upload 5GB and then confirm.
    Measuring that by downloading it would mean the *rejection* path is the
    expensive one — an authenticated remote DoS from the cheapest tier, which
    quota by definition cannot stop. ``HEAD`` answers in one round trip and
    constant memory.

    Returns ``None`` when the object does not exist, which is how an
    unconfirmed upload is detected.

    Raises ClientError if the error is not a missing-object (404/NoSuchKey):
    transient failures (AccessDenied, throttling, internal errors) must surface
    as real failures, not silently tell users their upload failed.
    """
    try:
        response = s3.head_object(Bucket=ARTIFACT_BUCKET, Key=key)
    except ClientError as e:
        error_code = e.response.get("Error", {}).get("Code") if e.response else None
        if error_code in ("404", "NoSuchKey"):
            return None
        raise
    return int(response["ContentLength"])


def checksum(s3: Any, key: str, *, chunk_bytes: int = CHECKSUM_CHUNK_BYTES) -> str:
    """SHA-256 of a stored object, computed by streaming it in chunks.

    A client-supplied checksum could not be trusted and dedup depends on this
    one, so the bytes really do have to move — but they are hashed as they
    arrive and never accumulated. ``StreamingBody.iter_chunks`` is botocore's
    incremental reader; ``.read()`` would buffer the whole object and is the
    thing this function exists to avoid.
    """
    digest = hashlib.sha256()
    body = s3.get_object(Bucket=ARTIFACT_BUCKET, Key=key)["Body"]
    for chunk in body.iter_chunks(chunk_size=chunk_bytes):
        digest.update(chunk)
    return digest.hexdigest()


def head(s3: Any, key: str) -> Optional[dict]:
    """Server-side measurement of a stored object: size AND checksum.

    Kept for callers that want both in one go. It is now composed from
    :func:`head_size` and :func:`checksum`, so it never buffers the object
    either. Callers that must enforce a size limit should call the two
    separately and check the limit in between — paying for a full stream of an
    object that is about to be rejected is exactly the failure mode
    ``head_size`` exists to prevent.

    Returns ``None`` when the object does not exist. Raises ClientError for any
    non-404 failure, same contract as before.
    """
    size = head_size(s3, key)
    if size is None:
        return None
    return {"size_bytes": size, "checksum": checksum(s3, key)}


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
