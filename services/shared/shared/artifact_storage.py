"""Object storage for Klio artifacts.

The only artifact module that imports boto3. Kept separate from
:mod:`shared.artifacts` (which owns the rows) so the lifecycle tests never
need an S3 fake, mirroring how ``storage-worker`` separates ``s3_client`` from
``worker``.

The artifact bucket is deliberately NOT ``agentguard-traces``: traces are
internal telemetry, artifacts are customer data with different retention and a
different blast radius.
"""

import codecs
import hashlib
import os
from typing import Any, Optional

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError

ARTIFACT_BUCKET = os.environ.get("KLIO_ARTIFACT_BUCKET", "klio-artifacts")

# SigV4 must be pinned EXPLICITLY, not left to resolve.
#
# botocore reports ``client.meta.config.signature_version == 's3v4'`` even when
# the caller sets nothing — but it still emits a *SigV2* presigned URL
# (``AWSAccessKeyId``/``Signature``/``Expires``) in that case. Two consequences,
# both load-bearing:
#
#   1. MinIO rejects SigV2 presigned URLs outright (``SignatureDoesNotMatch``),
#      so the presigned upload and download paths do not work against the dev
#      or self-hosted stack at all without this.
#   2. SigV2 query auth has no ``SignedHeaders`` concept, so the
#      ``ContentLength`` bound in :func:`presign_put` would be silently dropped
#      and the per-URL size cap would be a no-op.
#
# Region is likewise pinned: presigning needs one, and inheriting it from the
# ambient AWS profile makes the signature depend on the operator's shell.
ARTIFACT_S3_REGION = os.environ.get("S3_REGION", "us-east-1")

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
        region_name=ARTIFACT_S3_REGION,
        config=Config(
            signature_version="s3v4",
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


def presign_put(s3: Any, key: str, *, content_length: int) -> str:
    """Presigned upload, capped at exactly ``content_length`` bytes.

    ``content_length`` is signed, not merely advertised. Under SigV4 it lands
    in the request's ``SignedHeaders`` as ``content-length``, so the storage
    layer recomputes the signature over the length the client actually sends: a
    PUT of any other size fails verification and is rejected with a 403 before
    the object is stored. (Verified against live MinIO, not just a mock — see
    ``test_artifact_e2e.test_presigned_put_is_capped_at_the_declared_size``.)

    This is what makes :data:`shared.artifacts.MAX_PENDING_PER_ORG` a real
    bound. Signing only ``Bucket`` and ``Key`` — as this did before — capped
    nothing server-side: every outstanding reservation was a URL through which
    an object of *any* size could be uploaded, so the cap bounded the number of
    objects but not a single byte of storage.

    ``content_length`` must be positive. Zero would sign a URL that accepts
    only an empty body, and both zero and negative mean the caller has no real
    declared size — exactly the case that must not reach a signature, because
    an unbounded URL is the thing this parameter exists to make impossible.
    """
    if content_length <= 0:
        raise ValueError(
            f"content_length must be positive, got {content_length}: a presigned "
            "PUT with no declared size is uncapped server-side"
        )
    return s3.generate_presigned_url(
        "put_object",
        Params={
            "Bucket": ARTIFACT_BUCKET,
            "Key": key,
            "ContentLength": content_length,
        },
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


# The longest a single character can be in UTF-8. Used to turn a CHARACTER
# window into a byte ``Range`` that is guaranteed to contain it.
UTF8_MAX_BYTES_PER_CHAR = 4

# Bytes pulled per iteration while decoding a text page. Small relative to a
# page so the early abort below actually saves a transfer, large enough that a
# page is a handful of round trips.
TEXT_STREAM_CHUNK_BYTES = 64 * 1024


def get_text_page(
    s3: Any,
    key: str,
    *,
    offset: int,
    limit: int,
    chunk_bytes: int = TEXT_STREAM_CHUNK_BYTES,
) -> tuple[str, bool]:
    """One bounded page of a stored text object: ``(page, has_more)``.

    Replaces a whole-object ``.read().decode()``. Serving 51,200 characters of
    a 900MB ``text/plain`` artifact used to buffer all 900MB plus its decode
    into the server's RAM, on the event loop, with no size gate — an
    authenticated caller on a plan that allows 1GB files could stall every
    other request on the replica just by reading page one.

    **Why the page cannot simply be a byte range.** The page unit is
    CHARACTERS (``artifacts.TEXT_PAGE_CHARS``) and ``Range`` is BYTES, and in
    UTF-8 the two only coincide for pure ASCII. Worse, a byte range starting at
    byte *N* tells you nothing about which character index sits there: recovering
    that requires decoding from an anchor. There is exactly one anchor in a
    stored object — byte 0 — so a correct character-addressed page must be
    decoded from the start of the object.

    **What makes that bounded anyway.** A character occupies at most
    ``UTF8_MAX_BYTES_PER_CHAR`` bytes, so the first ``offset + limit + 1``
    characters are certainly contained in the first
    ``4 * (offset + limit + 1)`` bytes. Requesting exactly that range gives a
    hard transfer ceiling proportional to the page, never to the object: page
    one of that 900MB artifact ranges over ~200KB. S3 clamps a range that
    overruns the object, so no special case is needed for short objects.

    Within the range the body is consumed through ``iter_chunks`` and fed to an
    incremental UTF-8 decoder, so:

    * peak memory is ``O(limit)`` — one page plus one chunk — no matter what
      ``offset`` is or how large the object is;
    * a character split across a chunk boundary is buffered by the decoder and
      completed on the next chunk, never mis-decoded;
    * a character split by the *range* boundary is left as undecoded trailing
      bytes and dropped, which is safe because the range is sized to end beyond
      the last character this page can return;
    * the stream is closed as soon as the page is filled, so the common case
      transfers far less than the ceiling and releases the connection early.

    Invalid UTF-8 raises, exactly as the strict ``.decode("utf-8")`` it
    replaces did: a text artifact that is not valid text is a real failure, not
    something to paper over with replacement characters.

    ``has_more`` is decided by reading one character PAST the page. If the
    range is exhausted without producing it, the object genuinely has no more
    characters — the range was sized so that character would have been inside
    it had it existed.
    """
    if offset < 0:
        raise ValueError(f"offset must be non-negative, got {offset}")
    if limit <= 0:
        raise ValueError(f"limit must be positive, got {limit}")

    # One character past the page: enough to decide has_more, never more.
    range_end = UTF8_MAX_BYTES_PER_CHAR * (offset + limit + 1) - 1
    try:
        body = s3.get_object(
            Bucket=ARTIFACT_BUCKET, Key=key, Range=f"bytes=0-{range_end}"
        )["Body"]
    except ClientError as e:
        error_code = e.response.get("Error", {}).get("Code") if e.response else None
        if error_code in ("InvalidRange", "416"):
            # A zero-length object: ``bytes=0-N`` has no satisfiable start.
            # An empty artifact is an empty page, not a failure.
            return "", False
        raise

    decoder = codecs.getincrementaldecoder("utf-8")()
    parts: list = []
    skipped = 0
    taken = 0
    has_more = False
    try:
        for chunk in body.iter_chunks(chunk_size=chunk_bytes):
            # ``final=False`` throughout: incomplete trailing bytes stay in the
            # decoder and are discarded with it, which is the correct handling
            # for a character the range cut in half.
            decoded = decoder.decode(chunk)
            if not decoded:
                continue

            cursor = 0
            if skipped < offset:
                skip = min(offset - skipped, len(decoded))
                skipped += skip
                cursor = skip

            wanted = limit - taken
            piece = decoded[cursor : cursor + wanted]
            if piece:
                parts.append(piece)
                taken += len(piece)

            if taken >= limit and cursor + len(piece) < len(decoded):
                # At least one character exists beyond the page.
                has_more = True
                break
    finally:
        # Abandoning a ranged read must release the connection rather than
        # leave it to the garbage collector to reap under load.
        close = getattr(body, "close", None)
        if close is not None:
            close()

    return "".join(parts), has_more


def delete(s3: Any, key: str) -> None:
    s3.delete_object(Bucket=ARTIFACT_BUCKET, Key=key)
