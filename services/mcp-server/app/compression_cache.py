"""The originals cache: what a lossy compressor promised it could give back.

Structural compression (stage 1) is lossless and needs nothing here. This
exists so that the lossy compressors of stage 4 have a recovery contract
before they are allowed to drop a single byte.

**It is a tenancy boundary, not a convenience store.** An original is whatever
the caller compressed — a log, a source file, an API response — and is at
least as sensitive as a memory. Reads filter on ``org_id``, ``user_id`` and
expiry as SQL predicates, never by fetching and filtering in Python, and every
statement uses bound parameters.

**A miss is a miss.** ``load_original`` returns ``None`` for a handle that
never existed, one that expired, and one that belongs to somebody else. The
caller cannot tell those apart from the return value alone, which is the point:
distinguishing "not yours" from "no such thing" confirms the handle exists.

**The unattributed principal is a real principal.** ``resolve_request_user_id``
returns ``None`` by design so legacy API keys keep working, and that caller
calls ``compress`` like any other. Matching on ``user_id = :user`` is never
true against a NULL parameter, so such a caller was handed a handle it could
never redeem — after the lossy compressor had already dropped its content.
That is the exact broken promise this cache exists to prevent.

The predicate is therefore ``user_id IS NOT DISTINCT FROM :user``: NULL matches
NULL and nothing else. This does not widen tenancy, and the widening is what
the tests here attack rather than what the comment asserts:

* a NULL-user row is its own bucket inside one org. An attributed user's
  parameter is their own id, which is distinct from NULL, so they cannot read
  it; an unattributed caller's NULL is distinct from any real id, so they
  cannot read anybody's.
* the bucket is org-scoped, which is precisely how the identity design already
  treats everything an unattributed caller writes (see ``app/identity.py``):
  org-scoped, never private to a person who was never identified.
* redemption still requires the handle, which is salted by the full principal
  and by the content itself.

The alternative — refusing the lossy path outright for unattributed callers —
was rejected because it would take a working feature away from every legacy
key to close a hole that ``IS NOT DISTINCT FROM`` does not open.

**Why the handle is salted by (org_id, user_id) and not (org_id, content).**
The design spec says org-salted, so that one tenant's content can never satisfy
another's lookup. That is necessary but not sufficient. The handle is also the
row's primary key, so with an org-only salt two users in one org compressing
the same bytes collide: the first writer's row wins, and the second user reads
back nothing through a handle the API just handed them — locked out of their
own content by a dedup they cannot see. Adding ``user_id`` to the salt keeps
every property the spec asked for (deterministic, dedups within a tenant,
cannot collide across tenants) and removes that failure. Dedup is therefore
per (org, user), which is also the only scope in which a shared row would ever
be readable.

**Storage is inline for now.** The spec's inline/object-storage split mirrors
artifacts, and the column pair exists in migration 034 for it. Nothing in
stages 1-2 produces an original large enough to need the object path — the
lossy compressors that will are stage 4 — so the offload is deliberately not
built ahead of its first caller. ``store_original`` refuses an oversized
original rather than silently truncating it, because a cache that returns
something other than what went in is worse than one that declines.
"""

from __future__ import annotations

import hashlib
import logging
from typing import Any, Optional

import sqlalchemy as sa

logger = logging.getLogger("klio.compression.cache")

HANDLE_PREFIX = "kc_"
HANDLE_HEX_CHARS = 16

# Retention is 24 hours: retrieval happens within a session or not at all.
DEFAULT_TTL_SECONDS = 24 * 60 * 60

# Matches the artifacts inline ceiling (64KB). Beyond this an original belongs
# in object storage, which is stage 4's problem and is refused rather than
# truncated until then.
INLINE_MAX_BYTES = 64 * 1024


class OriginalTooLarge(ValueError):
    """The original exceeds what the inline path can hold."""


def make_handle(*, org_id: str, user_id: Optional[str], text: str) -> str:
    """Return the deterministic handle for this principal's copy of ``text``.

    Salted by the full principal so the same bytes dedup for one user within
    one org, and never collide across either boundary. See the module
    docstring for why ``user_id`` is in the salt.
    """
    digest = hashlib.sha256()
    # Length-prefixed so ("ab", "c") and ("a", "bc") cannot hash alike.
    for part in (org_id, user_id or "", text):
        encoded = part.encode("utf-8")
        digest.update(str(len(encoded)).encode("ascii"))
        digest.update(b":")
        digest.update(encoded)
    return HANDLE_PREFIX + digest.hexdigest()[:HANDLE_HEX_CHARS]


def store_original(
    db: Any,
    *,
    org_id: str,
    user_id: Optional[str],
    text: str,
    ttl_seconds: int = DEFAULT_TTL_SECONDS,
) -> str:
    """Store ``text`` and return the handle that recovers it.

    Idempotent: storing the same bytes as the same principal returns the same
    handle and refreshes the expiry, so a file an agent re-reads through a
    session stays recoverable for the whole of it rather than expiring on the
    first read's clock.

    Raises:
        OriginalTooLarge: the original does not fit the inline path.
    """
    size_bytes = len(text.encode("utf-8"))
    if size_bytes > INLINE_MAX_BYTES:
        raise OriginalTooLarge(
            f"original is {size_bytes} bytes; the inline cache holds {INLINE_MAX_BYTES}"
        )

    handle = make_handle(org_id=org_id, user_id=user_id, text=text)
    # content_hash is the unsalted digest: it is the dedup/diagnostic key for
    # "are these the same bytes", which the handle deliberately is not.
    content_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()

    db.execute(
        sa.text(
            """
            INSERT INTO compression_cache (
                id, org_id, user_id, content_hash, inline_text,
                size_bytes, expires_at
            ) VALUES (
                :id, :org, :user, :hash, :body,
                :size, NOW() + make_interval(secs => :ttl)
            )
            ON CONFLICT (id) DO UPDATE SET
                expires_at = EXCLUDED.expires_at
            """
        ),
        {
            "id": handle,
            "org": org_id,
            "user": user_id,
            "hash": content_hash,
            "body": text,
            "size": size_bytes,
            "ttl": ttl_seconds,
        },
    )
    db.commit()
    return handle


def load_original(
    db: Any,
    *,
    org_id: str,
    user_id: Optional[str],
    handle: str,
) -> Optional[str]:
    """Return the original behind ``handle``, or ``None``.

    ``None`` covers unknown, expired and not-yours alike — see the module
    docstring. The three predicates below are the boundary; removing any one
    of them is a privacy breach, not a bad result.

    ``user_id IS NOT DISTINCT FROM :user`` is NULL-aware equality: an
    unattributed caller matches only rows that are themselves unattributed,
    and an attributed one matches only their own. Neither degrades into
    match-anything. ``test_unattributed_caller_cannot_read_an_owned_entry``
    and ``test_an_attributed_user_cannot_read_an_unattributed_entry`` assert
    both directions rather than trusting this note.
    """
    row = db.execute(
        sa.text(
            """
            SELECT inline_text
            FROM compression_cache
            WHERE id = :handle
              AND org_id = :org
              AND user_id IS NOT DISTINCT FROM :user
              AND expires_at > NOW()
            """
        ),
        {"handle": handle, "org": org_id, "user": user_id},
    ).fetchone()
    if row is None:
        return None
    return row[0]


def handle_is_expired(
    db: Any,
    *,
    org_id: str,
    user_id: Optional[str],
    handle: str,
) -> bool:
    """Whether this principal's handle exists but has passed its expiry.

    Only ever consulted after :func:`load_original` returned ``None``, and
    only for a principal who owns the row — so it distinguishes "expired" from
    "never existed" for the owner without telling anybody else that somebody
    else's handle is real. Uses the same NULL-aware predicate, so the
    unattributed principal is told the truth about its own expiry too.
    """
    row = db.execute(
        sa.text(
            """
            SELECT 1
            FROM compression_cache
            WHERE id = :handle
              AND org_id = :org
              AND user_id IS NOT DISTINCT FROM :user
              AND expires_at <= NOW()
            """
        ),
        {"handle": handle, "org": org_id, "user": user_id},
    ).fetchone()
    return row is not None
