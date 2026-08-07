"""A deterministic stand-in for the embedding backend, plus the corpus that
demonstrates why a lexical arm is needed at all.

**Why a stand-in and not the real embedder.** ``shared.memory.embed_text``
calls ``litellm.aembedding()``. A test suite that reached a hosted model would
be neither hermetic nor reproducible, and the assertion here is about *which
rows a retrieval strategy returns*, not about any particular vendor's weights.
Every other database-backed suite in this repo already patches ``embed_text``;
those suites hand every row the SAME vector, deliberately, so cosine ranking
cannot mask an admission bug. That trick is exactly wrong here, because
*ranking* is the thing under test.

**What the stand-in models, and why it is not a rig.** The design doc's claim
is one specific, well-documented property of dense retrieval: a rare literal
does not get a private direction in the embedding space, it is *compressed
toward the neighbourhood of the tokens it resembles*. So:

* every in-vocabulary word gets its own fixed pseudo-random unit direction;
* an out-of-vocabulary identifier is represented as the **mean of the
  in-vocabulary parts it decomposes into** — ``KLIO_METRICS_TOKEN`` becomes
  the midpoint of ``metrics`` and ``token``, which is precisely "compressed
  toward its neighbourhood";
* a document is the normalised mean of its token vectors.

Nothing in this model knows which row a test wants to win. The consequence —
that a short, purely topical paraphrase outranks a longer row that actually
contains the literal — falls out of the geometry, and is the same complaint
the design doc cites from a production system. If the model were tuned to
produce a desired answer it would need a notion of "the answer"; it has none.

The lexical arm, by contrast, sees ``KLIO_METRICS_TOKEN`` as the phrase
``'klio' <-> 'metric' <-> 'token'`` — adjacent, in order — and only one row
contains it.
"""

from __future__ import annotations

import math
import random
import re

#: Matches the production embedding column, ``vector(1024)``.
EMBED_DIM = 1024

#: The words the stand-in "knows". Everything else is out of vocabulary and is
#: represented by its in-vocabulary parts. Listed explicitly rather than
#: derived from the corpus so that adding a row cannot silently change how an
#: existing row is encoded.
VOCAB = frozenset(
    {
        "a",
        "access",
        "an",
        "and",
        "are",
        "auth",
        "based",
        "bearer",
        "before",
        "by",
        "collection",
        "config",
        "controls",
        "counters",
        "deploy",
        "endpoint",
        "endpoints",
        "environment",
        "every",
        "for",
        "gated",
        "guards",
        "header",
        "in",
        "is",
        "issued",
        "job",
        "klio",
        "manager",
        "metric",
        "metrics",
        "needs",
        "need",
        "new",
        "note",
        "observability",
        "of",
        "on",
        "platform",
        "prometheus",
        "protects",
        "quarterly",
        "railway",
        "reads",
        "required",
        "requires",
        "rotate",
        "rotated",
        "route",
        "scrape",
        "scraping",
        "secret",
        "set",
        "stored",
        "team",
        "the",
        "to",
        "token",
        "tokens",
        "tuesday",
        "value",
        "variable",
        "was",
        "with",
    }
)


def _tokens(text: str) -> list[str]:
    """Lowercase word tokens, keeping ``_`` so identifiers stay whole."""
    return [t for t in re.split(r"[^a-z0-9_]+", text.lower()) if t]


def _axis(token: str) -> list[float]:
    """The fixed pseudo-random direction for ``token``. Seeded by the token,
    so it is identical across processes and runs."""
    rng = random.Random(f"klio-lexical-axis-{token}")
    return [rng.gauss(0.0, 1.0) for _ in range(EMBED_DIM)]


def _normalize(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(x * x for x in vector)) or 1.0
    return [x / norm for x in vector]


def _token_vector(token: str) -> list[float]:
    if token in VOCAB:
        return _normalize(_axis(token))
    parts = [part for part in token.split("_") if part in VOCAB]
    if not parts:
        # Nothing to compress toward: an entirely unknown token keeps its own
        # direction. This is the case the lexical arm does NOT need help with.
        return _normalize(_axis(token))
    accumulator = [0.0] * EMBED_DIM
    for part in parts:
        part_vector = _normalize(_axis(part))
        for index in range(EMBED_DIM):
            accumulator[index] += part_vector[index]
    return _normalize(accumulator)


def embed(text: str) -> list[float]:
    """The stand-in embedding: the normalised mean of the token directions."""
    accumulator = [0.0] * EMBED_DIM
    for token in _tokens(text):
        token_vector = _token_vector(token)
        for index in range(EMBED_DIM):
            accumulator[index] += token_vector[index]
    return _normalize(accumulator)


def embedding_literal(text: str) -> str:
    """``embed(text)`` rendered the way pgvector accepts as a bound parameter."""
    return "[" + ",".join(repr(x) for x in embed(text)) + "]"


def cosine(left: list[float], right: list[float]) -> float:
    return sum(x * y for x, y in zip(left, right))


# ── the corpus ────────────────────────────────────────────────────────────

#: The rare literal. Postgres turns it into the phrase query
#: ``'klio' <-> 'metric' <-> 'token'``, and ``klio`` appears in no other row.
LITERAL = "KLIO_METRICS_TOKEN"

#: The one row that contains the literal. Long and only incidentally topical,
#: which is what an operational note actually looks like.
TARGET_CONTENT = (
    "Railway deploy note: the value of KLIO_METRICS_TOKEN was rotated on "
    "Tuesday by the platform team and the new secret is in the manager."
)

#: Rows about the same subject that do NOT contain the literal. Short and
#: purely topical, which is what the dense arm rewards.
DISTRACTOR_CONTENTS = (
    "The metrics token guards the endpoint.",
    "A metrics token is required.",
    "Metrics endpoints need a token.",
    "Every metrics route needs the token.",
    "The metrics scrape job reads counters with a bearer token.",
)
