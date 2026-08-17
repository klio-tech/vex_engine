"""How a SEARCH STRING is phrased to the embedding model. Read path only.

The production model is ``intfloat/multilingual-e5-large-instruct``. The
2026-08-16 ranking diagnosis identified a real defect here — the engine sends
bare text and E5 is trained on asymmetric prompts — and proposed the ``e5-base``
convention: ``passage: `` on every stored document, ``query: `` on every
search. That fix requires re-embedding all 63,196 rows, ~67 minutes at the
gateway's measured throughput, during which recall is actively broken.

It was measured before it was built, and the measurement said something else.
Three conventions, same 11,648-row production sample, same query set of 17
(the diagnosis's four plus thirteen taken verbatim from real sessions across
seven projects):

======================================  ==============  =====================
convention                              on-vs-off       nonsense query's
                                        cosine margin   best match
======================================  ==============  =====================
bare query / bare documents (shipped)         -0.0159                  0.8279
``query: `` / ``passage: `` (proposed)        -0.0121                  0.8158
``Instruct:`` query / bare documents          -0.0080                  0.8009
======================================  ==============  =====================

Two things fall out, and neither was the expected answer.

**The proposed fix is the worst of the two changes and the only one that costs
a migration.** ``query: ``/``passage: `` is the convention for plain ``e5-*``.
The ``-instruct`` variants are trained differently: the *document* side takes
no prefix at all, and the *query* side takes ``Instruct: {task}\\nQuery: {q}``.
So the correct convention for THIS model leaves stored vectors exactly as they
already are. There is nothing to re-embed, no second column, no backfill
window, and the change is reversible by an environment variable rather than by
an hour of re-embedding.

**Neither convention rescues ranking, because the model cannot separate this
corpus.** Every margin above is NEGATIVE: for most queries the single
best-matching row in the whole corpus belongs to a *different project* than the
session asking. That is not a prompt-format problem. It is 90 % auto-mined
session facts that all read alike — "The X is configured to Y" — where a query
about a proxy is genuinely as close to one project's phrasing as another's. The
control query in the evaluation set, which is about nothing in the corpus at
all, still finds a 0.80 match.

So this module ships the better convention, and the honest conclusion goes with
it: **the project fence in ``capture.py`` and ``retrieve_memories_scoped`` is
what actually separates projects, not the embedding.** This is a real but small
improvement (margin −0.0159 → −0.0080, and the nonsense query's best match
falls by 0.027), taken because it is free.

**Default is ``bare`` — unchanged behaviour.** A 0.008 aggregate improvement
that is better on 8 of 17 queries and worse on 4 is not enough evidence to
change what production does by default. The mechanism is here, measured and
switchable, so flipping it is an observation away rather than a rebuild away.

Documents are never prefixed by anything in this module, in either mode. That
is not an omission — it is the ``-instruct`` convention, and it is also what
every existing vector in the corpus already is. A passage prefix would silently
invalidate every stored embedding.
"""

from __future__ import annotations

import os

from shared.env_config import parse_str

__all__ = [
    "TEMPLATE_BARE",
    "TEMPLATE_INSTRUCT",
    "TEMPLATES",
    "ROLE_QUERY",
    "ROLE_PASSAGE",
    "ROLES",
    "TEMPLATE_ENV",
    "TASK_ENV",
    "DEFAULT_TASK",
    "active_template",
    "task_description",
    "prefix_for",
    "apply_role",
]

TEMPLATE_BARE = "bare"
TEMPLATE_INSTRUCT = "instruct"
TEMPLATES = (TEMPLATE_BARE, TEMPLATE_INSTRUCT)

ROLE_QUERY = "query"
ROLE_PASSAGE = "passage"
ROLES = (ROLE_QUERY, ROLE_PASSAGE)

TEMPLATE_ENV = "KLIO_EMBED_QUERY_TEMPLATE"
TASK_ENV = "KLIO_EMBED_QUERY_TASK"

#: The task sentence in the ``Instruct:`` template. E5-instruct conditions on
#: this text, so it is part of the vector, and changing it changes every query
#: embedding — which is free (queries are embedded per request) but is still a
#: ranking change and should be measured like one.
DEFAULT_TASK = (
    "Given a question from a software engineering session, "
    "retrieve memories that answer it"
)


def active_template() -> str:
    """Which query phrasing this process uses. ``bare`` unless told otherwise.

    Read per call rather than frozen at import, matching
    :func:`shared.lexical.lexical_enabled`: this is a knob designed to be
    flipped and un-flipped while observing recall, and a value that only takes
    effect on restart is a knob nobody turns.
    """
    return parse_str(
        os.environ.get(TEMPLATE_ENV),
        default=TEMPLATE_BARE,
        allowed=TEMPLATES,
        name=TEMPLATE_ENV,
    )


def task_description() -> str:
    """The task sentence for the ``Instruct:`` template."""
    return os.environ.get(TASK_ENV, "").strip() or DEFAULT_TASK


def prefix_for(role: str) -> str:
    """The prefix for *role*. Always ``""`` for a passage — see the module docs.

    Raises:
        ValueError: on an unknown role. A typo is silent otherwise: the text
            embeds fine, into the wrong half of the space, and the only symptom
            is recall quality nobody can attribute.
    """
    if role not in ROLES:
        raise ValueError(f"unknown embedding role: {role!r}")
    if role == ROLE_PASSAGE:
        # Never prefixed. Every vector in the corpus was written this way, and
        # this is also the -instruct convention, so the two agree. Prefixing
        # documents here would invalidate 63,196 stored vectors silently.
        return ""
    if active_template() == TEMPLATE_BARE:
        return ""
    return f"Instruct: {task_description()}\nQuery: "


def apply_role(text_input: str, role: str) -> str:
    """*text_input* phrased for *role*."""
    return prefix_for(role) + text_input
