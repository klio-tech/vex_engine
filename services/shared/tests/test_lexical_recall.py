"""The lexical recall arm, against a real Postgres.

The premise of the whole feature is a *negative result*, so it is asserted
directly rather than assumed: a corpus is seeded in which one row contains a
rare literal and five topically similar rows do not, and the dense arm ranks
the row containing the literal LAST. See ``lexical_fixtures`` for why the
embedding stand-in models that honestly rather than arranging it.

Real Postgres and not a fake session, for the same reason the tenancy suites
use one: the behaviour under test is what ``to_tsvector``, ``ts_rank`` and
``websearch_to_tsquery`` actually do, and a fake that records the emitted SQL
proves only that a string was written.
"""

import uuid
from datetime import datetime, timezone
from unittest.mock import AsyncMock, patch

import pytest
from sqlalchemy import create_engine, text

import lexical_fixtures as corpus
from shared import metrics
from shared.memory import retrieve_memories_scoped
from shared.testing import database_url, requires_db

pytestmark = requires_db("the lexical recall arm tests")

DB_URL = database_url()


@pytest.fixture()
def db():
    """A real Connection whose rows are removed afterwards, org by org."""
    engine = create_engine(DB_URL)
    conn = engine.connect()
    orgs: set[str] = set()
    conn.created_org_ids = orgs  # type: ignore[attr-defined]
    try:
        yield conn
    finally:
        try:
            conn.rollback()
            for org_id in orgs:
                conn.execute(
                    text(
                        """
                        DELETE FROM project_members WHERE project_id IN (
                            SELECT id FROM projects WHERE org_id = :o)
                        """
                    ),
                    {"o": org_id},
                )
                conn.execute(
                    text("DELETE FROM session_memories WHERE org_id = :o"),
                    {"o": org_id},
                )
                conn.execute(
                    text("DELETE FROM projects WHERE org_id = :o"), {"o": org_id}
                )
            conn.commit()
        finally:
            conn.close()
            engine.dispose()


def _org(db) -> str:
    org_id = f"org_{uuid.uuid4().hex[:8]}"
    db.created_org_ids.add(org_id)
    return org_id


def _seed(
    db,
    *,
    org_id: str,
    content: str,
    scope: str = "org",
    user_id: str | None = None,
    project_id: str | None = None,
    recall_hidden: bool = False,
    agent_id: str = "agent-1",
) -> str:
    """Insert one memory whose embedding is the stand-in's vector for it."""
    memory_id = str(uuid.uuid4())
    now = datetime.now(timezone.utc)
    db.execute(
        text("""
            INSERT INTO session_memories (
                id, session_id, org_id, agent_id, execution_id,
                memory_type, content, embedding, scope, status,
                user_id, provenance, project_id, recall_hidden, metadata,
                created_at, updated_at
            ) VALUES (
                :id, :session_id, :org_id, :agent_id, NULL,
                'fact', :content, :embedding, :scope, 'active',
                :user_id, 'EXTRACTED', :project_id, :recall_hidden, '{}'::jsonb,
                :created_at, :updated_at
            )
        """),
        {
            "id": memory_id,
            "session_id": f"sess-{uuid.uuid4().hex[:6]}",
            "org_id": org_id,
            "agent_id": agent_id,
            "content": content,
            "embedding": corpus.embedding_literal(content),
            "scope": scope,
            "user_id": user_id,
            "project_id": project_id,
            "recall_hidden": recall_hidden,
            "created_at": now,
            "updated_at": now,
        },
    )
    db.commit()
    return memory_id


def _seed_corpus(db, org_id: str) -> str:
    """The target row plus its five topical decoys. Returns the target's id."""
    target_id = _seed(db, org_id=org_id, content=corpus.TARGET_CONTENT)
    for content in corpus.DISTRACTOR_CONTENTS:
        _seed(db, org_id=org_id, content=content)
    return target_id


def _embed_as(text_value: str):
    """Patch ``embed_text`` with the stand-in, which is what seeded the rows."""
    return patch(
        "shared.memory.embed_text",
        new_callable=AsyncMock,
        return_value=corpus.embed(text_value),
    )


def _project(db, *, org_id: str, name: str = "apollo") -> str:
    project_id = str(uuid.uuid4())
    db.execute(
        text(
            "INSERT INTO projects (id, org_id, git_remote, display_name) "
            "VALUES (:id, :org_id, :remote, :name)"
        ),
        {
            "id": project_id,
            "org_id": org_id,
            "remote": f"git@example.com:{uuid.uuid4().hex[:8]}.git",
            "name": name,
        },
    )
    db.commit()
    return project_id


def _add_member(db, *, project_id: str, user_id: str) -> None:
    db.execute(
        text(
            "INSERT INTO project_members (project_id, user_id, role) "
            "VALUES (:p, :u, 'member')"
        ),
        {"p": project_id, "u": user_id},
    )
    db.commit()


# ── the premise, and the fix ──────────────────────────────────────────────


@pytest.mark.asyncio
async def test_exact_literal_ranks_first(db):
    """A query for a rare literal returns the row containing it, at rank 1.

    This is the observed RED: before the lexical arm existed the dense arm
    ranked the target sixth of six and ``limit=5`` dropped it entirely.
    """
    org = _org(db)
    target_id = _seed_corpus(db, org)

    with _embed_as(corpus.LITERAL):
        results = await retrieve_memories_scoped(db, org, corpus.LITERAL, limit=5)

    assert results, "recall returned nothing at all"
    assert str(results[0].id) == target_id, (
        "the row containing the literal is not first; order was: "
        + " | ".join(r.content[:40] for r in results)
    )


@pytest.mark.asyncio
async def test_target_row_is_otherwise_findable(db):
    """Positive control. Without this, the test above could pass against a
    corpus where the target is the only row that exists at all."""
    org = _org(db)
    target_id = _seed_corpus(db, org)

    probe = "Railway deploy note the platform team rotated the secret"
    with _embed_as(probe):
        results = await retrieve_memories_scoped(db, org, probe, limit=6)

    assert target_id in {str(r.id) for r in results}


# ── adversarial tenancy ───────────────────────────────────────────────────
#
# Every test here seeds a row containing the exact literal that the caller must
# NOT be able to read, and asks for that literal by name. Each carries a
# positive control seeding the same literal somewhere the caller IS entitled
# to, so "returned nothing" can never be mistaken for "the boundary held".

ALICE = "11111111-1111-1111-1111-111111111111"
MALLORY = "22222222-2222-2222-2222-222222222222"

FORBIDDEN = "Do not read: KLIO_METRICS_TOKEN is bd7f21c9."
PERMITTED = "Allowed copy: KLIO_METRICS_TOKEN is documented here."


async def _recall_literal(db, org, **kwargs):
    with _embed_as(corpus.LITERAL):
        return await retrieve_memories_scoped(
            db, org, corpus.LITERAL, limit=50, **kwargs
        )


@pytest.mark.asyncio
async def test_another_users_private_row_is_not_returned(db):
    org = _org(db)
    _seed(db, org_id=org, content=FORBIDDEN, scope="private", user_id=ALICE)

    results = await _recall_literal(db, org, user_id=MALLORY)

    assert [r.content for r in results] == []


@pytest.mark.asyncio
async def test_control_the_literal_is_findable_when_it_is_your_own(db):
    """Positive control for the test above."""
    org = _org(db)
    _seed(db, org_id=org, content=FORBIDDEN, scope="private", user_id=ALICE)
    _seed(db, org_id=org, content=PERMITTED, scope="private", user_id=MALLORY)

    results = await _recall_literal(db, org, user_id=MALLORY)

    assert [r.content for r in results] == [PERMITTED]


@pytest.mark.asyncio
async def test_a_recall_hidden_row_is_not_returned(db):
    org = _org(db)
    _seed(db, org_id=org, content=FORBIDDEN, recall_hidden=True)

    results = await _recall_literal(db, org)

    assert [r.content for r in results] == []


@pytest.mark.asyncio
async def test_control_the_same_literal_is_findable_when_not_hidden(db):
    """Positive control: only ``recall_hidden`` separates these two rows."""
    org = _org(db)
    _seed(db, org_id=org, content=FORBIDDEN, recall_hidden=True)
    _seed(db, org_id=org, content=PERMITTED, recall_hidden=False)

    results = await _recall_literal(db, org)

    assert [r.content for r in results] == [PERMITTED]


@pytest.mark.asyncio
async def test_a_non_members_project_row_is_not_returned(db):
    org = _org(db)
    theirs = _project(db, org_id=org, name="theirs")
    _add_member(db, project_id=theirs, user_id=ALICE)
    _seed(db, org_id=org, content=FORBIDDEN, scope="project", project_id=theirs)

    results = await _recall_literal(db, org, user_id=MALLORY)

    assert [r.content for r in results] == []


@pytest.mark.asyncio
async def test_control_a_project_row_is_findable_by_a_member(db):
    """Positive control: membership is the only difference."""
    org = _org(db)
    theirs = _project(db, org_id=org, name="theirs")
    _add_member(db, project_id=theirs, user_id=ALICE)
    mine = _project(db, org_id=org, name="mine")
    _add_member(db, project_id=mine, user_id=MALLORY)
    _seed(db, org_id=org, content=FORBIDDEN, scope="project", project_id=theirs)
    _seed(db, org_id=org, content=PERMITTED, scope="project", project_id=mine)

    results = await _recall_literal(db, org, user_id=MALLORY)

    assert [r.content for r in results] == [PERMITTED]


@pytest.mark.asyncio
async def test_membership_in_another_org_does_not_admit_the_row(db):
    """The JOIN to ``projects`` on ``org_id`` is what stops this.

    ``project_members`` carries no ``org_id`` of its own, so a membership row
    in another tenant would otherwise satisfy the project arm for a row in
    this one — the outer ``org_id`` filter constrains the memory, not the
    entitlement.
    """
    victim_org = _org(db)
    attacker_org = _org(db)
    victim_project = _project(db, org_id=victim_org, name="victim")
    _seed(
        db,
        org_id=victim_org,
        content=FORBIDDEN,
        scope="project",
        project_id=victim_project,
    )
    # Mallory is a member of that project id, but via the OTHER org.
    _add_member(db, project_id=victim_project, user_id=MALLORY)
    attacker_project = _project(db, org_id=attacker_org, name="attacker")
    _add_member(db, project_id=attacker_project, user_id=MALLORY)

    results = await _recall_literal(db, attacker_org, user_id=MALLORY)

    assert [r.content for r in results] == []


@pytest.mark.asyncio
async def test_another_orgs_row_is_not_returned(db):
    org = _org(db)
    other = _org(db)
    _seed(db, org_id=other, content=FORBIDDEN)

    results = await _recall_literal(db, org)

    assert [r.content for r in results] == []


@pytest.mark.asyncio
async def test_control_the_literal_is_findable_inside_your_own_org(db):
    """Positive control for the cross-org test."""
    org = _org(db)
    other = _org(db)
    _seed(db, org_id=other, content=FORBIDDEN)
    _seed(db, org_id=org, content=PERMITTED)

    results = await _recall_literal(db, org)

    assert [r.content for r in results] == [PERMITTED]


# ── the project two-tier fill, with the lexical arm active ────────────────


@pytest.mark.asyncio
async def test_project_two_tier_fill_still_holds_with_lexical_active(db):
    """Tier 1 is same-project rows, tier 2 is unfiled rows, and rows filed
    under ANOTHER project never surface — including via the lexical arm."""
    org = _org(db)
    mine = _project(db, org_id=org, name="mine")
    theirs = _project(db, org_id=org, name="theirs")
    _add_member(db, project_id=mine, user_id=ALICE)
    _add_member(db, project_id=theirs, user_id=MALLORY)

    in_project = _seed(
        db,
        org_id=org,
        content="Tier one: KLIO_METRICS_TOKEN lives in this project.",
        project_id=mine,
    )
    unfiled = _seed(db, org_id=org, content="Tier two: KLIO_METRICS_TOKEN unfiled.")
    _seed(
        db,
        org_id=org,
        content="Other project: KLIO_METRICS_TOKEN must not surface.",
        scope="project",
        project_id=theirs,
    )

    results = await _recall_literal(db, org, user_id=ALICE, project_id=mine)

    ids = [str(r.id) for r in results]
    assert ids[0] == in_project, "the same-project row must fill the first slot"
    assert unfiled in ids
    assert not any("Other project" in r.content for r in results)


# ── the single-arm invariant, against the database ────────────────────────


@pytest.mark.asyncio
async def test_an_empty_lexical_arm_leaves_the_order_untouched(db, monkeypatch):
    """Twelve rows, none of which contain the literal, so the lexical arm
    matches nothing. The fused order must equal the order the kill switch
    produces — over a list long enough that agreement cannot be coincidence.
    """
    org = _org(db)
    for index in range(12):
        _seed(
            db,
            org_id=org,
            content=f"Row {index}: the metrics endpoint and its token, note {index}.",
        )

    monkeypatch.setenv("KLIO_LEXICAL_RECALL_ENABLED", "0")
    with _embed_as(corpus.LITERAL):
        without = await retrieve_memories_scoped(db, org, corpus.LITERAL, limit=12)

    monkeypatch.setenv("KLIO_LEXICAL_RECALL_ENABLED", "1")
    with _embed_as(corpus.LITERAL):
        with_arm = await retrieve_memories_scoped(db, org, corpus.LITERAL, limit=12)

    assert len(without) == 12
    assert [str(r.id) for r in with_arm] == [str(r.id) for r in without]
    assert [r.relevance for r in with_arm] == [r.relevance for r in without]


@pytest.mark.asyncio
async def test_a_stopword_only_query_leaves_the_order_untouched(db, monkeypatch):
    """The other route to an empty arm: a query that reduces to no lexemes.
    Same corpus, same embedding, only the arm differs."""
    org = _org(db)
    for index in range(12):
        _seed(db, org_id=org, content=f"Row {index}: metrics and tokens, note {index}.")

    query = "the a of and"
    monkeypatch.setenv("KLIO_LEXICAL_RECALL_ENABLED", "0")
    with _embed_as(query):
        without = await retrieve_memories_scoped(db, org, query, limit=12)

    monkeypatch.setenv("KLIO_LEXICAL_RECALL_ENABLED", "1")
    with _embed_as(query):
        with_arm = await retrieve_memories_scoped(db, org, query, limit=12)

    assert len(without) == 12
    assert [str(r.id) for r in with_arm] == [str(r.id) for r in without]


# ── hostile input, and failure isolation ──────────────────────────────────


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "query",
    [
        '&& || !! ((( "unclosed',
        "the a of and",
        "",
        "   ",
        "'; DROP TABLE session_memories; --",
        "a" * 5000,
        "🙂 ✅ ᾏ",
    ],
)
async def test_hostile_queries_do_not_raise(db, query):
    """``to_tsquery`` raises a syntax error on arbitrary input, which is why
    the arm uses ``websearch_to_tsquery``. This text comes from an LLM."""
    org = _org(db)
    _seed_corpus(db, org)

    with _embed_as("metrics token"):
        results = await retrieve_memories_scoped(db, org, query, limit=5)

    assert isinstance(results, list)


@pytest.mark.asyncio
async def test_a_lexical_failure_returns_the_vector_results_unchanged(db, monkeypatch):
    """A broken lexical arm must degrade exactly as graph expansion does.

    The document expression is replaced with a call to a function that does
    not exist, so the lexical statement raises inside Postgres — a real
    failure that aborts the transaction, not a Python-side stub.
    """
    org = _org(db)
    _seed_corpus(db, org)

    with _embed_as(corpus.LITERAL):
        healthy = await retrieve_memories_scoped(db, org, corpus.LITERAL, limit=5)

    monkeypatch.setattr(
        "shared.lexical.fts_document_expression",
        lambda column="content": f"no_such_function_xyz({column})",
    )
    with _embed_as(corpus.LITERAL):
        degraded = await retrieve_memories_scoped(db, org, corpus.LITERAL, limit=5)

    assert degraded, "a lexical failure must not empty the result"
    assert all(record.via is None for record in degraded)
    assert [str(r.id) for r in degraded] != [str(r.id) for r in healthy], (
        "the healthy run should have been improved by the arm; if these agree "
        "the degradation test is not proving anything"
    )


@pytest.mark.asyncio
async def test_a_row_the_arm_promoted_from_inside_the_pool_keeps_its_similarity(db):
    """The common case. The target IS in the vector candidate pool — just far
    down it — so fusion promotes a row that already has an honest cosine
    score, and it keeps it. Nothing is marked, because nothing needs to be:
    the row was a semantic match that lexical merely agreed with."""
    org = _org(db)
    target_id = _seed_corpus(db, org)

    with _embed_as(corpus.LITERAL):
        results = await retrieve_memories_scoped(db, org, corpus.LITERAL, limit=5)

    target = next(r for r in results if str(r.id) == target_id)
    assert target.via is None
    assert target.relevance is not None


@pytest.mark.asyncio
async def test_a_lexical_only_row_is_marked_and_carries_no_fake_relevance(
    db, monkeypatch
):
    """``ts_rank`` is not a cosine similarity, so it is not written into a
    field documented as one. The positive marker is what tells them apart.

    Reaching this branch takes two things, and both are worth stating because
    together they bound what the feature can do at its default settings:

    * the row must be OUTSIDE the vector candidate pool — with ``limit=2`` the
      pool is 8, so the corpus below is 13 rows and the target is last;
    * the lexical weight must be high enough for a row with no vector rank at
      all to survive the final truncation. At the default 0.5 a lexical-only
      row scores 0.5/61 = 0.0082 while every vector row scores at least
      1/(60+pool), so at the default the arm REORDERS the pool rather than
      adding to it. At 1.0 a lexical rank-1 row ties the vector's own leader
      and lands second.
    """
    org = _org(db)
    for index in range(12):
        _seed(db, org_id=org, content=f"The metrics token, note {index}.")
    target_id = _seed(db, org_id=org, content=corpus.TARGET_CONTENT)

    monkeypatch.setenv("KLIO_LEXICAL_WEIGHT", "1.0")
    with _embed_as(corpus.LITERAL):
        results = await retrieve_memories_scoped(db, org, corpus.LITERAL, limit=2)

    target = next((r for r in results if str(r.id) == target_id), None)
    assert target is not None, "the lexical-only row was not returned at all"
    assert target.via == "lexical"
    assert target.relevance is None
    assert any(r.via is None and r.relevance is not None for r in results), (
        "positive control: a vector-matched row must still carry its score"
    )


# ── instrumentation ───────────────────────────────────────────────────────


@pytest.fixture()
def clean_metrics():
    metrics.reset_for_test()
    yield
    metrics.reset_for_test()


@pytest.mark.asyncio
async def test_a_contributing_query_is_counted_as_contributing(db, clean_metrics):
    """THE number: recalls where lexical returned a row the vector arm did
    not. Here the corpus is larger than the candidate pool, so the target is
    genuinely absent from the vector arm's rows."""
    org = _org(db)
    for index in range(12):
        _seed(db, org_id=org, content=f"The metrics token, note {index}.")
    _seed(db, org_id=org, content=corpus.TARGET_CONTENT)

    with _embed_as(corpus.LITERAL):
        await retrieve_memories_scoped(db, org, corpus.LITERAL, limit=2)

    assert metrics.LEXICAL_ARM_QUERIES.value(outcome="rows") == 1.0
    assert metrics.LEXICAL_ARM_CONTRIBUTED.value() == 1.0
    assert metrics.LEXICAL_ARM_SECONDS.snapshot().count == 1


@pytest.mark.asyncio
async def test_an_agreeing_query_is_counted_but_not_as_contributing(db, clean_metrics):
    """Positive control for the counter above: the arm ran and matched, but
    every row it found the vector arm had already returned, so it changed
    nothing that the vector arm did not already know."""
    org = _org(db)
    _seed(db, org_id=org, content=corpus.TARGET_CONTENT)

    with _embed_as(corpus.LITERAL):
        await retrieve_memories_scoped(db, org, corpus.LITERAL, limit=5)

    assert metrics.LEXICAL_ARM_QUERIES.value(outcome="rows") == 1.0
    assert metrics.LEXICAL_ARM_CONTRIBUTED.value() == 0.0


@pytest.mark.asyncio
async def test_a_matchless_query_counts_as_empty_not_as_an_error(db, clean_metrics):
    org = _org(db)
    _seed(db, org_id=org, content="Nothing here resembles the identifier.")

    with _embed_as(corpus.LITERAL):
        await retrieve_memories_scoped(db, org, corpus.LITERAL, limit=5)

    assert metrics.LEXICAL_ARM_QUERIES.value(outcome="empty") == 1.0
    assert metrics.LEXICAL_ARM_QUERIES.value(outcome="error") == 0.0


@pytest.mark.asyncio
async def test_each_skip_reason_is_reported_under_its_own_label(
    db, clean_metrics, monkeypatch
):
    org = _org(db)
    _seed_corpus(db, org)

    monkeypatch.setenv("KLIO_LEXICAL_RECALL_ENABLED", "0")
    with _embed_as(corpus.LITERAL):
        await retrieve_memories_scoped(db, org, corpus.LITERAL, limit=5)
    assert metrics.LEXICAL_ARM_SKIPPED.value(reason="disabled") == 1.0

    monkeypatch.setenv("KLIO_LEXICAL_RECALL_ENABLED", "1")
    monkeypatch.setenv("KLIO_LEXICAL_WEIGHT", "0")
    with _embed_as(corpus.LITERAL):
        await retrieve_memories_scoped(db, org, corpus.LITERAL, limit=5)
    assert metrics.LEXICAL_ARM_SKIPPED.value(reason="zero_weight") == 1.0

    monkeypatch.delenv("KLIO_LEXICAL_WEIGHT")
    query = "the a of and"
    with _embed_as(query):
        await retrieve_memories_scoped(db, org, query, limit=5)
    assert metrics.LEXICAL_ARM_SKIPPED.value(reason="empty_query") == 1.0

    assert metrics.LEXICAL_ARM_QUERIES.value(outcome="rows") == 0.0


@pytest.mark.asyncio
async def test_a_lexical_failure_counts_as_an_error(db, clean_metrics, monkeypatch):
    org = _org(db)
    _seed_corpus(db, org)
    monkeypatch.setattr(
        "shared.lexical.fts_document_expression",
        lambda column="content": f"no_such_function_xyz({column})",
    )

    with _embed_as(corpus.LITERAL):
        await retrieve_memories_scoped(db, org, corpus.LITERAL, limit=5)

    assert metrics.LEXICAL_ARM_QUERIES.value(outcome="error") == 1.0


@pytest.mark.asyncio
async def test_the_two_tier_fill_reports_one_recall_not_two(db, clean_metrics):
    """The arm runs twice on the project path. The series are per recall, so
    a rate against the recall count stays interpretable."""
    org = _org(db)
    mine = _project(db, org_id=org, name="mine")
    _add_member(db, project_id=mine, user_id=ALICE)
    _seed(db, org_id=org, content=corpus.TARGET_CONTENT, project_id=mine)
    _seed(db, org_id=org, content="Unfiled: KLIO_METRICS_TOKEN again.")

    with _embed_as(corpus.LITERAL):
        await retrieve_memories_scoped(
            db, org, corpus.LITERAL, limit=5, user_id=ALICE, project_id=mine
        )

    total = sum(
        metrics.LEXICAL_ARM_QUERIES.value(outcome=outcome)
        for outcome in metrics.LEXICAL_OUTCOMES
    )
    assert total == 1.0
