"""The memory taxonomy the product reads back, and the durability line.

Two separate jobs live here, deliberately kept apart from the LLM plumbing in
:mod:`shared.memory` so they are testable without a model and importable
without a cycle:

1. **What kind of thing is this?** — the closed set of ``memory_type`` values
   the Context workspace groups by (``decision`` → Decisions, ``plan`` →
   Plans, ``fact`` → Constraints, everything else → Recent). Before this
   module existed, transcript distillation could only ever emit ``fact`` and
   ``summary`` and the curator could only ever emit ``fact``, so three of the
   workspace's four sections were structurally empty: ``decision`` and
   ``plan`` were reachable only by an agent explicitly calling the ``decide``
   / ``plan`` MCP tools, which agents essentially never do unprompted (0 rows
   of each against 5,197 of the rest on the reference org).

2. **Is this worth keeping at all?** — the durability line. The corpus problem
   is not that junk was labelled ``fact`` rather than ``observation``; it is
   that junk was stored. Better-classified junk is still junk, so the rule
   below is applied *before* the type is chosen.

The durability rule, stated once so the prompt and the code agree:

    Store a memory only if it would still be true, and still useful to a
    different agent in a different session, a week from now.

Its contrapositive is the thing the extractor is actually asked to drop:
anything whose truth is bound to a **process**, a **machine**, a **moment**,
or a **step** — a running container or port, a tool invocation and its output,
a command's exit code, a directory listing, a test-run tally, a commit SHA or
"branch X was created", and narration of the conversation itself ("the agent
then read the file", "the agent responded with 'pong'").

Two mechanisms enforce it and they are not redundant:

* :data:`DURABILITY_RULE` / :data:`TAXONOMY_RULE` go into the extraction
  prompt. This is the primary mechanism and the only one with judgment.
* :func:`is_transient` is a deterministic backstop over the model's output. It
  is deliberately **narrow**: it matches the handful of surface shapes that
  dominate the existing corpus and nothing more. A wide regex here would
  silently delete real knowledge, which is the failure mode this whole change
  exists to avoid — so when the two disagree about a borderline item, the
  prompt wins by default and only these named shapes are overridden.
"""

import re
from typing import NamedTuple, Optional

__all__ = [
    "MEMORY_TAXONOMY",
    "DEFAULT_MEMORY_TYPE",
    "DURABILITY_RULE",
    "TAXONOMY_RULE",
    "TYPED_OUTPUT_SHAPE",
    "ExtractedMemory",
    "is_transient",
    "coerce_memory_type",
    "parse_typed_items",
]


#: The closed set an extractor may assign. Mirrors what the Context workspace
#: groups by; anything outside it is coerced to :data:`DEFAULT_MEMORY_TYPE`.
#: ``summary`` is absent on purpose — a summary is produced by its own pass,
#: not chosen per item.
MEMORY_TAXONOMY: tuple[str, ...] = ("decision", "plan", "fact", "note", "observation")

#: Where an unrecognised label lands. ``observation`` stays available as the
#: bucket for durable-but-uncategorisable context; what changed is that it is
#: no longer where an unopinionated extractor dumps everything, because the
#: prompt now names the four alternatives and the durability rule removes the
#: material that used to fill it.
DEFAULT_MEMORY_TYPE: str = "observation"


DURABILITY_RULE: str = (
    "DURABILITY TEST — apply this FIRST, to every candidate, before choosing a "
    "kind. Keep an item ONLY if it would still be true, and still useful to a "
    "different agent working in a different session, a week from now. "
    "If it fails the test, omit it entirely; do not downgrade it to a lesser "
    "kind. Omitting everything is a valid and common answer. "
    "Specifically DROP anything whose truth is tied to a process, a machine, a "
    "moment, or a step: running processes, containers, ports, PIDs or URLs of "
    "local servers; tool invocations and their output; command exit codes and "
    "one-off command results; file or directory listings and 'file X exists' "
    "probes; test-run tallies and 'all tests pass'; commit SHAs, 'branch X was "
    "created', and other point-in-time VCS state; progress narration about the "
    "session itself ('the agent then read the file', 'the agent responded', "
    "'no crash occurred'). "
    "KEEP the durable residue instead: what was decided, what is planned, what "
    "constrains the system, and the conventions and context a newcomer would "
    "need. "
)

TAXONOMY_RULE: str = (
    "KINDS — for each item that survives the durability test, choose exactly "
    "one, using the FIRST that fits:\n"
    '- "decision": a choice that was made and settled, which constrains '
    "future work. It resolves alternatives. "
    "e.g. 'Deploys go through GitHub source builds, not Docker Hub images.' "
    "Do NOT use this for something merely proposed, considered, or still open.\n"
    '- "plan": stated forward intent that has not happened yet — the next '
    "step, the agreed sequence. e.g. 'Next the proposals UI gets wired to the "
    "approve endpoint.' Once the work is done it is no longer a plan.\n"
    '- "fact": a durable property or constraint of the system, product, '
    "tooling, or environment that was DISCOVERED rather than chosen — how "
    "something behaves, what it refuses, where it lives. e.g. 'The pooled "
    "Postgres endpoint runs PgBouncer in transaction mode and refuses libpq "
    "startup parameters.'\n"
    '- "note": durable context that is none of the above — a person\'s '
    "preference or convention, ownership, background, rationale worth "
    "remembering on its own.\n"
    '- "observation": last resort only, for something durable that genuinely '
    "fits no kind above. Prefer omitting an item over labelling it "
    '"observation".\n'
)

#: The exact JSON shape a typed extraction must answer with. Kept next to the
#: rules so a prompt change and a parser change are visible in one diff.
TYPED_OUTPUT_SHAPE: str = (
    '{"memories": [{"kind": "<decision|plan|fact|note|observation>", '
    '"content": "<one short self-contained statement>"}, ...]}'
)


class ExtractedMemory(NamedTuple):
    """One classified, durability-screened item from an extraction pass."""

    memory_type: str
    content: str


# ---------------------------------------------------------------------------
# Deterministic backstop
# ---------------------------------------------------------------------------

# Each pattern below is anchored to a surface shape that is transient *by
# construction*, not merely by topic. "The agent ran the test suite" is
# narration whatever it narrates; "port 54422" is a machine's current state
# whatever is listening on it. Topic-based matching is exactly what this must
# not do — "the deploy" and "the database" are the subjects of most of the
# durable knowledge in the corpus.
_TRANSIENT_PATTERNS: tuple[re.Pattern[str], ...] = (
    # Verbatim tool-invocation logs. 2,675 of 5,253 rows on the reference org
    # are literally this shape: `Used tool Bash input: {"command": ...}`.
    re.compile(r"^\s*(used|called|invoked|ran)\s+(the\s+)?tool\b", re.I),
    re.compile(r"\btool\s+(call|invocation|result|output)\b.*\binput\s*:", re.I),
    # Narration of the session rather than of the system.
    re.compile(
        r"\bthe\s+agent\s+(then\s+)?"
        r"(responded|replied|answered|read|opened|listed|printed|output|"
        r"ran|executed|invoked|reported|summari[sz]ed|confirmed)\b",
        re.I,
    ),
    re.compile(r"\b(no\s+(crash|error)s?\s+occurred|nothing\s+crashed)\b", re.I),
    re.compile(r"\bduring\s+(the|this)\s+(session|run|turn)\b", re.I),
    # Machine state: what is listening where, right now.
    re.compile(
        r"\b(is\s+)?(running|listening|served|exposed|bound)\b[^.]{0,40}"
        r"\b(on\s+)?port\s+\d{2,5}\b",
        re.I,
    ),
    re.compile(r"\b(container|process|pid)\b[^.]{0,40}\bport\s+\d{2,5}\b", re.I),
    re.compile(r"\bhttps?://(localhost|127\.0\.0\.1)[:/]", re.I),
    # Command outcomes.
    re.compile(r"\bexit(ed\s+with)?\s+(code|status)\s+\d+\b", re.I),
    re.compile(r"\bcommand\s+(succeeded|failed|returned|printed)\b", re.I),
    # Point-in-time VCS state.
    re.compile(r"\bHEAD\s+(is\s+|was\s+)?at\s+[0-9a-f]{7,40}\b", re.I),
    re.compile(r"\bbranch\s+\S+\s+(was\s+)?(created|checked\s+out|deleted)\b", re.I),
    # Test-run tallies. `\d+ tests` is the tell; a durable claim about coverage
    # policy ("the suite must stay above 80%") has no run count in it.
    re.compile(r"\b\d+\s+(new\s+)?tests?\b[^.]{0,30}\b(pass|passed|green|fail)", re.I),
    re.compile(r"\ball\s+(the\s+)?tests?\s+(pass|passed|are\s+green)\b", re.I),
    # Existence probes over the filesystem.
    re.compile(r"^\s*no\s+file(s)?\s+(named|matching|exist)", re.I),
    re.compile(r"\bfile\s+\S+\s+(exists|does\s+not\s+exist)\b", re.I),
)

#: Below this, an "item" carries no recallable content whatever its shape
#: ("ok", "done", "pong"). Chosen to be shorter than any real statement in the
#: sampled corpus rather than tuned — it is a floor, not a filter.
_MIN_DURABLE_CHARS: int = 25


def is_transient(content: Optional[str]) -> bool:
    """Whether ``content`` is session exhaust rather than durable knowledge.

    A deliberately narrow deterministic backstop behind the prompt; see the
    module docstring for why it is not the primary mechanism. Empty, blank,
    and sub-:data:`_MIN_DURABLE_CHARS` strings are transient by definition.
    """
    if not content:
        return True
    text = content.strip()
    if len(text) < _MIN_DURABLE_CHARS:
        return True
    return any(pattern.search(text) for pattern in _TRANSIENT_PATTERNS)


def coerce_memory_type(raw: object) -> str:
    """Map a model-supplied kind onto :data:`MEMORY_TAXONOMY`.

    Unknown, missing, or non-string labels become :data:`DEFAULT_MEMORY_TYPE`
    rather than raising: a single malformed label must cost one item's
    precision, never the batch.
    """
    if not isinstance(raw, str):
        return DEFAULT_MEMORY_TYPE
    candidate = raw.strip().lower()
    # Tolerate the plural/section names a model may echo back from the prompt.
    aliases = {
        "decisions": "decision",
        "plans": "plan",
        "facts": "fact",
        "constraint": "fact",
        "constraints": "fact",
        "notes": "note",
        "observations": "observation",
    }
    candidate = aliases.get(candidate, candidate)
    return candidate if candidate in MEMORY_TAXONOMY else DEFAULT_MEMORY_TYPE


def parse_typed_items(raw_items: object) -> list[ExtractedMemory]:
    """Normalise a model's ``memories`` array into screened, typed items.

    Accepts the documented shape (``{"kind": ..., "content": ...}``) and
    tolerates a bare string, which is treated as an untyped item and coerced.
    Malformed entries and items failing :func:`is_transient` are dropped one by
    one; they never take the batch down.
    """
    if not isinstance(raw_items, list):
        return []

    items: list[ExtractedMemory] = []
    for entry in raw_items:
        if isinstance(entry, str):
            kind, content = DEFAULT_MEMORY_TYPE, entry
        elif isinstance(entry, dict):
            content = entry.get("content")
            if not isinstance(content, str):
                continue
            kind = coerce_memory_type(entry.get("kind", entry.get("type")))
        else:
            continue

        content = content.strip()
        if is_transient(content):
            continue
        items.append(ExtractedMemory(memory_type=kind, content=content))
    return items
