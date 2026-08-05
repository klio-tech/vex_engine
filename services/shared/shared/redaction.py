"""Secret/PII detection and masking for memory content.

Pure functions, no I/O. Every memory write funnels through
:func:`shared.brain.write_scoped_memory`, which calls :func:`redact` BEFORE
embedding — neither stored text nor vectors ever contain a detected secret.

Detection is pattern + entropy based (the detect-secrets approach this repo
already uses in pre-commit). The generic entropy detector only fires inside a
credential-shaped context (key-value assignment), because entropy alone
misfires on UUIDs, git SHAs, and hashes — which are fine to store.

Placeholder format: ``[REDACTED:<kind>:<fingerprint>]`` where fingerprint is
the last 4 hex chars of SHA-256 of the matched secret, so repeated mentions of
the same secret correlate without exposure. Placeholders never re-match
(idempotent).
"""

import hashlib
import math
import re
from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Detection patterns
#
# Order matters: more specific kinds first so e.g. a Stripe key is tagged
# stripe_secret_key, not generic credential_assignment.  Anthropic keys must
# come before OpenAI keys because both share the "sk-" prefix and the
# Anthropic pattern is more specific (sk-ant-…).
# ---------------------------------------------------------------------------

_PATTERNS: tuple[tuple[str, re.Pattern], ...] = (
    # --- Stripe ---------------------------------------------------------------
    ("stripe_secret_key", re.compile(r"\b[rs]k_(?:live|test)_[A-Za-z0-9]{20,}\b")),
    ("stripe_publishable_key", re.compile(r"\bpk_(?:live|test)_[A-Za-z0-9]{20,}\b")),
    ("stripe_webhook_secret", re.compile(r"\bwhsec_[A-Za-z0-9]{20,}\b")),
    # --- AWS ------------------------------------------------------------------
    ("aws_access_key_id", re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b")),
    # The 40-char secret that pairs with the access key id above. It has no
    # prefix to anchor on, so the regex matches the *shape* (exactly 40 chars
    # of the base64 alphabet, not embedded in a longer run of them) and
    # _aws_secret_guard rejects anything that is not mixed-case-plus-digit with
    # a base64-only character. That guard is what keeps 40-char lowercase hex —
    # a git tree hash — out of the net.
    (
        "aws_secret_access_key",
        re.compile(r"(?<![A-Za-z0-9/+=])[A-Za-z0-9/+=]{40}(?![A-Za-z0-9/+=])"),
    ),
    # --- GitHub ---------------------------------------------------------------
    (
        "github_token",
        re.compile(r"\b(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{30,})\b"),
    ),
    # --- Anthropic — must precede openai (both start with sk-) ---------------
    ("anthropic_api_key", re.compile(r"\bsk-ant-[A-Za-z0-9\-_]{20,}\b")),
    # --- OpenAI ---------------------------------------------------------------
    # Require the "proj-" infix OR a minimum length of 32 chars after "sk-" to
    # avoid matching short test tokens or the Anthropic pattern above.
    (
        "openai_api_key",
        re.compile(r"\bsk-(?:proj-[A-Za-z0-9\-_]{20,}|[A-Za-z0-9\-_]{32,})\b"),
    ),
    # --- Slack ----------------------------------------------------------------
    ("slack_token", re.compile(r"\bxox[baprs]-[A-Za-z0-9\-]{10,}\b")),
    # --- Google ---------------------------------------------------------------
    ("google_api_key", re.compile(r"\bAIza[0-9A-Za-z\-_]{30,}\b")),
    # --- Supabase -------------------------------------------------------------
    ("supabase_key", re.compile(r"\bsbp_[A-Za-z0-9]{20,}\b")),
    # --- JWT — three dot-separated base64url segments, each ≥ 8 chars --------
    (
        "jwt",
        re.compile(
            r"\beyJ[A-Za-z0-9_\-]{8,}\.[A-Za-z0-9_\-]{8,}\.[A-Za-z0-9_\-]{8,}\b"
        ),
    ),
    # --- PEM private key block -----------------------------------------------
    (
        "private_key_block",
        re.compile(
            r"-----BEGIN [A-Z ]*PRIVATE KEY-----.*?-----END [A-Z ]*PRIVATE KEY-----",
            re.DOTALL,
        ),
    ),
    # --- Database / service connection strings (only when password present) --
    # Capture group 1 = the password, which is the segment between the LAST ":"
    # and the LAST "@" of the authority.  The optional first sub-group allows an
    # email address as the username (e.g. Cloud SQL IAM auth, MongoDB Atlas):
    #   mongodb://jane.doe@example.com:p4ssw0rdXYZ@host:27017/db
    # The host segment is ``*`` and not ``+`` on purpose: a Redis AUTH URL has
    # no username at all (``redis://:password@host``), and requiring one
    # non-colon character there made the single most common password-carrying
    # URL shape unmatchable.
    (
        "connection_string",
        re.compile(
            r"(?:\b[a-z][a-z0-9+.\-]*://)(?:[^\s/@]+@)?[^\s:/@]*:([^\s@]+)@[^\s]+"
        ),
    ),
    # --- HTTP Bearer token ---------------------------------------------------
    # Capture group 1 = the token value.
    ("bearer_token", re.compile(r"(?i)\bBearer\s+([A-Za-z0-9\-._~+/]{16,}=*)")),
    # --- Generic credential assignment (entropy-gated) -----------------------
    # Capture group 1 = the WHOLE key token, group 2 = value.
    #
    # Two things here look redundant and are not:
    #
    # * The key group captures the entire identifier and merely *contains* one
    #   of the sensitive words. A bare ``\b(secret|...)`` cannot match
    #   ``AWS_SECRET_ACCESS_KEY``: ``_`` is a word character, so there is no
    #   word boundary before ``SECRET``, and even at a boundary the required
    #   ``\s*[:=]`` would land on ``_``. SCREAMING_SNAKE_CASE is the single
    #   most common shape a real credential is written in.
    # * The optional quotes around the key name are what let a JSON or Python
    #   dict literal match. In ``{"password": "..."}`` the character after the
    #   key is a closing quote, which broke the ``\s*[:=]`` follow outright.
    (
        "credential_assignment",
        re.compile(
            r"(?i)(?<![A-Za-z0-9_])[\"']?"
            r"([A-Za-z0-9_.\-]*"
            r"(?:password|passwd|pwd|secret|token|api[_-]?key|access[_-]?key)"
            r"[A-Za-z0-9_.\-]*)"
            # The value class must NOT exclude ``]``: an already-placed
            # ``[REDACTED:...]`` placeholder ends in one, and stopping short of
            # it would hide the placeholder from the idempotency guard and
            # re-redact on every pass.
            r"[\"']?\s*[:=]\s*[\"']?([^\s\"',;]{8,})[\"']?"
        ),
    ),
    # --- PII ------------------------------------------------------------------
    ("email", re.compile(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b")),
    # credit_card: Luhn-validated in _should_redact_match; length 13–19 digits.
    # Pattern is digit-terminated (\b\d…\d\b) so a trailing separator (space,
    # dash) is never consumed — the character after the card number is preserved.
    (
        "credit_card",
        re.compile(r"\b\d(?:[ \-]?\d){12,18}\b"),
    ),
    # phone: international format with "+" prefix, 10–14 digits total.
    ("phone", re.compile(r"(?<![\d.])\+\d{10,14}\b")),
)

# For credential_assignment / bearer_token / connection_string, only the
# captured VALUE group is the secret; everything else replaces the full match.
_VALUE_GROUP_KINDS: dict[str, int] = {
    "credential_assignment": 2,
    "bearer_token": 1,
    "connection_string": 1,
}

# Matches any already-placed redaction placeholder so we never double-redact.
_PLACEHOLDER_RE = re.compile(r"\[REDACTED:[a-z_]+:[0-9a-f]{4}\]")

# Entropy thresholds used for the credential_assignment entropy gate.
# Values of 8–11 chars need only entropy ≥ 2.5; values of 12+ chars must
# meet the stricter ≥ 3.0 bar (long low-entropy strings are usually prose).
_ENTROPY_THRESHOLD = 3.0
_MIN_VALUE_LEN_FOR_ENTROPY = 12

# Key names whose value is a password by definition, so the entropy gate is
# skipped entirely for them.
#
# The gate exists to stop prose-shaped values from being mangled, and it is
# strictest (3.0 bits at length >= 12) exactly where real passwords live:
# ``Passw0rdPassw0rd`` measures 2.7500 and ``hunter2hunter2`` 2.8074, so both
# passed straight through. On these key names there is no prose to protect —
# the key has already declared what the value is. Every other key name
# (``token``, ``api_key``, ``access_key``) keeps the gate.
_ALWAYS_REDACT_KEY_WORDS: tuple[str, ...] = ("password", "passwd", "pwd", "secret")


# ---------------------------------------------------------------------------
# Public dataclasses (frozen → immutable)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Finding:
    """A single detected secret/PII item."""

    kind: str
    fingerprint: str


@dataclass(frozen=True)
class RedactionResult:
    """The outcome of a :func:`redact` call."""

    clean_text: str
    findings: tuple[Finding, ...]
    redacted: bool


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _fingerprint(secret: str) -> str:
    """Return the last 4 hex chars of SHA-256(secret)."""
    return hashlib.sha256(secret.encode("utf-8")).hexdigest()[-4:]


def _shannon_entropy(value: str) -> float:
    """Return the Shannon entropy (bits) of *value*."""
    if not value:
        return 0.0
    counts: dict[str, int] = {}
    for ch in value:
        counts[ch] = counts.get(ch, 0) + 1
    total = len(value)
    return -sum((c / total) * math.log2(c / total) for c in counts.values())


def _luhn_valid(digits: str) -> bool:
    """Return True if *digits* passes the Luhn checksum."""
    total, alt = 0, False
    for d in reversed(digits):
        n = int(d)
        if alt:
            n *= 2
            if n > 9:
                n -= 9
        total += n
        alt = not alt
    return total % 10 == 0


def _credit_card_guard(match: re.Match) -> bool:
    """Return True when the match looks like a real credit-card number.

    Strips separators then validates digit count and Luhn checksum.
    """
    digits = re.sub(r"[ \-]", "", match.group(0))
    if not (13 <= len(digits) <= 19):
        return False
    return _luhn_valid(digits)


def _key_name_is_unconditional(key_name: str) -> bool:
    """Return True when *key_name* names a value that is a secret by definition."""
    lowered = key_name.lower()
    return any(word in lowered for word in _ALWAYS_REDACT_KEY_WORDS)


def _credential_assignment_guard(key_name: str, secret: str) -> bool:
    """Return True when this assignment should be masked.

    ``password``/``passwd``/``pwd``/``secret`` key names bypass the entropy
    gate entirely — see :data:`_ALWAYS_REDACT_KEY_WORDS`. Everywhere else the
    gate holds: short values (8–11 chars) require entropy ≥ 2.5 bits; longer
    values require entropy ≥ _ENTROPY_THRESHOLD (3.0 bits).
    """
    if _key_name_is_unconditional(key_name):
        return True
    entropy = _shannon_entropy(secret)
    if len(secret) >= _MIN_VALUE_LEN_FOR_ENTROPY:
        return entropy >= _ENTROPY_THRESHOLD
    return entropy >= 2.5


def _aws_secret_guard(secret: str) -> bool:
    """Return True when a 40-char base64-alphabet run looks like an AWS secret.

    The regex can only match the shape, which 40-char lowercase hex (a git
    tree hash) also satisfies. A real AWS secret access key is generated from
    the full base64 alphabet, so requiring an upper, a lower, a digit and at
    least one ``/``, ``+`` or ``=`` separates it from hashes and from ordinary
    identifiers without needing surrounding context.
    """
    return (
        any(c.isupper() for c in secret)
        and any(c.islower() for c in secret)
        and any(c.isdigit() for c in secret)
        and any(c in "/+=" for c in secret)
    )


def _normalize_secret_for_fingerprint(kind: str, secret: str) -> str:
    """Return the canonical form of *secret* used for fingerprinting.

    Credit-card numbers may appear with or without separators; normalising to
    digits-only ensures the same card always yields the same fingerprint
    regardless of formatting.
    """
    if kind == "credit_card":
        return re.sub(r"[ \-]", "", secret)
    return secret


def _should_redact_match(kind: str, match: re.Match, secret: str) -> bool:
    """Apply kind-specific guards beyond the regex match.

    Returns True when the match should be redacted, False when it should be
    treated as a false positive and left in place.
    """
    # Never re-redact an already-masked placeholder.
    if _PLACEHOLDER_RE.search(match.group(0)):
        return False

    if kind == "credit_card" and not _credit_card_guard(match):
        return False

    if kind == "credential_assignment" and not _credential_assignment_guard(
        match.group(1), secret
    ):
        return False

    if kind == "aws_secret_access_key" and not _aws_secret_guard(secret):
        return False

    return True


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def redact(content: str) -> RedactionResult:
    """Detect and mask secrets/PII in *content*.

    This function is:

    - **Pure** — no I/O, no side effects, deterministic.
    - **Idempotent** — calling it a second time on ``result.clean_text``
      returns ``(clean_text=same, findings=(), redacted=False)``.
    - **Safe on empty input** — never raises for string arguments.

    Parameters
    ----------
    content:
        The raw text to scan.

    Returns
    -------
    RedactionResult
        Frozen dataclass with ``clean_text``, ``findings``, and ``redacted``.
    """
    if not content:
        return RedactionResult(clean_text=content or "", findings=(), redacted=False)

    findings: list[Finding] = []
    clean = content

    for kind, pattern in _PATTERNS:
        out: list[str] = []
        last = 0

        for match in pattern.finditer(clean):
            value_group = _VALUE_GROUP_KINDS.get(kind)
            secret = match.group(value_group) if value_group else match.group(0)

            if not _should_redact_match(kind, match, secret):
                continue

            fp = _fingerprint(_normalize_secret_for_fingerprint(kind, secret))
            findings.append(Finding(kind=kind, fingerprint=fp))
            placeholder = f"[REDACTED:{kind}:{fp}]"

            if value_group:
                # Replace only the secret value portion within the full match.
                s, e = match.span(value_group)
                out.append(clean[last:s])
                out.append(placeholder)
                last = e
            else:
                s, e = match.span(0)
                out.append(clean[last:s])
                out.append(placeholder)
                last = e

        out.append(clean[last:])
        clean = "".join(out)

    return RedactionResult(
        clean_text=clean,
        findings=tuple(findings),
        redacted=bool(findings),
    )
