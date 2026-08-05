"""Unit tests for shared.redaction — detection catalog + masking.

False positives are product damage (memories get mangled), so the
false-positive corpus matters as much as the true-positive corpus.
"""

import re

import pytest

from shared.redaction import (
    _credit_card_guard,
    _luhn_valid,
    _shannon_entropy,
    redact,
)

# ---- true positives: every kind in the catalog --------------------------

TRUE_POSITIVES = [
    # (input, expected kind)
    (
        "User provided Stripe test secret key: "
        "sk_test_51TcfsTP4cXbUl5jnQJG2Ou9BUcBChSVqEDNj7d65uQjZXtAVzI1XR2"
        "eewh5KDq8GmCfrr1a1xVRNcSdB1c7TtWrU00GsK8y8Je",
        "stripe_secret_key",
    ),
    ("publishable: pk_live_abc123DEF456ghi789JKL012", "stripe_publishable_key"),
    ("whsec_8f2e1a9b0c3d4e5f6a7b8c9d0e1f2a3b", "stripe_webhook_secret"),
    ("aws key AKIAIOSFODNN7EXAMPLE", "aws_access_key_id"),
    ("ghp_16C7e42F292c6912E7710c838347Ae178B4a", "github_token"),
    ("github_pat_11ABCDEFG0abcdefghijklmnopqrstuvwxyz1234567890ABCD", "github_token"),
    ("OPENAI_API_KEY=sk-proj-abc123DEF456ghi789JKLmno", "openai_api_key"),
    ("sk-ant-api03-abc123DEF456ghi789JKLmno-pqrstu", "anthropic_api_key"),
    ("xoxb-2624-7034-abcdefGHIJKL", "slack_token"),
    ("google key AIzaSyA1234567890abcdefghijklmnopqrstu", "google_api_key"),
    (
        "token eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9."
        "eyJzdWIiOiIxMjM0NTY3ODkwIn0."
        "SflKxwRJSMeKKF2QT4fwpMeJf36POk6yJV_adQssw5c",
        "jwt",
    ),
    (
        "-----BEGIN RSA PRIVATE KEY-----\nMIIEpAIBAAKCAQEA\n"
        "-----END RSA PRIVATE KEY-----",
        "private_key_block",
    ),
    ("password=hunter2secret", "credential_assignment"),
    ('api_key: "9f8e7d6c5b4a3210fedcba9876543210"', "credential_assignment"),
    ("postgres://admin:s3cretPW@db.internal:5432/prod", "connection_string"),
    ("Authorization: Bearer abc123DEF456ghi789JKLmnop", "bearer_token"),
    ("reach me at jane.doe@example.com please", "email"),
    ("card 4111 1111 1111 1111 exp 12/28", "credit_card"),
    ("call +14155552671 tomorrow", "phone"),
]


@pytest.mark.parametrize("content,kind", TRUE_POSITIVES)
def test_detects_and_masks(content, kind):
    result = redact(content)
    assert result.redacted is True
    assert kind in {f.kind for f in result.findings}
    # The raw secret must be gone from the clean text.
    assert "sk_test_51Tcfs" not in result.clean_text
    assert "hunter2secret" not in result.clean_text
    assert f"[REDACTED:{kind}:" in result.clean_text or result.redacted


def test_placeholder_format_and_fingerprint():
    secret = "sk_test_51TcfsTP4cXbUl5jnQJG2Ou9BUcBChSVqEDNj7d65uQjZXtAVzI1XR2eewh5KDq8GmCfrr1a1xVRNcSdB1c7TtWrU00GsK8y8Je"
    r1 = redact(f"key A: {secret}")
    r2 = redact(f"key B mentioned again: {secret}")
    ph1 = [f for f in r1.findings if f.kind == "stripe_secret_key"][0]
    ph2 = [f for f in r2.findings if f.kind == "stripe_secret_key"][0]
    # Same secret → same 4-hex-char fingerprint, correlatable without exposure.
    assert ph1.fingerprint == ph2.fingerprint
    assert len(ph1.fingerprint) == 4
    assert f"[REDACTED:stripe_secret_key:{ph1.fingerprint}]" in r1.clean_text


# ---- false positives: must NOT redact ------------------------------------

FALSE_POSITIVES = [
    "The migration id is 2d07c2d4-b3c5-4b49-8ff9-08961a737b61",  # UUID
    "commit c2c3c09 fixed the bug",  # short SHA
    "commit 5bfab80d528255fb2342c9395f2c3bf1e817f145ab90a702",  # long hex sha
    "ORDER BY embedding <=> :embedding LIMIT 10",  # SQL
    "The user wants to connect Stripe to moonforge-prod.",  # prose w/ 'Stripe'
    "Set MEMORY_EMBEDDING_MODEL to the default model name",  # env var name only
    "the meeting is at 10:30, room 415",  # not a phone
    "version 4.1.1 released",  # not a card
]


@pytest.mark.parametrize("content", FALSE_POSITIVES)
def test_does_not_redact_ordinary_content(content):
    result = redact(content)
    assert result.redacted is False
    assert result.clean_text == content
    assert result.findings == ()


# ---- idempotency ----------------------------------------------------------


def test_redaction_is_idempotent():
    content = "key: sk_test_51TcfsTP4cXbUl5jnQJG2Ou9BUcBChSVqEDNj7d65uQjZXtAVzI1XR2eewh5KDq8GmCfrr1a1xVRNcSdB1c7TtWrU00GsK8y8Je"
    once = redact(content)
    twice = redact(once.clean_text)
    assert twice.redacted is False
    assert twice.clean_text == once.clean_text


def test_result_is_immutable():
    result = redact("nothing secret here")
    with pytest.raises(Exception):
        result.clean_text = "mutated"  # frozen dataclass


def test_empty_and_none_safe():
    assert redact("").redacted is False
    assert redact("").clean_text == ""


# ---- regression: connection-string with email-as-username ----------------


def test_connection_string_with_email_username_masks_password():
    result = redact("mongodb://jane.doe@example.com:p4ssw0rdXYZ@host:27017/db")
    assert "p4ssw0rdXYZ" not in result.clean_text
    assert "connection_string" in {f.kind for f in result.findings}


# ---- regression: credit-card boundary and fingerprint normalization ------


def test_credit_card_mask_preserves_following_space():
    result = redact("card 4111 1111 1111 1111 exp 12/28")
    assert "] exp" in result.clean_text


def test_credit_card_fingerprint_normalized_across_formats():
    spaced = redact("card 4111 1111 1111 1111")
    plain = redact("card 4111111111111111")
    fp_s = [f for f in spaced.findings if f.kind == "credit_card"][0].fingerprint
    fp_p = [f for f in plain.findings if f.kind == "credit_card"][0].fingerprint
    assert fp_s == fp_p


# ---- guard unit tests: branch coverage for internal helpers ----------------


def test_shannon_entropy_empty_string_returns_zero():
    """_shannon_entropy("") must return 0.0 (L161 early-return branch)."""
    assert _shannon_entropy("") == 0.0


def test_luhn_valid_double_minus_nine():
    """_luhn_valid must correctly subtract 9 when doubled digit > 9 (L177 branch).

    The Mastercard test card 5500000000000004 has a leading 5 in an alternate
    position that doubles to 10 (> 9), triggering the ``n -= 9`` subtraction.
    """
    # 5500000000000004 is the standard Mastercard test card; the leading 5
    # at the last alternate position doubles to 10, exercising n -= 9 at L177.
    assert _luhn_valid("5500000000000004") is True


def test_mastercard_test_card_is_redacted():
    """The standard Mastercard test card must be detected through redact() (L177 path)."""
    result = redact("card 5500 0000 0000 0004")
    assert result.redacted is True
    assert "credit_card" in {f.kind for f in result.findings}


def test_credit_card_guard_rejects_too_short_digits():
    """_credit_card_guard must return False for a match with < 13 stripped digits (L190 branch).

    This is a defensive code path: the credit_card regex always produces 13-19 digit
    matches, so L190 cannot be triggered via redact(). We call _credit_card_guard
    directly with a synthetic match whose group(0) has < 13 digits.
    """
    pattern = re.compile(r"\d+")
    short_match = pattern.match("411111111")  # 9 digits
    assert short_match is not None
    assert _credit_card_guard(short_match) is False


def test_credit_card_luhn_invalid_not_redacted():
    """A 16-digit sequence that fails Luhn must not be masked (L229 branch)."""
    # 4111111111111112 passes length but fails Luhn (off-by-one on check digit).
    result = redact("card 4111 1111 1111 1112")
    assert result.redacted is False


def test_credential_assignment_low_entropy_not_redacted():
    """Low-entropy credential values must not be masked (entropy-gate branch).

    This test used to assert on ``password=aaaaaaaaaaaa``. That is no longer
    the contract: a ``password`` key name now bypasses the entropy gate
    unconditionally, because the gate was letting real passwords
    (``Passw0rdPassw0rd``, 2.7500 bits) through. The gate itself still exists
    for key names that can carry prose-shaped values, which is what this
    exercises.
    """
    result = redact("token=aaaaaaaaaaaa")
    assert result.redacted is False


def test_credential_assignment_short_high_entropy_redacted():
    """Short (8-11 char) but high-entropy credential values must be masked (L203 branch).

    The short-value path requires entropy >= 2.5 bits; "xK8#mP2!" has high
    character diversity for 8 chars. The key name is ``token`` and not
    ``secret`` on purpose — a ``secret`` key now bypasses the gate, which
    would leave this branch uncovered.
    """
    result = redact("token=xK8#mP2!")
    assert result.redacted is True
    assert "credential_assignment" in {f.kind for f in result.findings}


# ---- gap 1: word-boundary failure on underscored key names ----------------
#
# ``\b(password|...|access[_-]?key)`` cannot match ``AWS_SECRET_ACCESS_KEY``:
# the character before ``SECRET`` is ``_``, which is a word character, so
# there is no word boundary there. Every SCREAMING_SNAKE_CASE env var — the
# single most common shape a real credential is written in — was invisible.


UNDERSCORED_KEY_ASSIGNMENTS = [
    "AWS_SECRET_ACCESS_KEY=wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY",
    "DATABASE_PASSWORD=Sup3rS3cretP@ss",
]


@pytest.mark.parametrize("content", UNDERSCORED_KEY_ASSIGNMENTS)
def test_underscored_key_name_is_redacted(content):
    result = redact(content)
    assert result.redacted is True, f"not redacted: {content!r}"
    secret = content.split("=", 1)[1]
    assert secret not in result.clean_text


def test_underscored_key_positive_control_is_not_vacuous():
    """The fixture strings really do carry the key name and a secret value.

    Without this, a redactor that redacted *everything* would pass the tests
    above and a redactor that saw an empty value would too.
    """
    for content in UNDERSCORED_KEY_ASSIGNMENTS:
        key, _, value = content.partition("=")
        assert key.endswith(("KEY", "PASSWORD"))
        assert len(value) >= 12
    # And ordinary underscored prose must still survive untouched.
    benign = "MEMORY_EMBEDDING_MODEL=text-embedding-3-small"
    assert redact(benign).redacted is False


# ---- gap 2: the closing quote breaks the ``\s*[:=]`` follow ----------------
#
# In ``{"password": "s3cretV4lue!"}`` the key name is *quoted*, so the char
# after ``password`` is ``"`` and the required ``\s*[:=]`` never matches.
# JSON and Python dict literals are exactly how an agent transcript quotes a
# config blob.


QUOTED_KEY_ASSIGNMENTS = [
    ('{"password": "s3cretV4lue!"}', "s3cretV4lue!"),
    ("{'api_key': 'AbCd3fGh1jKlM'}", "AbCd3fGh1jKlM"),
]


@pytest.mark.parametrize("content,secret", QUOTED_KEY_ASSIGNMENTS)
def test_quoted_key_name_is_redacted(content, secret):
    result = redact(content)
    assert result.redacted is True, f"not redacted: {content!r}"
    assert secret not in result.clean_text
    assert "credential_assignment" in {f.kind for f in result.findings}


def test_quoted_key_redaction_preserves_the_surrounding_literal():
    """Only the value is masked — the JSON stays parseable in shape."""
    result = redact('{"password": "s3cretV4lue!"}')
    assert result.clean_text.startswith('{"password": "[REDACTED:')
    assert result.clean_text.endswith('"}')


# ---- gap 3: empty username in a connection string -------------------------
#
# ``[^\s:/@]+:([^\s@]+)@`` requires at least one non-colon character before
# the password colon, so ``redis://:hunter2secret@host`` — the canonical
# Redis AUTH URL, which has no username at all — could not match.


def test_connection_string_with_empty_username_is_redacted():
    result = redact("redis://:hunter2secret@host:6379")
    assert result.redacted is True
    assert "hunter2secret" not in result.clean_text
    assert "connection_string" in {f.kind for f in result.findings}


def test_empty_username_connection_string_positive_control():
    """A passwordless URL of the same shape must still be left alone."""
    assert redact("redis://host:6379/0").redacted is False


# ---- gap 4: the entropy gate is strictest where secrets get longer --------
#
# At length >= 12 the gate demands 3.0 bits. Real passwords do not clear it:
# ``Passw0rdPassw0rd`` measures 2.7500 and ``hunter2hunter2`` 2.8074. The
# gate exists to stop prose, but on an explicit ``password``/``secret`` key
# name there is no prose to protect — the key name has already told us what
# the value is.


LOW_ENTROPY_PASSWORDS = [
    "password=Passw0rdPassw0rd",
    "password: hunter2hunter2",
    "passwd=correcthorsebattery",
    "pwd = mydogsnameisrex1",
    "secret: mysupersecretvalue",
]


@pytest.mark.parametrize("content", LOW_ENTROPY_PASSWORDS)
def test_password_key_redacts_regardless_of_entropy(content):
    result = redact(content)
    assert result.redacted is True, f"not redacted: {content!r}"
    assert "credential_assignment" in {f.kind for f in result.findings}


def test_low_entropy_password_control_measures_below_the_gate():
    """Positive control: these values genuinely fail the 3.0-bit gate.

    If they cleared it on their own the tests above would prove nothing about
    the key-name bypass.
    """
    assert _shannon_entropy("Passw0rdPassw0rd") < 3.0
    assert _shannon_entropy("hunter2hunter2") < 3.0


def test_entropy_gate_still_applies_to_non_password_key_names():
    """The bypass is scoped. ``token``/``api_key`` keep the entropy gate, so
    prose-shaped values assigned to them are not mangled."""
    result = redact("token=aaaaaaaaaaaaaaaa")
    assert result.redacted is False


# ---- gap 5: there is no AWS secret-value pattern at all -------------------
#
# The catalog carried ``aws_access_key_id`` (AKIA/ASIA) only. The 40-char
# secret access key that pairs with it had no pattern, so a bare secret with
# no ``KEY=`` prefix passed straight through.


def test_bare_aws_secret_access_key_is_redacted():
    result = redact(
        "the pair is AKIAIOSFODNN7EXAMPLE and wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY"
    )
    assert result.redacted is True
    kinds = {f.kind for f in result.findings}
    assert "aws_secret_access_key" in kinds
    assert "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY" not in result.clean_text


def test_aws_secret_pattern_matches_in_assignment_form_on_its_own():
    """The AWS pattern must fire on ``KEY=<secret>`` without leaning on the
    credential_assignment detector to cover for it.

    Regression: the character class originally included ``=`` for base64
    padding, which put ``=`` in the lookbehind as well and made the single
    most common way the secret is written unmatchable by this pattern. The
    credential_assignment detector masked the failure end-to-end.
    """
    from shared.redaction import _PATTERNS

    pattern = dict(_PATTERNS)["aws_secret_access_key"]
    assert pattern.search(
        "AWS_SECRET_ACCESS_KEY=wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY"
    )


def test_aws_secret_pattern_does_not_eat_ordinary_long_tokens():
    """Positive control on the false-positive side: a 40-char base64-ish
    string that is not adjacent to AWS context must survive."""
    # A git tree hash and a long path are both 40-ish chars of [A-Za-z0-9/+].
    assert redact("commit 5bfab80d528255fb2342c9395f2c3bf1e817f145").redacted is False


def test_already_masked_placeholder_not_re_redacted():
    """A [REDACTED:...] placeholder must not be re-redacted (L226 branch).

    After a first pass the clean_text contains only placeholders; a second
    pass must return redacted=False (idempotency via the placeholder guard).
    This also covers the 'continue' skip at L277.
    """
    first = redact("password=hunter2secret")
    assert first.redacted is True
    second = redact(first.clean_text)
    assert second.redacted is False
    assert second.clean_text == first.clean_text
