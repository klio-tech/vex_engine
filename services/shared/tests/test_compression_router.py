"""The router decides what a blob is and who gets to compress it.

Two decisions carry the risk:

* **The size floor.** Below ~500 tokens there is nothing worth winning, and
  the attempt costs a model call or a parse for no gain — while still carrying
  every risk of mangling something. Small input is passthrough, always.
* **Unrecognised means passthrough.** Not "closest match". A router that
  guesses feeds a Ruby file to the Python grammar or a log file to the JSON
  compressor, and the caller has no signal that anything went wrong.
"""

import json

import pytest

from shared.compression.router import (
    MIN_TOKENS_TO_COMPRESS,
    Compressor,
    compress,
    estimate_tokens,
    select_compressor,
)

# Comfortably over the size floor.
_FILLER = (
    "The deployment failed because the database connection pool was exhausted. "
    "The error surfaced in the checkout service during the evening peak load. "
)
LONG_PROSE = _FILLER * 20

LONG_PYTHON = "import os\n\n\n" + "".join(
    f"def handler_{i}(request, timeout=30):\n"
    f"    payload = request.json()\n"
    f"    result = process(payload, timeout)\n"
    f"    return result\n\n\n"
    for i in range(40)
)

LONG_JSON = json.dumps(
    {
        "memories": [
            {
                "id": f"row-{i}",
                "content": f"memory number {i} about the deployment and the pool",
                "scope": "org",
                "provenance": "EXTRACTED",
                "confidence": None,
                "space_id": None,
            }
            for i in range(30)
        ]
    }
)


# --------------------------------------------------------------------------
# The size floor
# --------------------------------------------------------------------------


def test_the_floor_is_the_documented_500_tokens():
    assert MIN_TOKENS_TO_COMPRESS == 500


@pytest.mark.parametrize(
    "text",
    [
        "def f(a):\n    return a + 1\n",
        '{"memories": [{"id": "a", "content": "b"}]}',
        "A short paragraph that is not worth anybody's time to compress.",
        "",
    ],
)
def test_anything_under_the_floor_is_passthrough(text):
    assert select_compressor(text) is Compressor.PASSTHROUGH
    result = compress(text)
    assert result.text == text
    assert result.compressor == Compressor.PASSTHROUGH.value
    assert result.lossy is False


def test_the_floor_is_measured_in_estimated_tokens():
    assert estimate_tokens("") == 0
    assert estimate_tokens("a" * 400) == 100


# --------------------------------------------------------------------------
# Selection
# --------------------------------------------------------------------------


def test_json_routes_to_the_structural_compressor():
    assert select_compressor(LONG_JSON) is Compressor.STRUCTURAL


def test_source_code_routes_to_the_code_compressor():
    assert select_compressor(LONG_PYTHON) is Compressor.CODE


def test_prose_routes_to_the_prose_compressor():
    assert select_compressor(LONG_PROSE) is Compressor.PROSE


def test_a_diff_is_not_claimed_by_the_code_compressor():
    """Diffs are line-oriented text about code, not code. The AST compressor
    cannot parse one, so it must not be handed one."""
    diff = "diff --git a/x.py b/x.py\n" + "-old line here\n+new line here\n" * 60
    assert select_compressor(diff) is not Compressor.CODE


# --------------------------------------------------------------------------
# Results
# --------------------------------------------------------------------------


def test_structural_reports_itself_lossless_and_shrinks_the_payload():
    result = compress(LONG_JSON)
    assert result.compressor == Compressor.STRUCTURAL.value
    assert result.lossy is False
    assert result.compressed_tokens < result.original_tokens


def test_structural_compression_is_reversible():
    """A lossless claim is tested, not asserted.

    Every row of the original is reconstructible from the compressed payload:
    hoisted constants come back, and a dropped key means it was null.
    """
    from shared.compression.structural import expand_rows

    original = json.loads(LONG_JSON)
    compressed = json.loads(compress(LONG_JSON).text)
    assert expand_rows(compressed) == original


def test_code_reports_itself_lossy_and_keeps_signatures():
    result = compress(LONG_PYTHON)
    assert result.compressor == Compressor.CODE.value
    assert result.lossy is True
    assert "def handler_0(request, timeout=30):" in result.text
    assert "request.json()" not in result.text
    assert result.compressed_tokens < result.original_tokens


def test_prose_reports_itself_lossy(monkeypatch):
    from shared.compression import prose as prose_module

    monkeypatch.setattr(
        prose_module,
        "_score_tokens",
        lambda text: [(0, len(text) // 2, 0.9), (len(text) // 2, len(text), 0.0)],
    )
    result = compress(LONG_PROSE)
    assert result.compressor == Compressor.PROSE.value
    assert result.lossy is True


def test_a_compressor_that_achieves_nothing_reports_no_loss(monkeypatch):
    """If the output is byte-identical to the input, nothing was lost — and
    the caller must not be handed a lossy flag it would then honour with a
    pointless cache write."""
    from shared.compression import prose as prose_module

    monkeypatch.setattr(prose_module, "_score_tokens", lambda _text: None)
    result = compress(LONG_PROSE)
    assert result.text == LONG_PROSE
    assert result.lossy is False


def test_a_compressor_that_inflates_is_rejected(monkeypatch):
    """The guard is `no bigger than it went in`, not `not identical`.

    `out == text` only catches a compressor that did literally nothing. A
    compressor that made the content LARGER passed straight through and the
    inflation was reported as a saving. This is the backstop for every
    compressor, present and future: whatever it does, the caller is never
    handed more tokens than it asked to be relieved of.
    """
    from shared.compression import router as router_module

    monkeypatch.setattr(
        router_module, "compress_json_text", lambda text: text + "x" * 100
    )
    result = compress(LONG_JSON)
    assert result.text == LONG_JSON
    assert result.compressor == Compressor.PASSTHROUGH.value
    assert result.compressed_tokens == result.original_tokens


def test_a_lossy_compressor_that_inflates_is_also_rejected(monkeypatch):
    from shared.compression import router as router_module

    monkeypatch.setattr(
        router_module, "compress_code", lambda text: text + "# padding\n" * 50
    )
    result = compress(LONG_PYTHON)
    assert result.text == LONG_PYTHON
    assert result.lossy is False, "an inflated result must not claim a handle"


def test_compression_never_returns_more_than_it_was_given():
    """The property, over every fixture in this module and a non-ASCII one."""
    cjk = json.dumps(
        {
            "memories": [
                {"id": f"id-{i}", "content": "配置文件已更新，请检查数据库连接池设置。"}
                for i in range(40)
            ],
            "count": 40,
        },
        ensure_ascii=False,
        indent=2,
    )
    for text in (LONG_JSON, LONG_PYTHON, LONG_PROSE, cjk):
        result = compress(text)
        assert len(result.text) <= len(text), (
            f"{len(text)} chars in, {len(result.text)} chars out"
        )
        assert result.compressed_tokens <= result.original_tokens


def test_token_counts_describe_the_actual_texts():
    result = compress(LONG_PYTHON)
    assert result.original_tokens == estimate_tokens(LONG_PYTHON)
    assert result.compressed_tokens == estimate_tokens(result.text)


# --------------------------------------------------------------------------
# Fail-open
# --------------------------------------------------------------------------


def test_a_compressor_that_raises_returns_the_input_unchanged(monkeypatch):
    from shared.compression import router as router_module

    def boom(*_args, **_kwargs):
        raise RuntimeError("grammar exploded")

    monkeypatch.setattr(router_module, "compress_code", boom)
    result = compress(LONG_PYTHON)
    assert result.text == LONG_PYTHON
    assert result.lossy is False


def test_selection_that_raises_falls_back_to_passthrough(monkeypatch):
    from shared.compression import router as router_module

    def boom(_text):
        raise RuntimeError("detector exploded")

    monkeypatch.setattr(router_module, "select_compressor", boom)
    result = compress(LONG_PYTHON)
    assert result.text == LONG_PYTHON
    assert result.compressor == Compressor.PASSTHROUGH.value


def test_non_string_input_is_returned_unchanged():
    result = compress(None)  # type: ignore[arg-type]
    assert result.text is None
    assert result.compressor == Compressor.PASSTHROUGH.value


# --------------------------------------------------------------------------
# The correctness floor, end to end through the router
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "needle",
    [
        "/Users/dev/app/loader.py",
        "https://api.klio.tech/v1/recall",
        "MAX_POOL_SIZE",
        "parse_config(path, strict=True)",
    ],
)
def test_technical_spans_survive_whichever_compressor_is_chosen(monkeypatch, needle):
    from shared.compression import prose as prose_module

    monkeypatch.setattr(
        prose_module, "_score_tokens", lambda text: [(0, len(text), 0.0)]
    )
    text = LONG_PROSE + f" The detail was {needle} throughout the incident."
    assert needle in compress(text).text


def test_a_stack_trace_survives_the_router(monkeypatch):
    from shared.compression import prose as prose_module

    monkeypatch.setattr(
        prose_module, "_score_tokens", lambda text: [(0, len(text), 0.0)]
    )
    trace = (
        "Traceback (most recent call last):\n"
        '  File "/Users/dev/app/main.py", line 42, in handler\n'
        "    return loader.load(path)\n"
        "ValueError: empty\n"
    ) * 12
    out = compress(trace).text
    assert "/Users/dev/app/main.py" in out
    assert "loader.load(path)" in out


# --------------------------------------------------------------------------
# What "prose" is, and what only looks like it
# --------------------------------------------------------------------------


def test_a_log_file_is_prose():
    """Logs are the spec's primary prose target and are mostly not words.

    An earlier gate measured the fraction of letters in the text and ranked a
    base64 blob (0.82) as more prose-like than a log file (0.63). Function
    words are what actually separates them.
    """
    logs = "".join(
        f"2026-08-01T14:22:{i:02d} INFO  checkout.service pool acquire "
        f"waited 12ms for a free connection\n"
        f"2026-08-01T14:22:{i:02d} ERROR checkout.service timed out while "
        f"waiting on the pool\n"
        for i in range(40)
    )
    assert select_compressor(logs) is Compressor.PROSE


def test_a_base64_blob_is_not_prose():
    import base64

    blob = base64.b64encode(bytes(range(256)) * 20).decode()
    assert select_compressor(blob) is Compressor.PASSTHROUGH


def test_a_csv_of_identifiers_is_not_prose():
    csv = "\n".join(f"{i},{i * 3},alpha,beta,gamma,delta" for i in range(400))
    assert select_compressor(csv) is Compressor.PASSTHROUGH


def test_a_module_of_only_class_definitions_is_still_code():
    """Detection asks "is this Python?", not "is there a body to drop?".

    A file of nothing but dataclasses has nothing to compress. It must still
    be recognised as code, or it falls through to the prose model — which is
    the guess the whole design refuses to make.
    """
    source = "from dataclasses import dataclass\n\n\n" + "".join(
        f"@dataclass\nclass Row{i}:\n    identifier: str\n    label: str\n"
        f"    weight: float\n\n\n"
        for i in range(40)
    )
    assert select_compressor(source) is Compressor.CODE
    result = compress(source)
    assert result.text == source
    assert result.lossy is False


def test_allow_lossy_false_declines_before_the_compressor_runs(monkeypatch):
    """Suppressing a lossy result afterwards would still have paid for it."""
    from shared.compression import router as router_module

    def must_not_run(*_args, **_kwargs):
        raise AssertionError("the lossy compressor was called anyway")

    monkeypatch.setattr(router_module, "compress_code", must_not_run)
    result = compress(LONG_PYTHON, allow_lossy=False)
    assert result.text == LONG_PYTHON
    assert result.lossy is False


def test_allow_lossy_false_still_runs_the_lossless_compressor():
    result = compress(LONG_JSON, allow_lossy=False)
    assert result.compressor == Compressor.STRUCTURAL.value
    assert result.compressed_tokens < result.original_tokens
