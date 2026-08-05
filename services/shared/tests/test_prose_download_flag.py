"""``KLIO_PROSE_ALLOW_DOWNLOAD`` used the same bare ``== "1"`` shape.

Executed against the pre-fix code::

    'true' -> allow_download=False
    'yes'  -> allow_download=False
    'on'   -> allow_download=False
    '1'    -> allow_download=True

The consequence is quieter than the OAuth one but the same defect: an operator
who set ``KLIO_PROSE_ALLOW_DOWNLOAD=true`` out of band to warm the weights got
``local_files_only=True`` and the "weights are not available locally; prose
compression is off" warning, with nothing pointing at the flag.
"""

from unittest.mock import patch

import pytest

from shared.compression import prose


def _resolve_with(monkeypatch, flag):
    """Run ``_resolve_weights`` far enough to observe ``local_files_only``."""
    monkeypatch.delenv("KLIO_PROSE_MODEL_PATH", raising=False)
    if flag is None:
        monkeypatch.delenv("KLIO_PROSE_ALLOW_DOWNLOAD", raising=False)
    else:
        monkeypatch.setenv("KLIO_PROSE_ALLOW_DOWNLOAD", flag)

    seen = {}

    def _fake_download(model_id, filename, local_files_only=True):
        seen["local_files_only"] = local_files_only
        return f"/tmp/{filename}"

    import huggingface_hub

    with patch.object(huggingface_hub, "hf_hub_download", _fake_download):
        prose._resolve_weights()
    return seen.get("local_files_only")


@pytest.mark.parametrize("flag", ["1", "true", "TRUE", "yes", "on", " true "])
def test_every_reasonable_way_to_say_yes_allows_the_download(monkeypatch, flag):
    assert _resolve_with(monkeypatch, flag) is False, (
        "allow_download must translate to local_files_only=False"
    )


@pytest.mark.parametrize("flag", [None, "", "0", "false", "no", "off"])
def test_absent_or_falsy_keeps_downloads_off(monkeypatch, flag):
    """Positive control: the default stays local-only.

    This is the important half — the flag exists because a download on a
    request path is unacceptable, so widening the truthy set must not widen
    the default.
    """
    assert _resolve_with(monkeypatch, flag) is True
