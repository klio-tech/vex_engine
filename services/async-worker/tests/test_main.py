"""Tests for app.main's M5 S3-env startup observability check.

No fail-fast: this must only ever log, never raise or block startup.
"""

from app import main


def test_warns_when_s3_endpoint_unset(monkeypatch, caplog):
    monkeypatch.delenv("S3_ENDPOINT", raising=False)
    monkeypatch.delenv("S3_ACCESS_KEY", raising=False)
    monkeypatch.delenv("S3_SECRET_KEY", raising=False)

    with caplog.at_level("ERROR"):
        main._warn_if_s3_env_missing()

    assert any("S3 env not fully configured" in rec.message for rec in caplog.records)


def test_warns_when_s3_endpoint_is_the_localhost_dev_default(monkeypatch, caplog):
    monkeypatch.setenv("S3_ENDPOINT", "http://localhost:9000")
    monkeypatch.setenv("S3_ACCESS_KEY", "agentguard")
    monkeypatch.setenv("S3_SECRET_KEY", "agentguard_dev")

    with caplog.at_level("ERROR"):
        main._warn_if_s3_env_missing()

    assert any("S3 env not fully configured" in rec.message for rec in caplog.records)


def test_no_warning_when_s3_env_is_fully_and_explicitly_configured(monkeypatch, caplog):
    monkeypatch.setenv("S3_ENDPOINT", "https://s3.us-east-1.amazonaws.com")
    monkeypatch.setenv("S3_ACCESS_KEY", "AKIAEXAMPLE")
    monkeypatch.setenv("S3_SECRET_KEY", "secret-example")

    with caplog.at_level("ERROR"):
        main._warn_if_s3_env_missing()

    assert not any("S3 env not fully configured" in rec.message for rec in caplog.records)


def test_warn_never_raises_and_does_not_block_startup(monkeypatch):
    """No fail-fast: even with everything unset, this must not raise."""
    monkeypatch.delenv("S3_ENDPOINT", raising=False)
    monkeypatch.delenv("S3_ACCESS_KEY", raising=False)
    monkeypatch.delenv("S3_SECRET_KEY", raising=False)
    main._warn_if_s3_env_missing()  # must not raise
