"""Sliding-window rate limit per (user, client) for OAuth principals."""

from app.oauth import OAuthRateLimiter


def test_allows_under_limit():
    limiter = OAuthRateLimiter()
    for _ in range(5):
        retry_after = limiter.check("u-1:c-1", rpm=5, now=100.0)
        assert retry_after is None


def test_blocks_over_limit_with_retry_after():
    limiter = OAuthRateLimiter()
    for _ in range(5):
        assert limiter.check("u-1:c-1", rpm=5, now=100.0) is None
    retry_after = limiter.check("u-1:c-1", rpm=5, now=100.0)
    assert retry_after is not None and retry_after > 0


def test_window_slides():
    limiter = OAuthRateLimiter()
    for _ in range(5):
        assert limiter.check("u-1:c-1", rpm=5, now=100.0) is None
    assert limiter.check("u-1:c-1", rpm=5, now=100.0) is not None
    # 61 seconds later the window has slid past the burst.
    assert limiter.check("u-1:c-1", rpm=5, now=161.0) is None


def test_keys_are_isolated():
    limiter = OAuthRateLimiter()
    for _ in range(5):
        assert limiter.check("u-1:c-1", rpm=5, now=100.0) is None
    assert limiter.check("u-2:c-1", rpm=5, now=100.0) is None
