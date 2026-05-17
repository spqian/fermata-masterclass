"""Cached-token discount in cost estimators (Gemini implicit caching)."""
from __future__ import annotations


def test_estimate_cost_usd_applies_cache_discount():
    from masterclass.agent.usage import estimate_cost_usd

    # gemini-2.5-pro: input $1.25 / Mtok, output $10 / Mtok.
    base = estimate_cost_usd("gemini-2.5-pro", 100_000, 1_000)
    assert abs(base - 0.135) < 1e-6
    discounted = estimate_cost_usd("gemini-2.5-pro", 100_000, 1_000, cached_tokens=50_000)
    # 50k uncached * 1.25 + 50k cached * 0.3125 (25% of 1.25) + 1k * 10, /1M
    # = (62500 + 15625 + 10000) / 1_000_000 = 0.088125
    assert abs(discounted - 0.088125) < 1e-6
    assert discounted < base


def test_estimate_chat_cost_applies_cache_discount():
    from masterclass.engine.teach_chat import estimate_chat_cost

    base = estimate_chat_cost("gemini-2.5-pro", 21_000, 500)
    cached = estimate_chat_cost("gemini-2.5-pro", 21_000, 500, cached_tokens=15_000)
    assert cached < base


def test_cached_tokens_propagates_through_llm_usage():
    from masterclass.agent.llm import LlmUsage
    u = LlmUsage(provider="gemini", model="gemini-2.5-pro", input_tokens=1000, output_tokens=100, estimated_cost_usd=0.01, cached_tokens=500)
    assert u.cached_tokens == 500
    u2 = LlmUsage(provider="gemini", model="gemini-2.5-pro", input_tokens=1000, output_tokens=100, estimated_cost_usd=0.01)
    assert u2.cached_tokens == 0
