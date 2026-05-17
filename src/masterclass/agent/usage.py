from __future__ import annotations


_PRICING_PER_MILLION = {
    "gemini-3.1-pro": (1.25, 12.00),
    "gemini-3.1-pro-preview": (1.25, 12.00),
    "gemini-3-pro": (1.25, 12.00),
    "gemini-3-pro-preview": (1.25, 12.00),
    "gemini-pro-latest": (1.25, 12.00),
    "gemini-3-flash": (0.30, 2.50),
    "gemini-3-flash-preview": (0.30, 2.50),
    "gemini-3.1-flash": (0.30, 2.50),
    "gemini-3.1-flash-lite": (0.10, 0.40),
    "gemini-flash-latest": (0.30, 2.50),
    "gemini-flash-lite-latest": (0.10, 0.40),
    "gemini-2.5-pro": (1.25, 10.00),
    "gemini-2.5-flash": (0.30, 2.50),
    "gemini-2.5-flash-lite": (0.10, 0.40),
    "gemini-2.0-flash": (0.10, 0.40),
}


def estimate_cost_usd(
    model: str,
    input_tokens: int | None,
    output_tokens: int | None,
    *,
    cached_tokens: int = 0,
) -> float | None:
    """Estimate USD cost for one generation.

    ``cached_tokens`` is the subset of ``input_tokens`` that Gemini billed at
    the implicit-cache rate (25% of normal input). When present, those
    tokens are charged at the lower rate instead of double-counting at full
    price.
    """
    if input_tokens is None or output_tokens is None:
        return None
    rates = None
    for prefix, candidate in _PRICING_PER_MILLION.items():
        if model.startswith(prefix):
            rates = candidate
            break
    if rates is None:
        return None
    input_rate, output_rate = rates
    cached = max(0, min(int(cached_tokens or 0), int(input_tokens)))
    uncached = int(input_tokens) - cached
    cached_rate = input_rate * 0.25
    return round(
        (uncached * input_rate + cached * cached_rate) / 1_000_000
        + output_tokens * output_rate / 1_000_000,
        6,
    )


def summarize_usage(model: str, usage_log: list[dict]) -> dict:
    total_input = sum(int(u.get("prompt_token_count", 0) or 0) for u in usage_log)
    total_output = sum(int(u.get("candidates_token_count", 0) or 0) for u in usage_log)
    total_cached = sum(int(u.get("cached_content_token_count", 0) or 0) for u in usage_log)
    return {
        "model": model,
        "per_turn": usage_log,
        "total_input": total_input,
        "total_output": total_output,
        "total_cached": total_cached,
        "estimated_cost_usd": estimate_cost_usd(model, total_input, total_output, cached_tokens=total_cached),
    }

