from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Protocol


@dataclass(frozen=True)
class LlmUsage:
    provider: str
    model: str
    input_tokens: int | None
    output_tokens: int | None
    estimated_cost_usd: float | None
    # Implicit-cache hits: subset of input_tokens that Gemini billed at the
    # 25% cached-input rate. Surfaced separately so callers can show savings
    # and the cost estimator can apply the discount.
    cached_tokens: int = 0


class LlmProvider(Protocol):
    """Provider boundary for shared-key Gemini now and BYO/Vertex later."""

    provider_name: str

    def generate_with_tools(
        self,
        *,
        model: str,
        system_instruction: str,
        contents: list[Any],
        tools: list[dict[str, Any]],
        max_tool_calls: int,
        tool_executor: Callable[[str, dict[str, Any]], dict[str, Any]] | None = None,
    ) -> tuple[str, LlmUsage, list[dict[str, Any]]]:
        raise NotImplementedError

    def generate_json(
        self,
        *,
        model: str,
        system_instruction: str,
        contents: list[Any],
        response_schema: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], LlmUsage]:
        """Multimodal structured-output call.

        contents is a list of strings and/or image parts of the form
        ``{"mime_type": "image/png", "data": bytes, "label": "page-3"}``.
        Implementations must return parsed JSON.
        """

        raise NotImplementedError

    def search_json(
        self,
        *,
        model: str,
        system_instruction: str,
        contents: list[Any],
    ) -> tuple[dict[str, Any], LlmUsage]:
        """Google-Search-grounded JSON call.

        Implementations should enable the provider's web-search tool and parse
        the model's JSON response. Used for catalog lookups (MIDI URLs,
        composer disambiguation, etc.) where freshness matters.
        """

        raise NotImplementedError


@dataclass(frozen=True)
class SharedKeyGeminiConfig:
    model: str = "gemini-2.5-pro"
    request_timeout_sec: int = 90
    max_tool_calls: int = 15


ToolExecutor = Callable[[str, dict[str, Any]], dict[str, Any]]

