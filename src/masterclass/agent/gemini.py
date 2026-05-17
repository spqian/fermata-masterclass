from __future__ import annotations

import json
import os
import time
from typing import Any

from masterclass.agent.llm import LlmUsage, SharedKeyGeminiConfig, ToolExecutor
from masterclass.agent.usage import estimate_cost_usd


class SharedKeyGeminiProvider:
    """Google Gemini provider using the app's shared API key.

    BYO user keys and Vertex tenant credentials can implement the same provider
    protocol later without changing the teacher-agent/tool layers.
    """

    provider_name = "google-gemini-shared-key"

    def __init__(self, *, api_key: str | None = None, config: SharedKeyGeminiConfig | None = None) -> None:
        self.api_key = (api_key or os.environ.get("GEMINI_API_KEY") or "").strip()
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY is required for SharedKeyGeminiProvider")
        self.config = config or SharedKeyGeminiConfig()

    def generate_with_tools(
        self,
        *,
        model: str,
        system_instruction: str,
        contents: list[Any],
        tools: list[dict[str, Any]],
        max_tool_calls: int,
        tool_executor: ToolExecutor | None = None,
    ) -> tuple[str, LlmUsage, list[dict[str, Any]]]:
        try:
            from google import genai
            from google.genai import types
        except ImportError as exc:
            raise RuntimeError("Install LLM dependencies with: pip install -e .[llm]") from exc

        client = genai.Client(api_key=self.api_key)
        tool_config = types.Tool(function_declarations=tools) if tools else None
        config_kwargs: dict[str, Any] = {
            "system_instruction": system_instruction,
            "temperature": 0.4,
            "http_options": types.HttpOptions(timeout=self.config.request_timeout_sec * 1000),
        }
        if tool_config is not None:
            config_kwargs["tools"] = [tool_config]
        config = types.GenerateContentConfig(**config_kwargs)

        history: list[Any] = []
        user_content = types.UserContent(parts=[_to_part(types, part) for part in contents])
        response = self._send(client, model, history, user_content, config)
        usage_log = [_usage_record(response, "initial")]
        tool_calls_log: list[dict[str, Any]] = []

        for turn in range(1, max_tool_calls + 1):
            calls, text = _extract_calls_and_text(response)
            if not calls:
                usage = _usage_from_log(self.provider_name, model, usage_log)
                return text, usage, tool_calls_log
            if tool_executor is None:
                usage = _usage_from_log(self.provider_name, model, usage_log)
                tool_calls_log.extend({"turn": turn, "tool": c.name, "args": dict(c.args or {}), "status": "not_executed"} for c in calls)
                return text, usage, tool_calls_log

            response_parts = []
            for call in calls:
                args = dict(call.args or {})
                started = time.time()
                try:
                    result = tool_executor(call.name, args)
                    status = "error" if isinstance(result, dict) and "error" in result else "ok"
                except Exception as exc:
                    result = {"error": f"{type(exc).__name__}: {exc}"}
                    status = "error"
                result = _cap_result(result)
                tool_calls_log.append({
                    "turn": turn,
                    "tool": call.name,
                    "args": args,
                    "status": status,
                    "duration_sec": round(time.time() - started, 3),
                })
                response_parts.append(types.Part.from_function_response(name=call.name, response={"result": result}))
            response = self._send(client, model, history, types.UserContent(parts=response_parts), config)
            usage_log.append(_usage_record(response, f"turn{turn}"))

        force = types.UserContent(parts=[types.Part(text="Stop calling tools and produce the final answer now.")])
        response = self._send(client, model, history, force, config)
        usage_log.append(_usage_record(response, "force-final"))
        _, text = _extract_calls_and_text(response)
        return text, _usage_from_log(self.provider_name, model, usage_log), tool_calls_log

    def generate_json(
        self,
        *,
        model: str,
        system_instruction: str,
        contents: list[Any],
        response_schema: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], LlmUsage]:
        return self._generate_json_inner(
            model=model,
            system_instruction=system_instruction,
            contents=contents,
            response_schema=response_schema,
            enable_search=False,
        )

    def search_json(
        self,
        *,
        model: str,
        system_instruction: str,
        contents: list[Any],
    ) -> tuple[dict[str, Any], LlmUsage]:
        """Issue a Google-Search-grounded query and parse the JSON response.

        Gemini disallows ``response_schema`` together with the search tool, so we
        ask the model to emit JSON in plain text and parse it ourselves. The
        caller's prompt should specify the exact JSON shape it expects.
        """

        return self._generate_json_inner(
            model=model,
            system_instruction=system_instruction,
            contents=contents,
            response_schema=None,
            enable_search=True,
        )

    def _generate_json_inner(
        self,
        *,
        model: str,
        system_instruction: str,
        contents: list[Any],
        response_schema: dict[str, Any] | None,
        enable_search: bool,
    ) -> tuple[dict[str, Any], LlmUsage]:
        try:
            from google import genai
            from google.genai import types
        except ImportError as exc:
            raise RuntimeError("Install LLM dependencies with: pip install -e .[llm]") from exc

        client = genai.Client(api_key=self.api_key)
        config_kwargs: dict[str, Any] = {
            "system_instruction": system_instruction,
            "temperature": 0.1,
            "http_options": types.HttpOptions(timeout=self.config.request_timeout_sec * 1000),
        }
        if enable_search:
            config_kwargs["tools"] = [types.Tool(google_search=types.GoogleSearch())]
        else:
            config_kwargs["response_mime_type"] = "application/json"
            if response_schema is not None:
                config_kwargs["response_schema"] = response_schema
        config = types.GenerateContentConfig(**config_kwargs)
        user_content = types.UserContent(parts=[_to_part(types, part) for part in contents])
        response = self._send(client, model, [], user_content, config)
        _, text = _extract_calls_and_text(response)
        if enable_search:
            text = _strip_code_fence(text)
        try:
            parsed = json.loads(text) if text else {}
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"Gemini returned non-JSON response: {exc}: {text[:500]}") from exc
        usage_log = [_usage_record(response, "json")]
        return parsed, _usage_from_log(self.provider_name, model, usage_log)

    def _send(self, client: Any, model: str, history: list[Any], content: Any, config: Any) -> Any:
        from google.genai import errors as genai_errors

        last_error: Exception | None = None
        max_attempts = 5
        for attempt in range(1, max_attempts + 1):
            try:
                response = client.models.generate_content(model=model, contents=history + [content], config=config)
                history.append(content)
                if response.candidates and response.candidates[0].content:
                    history.append(response.candidates[0].content)
                return response
            except genai_errors.ServerError as exc:
                last_error = exc
                delay = min(90, 10 * (2 ** (attempt - 1)))  # 10, 20, 40, 80, 90
                time.sleep(delay)
            except genai_errors.ClientError as exc:
                last_error = exc
                if getattr(exc, "code", None) == 429:
                    delay = min(90, 15 * (2 ** (attempt - 1)))  # 15, 30, 60, 90, 90
                    time.sleep(delay)
                    continue
                raise
        raise RuntimeError(f"Gemini request failed after {max_attempts} retries: {type(last_error).__name__}: {last_error}")


def _to_part(types: Any, part: Any) -> Any:
    if hasattr(part, "parts"):
        return part
    if isinstance(part, str):
        return types.Part(text=part)
    if getattr(part, "uri", None) and getattr(part, "mime_type", None):
        return types.Part.from_uri(file_uri=part.uri, mime_type=part.mime_type)
    if isinstance(part, dict) and "file_uri" in part and "mime_type" in part:
        return types.Part.from_uri(file_uri=part["file_uri"], mime_type=part["mime_type"])
    if isinstance(part, dict) and "data" in part and "mime_type" in part:
        return types.Part.from_bytes(data=part["data"], mime_type=part["mime_type"])
    if isinstance(part, bytes):
        raise TypeError("raw bytes require a MIME type wrapper before calling Gemini")
    return types.Part(text=str(part))


def _strip_code_fence(text: str) -> str:
    """Pull JSON out of a ```json ... ``` fence if Gemini wrapped it."""

    stripped = (text or "").strip()
    if stripped.startswith("```"):
        stripped = stripped.lstrip("`")
        # remove optional language tag like "json\n"
        nl = stripped.find("\n")
        if nl != -1:
            stripped = stripped[nl + 1:]
        if stripped.endswith("```"):
            stripped = stripped[:-3]
    return stripped.strip()


def _extract_calls_and_text(response: Any) -> tuple[list[Any], str]:
    calls: list[Any] = []
    text_parts: list[str] = []
    for candidate in response.candidates or []:
        for part in candidate.content.parts or []:
            if getattr(part, "function_call", None):
                calls.append(part.function_call)
            if getattr(part, "text", None):
                text_parts.append(part.text)
    return calls, "".join(text_parts).strip()


def _usage_record(response: Any, label: str) -> dict[str, int | str]:
    usage = getattr(response, "usage_metadata", None)
    return {
        "label": label,
        "prompt_token_count": int(getattr(usage, "prompt_token_count", 0) or 0) if usage else 0,
        "candidates_token_count": int(getattr(usage, "candidates_token_count", 0) or 0) if usage else 0,
        "cached_content_token_count": int(getattr(usage, "cached_content_token_count", 0) or 0) if usage else 0,
    }


def _usage_from_log(provider: str, model: str, usage_log: list[dict]) -> LlmUsage:
    input_tokens = sum(int(u.get("prompt_token_count", 0) or 0) for u in usage_log)
    output_tokens = sum(int(u.get("candidates_token_count", 0) or 0) for u in usage_log)
    return LlmUsage(
        provider=provider,
        model=model,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        estimated_cost_usd=estimate_cost_usd(model, input_tokens, output_tokens),
    )


def _cap_result(result: dict[str, Any]) -> dict[str, Any]:
    encoded = json.dumps(result, default=str)
    if len(encoded) <= 8000:
        return json.loads(encoded)
    return {
        "_truncated": True,
        "_original_size": len(encoded),
        "summary": encoded[:4000] + "...[truncated]..." + encoded[-2000:],
    }

