from __future__ import annotations

import hashlib
import os
import threading
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

from masterclass.agent.llm import LlmProvider, LlmUsage
from masterclass.storage.base import ObjectStorage

MAX_MESSAGE_BYTES = 2 * 1024
DAILY_USER_TURN_CAP = 50
CONVERSATION_USER_TURN_CAP = 20
TOPIC_GUARD_MODEL = "gemini-2.5-flash"
TOPIC_GUARD_CACHE_TTL_SEC = 60 * 60
TOPIC_REFUSAL = "I'm here to help with this music lesson. Could you ask something about your performance, the score, or what we discussed?"

_locks: dict[str, threading.Lock] = {}
_locks_guard = threading.Lock()


class ChatGuardrailError(Exception):
    status_code = 400

    def __init__(self, detail: str, *, usage: dict[str, Any] | None = None) -> None:
        super().__init__(detail)
        self.detail = detail
        self.usage = usage


class MessageTooLargeError(ChatGuardrailError):
    status_code = 413


class ChatRateLimitError(ChatGuardrailError):
    status_code = 429


class OffTopicError(ChatGuardrailError):
    status_code = 422


@dataclass(frozen=True)
class TopicDecision:
    allowed: bool
    usage: LlmUsage | None = None
    cached: bool = False


def check_message_size(message: str, max_bytes: int = MAX_MESSAGE_BYTES) -> None:
    if len((message or "").encode("utf-8")) > max_bytes:
        raise MessageTooLargeError(f"Message is too large. Please keep chat questions under {max_bytes} bytes.")


def check_conversation_turn_cap(current_user_messages: int, cap: int = CONVERSATION_USER_TURN_CAP) -> None:
    if current_user_messages >= cap:
        raise ChatRateLimitError(f"This chat has reached the {cap}-message limit. Start a new chat thread for more questions.")


def check_user_quota(storage: ObjectStorage, tenant_id: str, user_id: str, cap: int = DAILY_USER_TURN_CAP) -> dict[str, Any]:
    """Atomically increment today's per-user chat quota and return the quota row."""

    today = datetime.now(UTC).strftime("%Y%m%d")
    safe_user = _safe_id(user_id)
    key = f"tenant/{_safe_id(tenant_id)}/users/{safe_user}/sessions/_user_quotas/{safe_user}_{today}.json"
    lock = _lock_for(key)
    with lock:
        row: dict[str, Any]
        if storage.exists(key):
            try:
                row = storage.read_json(key)
            except Exception:
                row = {}
        else:
            row = {}
        count = int(row.get("count") or 0)
        if count >= cap:
            raise ChatRateLimitError(f"Daily chat limit reached ({cap} messages). Your quota resets at midnight UTC.")
        row = {
            "schema_version": 1,
            "user_id": user_id,
            "date_utc": today,
            "count": count + 1,
            "limit": cap,
            "updated_at": datetime.now(UTC).isoformat(),
        }
        storage.write_json(key, row)
        return row


def topic_guard(storage: ObjectStorage, tenant_id: str, user_id: str, message: str, provider: LlmProvider) -> TopicDecision:
    if os.environ.get("DISABLE_TOPIC_GUARD", "").lower() == "true":
        return TopicDecision(True, None, False)

    cached = _read_topic_cache(storage, tenant_id, user_id, message)
    if cached is not None:
        return TopicDecision(cached, None, True)

    allowed, usage = _ask_topic_guard(message, provider)
    _write_topic_cache(storage, tenant_id, user_id, message, allowed)
    if not allowed:
        raise OffTopicError(TOPIC_REFUSAL, usage=_usage_dict(usage))
    return TopicDecision(True, usage, False)


def _ask_topic_guard(message: str, provider: LlmProvider) -> tuple[bool, LlmUsage | None]:
    prompt = (
        "User asked the music teacher: "
        f"'{message}'. Is this question about analyzing or improving a music performance, "
        "asking for music-pedagogy clarification, or asking about the just-completed lesson? "
        "Answer with one word: yes or no."
    )
    if getattr(provider, "provider_name", "") == "dry-run":
        lowered = message.lower()
        allowed = not any(word in lowered for word in ("weather", "stock", "bitcoin", "recipe", "tokyo"))
        return allowed, LlmUsage(provider="dry-run", model=TOPIC_GUARD_MODEL, input_tokens=0, output_tokens=0, estimated_cost_usd=0.0)

    # Use the already-constructed provider's API key today. TODO(auth-and-byok): replace
    # this direct shared-key client path with the per-user provider from use_for_user(user_id).
    api_key = (getattr(provider, "api_key", None) or os.environ.get("GEMINI_API_KEY") or "").strip()
    if not api_key:
        # If there is no key, let the main request fail with the normal provider error.
        return True, None
    try:
        from google import genai
        from google.genai import types
    except ImportError:
        return True, None
    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model=TOPIC_GUARD_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0,
            http_options=types.HttpOptions(timeout=30_000),
        ),
    )
    text = (getattr(response, "text", "") or "").strip().lower()
    meta = getattr(response, "usage_metadata", None)
    input_tokens = int(getattr(meta, "prompt_token_count", 0) or 0) if meta else 0
    output_tokens = int(getattr(meta, "candidates_token_count", 0) or 0) if meta else 0
    usage = LlmUsage(
        provider=getattr(provider, "provider_name", "google-gemini-shared-key"),
        model=TOPIC_GUARD_MODEL,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        estimated_cost_usd=round((input_tokens * 0.075 + output_tokens * 0.30) / 1_000_000, 6),
    )
    return text.startswith("yes"), usage


def _read_topic_cache(storage: ObjectStorage, tenant_id: str, user_id: str, message: str) -> bool | None:
    key = _topic_cache_key(tenant_id, user_id, message)
    if not storage.exists(key):
        return None
    try:
        row = storage.read_json(key)
        expires = datetime.fromisoformat(str(row.get("expires_at")))
        if expires <= datetime.now(UTC):
            return None
        return bool(row.get("allowed"))
    except Exception:
        return None


def _write_topic_cache(storage: ObjectStorage, tenant_id: str, user_id: str, message: str, allowed: bool) -> None:
    key = _topic_cache_key(tenant_id, user_id, message)
    storage.write_json(key, {
        "schema_version": 1,
        "user_id": user_id,
        "message_sha256": hashlib.sha256(message.encode("utf-8")).hexdigest(),
        "allowed": bool(allowed),
        "expires_at": (datetime.now(UTC) + timedelta(seconds=TOPIC_GUARD_CACHE_TTL_SEC)).isoformat(),
        "created_at": datetime.now(UTC).isoformat(),
    })


def _topic_cache_key(tenant_id: str, user_id: str, message: str) -> str:
    digest = hashlib.sha256(message.encode("utf-8")).hexdigest()
    return f"tenant/{_safe_id(tenant_id)}/users/{_safe_id(user_id)}/sessions/_topic_guard_cache/{_safe_id(user_id)}_{digest}.json"


def _lock_for(key: str) -> threading.Lock:
    with _locks_guard:
        lock = _locks.get(key)
        if lock is None:
            lock = threading.Lock()
            _locks[key] = lock
        return lock


def _safe_id(value: str) -> str:
    value = (value or "").strip()
    if not value or "/" in value or "\\" in value or ".." in value:
        raise ValueError("path-unsafe id")
    return value


def _usage_dict(usage: LlmUsage | None) -> dict[str, Any] | None:
    if usage is None:
        return None
    return {
        "input_tokens": usage.input_tokens,
        "output_tokens": usage.output_tokens,
        "estimated_cost_usd": usage.estimated_cost_usd,
        "model": usage.model,
    }
