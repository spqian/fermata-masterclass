from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from masterclass.agent.llm import LlmProvider, LlmUsage
from masterclass.agent_tools.catalog import tool_catalog_text
from masterclass.agent_tools.registry import default_tool_registry
from masterclass.core.chat_models import ChatConversation, ChatMessage, get_or_create_conversation, save_conversation
from masterclass.core.masterclasses import MasterclassStore
from masterclass.core.models import SessionManifest, TenantContext
from masterclass.core.sessions import SessionStore
from masterclass.engine.instruments import load_instrument_profile, system_instruction_for_profile
from masterclass.engine.prompt_evidence import build_evidence_digest
from masterclass.engine.prompt_inventory import build_score_note_inventory
from masterclass.engine.score_prep import select_score_pages_for_lesson
from masterclass.engine.teach_lesson import _delete_uploaded_files, _file_part, _frame_keys, _read_score_map, _score_image_keys
from masterclass.storage.base import ObjectStorage


GEMINI_CHAT_PRICING_PER_MILLION: dict[str, dict[str, float]] = {
    "gemini-3.1-pro": {
        "input_le_200k": 1.25,
        "input_gt_200k": 5.00,
        "output": 12.00,
    },
    "gemini-2.5-pro": {
        "input_le_200k": 1.25,
        "input_gt_200k": 5.00,
        "output": 10.00,
    },
    "gemini-2.5-flash": {
        "input": 0.075,
        "output": 0.30,
    },
}
CHAT_TOPIC_REFUSAL = "I'm here to help with this music lesson. Could you ask something about your performance, the score, or what we discussed?"


@dataclass(frozen=True)
class ChatConfig:
    model: str = "gemini-2.5-pro"
    max_score_pages: int = 8
    max_video_frames: int = 8
    max_tool_calls: int = 5
    use_files_api: bool = True


@dataclass(frozen=True)
class ChatTurnResult:
    conversation_id: str
    reply: str
    tool_calls: list[dict[str, Any]]
    usage: dict[str, Any]


def run_chat_turn(
    *,
    storage: ObjectStorage,
    store: SessionStore,
    manifest: SessionManifest,
    provider: LlmProvider,
    message: str,
    conversation_id: str | None = None,
    masterclasses: MasterclassStore | None = None,
    config: ChatConfig | None = None,
    topic_guard_usage: LlmUsage | None = None,
) -> ChatTurnResult:
    """Run one synchronous follow-up teacher chat turn and persist the conversation."""

    config = config or ChatConfig()
    conversation = get_or_create_conversation(storage, store, manifest, conversation_id)
    conversation.append(ChatMessage(role="user", content=message))

    profile = load_instrument_profile(manifest.instrument_profile)
    registry = default_tool_registry(profile)
    system_instruction = build_chat_system_instruction(storage, store, manifest, conversation, profile=profile)
    contents, uploaded = _build_chat_contents(
        storage=storage,
        store=store,
        manifest=manifest,
        provider=provider,
        config=config,
        masterclasses=masterclasses,
        user_message=message,
    )

    try:
        def run_tool(tool_name: str, args: dict[str, Any]) -> dict[str, Any]:
            return registry.call(storage, manifest.session, tool_name, args)

        text, usage, tool_calls = provider.generate_with_tools(
            model=config.model,
            system_instruction=system_instruction,
            contents=contents,
            tools=registry.declarations(),
            max_tool_calls=config.max_tool_calls,
            tool_executor=run_tool,
        )
    finally:
        _delete_uploaded_files(uploaded)

    reply = (text or "").strip() or "I couldn't produce a reply for that turn. Please try again."
    usage_dict = chat_usage_dict(usage, guard_usage=topic_guard_usage)
    conversation.append(ChatMessage(role="teacher", content=reply, tool_calls=tool_calls, usage=usage_dict))
    save_conversation(storage, store, manifest, conversation)
    return ChatTurnResult(conversation.conversation_id, reply, tool_calls, usage_dict)


def build_chat_system_instruction(
    storage: ObjectStorage | None = None,
    store: SessionStore | None = None,
    manifest: SessionManifest | None = None,
    conversation: ChatConversation | None = None,
    *,
    profile: Any | None = None,
) -> str:
    """Build the instrument-aware teacher prompt plus lesson-scoped chat rules."""

    if profile is None and manifest is not None:
        profile = load_instrument_profile(manifest.instrument_profile)
    base = system_instruction_for_profile(profile, tool_catalog=tool_catalog_text(profile)) if profile is not None else "You are a careful music teacher."
    takeaway, comments_digest = _lesson_context(storage, store, manifest) if storage and store and manifest else ({}, [])
    history = _chat_history_text(conversation) if conversation else "(no previous chat messages)"
    return base + "\n\n" + (
        "## Chat mode\n\n"
        "You already gave the student a structured critique of THIS performance (their original lesson takeaway and comments are below). "
        "The student is now asking a follow-up question.\n\n"
        "- Stay STRICTLY scoped to this lesson and this piece. Do not invent new analyses of unrelated topics.\n"
        "- You may consult the audio (listen tool), video frames (watch / get_frames tools), or any inspection tool if it helps you give a better answer. Use them sparingly — at most 5 tool calls per response.\n"
        "- When citing measures, use the same measure numbering convention as the lesson comments.\n"
        "- If the question is off-topic for music performance, politely redirect.\n"
        "- Be concise — chat responses should be 1-3 paragraphs unless a longer explanation is genuinely needed.\n\n"
        "Original lesson takeaway:\n"
        f"{json.dumps(takeaway, ensure_ascii=False, indent=2)}\n\n"
        "Original lesson comments (severity, bar references, summary only):\n"
        f"{json.dumps(comments_digest, ensure_ascii=False, indent=2)}\n\n"
        "Conversation so far (most recent last):\n"
        f"{history}"
    )


def chat_usage_dict(usage: LlmUsage | None, *, guard_usage: LlmUsage | None = None) -> dict[str, Any]:
    model = usage.model if usage else "gemini-2.5-pro"
    input_tokens = usage.input_tokens if usage else 0
    output_tokens = usage.output_tokens if usage else 0
    estimated = estimate_chat_cost(model, input_tokens, output_tokens)
    if guard_usage is not None:
        guard_cost = estimate_chat_cost(guard_usage.model, guard_usage.input_tokens, guard_usage.output_tokens) or 0.0
        estimated = round((estimated or 0.0) + guard_cost, 6)
    data: dict[str, Any] = {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "estimated_cost_usd": estimated,
        "model": model,
    }
    if guard_usage is not None:
        data["topic_guard"] = {
            "input_tokens": guard_usage.input_tokens,
            "output_tokens": guard_usage.output_tokens,
            "estimated_cost_usd": estimate_chat_cost(guard_usage.model, guard_usage.input_tokens, guard_usage.output_tokens),
            "model": guard_usage.model,
        }
    return data


def estimate_chat_cost(model: str, input_tokens: int | None, output_tokens: int | None) -> float | None:
    if input_tokens is None or output_tokens is None:
        return None
    # Match model name to pricing table (handles versioned names like "gemini-3.1-pro-preview")
    for prefix in ("gemini-3.1-pro", "gemini-3-pro", "gemini-2.5-pro"):
        if model.startswith(prefix):
            table = GEMINI_CHAT_PRICING_PER_MILLION.get(prefix) or GEMINI_CHAT_PRICING_PER_MILLION.get("gemini-2.5-pro")
            input_rate = table["input_gt_200k"] if input_tokens > 200_000 else table["input_le_200k"]
            return round((input_tokens * input_rate + output_tokens * table["output"]) / 1_000_000, 6)
    if model.startswith("gemini-2.5-flash"):
        table = GEMINI_CHAT_PRICING_PER_MILLION["gemini-2.5-flash"]
        return round((input_tokens * table["input"] + output_tokens * table["output"]) / 1_000_000, 6)
    return usage_cost_fallback(input_tokens, output_tokens)


def usage_cost_fallback(input_tokens: int, output_tokens: int) -> float:
    return round((input_tokens * 1.25 + output_tokens * 10.0) / 1_000_000, 6)


def _build_chat_contents(
    *,
    storage: ObjectStorage,
    store: SessionStore,
    manifest: SessionManifest,
    provider: LlmProvider,
    config: ChatConfig,
    masterclasses: MasterclassStore | None,
    user_message: str,
) -> tuple[list[Any], list[Any]]:
    score_map = _read_score_map(storage, store, manifest)
    score_key = score_map.get("key") if score_map else None
    first_measure = _as_int(manifest.metadata.get("first_measure"), score_map.get("first_measure") if score_map else None)
    last_measure = _as_int(manifest.metadata.get("last_measure"), score_map.get("last_measure") if score_map else None)
    evidence_digest = build_evidence_digest(storage=storage, store=store, manifest=manifest, score_key=score_key)
    if score_key:
        evidence_digest = f"Key: {score_key}. Use this key's spelling in all replies; copy note spellings from the inventory.\n\n" + evidence_digest
    inventory = build_score_note_inventory(score_map, first_measure=first_measure, last_measure=last_measure) if score_map else ""
    audio_key = manifest.artifacts.get("artifacts/audio_16k.wav") or manifest.artifacts.get("artifacts/audio.wav")
    if not audio_key:
        raise ValueError("manifest is missing lesson audio")

    uploaded: list[Any] = []
    contents: list[Any] = [
        "# Evidence digest\n\n" + (evidence_digest or "(no deterministic evidence digest available)") + "\n",
        "# Recording briefing\n\n"
        f"Repertoire: {manifest.repertoire or '(unknown)'}\n"
        f"Movement: {manifest.movement or '(unknown)'}\n"
        f"Instrument: {manifest.instrument or manifest.instrument_profile or '(unspecified)'}\n"
        f"Measures: {first_measure or manifest.metadata.get('first_measure')}–{last_measure or manifest.metadata.get('last_measure')}\n"
        f"Student notes: {(manifest.notes or '(none)').strip()}\n",
        "# Recording (audio)\n",
    ]
    audio_part = _file_part(storage, provider, audio_key, "audio/wav", config=config, uploaded=uploaded)
    contents.append(audio_part if audio_part is not None else {"mime_type": "audio/wav", "data": storage.read_bytes(audio_key), "label": "lesson-audio"})

    contents.append("\n# Score (system images, in order of appearance)\n")
    score_image_keys = _score_image_keys(storage, store, manifest, score_map)
    if score_image_keys:
        for index, key in enumerate(score_image_keys[: config.max_score_pages], start=1):
            contents.append(f"score image {index}: {key.rsplit('/', 1)[-1]}")
            part = _file_part(storage, provider, key, "image/png", config=config, uploaded=uploaded)
            contents.append(part if part is not None else {"mime_type": "image/png", "data": storage.read_bytes(key), "label": f"score-{index}"})
    else:
        pngs, layout = _selected_score_pages(storage, manifest, masterclasses, first_measure, last_measure, config.max_score_pages)
        if layout:
            contents.append("\n--- score layout ---\n" + json.dumps(layout, indent=2))
        for index, png in enumerate(pngs, start=1):
            contents.append({"mime_type": "image/png", "data": png, "label": f"score-page-{index}"})
        if not pngs:
            contents.append("(no score images available)\n")

    contents.append("\n# Sample video frames (for technique/visual observations)\n")
    frame_keys = _frame_keys(storage, manifest, config.max_video_frames)
    for index, key in enumerate(frame_keys, start=1):
        contents.append(f"video frame {index}: {key.rsplit('/', 1)[-1]}")
        part = _file_part(storage, provider, key, "image/jpeg", config=config, uploaded=uploaded)
        contents.append(part if part is not None else {"mime_type": "image/jpeg", "data": storage.read_bytes(key), "label": f"frame-{index}"})
    if not frame_keys:
        contents.append("(no frames available)\n")

    contents.append("\n# Score note inventory\n\n" + (inventory or "(no score note inventory available)") + "\n")
    contents.append(
        "# Current student follow-up\n\n"
        f"{user_message}\n\n"
        "Answer naturally as the same teacher. You may use tools, but do not produce JSON; write a concise chat reply."
    )
    return contents, uploaded


def _lesson_context(storage: ObjectStorage | None, store: SessionStore | None, manifest: SessionManifest | None) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if storage is None or store is None or manifest is None:
        return {}, []
    key = manifest.artifacts.get("lesson/comments_enriched.json") or store.artifact_key(manifest.session, "lesson/comments_enriched.json")
    if not key or not storage.exists(key):
        return {}, []
    try:
        lesson = storage.read_json(key)
    except Exception:
        return {}, []
    takeaway = {
        "summary": lesson.get("summary"),
        "progress_notes": lesson.get("progress_notes"),
        "lesson": lesson.get("lesson") or {},
    }
    comments = []
    for c in lesson.get("comments") or []:
        if not isinstance(c, dict):
            continue
        comments.append({
            "severity": c.get("severity"),
            "measure": c.get("measure"),
            "beat": c.get("beat"),
            "summary": c.get("summary"),
        })
    return takeaway, comments[:40]


def _chat_history_text(conversation: ChatConversation | None) -> str:
    if not conversation or not conversation.messages:
        return "(no previous chat messages)"
    lines: list[str] = []
    for msg in conversation.messages[-40:]:
        role = "Student" if msg.role == "user" else "Teacher" if msg.role == "teacher" else msg.role.title()
        lines.append(f"{role}: {msg.content}")
    return "\n".join(lines)


def _selected_score_pages(
    storage: ObjectStorage,
    manifest: SessionManifest,
    masterclasses: MasterclassStore | None,
    first_measure: int | None,
    last_measure: int | None,
    max_pages: int,
) -> tuple[list[bytes], list[dict[str, Any]]]:
    masterclass_id = manifest.metadata.get("masterclass_id")
    if not masterclasses or not masterclass_id:
        return [], []
    try:
        class_manifest = masterclasses.load_by_id(
            TenantContext(manifest.session.tenant_id, manifest.session.user_id),
            str(masterclass_id),
        )
    except Exception:
        return [], []
    return select_score_pages_for_lesson(
        storage=storage,
        masterclass=class_manifest,
        first_measure=first_measure,
        last_measure=last_measure,
        max_pages=max_pages,
    )


def _as_int(*values: Any) -> int | None:
    for value in values:
        if value is None:
            continue
        try:
            return int(value)
        except (TypeError, ValueError):
            continue
    return None
