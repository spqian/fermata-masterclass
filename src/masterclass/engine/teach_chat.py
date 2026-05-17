from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from masterclass.agent.llm import LlmProvider, LlmUsage
from masterclass.agent_tools.catalog import tool_catalog_text
from masterclass.agent_tools.registry import default_tool_registry
from masterclass.core.chat_models import ChatConversation, ChatMessage, conversation_key, get_or_create_conversation, load_conversation, save_conversation
from masterclass.core.conversation_lock import conversation_lock
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
    comment_id: str | None = None,
    masterclasses: MasterclassStore | None = None,
    config: ChatConfig | None = None,
    topic_guard_usage: LlmUsage | None = None,
) -> ChatTurnResult:
    """Run one synchronous follow-up teacher chat turn and persist the conversation."""

    config = config or ChatConfig()
    conversation = get_or_create_conversation(storage, store, manifest, conversation_id)
    # Snapshot prior history BEFORE appending the new message so we can pass
    # it to the model as multi-turn contents (which keeps the system prompt
    # stable and lets Gemini implicit caching kick in).
    prior_history = list(conversation.messages)
    conversation.append(ChatMessage(role="user", content=message))

    profile = load_instrument_profile(manifest.instrument_profile)
    registry = default_tool_registry(profile)
    system_instruction = build_chat_system_instruction(
        storage, store, manifest, conversation=None, profile=profile, comment_id=comment_id,
    )
    contents, uploaded = _build_chat_contents(
        storage=storage,
        store=store,
        manifest=manifest,
        provider=provider,
        config=config,
        masterclasses=masterclasses,
        user_message=message,
        prior_history=prior_history,
        comment_id=comment_id,
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
    user_msg = ChatMessage(role="user", content=message)
    teacher_msg = ChatMessage(role="teacher", content=reply, tool_calls=tool_calls, usage=usage_dict)
    # Merge-save under per-conversation lock so a concurrent drill
    # completion (which also appends to ``cmt_<id>.json``) can't race
    # with us and lose messages on the disk file.
    final_conv_id = conversation.conversation_id
    lock_key = conversation_key(store, manifest.session, final_conv_id)
    with conversation_lock(lock_key):
        try:
            fresh = load_conversation(storage, store, manifest, final_conv_id)
        except FileNotFoundError:
            fresh = ChatConversation(
                conversation_id=final_conv_id,
                session_id=manifest.session.session_id,
                user_id=manifest.session.user_id,
            )
        fresh.append(user_msg)
        fresh.append(teacher_msg)
        save_conversation(storage, store, manifest, fresh)
    return ChatTurnResult(final_conv_id, reply, tool_calls, usage_dict)


def build_chat_system_instruction(
    storage: ObjectStorage | None = None,
    store: SessionStore | None = None,
    manifest: SessionManifest | None = None,
    conversation: ChatConversation | None = None,
    *,
    profile: Any | None = None,
    comment_id: str | None = None,
) -> str:
    """Build the instrument-aware teacher prompt plus lesson-scoped chat rules.

    The system instruction is intentionally STABLE across chat turns within
    a lesson: it does NOT include the conversation history (history is
    passed as multi-turn contents instead) and it does NOT vary by
    comment_id (the comment context is injected into the user message via a
    `[Re: comment <id>]` prefix). Both choices preserve a stable ~5KB
    system prefix that Gemini implicit caching can reuse across turns at
    the 25% cached-input rate. Passing a ``conversation`` is accepted for
    backward compat but ignored.
    """
    del conversation, comment_id  # not used by the cache-stable prompt

    if profile is None and manifest is not None:
        profile = load_instrument_profile(manifest.instrument_profile)
    base = system_instruction_for_profile(profile, tool_catalog=tool_catalog_text(profile)) if profile is not None else "You are a careful music teacher."
    takeaway, comments_digest = _lesson_context(storage, store, manifest) if storage and store and manifest else ({}, [])
    return base + "\n\n" + (
        "## Chat mode\n\n"
        "You already gave the student a structured critique of THIS performance (their original lesson takeaway and comments are below). "
        "The student is now asking a follow-up question.\n\n"
        "- Stay STRICTLY scoped to this lesson and this piece. Do not invent new analyses of unrelated topics.\n"
        "- You may consult the audio (listen tool), video frames (watch / get_frames tools), or any inspection tool if it helps you give a better answer. Use them sparingly — at most 5 tool calls per response.\n"
        "- When citing measures, use the same measure numbering convention as the lesson comments.\n"
        "- If the question is off-topic for music performance, politely redirect.\n"
        "- Be concise — chat responses should be 1-3 paragraphs unless a longer explanation is genuinely needed.\n"
        "- A user message prefixed with `[Re: comment <id>] ` is a reply inside that comment's thread; "
        "keep your answer focused on that comment's scope. Without the prefix the message is a general lesson question.\n\n"
        "Original lesson takeaway:\n"
        f"{json.dumps(takeaway, ensure_ascii=False, indent=2)}\n\n"
        "Original lesson comments (severity, bar references, summary only):\n"
        f"{json.dumps(comments_digest, ensure_ascii=False, indent=2)}\n"
    )


def chat_usage_dict(usage: LlmUsage | None, *, guard_usage: LlmUsage | None = None) -> dict[str, Any]:
    model = usage.model if usage else "gemini-2.5-pro"
    input_tokens = usage.input_tokens if usage else 0
    output_tokens = usage.output_tokens if usage else 0
    cached_tokens = getattr(usage, "cached_tokens", 0) if usage else 0
    estimated = estimate_chat_cost(model, input_tokens, output_tokens, cached_tokens=cached_tokens)
    if guard_usage is not None:
        guard_cost = estimate_chat_cost(
            guard_usage.model,
            guard_usage.input_tokens,
            guard_usage.output_tokens,
            cached_tokens=getattr(guard_usage, "cached_tokens", 0),
        ) or 0.0
        estimated = round((estimated or 0.0) + guard_cost, 6)
    data: dict[str, Any] = {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cached_tokens": cached_tokens,
        "estimated_cost_usd": estimated,
        "model": model,
    }
    if guard_usage is not None:
        data["topic_guard"] = {
            "input_tokens": guard_usage.input_tokens,
            "output_tokens": guard_usage.output_tokens,
            "cached_tokens": getattr(guard_usage, "cached_tokens", 0),
            "estimated_cost_usd": estimate_chat_cost(
                guard_usage.model,
                guard_usage.input_tokens,
                guard_usage.output_tokens,
                cached_tokens=getattr(guard_usage, "cached_tokens", 0),
            ),
            "model": guard_usage.model,
        }
    return data


def estimate_chat_cost(
    model: str,
    input_tokens: int | None,
    output_tokens: int | None,
    *,
    cached_tokens: int = 0,
) -> float | None:
    """Estimate chat-turn cost. ``cached_tokens`` is the subset of
    ``input_tokens`` Gemini billed at the 25% implicit-cache rate."""
    if input_tokens is None or output_tokens is None:
        return None
    # Match model name to pricing table (handles versioned names like "gemini-3.1-pro-preview")
    for prefix in ("gemini-3.1-pro", "gemini-3-pro", "gemini-2.5-pro"):
        if model.startswith(prefix):
            table = GEMINI_CHAT_PRICING_PER_MILLION.get(prefix) or GEMINI_CHAT_PRICING_PER_MILLION.get("gemini-2.5-pro")
            input_rate = table["input_gt_200k"] if input_tokens > 200_000 else table["input_le_200k"]
            cached = max(0, min(int(cached_tokens or 0), int(input_tokens)))
            uncached = int(input_tokens) - cached
            return round(
                (uncached * input_rate + cached * input_rate * 0.25 + output_tokens * table["output"]) / 1_000_000,
                6,
            )
    if model.startswith("gemini-2.5-flash"):
        table = GEMINI_CHAT_PRICING_PER_MILLION["gemini-2.5-flash"]
        cached = max(0, min(int(cached_tokens or 0), int(input_tokens)))
        uncached = int(input_tokens) - cached
        return round(
            (uncached * table["input"] + cached * table["input"] * 0.25 + output_tokens * table["output"]) / 1_000_000,
            6,
        )
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
    prior_history: list[ChatMessage] | None = None,
    comment_id: str | None = None,
) -> tuple[list[Any], list[Any]]:
    """Build chat contents WITHOUT re-shipping audio / score / video binaries.

    Per-lesson context (evidence digest, briefing, score note inventory) is
    emitted FIRST in stable order so Gemini implicit caching can reuse the
    prefix across turns. Conversation history follows as alternating
    user/teacher turns. The new student message is last (and optionally
    prefixed with ``[Re: comment <id>] `` so the teacher knows the question
    is scoped to a specific lesson comment).
    """
    score_map = _read_score_map(storage, store, manifest)
    score_key = score_map.get("key") if score_map else None
    first_measure = _as_int(manifest.metadata.get("first_measure"), score_map.get("first_measure") if score_map else None)
    last_measure = _as_int(manifest.metadata.get("last_measure"), score_map.get("last_measure") if score_map else None)
    evidence_digest = build_evidence_digest(storage=storage, store=store, manifest=manifest, score_key=score_key)
    if score_key:
        evidence_digest = f"Key: {score_key}. Use this key's spelling in all replies; copy note spellings from the inventory.\n\n" + evidence_digest
    inventory = build_score_note_inventory(score_map, first_measure=first_measure, last_measure=last_measure) if score_map else ""

    uploaded: list[Any] = []
    # ---- Stable per-lesson preamble (this is what implicit caching will catch) ----
    contents: list[Any] = [
        "# Evidence digest\n\n" + (evidence_digest or "(no deterministic evidence digest available)") + "\n",
        "# Recording briefing\n\n"
        f"Repertoire: {manifest.repertoire or '(unknown)'}\n"
        f"Movement: {manifest.movement or '(unknown)'}\n"
        f"Instrument: {manifest.instrument or manifest.instrument_profile or '(unspecified)'}\n"
        f"Measures: {first_measure or manifest.metadata.get('first_measure')}\u2013{last_measure or manifest.metadata.get('last_measure')}\n"
        f"Student notes: {(manifest.notes or '(none)').strip()}\n",
        "# Score note inventory\n\n" + (inventory or "(no score note inventory available)") + "\n",
        "# Binary context (not inline — use tools to fetch on demand)\n\n"
        "To save tokens, the audio recording, score page images, and video frames "
        "are NOT included inline in this chat turn. They are still available via tools:\n"
        "- `listen(start_sec, end_sec)` to hear any audio window\n"
        "- `watch(start_sec, end_sec, question)` for a short video clip\n"
        "- `get_frames(start_sec, end_sec, fps)` for stills\n"
        "- `inspect_intonation`, `inspect_bar`, `inspect_chord`, `measure_tempo`, `measure_dynamics`, etc. for measurement queries\n"
        "Call them only when the student's question genuinely requires a fresh look at the audio/video; "
        "for follow-ups about comments you already made, the evidence digest + inventory above are usually enough.\n",
    ]

    # ---- Prior conversation history as alternating turns ----
    # We pass these as plain strings tagged with role markers so the
    # provider can construct multi-turn contents. The Gemini provider will
    # wrap a final user message; everything before it is conceptually part
    # of the prefix the model already saw. With a stable system prompt +
    # stable preamble + append-only history, the cached prefix grows
    # monotonically each turn.
    for msg in (prior_history or []):
        role = "Student" if msg.role == "user" else "Teacher" if msg.role == "teacher" else msg.role.title()
        contents.append(f"# {role} (previous turn)\n\n{msg.content}\n")

    # ---- Current student message ----
    prefix = f"[Re: comment {comment_id}] " if comment_id else ""
    contents.append(
        "# Current student follow-up\n\n"
        f"{prefix}{user_message}\n\n"
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
            "id": c.get("id"),
            "severity": c.get("severity"),
            "category": c.get("category"),
            "measure": c.get("measure"),
            "beat": c.get("beat"),
            "summary": c.get("summary"),
            "text": c.get("text"),
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
