"""End-to-end drill (practice clip) processing pipeline.

A drill is a short student recording of a teacher-prescribed practice
exercise. Unlike a lesson, we don't have a score reference for the
recording, so we can't run score-matching, alignment, evidence-packet
building, or the multimodal-teacher loop with score-anchored tools.

Stages (each persists ``{stage}_state`` on ``manifest.metadata``):

1. ``extract_media``: decode the upload to 16 kHz wav and pull a few
   sample video frames (max 4).
2. ``transcribe``: run basic-pitch and persist the raw note events.
3. ``drill_metrics``: compute IOI stats / tempo estimate / pitch
   distribution and persist ``analysis/drill_metrics.json``.
4. ``drill_feedback``: one Gemini call with the drill instruction,
   parent comment context, audio (Files API), one frame, and the
   drill metrics summary; saves the markdown feedback and appends a
   ``drill_result`` message to the parent comment's chat thread.
"""
from __future__ import annotations

import json
import logging
import os
import tempfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from masterclass.agent.llm import LlmProvider
from masterclass.core.chat_models import (
    ChatConversation,
    ChatMessage,
    conversation_key,
    load_conversation,
    safe_conversation_id,
    save_conversation,
)
from masterclass.core.conversation_lock import conversation_lock
from masterclass.core.models import JobState, SessionManifest, SessionRef, TenantContext
from masterclass.core.sessions import SessionStore
from masterclass.engine.drill_metrics import compute_drill_metrics
from masterclass.prompts import load_drill_evaluator_prompt
from masterclass.storage.base import ObjectStorage
from masterclass.toolchain.ffmpeg import FfmpegToolchain


_LOG = logging.getLogger(__name__)


@dataclass(frozen=True)
class DrillConfig:
    model: str = "gemini-2.5-flash"
    max_frames: int = 4
    use_files_api: bool = True
    request_timeout_sec: int = 90


# ---------------------------------------------------------------------------
# Stage state helpers
# ---------------------------------------------------------------------------
def _mark_stage(store: SessionStore, manifest: SessionManifest, stage: str, state: str, error: str | None = None) -> None:
    manifest.metadata[f"{stage}_state"] = state
    manifest.metadata[f"{stage}_error"] = error
    manifest.metadata[f"{stage}_updated_at"] = datetime.now(UTC).isoformat()
    store.save(manifest)


def _set_top_state(store: SessionStore, manifest: SessionManifest, state: str) -> None:
    manifest.metadata["drill_state"] = state
    store.save(manifest)


# ---------------------------------------------------------------------------
# Stages
# ---------------------------------------------------------------------------
def _stage_extract_media(
    *,
    storage: ObjectStorage,
    store: SessionStore,
    manifest: SessionManifest,
    ffmpeg: FfmpegToolchain,
    max_frames: int,
) -> None:
    source_key = manifest.artifacts.get("input/source_video")
    if not source_key:
        raise ValueError("drill manifest has no input/source_video artifact")

    with tempfile.TemporaryDirectory(prefix="masterclass-drill-") as tmp_raw:
        tmp = Path(tmp_raw)
        source = tmp / (manifest.source_filename or "source.bin")
        storage.read_to_file(source_key, source)

        audio_wav = tmp / "audio.wav"
        from masterclass.toolchain.process import run_process
        run_process(
            [
                str(ffmpeg.ffmpeg), "-y", "-i", str(source),
                "-ac", "1", "-ar", "16000", "-sample_fmt", "s16",
                str(audio_wav),
            ],
            timeout_sec=300,
        )
        audio_key = store.artifact_key(manifest.session, "artifacts/audio.wav")
        storage.write_file(audio_key, audio_wav, content_type="audio/wav")
        manifest.artifacts["artifacts/audio.wav"] = audio_key
        manifest.artifacts["artifacts/audio_16k.wav"] = audio_key

        # Frames: best-effort. If the upload was audio-only or the codec
        # has no video stream, the frame pass fails silently — drills are
        # short so we cap at max_frames sample points.
        frames_dir = tmp / "frames"
        frames_dir.mkdir(parents=True, exist_ok=True)
        try:
            frame_paths = ffmpeg.extract_frames(source, frames_dir, every_seconds=max(2.0, 30.0 / max(1, max_frames)))
        except Exception as exc:
            _LOG.info("drill extract_frames skipped: %s", exc)
            frame_paths = []

        frame_keys: list[str] = []
        for frame in frame_paths[:max_frames]:
            key = store.artifact_key(manifest.session, f"artifacts/frames/{frame.name}")
            storage.write_file(key, frame, content_type="image/jpeg")
            frame_keys.append(key)
        manifest.metadata["frames"] = frame_keys
        manifest.metadata["frame_count"] = len(frame_keys)
        store.save(manifest)


def _stage_transcribe(
    *,
    storage: ObjectStorage,
    store: SessionStore,
    manifest: SessionManifest,
) -> list[dict[str, Any]]:
    # basic-pitch only: drills are short, often pitched-but-not-piano, and
    # cheap to transcribe.
    from masterclass.engine.audio_truth import _transcribe_basic_pitch

    audio_key = manifest.artifacts.get("artifacts/audio.wav")
    if not audio_key:
        raise ValueError("drill manifest has no audio after extract_media")
    audio_bytes = storage.read_bytes(audio_key)
    notes = _transcribe_basic_pitch(audio_bytes)

    notes_key = store.artifact_key(manifest.session, "analysis/drill_audio_truth_notes.json")
    storage.write_json(
        notes_key,
        {
            "schema_version": 1,
            "method": "basic_pitch",
            "n_notes": len(notes),
            "generated_at": datetime.now(UTC).isoformat(),
            "notes": notes,
        },
    )
    manifest.artifacts["analysis/drill_audio_truth_notes.json"] = notes_key
    store.save(manifest)
    return notes


def _stage_drill_metrics(
    *,
    storage: ObjectStorage,
    store: SessionStore,
    manifest: SessionManifest,
    notes: list[dict[str, Any]],
) -> dict[str, Any]:
    metrics = compute_drill_metrics(notes)
    metrics_key = store.artifact_key(manifest.session, "analysis/drill_metrics.json")
    storage.write_json(metrics_key, metrics)
    manifest.artifacts["analysis/drill_metrics.json"] = metrics_key
    manifest.metadata["drill_metrics_summary"] = {
        "n_notes": metrics.get("n_notes"),
        "tempo_bpm_estimate": metrics.get("tempo_bpm_estimate"),
        "low_signal": metrics.get("low_signal"),
    }
    store.save(manifest)
    return metrics


def _stage_drill_feedback(
    *,
    storage: ObjectStorage,
    store: SessionStore,
    manifest: SessionManifest,
    provider: LlmProvider,
    metrics: dict[str, Any],
    config: DrillConfig,
) -> str:
    instruction = (manifest.metadata.get("drill_instruction") or "").strip()
    parent_comment = manifest.metadata.get("parent_comment") or {}

    audio_key = manifest.artifacts.get("artifacts/audio.wav")
    if not audio_key:
        raise ValueError("drill manifest has no audio for feedback stage")

    system_instruction = load_drill_evaluator_prompt()

    contents: list[Any] = []
    contents.append("# Drill instruction (what the student was asked to practise)\n\n" + (instruction or "(no instruction provided)"))
    if parent_comment:
        ctx_lines = ["# Parent lesson comment context\n"]
        for field in ("measure", "category", "severity", "summary", "text"):
            val = parent_comment.get(field)
            if val is not None and val != "":
                ctx_lines.append(f"- **{field}**: {val}")
        contents.append("\n".join(ctx_lines))
    contents.append(
        "# drill_metrics (machine-computed; treat as sanity check)\n\n"
        "```json\n" + json.dumps(metrics, indent=2) + "\n```"
    )
    contents.append("# Drill recording (audio attached below)")

    uploaded: list[Any] = []
    audio_part = _maybe_files_api_part(storage, provider, audio_key, "audio/wav", config=config, uploaded=uploaded)
    if audio_part is not None:
        contents.append(audio_part)
    else:
        contents.append({"mime_type": "audio/wav", "data": storage.read_bytes(audio_key), "label": "drill-audio"})

    frame_keys = list(manifest.metadata.get("frames") or [])[: max(1, config.max_frames)]
    # Drills are short; one frame is plenty.
    for index, key in enumerate(frame_keys[:1], start=1):
        if not storage.exists(key):
            continue
        contents.append(f"# Sample video frame {index}")
        part = _maybe_files_api_part(storage, provider, key, "image/jpeg", config=config, uploaded=uploaded)
        if part is not None:
            contents.append(part)
        else:
            contents.append({"mime_type": "image/jpeg", "data": storage.read_bytes(key), "label": f"drill-frame-{index}"})

    contents.append(
        "# Your task\n\n"
        "Write 1-2 short paragraphs of markdown feedback for the student, "
        "per the system instruction. No JSON, no headings."
    )

    try:
        from masterclass.agent_tools.registry import default_drill_tool_registry
        from masterclass.engine.instruments import load_instrument_profile
        profile = load_instrument_profile(manifest.instrument_profile)
        registry = default_drill_tool_registry(profile)
        def run_tool(name: str, args: dict[str, Any]) -> dict[str, Any]:
            return registry.call(storage, manifest.session, name, args)

        text, usage, tool_calls = provider.generate_with_tools(
            model=config.model,
            system_instruction=system_instruction,
            contents=contents,
            tools=registry.declarations(),
            max_tool_calls=4,
            tool_executor=run_tool,
        )
    finally:
        _delete_uploaded_files(uploaded)

    feedback = (text or "").strip()
    if not feedback:
        raise RuntimeError("drill feedback LLM returned no text")

    fb_key = store.artifact_key(manifest.session, "lesson/drill_feedback.md")
    storage.write_bytes(fb_key, feedback.encode("utf-8"), content_type="text/markdown")
    manifest.artifacts["lesson/drill_feedback.md"] = fb_key
    manifest.metadata["drill_feedback_excerpt"] = feedback[:300]
    manifest.llm_usage.append({
        "stage": "drill_feedback",
        "provider": usage.provider,
        "model": usage.model,
        "input_tokens": usage.input_tokens,
        "output_tokens": usage.output_tokens,
        "estimated_cost_usd": usage.estimated_cost_usd,
        "tool_calls": tool_calls,
        "at": datetime.now(UTC).isoformat(),
    })
    store.save(manifest)
    return feedback


def _maybe_files_api_part(
    storage: ObjectStorage,
    provider: LlmProvider,
    key: str,
    mime_type: str,
    *,
    config: DrillConfig,
    uploaded: list[Any],
) -> Any | None:
    if not config.use_files_api or provider.provider_name == "dry-run":
        return None
    resolver = getattr(storage, "resolve_local_path", None)
    if not callable(resolver):
        return None
    try:
        path = Path(resolver(key))
    except Exception:
        return None
    if not path.exists():
        return None
    try:
        from google import genai
        from google.genai import types
    except ImportError:
        return None
    api_key = getattr(provider, "api_key", None) or os.environ.get("GEMINI_API_KEY")
    if not api_key:
        return None
    client = genai.Client(api_key=str(api_key), http_options=types.HttpOptions(timeout=config.request_timeout_sec * 1000))
    uploaded_file = client.files.upload(file=str(path), config={"mime_type": mime_type})
    uploaded.append((client, uploaded_file))
    return uploaded_file


def _delete_uploaded_files(uploaded: list[Any]) -> None:
    for client, file_obj in uploaded:
        try:
            client.files.delete(name=file_obj.name)
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Conversation append (parent comment thread)
# ---------------------------------------------------------------------------
def _post_drill_result_to_parent(
    *,
    storage: ObjectStorage,
    store: SessionStore,
    manifest: SessionManifest,
    feedback: str,
) -> None:
    parent_session_id = manifest.metadata.get("parent_session_id")
    parent_comment_id = manifest.metadata.get("parent_comment_id")
    if not parent_session_id or not parent_comment_id:
        return
    ctx = TenantContext(tenant_id=manifest.session.tenant_id, user_id=manifest.session.user_id)
    try:
        parent_manifest = store.load_by_id(ctx, parent_session_id)
    except FileNotFoundError:
        _LOG.warning("drill parent session %s missing; skipping chat append", parent_session_id)
        return
    conv_id = _comment_conversation_id(parent_comment_id)
    key = conversation_key(store, parent_manifest.session, conv_id)
    with conversation_lock(key):
        try:
            conversation = load_conversation(storage, store, parent_manifest, conv_id)
        except FileNotFoundError:
            conversation = ChatConversation(
                conversation_id=conv_id,
                session_id=parent_manifest.session.session_id,
                user_id=parent_manifest.session.user_id,
            )
        conversation.append(ChatMessage(
            role="teacher",
            content=feedback,
            metadata={
                "type": "drill_result",
                "drill_session_id": manifest.session.session_id,
                "state": "ready",
            },
        ))
        # Best-effort: also flip the most recent matching drill_upload
        # bubble to ``state=ready`` so the UI doesn't keep showing a
        # spinner once we've appended the result.
        for msg in reversed(conversation.messages):
            md = msg.metadata or {}
            if md.get("type") == "drill_upload" and md.get("drill_session_id") == manifest.session.session_id:
                md["state"] = "ready"
                msg.metadata = md
                break
        save_conversation(storage, store, parent_manifest, conversation)


def _post_drill_failure_to_parent(
    *,
    storage: ObjectStorage,
    store: SessionStore,
    manifest: SessionManifest,
    error_message: str,
) -> None:
    parent_session_id = manifest.metadata.get("parent_session_id")
    parent_comment_id = manifest.metadata.get("parent_comment_id")
    if not parent_session_id or not parent_comment_id:
        return
    ctx = TenantContext(tenant_id=manifest.session.tenant_id, user_id=manifest.session.user_id)
    try:
        parent_manifest = store.load_by_id(ctx, parent_session_id)
    except FileNotFoundError:
        return
    conv_id = _comment_conversation_id(parent_comment_id)
    key = conversation_key(store, parent_manifest.session, conv_id)
    with conversation_lock(key):
        try:
            conversation = load_conversation(storage, store, parent_manifest, conv_id)
        except FileNotFoundError:
            return
        # Flip latest matching drill_upload to failed.
        for msg in reversed(conversation.messages):
            md = msg.metadata or {}
            if md.get("type") == "drill_upload" and md.get("drill_session_id") == manifest.session.session_id:
                md["state"] = "failed"
                md["error"] = error_message
                msg.metadata = md
                break
        conversation.append(ChatMessage(
            role="system",
            content=f"❌ Practice clip analysis failed: {error_message}",
            metadata={
                "type": "drill_result",
                "drill_session_id": manifest.session.session_id,
                "state": "failed",
                "error": error_message,
            },
        ))
        save_conversation(storage, store, parent_manifest, conversation)


def _comment_conversation_id(comment_id: str) -> str:
    """Mirror of ``main._comment_conversation_id`` to avoid circular import."""
    import re
    safe = re.sub(r"[^A-Za-z0-9_.-]", "_", (comment_id or "").strip())
    if not safe:
        raise ValueError("comment_id is required")
    return safe_conversation_id(f"cmt_{safe[:64]}")


# ---------------------------------------------------------------------------
# Public orchestrator
# ---------------------------------------------------------------------------
def run_drill_pipeline(
    *,
    storage: ObjectStorage,
    store: SessionStore,
    manifest: SessionManifest,
    provider: LlmProvider,
    config: DrillConfig | None = None,
    ffmpeg: FfmpegToolchain | None = None,
    resume: bool = False,
) -> SessionManifest:
    config = config or DrillConfig()
    ffmpeg = ffmpeg or FfmpegToolchain.discover()

    def _stage_done(stage: str) -> bool:
        if not resume:
            return False
        return (manifest.metadata.get(f"{stage}_state") or "") == "ready"

    _set_top_state(store, manifest, "processing")
    manifest.state = JobState.ANALYZING
    store.save(manifest)

    try:
        if not _stage_done("extract_media"):
            _mark_stage(store, manifest, "extract_media", "running")
            _stage_extract_media(storage=storage, store=store, manifest=manifest, ffmpeg=ffmpeg, max_frames=config.max_frames)
            _mark_stage(store, manifest, "extract_media", "ready")

        notes: list[dict[str, Any]]
        if _stage_done("transcribe"):
            notes_key = manifest.artifacts.get("analysis/drill_audio_truth_notes.json")
            payload = storage.read_json(notes_key) if notes_key else {}
            notes = list(payload.get("notes") or []) if isinstance(payload, dict) else []
        else:
            _mark_stage(store, manifest, "transcribe", "running")
            notes = _stage_transcribe(storage=storage, store=store, manifest=manifest)
            _mark_stage(store, manifest, "transcribe", "ready")

        if _stage_done("drill_metrics"):
            metrics_key = manifest.artifacts.get("analysis/drill_metrics.json")
            metrics = storage.read_json(metrics_key) if metrics_key else {}
        else:
            _mark_stage(store, manifest, "drill_metrics", "running")
            metrics = _stage_drill_metrics(storage=storage, store=store, manifest=manifest, notes=notes)
            _mark_stage(store, manifest, "drill_metrics", "ready")

        _mark_stage(store, manifest, "drill_feedback", "running")
        feedback = _stage_drill_feedback(
            storage=storage,
            store=store,
            manifest=manifest,
            provider=provider,
            metrics=metrics if isinstance(metrics, dict) else {},
            config=config,
        )
        _mark_stage(store, manifest, "drill_feedback", "ready")

        _post_drill_result_to_parent(storage=storage, store=store, manifest=manifest, feedback=feedback)

        _set_top_state(store, manifest, "ready")
        manifest.state = JobState.READY
        store.save(manifest)
        return manifest
    except Exception as exc:
        err = f"{type(exc).__name__}: {exc}"
        # Mark whatever stage is currently 'running' as failed.
        for stage in ("extract_media", "transcribe", "drill_metrics", "drill_feedback"):
            if (manifest.metadata.get(f"{stage}_state") or "") == "running":
                _mark_stage(store, manifest, stage, "failed", err)
                break
        manifest.errors.append({
            "stage": "drill",
            "error": err,
            "at": datetime.now(UTC).isoformat(),
        })
        _set_top_state(store, manifest, "failed")
        manifest.metadata["drill_error"] = err
        manifest.state = JobState.FAILED
        store.save(manifest)
        try:
            _post_drill_failure_to_parent(storage=storage, store=store, manifest=manifest, error_message=err)
        except Exception:
            _LOG.exception("posting drill failure to parent thread failed")
        raise
