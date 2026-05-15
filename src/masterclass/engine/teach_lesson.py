from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from masterclass.agent.llm import LlmProvider
from masterclass.agent_tools.catalog import tool_catalog_text
from masterclass.agent_tools.registry import default_tool_registry
from masterclass.core.models import JobState, SessionManifest
from masterclass.core.sessions import SessionStore
from masterclass.engine.instruments import BUILTIN_PROFILES, load_instrument_profile, system_instruction_for_profile
from masterclass.engine.prompt_evidence import build_evidence_digest
from masterclass.engine.prompt_inventory import build_score_note_inventory
from masterclass.storage.base import ObjectStorage


TEACH_PROFILES = BUILTIN_PROFILES


COMMENT_CATEGORIES = [
    "intonation",
    "voicing",
    "rhythm",
    "dynamics",
    "articulation",
    "phrasing",
    "interpretation",
    "technique",
    "progress",
    "encouragement",
    "musical",
]


TEACH_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "summary": {"type": "string"},
        "lesson": {"type": "object"},
        "measure_timestamps": {"type": "array"},
        "comments": {"type": "array"},
        "progress_notes": {"type": "string"},
    },
    "required": ["summary", "lesson", "comments", "progress_notes", "measure_timestamps"],
}


@dataclass(frozen=True)
class TeachConfig:
    model: str = "gemini-2.5-pro"
    max_score_pages: int = 8
    audio_max_bytes: int = 28 * 1024 * 1024
    max_tool_calls: int = 15
    max_video_frames: int = 16
    use_files_api: bool = True


def teach_lesson(
    *,
    storage: ObjectStorage,
    store: SessionStore,
    manifest: SessionManifest,
    provider: LlmProvider,
    score_pages: list[bytes] | None = None,
    score_layout: list[dict[str, Any]] | None = None,
    config: TeachConfig | None = None,
) -> SessionManifest:
    """Run the agentic Gemini-as-teacher loop and persist v2 lesson comments."""

    config = config or TeachConfig()
    manifest.state = JobState.TEACHING
    store.save(manifest)

    profile = load_instrument_profile(manifest.instrument_profile)
    registry = default_tool_registry(profile)
    system_instruction = system_instruction_for_profile(profile, tool_catalog=tool_catalog_text(profile))

    score_map = _read_score_map(storage, store, manifest)
    score_key = score_map.get("key") if score_map else None
    first_measure = _as_int(manifest.metadata.get("first_measure"), score_map.get("first_measure") if score_map else None)
    last_measure = _as_int(manifest.metadata.get("last_measure"), score_map.get("last_measure") if score_map else None)
    evidence_digest = build_evidence_digest(storage=storage, store=store, manifest=manifest, score_key=score_key)
    if score_key:
        evidence_digest = (
            f"Key: {score_key}. Use this key's spelling in all comments; copy note spellings from the inventory.\n\n"
            + evidence_digest
        )
    inventory = build_score_note_inventory(score_map, first_measure=first_measure, last_measure=last_measure) if score_map else ""
    prior_context = _read_prior_context(storage, manifest)

    audio_key = manifest.artifacts.get("artifacts/audio_16k.wav") or manifest.artifacts.get("artifacts/audio.wav")
    if not audio_key:
        raise ValueError("manifest is missing lesson audio")

    uploaded: list[Any] = []
    contents: list[Any] = []
    try:
        contents.extend(_build_user_contents(
            storage=storage,
            store=store,
            manifest=manifest,
            provider=provider,
            config=config,
            audio_key=audio_key,
            score_map=score_map,
            evidence_digest=evidence_digest,
            inventory=inventory,
            prior_context=prior_context,
            score_pages=score_pages or [],
            score_layout=score_layout or [],
            uploaded=uploaded,
        ))

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
        result = _extract_json_block(text)
        if result is None:
            raw_key = store.artifact_key(manifest.session, "llm/raw_teacher_response.txt")
            storage.write_bytes(raw_key, text.encode("utf-8"), content_type="text/plain")
            manifest.artifacts["llm/raw_teacher_response.txt"] = raw_key
            raise RuntimeError("Gemini teacher returned no parseable JSON")
        _validate_teach_response(result)
    except Exception as exc:
        _delete_uploaded_files(uploaded)
        manifest.metadata["teach_state"] = "failed"
        manifest.metadata["teach_error"] = f"{type(exc).__name__}: {exc}"
        manifest.errors.append({"stage": "teach", "error": manifest.metadata["teach_error"], "at": datetime.now(UTC).isoformat()})
        store.save(manifest)
        raise

    comments = _normalize_comments(result.get("comments") or [])
    measure_timestamps = _normalize_measure_timestamps(result.get("measure_timestamps") or [])
    summary = str(result.get("summary") or result.get("enrichment_notes") or "").strip()
    progress_notes = str(result.get("progress_notes") or "").strip()
    turns = _turn_count(tool_calls)
    tool_audit = {
        "schema_version": 1,
        "model": config.model,
        "generated_at": datetime.now(UTC).isoformat(),
        "turn_count": turns,
        "tool_calls": tool_calls,
    }

    enriched = {
        "schema_version": 4,
        "summary": summary,
        "lesson": result.get("lesson") or {},
        "progress_notes": progress_notes,
        "measure_timestamps": measure_timestamps,
        "comments": comments,
        "dropped": result.get("dropped") or [],
        "_meta": {
            "model": config.model,
            "provider": provider.provider_name,
            "input_tokens": usage.input_tokens,
            "output_tokens": usage.output_tokens,
            "estimated_cost_usd": usage.estimated_cost_usd,
            "score_image_count": _score_image_count(contents),
            "n_tool_calls": len(tool_calls),
            "n_turns": turns,
            "generated_at": datetime.now(UTC).isoformat(),
        },
    }

    enriched_key = store.artifact_key(manifest.session, "lesson/comments_enriched.json")
    comments_key = store.artifact_key(manifest.session, "lesson/comments.json")
    raw_key = store.artifact_key(manifest.session, "llm/raw_teacher_response.json")
    audit_key = store.artifact_key(manifest.session, "analysis/teach_tool_calls.json")
    storage.write_json(enriched_key, enriched)
    storage.write_json(comments_key, {"comments": comments})
    storage.write_json(raw_key, result)
    storage.write_json(audit_key, tool_audit)
    manifest.artifacts["lesson/comments_enriched.json"] = enriched_key
    manifest.artifacts["lesson/comments.json"] = comments_key
    manifest.artifacts["llm/raw_teacher_response.json"] = raw_key
    manifest.artifacts["analysis/teach_tool_calls.json"] = audit_key
    manifest.metadata["teach_state"] = "ready"
    manifest.metadata["teach_summary"] = summary
    manifest.metadata["teach_comment_count"] = len(comments)
    manifest.metadata["teach_turn_count"] = turns
    manifest.metadata["teach_tool_call_count"] = len(tool_calls)
    manifest.metadata["teach_tool_calls_artifact"] = audit_key
    manifest.llm_usage.append({
        "stage": "teach",
        "provider": usage.provider,
        "model": usage.model,
        "input_tokens": usage.input_tokens,
        "output_tokens": usage.output_tokens,
        "estimated_cost_usd": usage.estimated_cost_usd,
        "tool_calls": tool_calls,
        "turns": turns,
        "at": datetime.now(UTC).isoformat(),
    })
    manifest.state = JobState.READY
    store.save(manifest)
    _delete_uploaded_files(uploaded)
    return manifest


def _build_user_contents(
    *,
    storage: ObjectStorage,
    store: SessionStore,
    manifest: SessionManifest,
    provider: LlmProvider,
    config: TeachConfig,
    audio_key: str,
    score_map: dict[str, Any],
    evidence_digest: str,
    inventory: str,
    prior_context: str,
    score_pages: list[bytes],
    score_layout: list[dict[str, Any]],
    uploaded: list[Any],
) -> list[Any]:
    contents: list[Any] = [
        "# Evidence digest\n\n" + (evidence_digest or "(no deterministic evidence digest available)") + "\n",
        "# Recording briefing\n\n"
        f"Repertoire: {manifest.repertoire or '(unknown)'}\n"
        f"Movement: {manifest.movement or '(unknown)'}\n"
        f"Instrument: {manifest.instrument or manifest.instrument_profile or '(unspecified)'}\n"
        f"Measures: {manifest.metadata.get('first_measure')}–{manifest.metadata.get('last_measure')}\n"
        f"Student notes: {(manifest.notes or '(none)').strip()}\n",
        "# Prior takes of this piece\n\n" + (prior_context or "(no prior lessons)") + "\n",
        "# Recording (audio)\n",
    ]

    audio_part = _file_part(storage, provider, audio_key, "audio/wav", config=config, uploaded=uploaded)
    if audio_part is not None:
        contents.append(audio_part)
    else:
        contents.append({"mime_type": "audio/wav", "data": storage.read_bytes(audio_key), "label": "lesson-audio"})

    score_image_keys = _score_image_keys(storage, store, manifest, score_map)
    contents.append("\n# Score (system images, in order of appearance)\n")
    if score_image_keys:
        for index, key in enumerate(score_image_keys[: config.max_score_pages], start=1):
            contents.append(f"score image {index}: {key.rsplit('/', 1)[-1]}")
            part = _file_part(storage, provider, key, "image/png", config=config, uploaded=uploaded)
            contents.append(part if part is not None else {"mime_type": "image/png", "data": storage.read_bytes(key), "label": f"score-{index}"})
    elif score_pages:
        if score_layout:
            contents.append("\n--- score layout ---\n" + json.dumps(score_layout, indent=2))
        for index, png in enumerate(score_pages[: config.max_score_pages], start=1):
            contents.append({"mime_type": "image/png", "data": png, "label": f"score-page-{index}"})
    else:
        contents.append("(no score images available)\n")

    contents.append("\n# Sample video frames (for technique/visual observations)\n")
    frame_keys = _frame_keys(storage, manifest, config.max_video_frames)
    for index, key in enumerate(frame_keys, start=1):
        contents.append(f"video frame {index}: {key.rsplit('/', 1)[-1]}")
        part = _file_part(storage, provider, key, "image/jpeg", config=config, uploaded=uploaded)
        contents.append(part if part is not None else {"mime_type": "image/jpeg", "data": storage.read_bytes(key), "label": f"frame-{index}"})
    if not frame_keys:
        contents.append("(no frames available)\n")

    contents.append("\n# Score note inventory (for note_id citations in your final output)\n\n" + (inventory or "(no score note inventory available)") + "\n")
    contents.append(
        "# Your task\n\n"
        "Listen to the recording. Examine the score. **Examine the video frames** above — they are not decoration; "
        "comment on what you see (hand position, fingering, posture, pedaling, bow technique, etc.) when something is genuinely visible. "
        "Use `get_frames` to extract additional frames at any moment of interest (e.g. around tricky passages or suspicious sounds). "
        "Use investigation tools to fact-check measurable claims. "
        "Then produce the final v2 `comments_enriched.json` as a single JSON code block. "
        "The JSON must include `summary`, `lesson`, `progress_notes`, `measure_timestamps`, and `comments`; "
        "each comment must include `references` with measure/beat/note_name/page/system_index/note_id when possible. "
        "Aim for 5-12 tool calls and 8-15 final comments, with at least one technique/visual comment if anything is visible in the frames."
    )
    return contents


def _file_part(
    storage: ObjectStorage,
    provider: LlmProvider,
    key: str,
    mime_type: str,
    *,
    config: TeachConfig,
    uploaded: list[Any],
) -> Any | None:
    if not config.use_files_api or provider.provider_name == "dry-run":
        return None
    path = _local_path(storage, key)
    if path is None or not path.exists():
        return None
    try:
        from google import genai
        from google.genai import types
    except ImportError:
        return None
    api_key = getattr(provider, "api_key", None) or os.environ.get("GEMINI_API_KEY")
    if not api_key:
        return None
    client = genai.Client(api_key=str(api_key), http_options=types.HttpOptions(timeout=90_000))
    uploaded_file = client.files.upload(file=str(path), config={"mime_type": mime_type})
    uploaded.append((client, uploaded_file))
    return uploaded_file


def _delete_uploaded_files(uploaded: list[Any]) -> None:
    for client, file_obj in uploaded:
        try:
            client.files.delete(name=file_obj.name)
        except Exception:
            pass


def _local_path(storage: ObjectStorage, key: str) -> Path | None:
    resolver = getattr(storage, "resolve_local_path", None)
    if callable(resolver):
        try:
            return Path(resolver(key))
        except Exception:
            return None
    return None


def _read_score_map(storage: ObjectStorage, store: SessionStore, manifest: SessionManifest) -> dict[str, Any]:
    for key in (manifest.artifacts.get("score/score_map.json"), store.artifact_key(manifest.session, "score/score_map.json")):
        if key and storage.exists(key):
            return storage.read_json(key)
    return {}


def _read_prior_context(storage: ObjectStorage, manifest: SessionManifest) -> str:
    """Return the prior-lessons context formatted as Markdown the teacher can act on."""
    key = manifest.artifacts.get("context/prior_lessons.json")
    if not (key and storage.exists(key)):
        return "(no prior lessons — this is the first take of this piece)"
    try:
        data = storage.read_json(key)
    except Exception:
        return "(prior context unreadable)"
    lessons = data.get("lessons") or []
    if not lessons:
        return "(no prior lessons — this is the first take of this piece)"
    lines: list[str] = []
    lines.append(f"This is take #{len(lessons) + 1} of this piece. {len(lessons)} previous take(s) on record:")
    lines.append("")
    for idx, lesson in enumerate(lessons, start=1):
        when = (lesson.get("created_at") or "")[:10] or "(unknown date)"
        bars = ""
        f, l = lesson.get("first_measure"), lesson.get("last_measure")
        if f or l:
            bars = f" (bars {f or '?'}–{l or '?'})"
        lines.append(f"## Prior take #{idx} — {when}{bars}")
        student_notes = (lesson.get("notes") or "").strip()
        if student_notes:
            lines.append(f"Student's stated focus: {student_notes!r}")
        if lesson.get("summary"):
            lines.append(f"Teacher's summary last time: {lesson['summary']}")
        prior_lesson = lesson.get("lesson") or {}
        if prior_lesson.get("artistic_summary"):
            lines.append(f"Artistic reading last time: {prior_lesson['artistic_summary']}")
        what_works = prior_lesson.get("what_works") or []
        if what_works:
            lines.append("What was working last time:")
            for item in what_works[:8]:
                lines.append(f"  - {item}")
        areas = prior_lesson.get("areas_to_develop") or []
        if areas:
            lines.append("Areas to develop last time (CHECK IF THESE IMPROVED):")
            for a in areas[:8]:
                if isinstance(a, dict):
                    focus = a.get("focus") or "(unnamed)"
                    pri = a.get("priority") or "?"
                    ex = a.get("exercise") or ""
                    lines.append(f"  - [{pri}] {focus}: {ex}")
                else:
                    lines.append(f"  - {a}")
        practice = prior_lesson.get("this_week_practice") or []
        if practice:
            lines.append("Last time's prescribed practice (DID THE STUDENT FOLLOW IT?):")
            for p in practice[:8]:
                lines.append(f"  - {p}")
        next_take = prior_lesson.get("next_take")
        if next_take:
            lines.append(f"Last time's 'for next take' note: {next_take}")
        comments = lesson.get("teacher_comments") or []
        # Show only alert + warn comments (info is usually camera-angle noise).
        meaningful = [c for c in comments if (c.get("severity") or "").lower() in ("alert", "warn")]
        if meaningful:
            lines.append(f"Specific comments last time ({len(meaningful)} of {len(comments)} surfaced; alert/warn only):")
            for c in meaningful[:15]:
                sev = (c.get("severity") or "").upper()
                m = c.get("measure")
                summ = c.get("summary") or "(no summary)"
                lines.append(f"  - [{sev}] m{m}: {summ}")
        lines.append("")
    return "\n".join(lines).strip()


def _score_image_keys(storage: ObjectStorage, store: SessionStore, manifest: SessionManifest, score_map: dict[str, Any]) -> list[str]:
    keys: list[str] = []
    seen: set[str] = set()
    systems = score_map.get("systems") or []
    systems = sorted((s for s in systems if isinstance(s, dict)), key=lambda s: (int(s.get("page") or 0), int(s.get("system_on_page") or 0)))
    for system in systems:
        rel = system.get("image")
        key = manifest.artifacts.get(str(rel)) if rel else None
        if not key and rel:
            key = store.artifact_key(manifest.session, str(rel))
        if key and key not in seen and storage.exists(key):
            keys.append(key)
            seen.add(key)
    return keys


def _frame_keys(storage: ObjectStorage, manifest: SessionManifest, limit: int) -> list[str]:
    keys = [str(k) for k in (manifest.metadata.get("frames") or []) if isinstance(k, str)]
    if not keys:
        prefix = f"tenant/{manifest.session.tenant_id}/users/{manifest.session.user_id}/sessions/{manifest.session.session_id}/artifacts/frames"
        keys = sorted(storage.list_keys(prefix))
    return [k for k in keys if k.lower().endswith((".jpg", ".jpeg", ".png")) and storage.exists(k)][:limit]


def _extract_json_block(text: str) -> dict[str, Any] | None:
    import re

    for pattern in (r"```json\s*(\{.*?\})\s*```", r"```\s*(\{.*?\})\s*```"):
        match = re.search(pattern, text or "", re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                pass
    start = (text or "").find("{")
    if start < 0:
        return None
    depth = 0
    in_string = False
    escape = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_string:
            escape = (not escape and ch == "\\")
            if ch == '"' and not escape:
                in_string = False
            elif ch != "\\":
                escape = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[start : i + 1])
                except json.JSONDecodeError:
                    return None
    return None


def _validate_teach_response(result: dict[str, Any]) -> None:
    missing = [field for field in ("lesson", "comments", "progress_notes", "measure_timestamps") if field not in result]
    if missing:
        raise ValueError(f"teacher JSON missing required fields: {', '.join(missing)}")
    if not isinstance(result.get("lesson"), dict):
        raise ValueError("teacher JSON field `lesson` must be an object")
    if not isinstance(result.get("comments"), list):
        raise ValueError("teacher JSON field `comments` must be a list")
    for index, comment in enumerate(result.get("comments") or [], start=1):
        if isinstance(comment, dict) and "references" not in comment and "note_refs" not in comment:
            raise ValueError(f"teacher JSON comment {index} missing references")


def _normalize_comments(raw: list[Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for index, entry in enumerate(raw, start=1):
        if not isinstance(entry, dict):
            continue
        start = _as_float(entry.get("start", entry.get("start_sec")), 0.0) or 0.0
        end = _as_float(entry.get("end", entry.get("end_sec")), max(start + 2.0, start)) or max(start + 2.0, start)
        out.append({
            "id": str(entry.get("id") or f"g_{index:03d}"),
            "start": round(max(0.0, start), 3),
            "end": round(max(start, end), 3),
            "measure": entry.get("measure"),
            "beat": entry.get("beat"),
            "category": entry.get("category") or "interpretation",
            "severity": entry.get("severity") or "warn",
            "summary": str(entry.get("summary") or entry.get("title") or "").strip(),
            "text": str(entry.get("text") or entry.get("message") or "").strip(),
            "confidence": entry.get("confidence") or "medium",
            "evidence_ref": entry.get("evidence_ref"),
            "provenance": entry.get("provenance") or [],
            "references": _normalize_references(entry.get("references") or entry.get("note_refs") or []),
            "note_refs": entry.get("note_refs") or [],
        })
    out.sort(key=lambda c: c["start"])
    return out


def _normalize_references(raw: Any) -> list[dict[str, Any]]:
    refs: list[dict[str, Any]] = []
    if not isinstance(raw, list):
        return refs
    for ref in raw:
        if isinstance(ref, str):
            refs.append({"note_id": ref})
            continue
        if not isinstance(ref, dict):
            continue
        refs.append({
            "measure": ref.get("measure", ref.get("midi_measure")),
            "beat": ref.get("beat", ref.get("beat_in_bar")),
            "note_name": (ref.get("note_name") or ref.get("pitch_name") or ref.get("name") or "").strip() or None,
            "page": ref.get("page"),
            "system_index": ref.get("system_index", ref.get("system_on_page")),
            "hand": (ref.get("hand") or "").strip() or None,
            "note_id": ref.get("note_id"),
        })
    return refs


def _normalize_measure_timestamps(raw: Any) -> list[dict[str, float]]:
    pairs: list[tuple[int, float]] = []
    if isinstance(raw, dict):
        iterator = raw.items()
        for k, v in iterator:
            try:
                pairs.append((int(k), float(v)))
            except (TypeError, ValueError):
                continue
    elif isinstance(raw, list):
        for entry in raw:
            if not isinstance(entry, dict):
                continue
            try:
                pairs.append((int(entry.get("measure")), float(entry.get("start", entry.get("start_seconds")))))
            except (TypeError, ValueError):
                continue
    deduped: dict[int, float] = {}
    for measure, start in sorted(pairs):
        deduped.setdefault(measure, start)
    return [{"measure": m, "start": round(max(0.0, s), 3)} for m, s in sorted(deduped.items())]


def _turn_count(tool_calls: list[dict[str, Any]]) -> int:
    if not tool_calls:
        return 1
    return int(max((c.get("turn") or 0) for c in tool_calls) or 0) + 1


def _score_image_count(contents: list[Any]) -> int:
    return sum(
        1
        for part in contents
        if (isinstance(part, dict) and str(part.get("mime_type", "")).startswith("image/"))
        or (getattr(part, "mime_type", "") or "").startswith("image/")
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


def _as_float(value: Any, default: float | None = None) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default
