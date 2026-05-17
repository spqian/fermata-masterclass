"""Tests for the drill (practice clip) engine layer."""
from __future__ import annotations

from typing import Any

import pytest

from masterclass.agent.llm import LlmUsage
from masterclass.agent_tools._drill_guard import session_is_drill
from masterclass.agent_tools.inspect_bar import inspect_bar
from masterclass.agent_tools.inspect_chord import inspect_chord
from masterclass.agent_tools.inspect_note import inspect_note
from masterclass.agent_tools.registry import default_drill_tool_registry, default_tool_registry
from masterclass.core.models import SESSION_KIND_DRILL, SESSION_KIND_LESSON
from masterclass.engine.drill_metrics import compute_drill_metrics

from tests.conftest import make_session_manifest


# ---------------------------------------------------------------------------
# drill_metrics
# ---------------------------------------------------------------------------
def test_drill_metrics_empty_notes_marked_low_signal():
    m = compute_drill_metrics([])
    assert m["n_notes"] == 0
    assert m["low_signal"] is True
    assert "no notes detected" in m["low_signal_reason"]


def test_drill_metrics_low_signal_under_4_notes():
    notes = [{"performed_time_sec": 0.0, "dwell_sec": 0.3, "pitches_midi": [60], "amplitude": 0.5}]
    m = compute_drill_metrics(notes)
    assert m["n_notes"] == 1
    assert m["low_signal"] is True


def test_drill_metrics_computes_ioi_and_tempo_for_even_pulse():
    # 4 evenly-spaced quarter notes at 120 bpm -> IOI = 0.5s -> tempo ~120.
    notes = []
    for i, midi in enumerate([60, 62, 64, 65]):
        notes.append({
            "performed_time_sec": i * 0.5,
            "dwell_sec": 0.4,
            "pitches_midi": [midi],
            "amplitude": 0.5,
        })
    m = compute_drill_metrics(notes)
    assert m["n_notes"] == 4
    assert m["low_signal"] is False
    assert m["ioi_mean_sec"] == pytest.approx(0.5, abs=1e-3)
    assert m["ioi_median_sec"] == pytest.approx(0.5, abs=1e-3)
    assert m["ioi_stdev_sec"] == pytest.approx(0.0, abs=1e-3)
    assert m["tempo_bpm_estimate"] == pytest.approx(120.0, abs=0.5)
    assert m["n_unique_pitches"] == 4


def test_drill_metrics_uneven_pulse_has_nonzero_cv():
    notes = []
    onsets = [0.0, 0.4, 1.0, 1.8]  # uneven IOIs: 0.4, 0.6, 0.8
    for t, midi in zip(onsets, [60, 60, 60, 60]):
        notes.append({"performed_time_sec": t, "dwell_sec": 0.2, "pitches_midi": [midi], "amplitude": 0.4})
    m = compute_drill_metrics(notes)
    assert m["ioi_cv"] is not None
    assert m["ioi_cv"] > 0.1
    # Repeated pitch -> top pitch count == 4
    assert m["top_pitch_midi"] == 60
    assert m["top_pitch_count"] == 4


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
def test_default_drill_registry_excludes_score_anchored_tools():
    registry = default_drill_tool_registry(None)
    names = {decl["name"] for decl in registry.declarations()}
    for excluded in ("inspect_bar", "inspect_note", "inspect_chord", "inspect_voicing", "measure_vibrato", "list_frames"):
        assert excluded not in names, f"{excluded} should not be in drill registry"
    for included in ("listen", "get_frames", "watch", "measure_tempo", "measure_dynamics"):
        assert included in names, f"{included} should be in drill registry"


def test_default_lesson_registry_unchanged_still_has_score_tools():
    registry = default_tool_registry(None)
    names = {decl["name"] for decl in registry.declarations()}
    assert "inspect_bar" in names
    assert "inspect_note" in names


# ---------------------------------------------------------------------------
# Tool guards on drill sessions
# ---------------------------------------------------------------------------
def test_session_is_drill_detects_kind_field(local_storage, session_store, tenant_ctx):
    drill = make_session_manifest(local_storage, session_store, tenant_ctx)
    drill.kind = SESSION_KIND_DRILL
    session_store.save(drill)
    assert session_is_drill(local_storage, drill.session) is True

    lesson = make_session_manifest(local_storage, session_store, tenant_ctx)
    assert session_is_drill(local_storage, lesson.session) is False


def test_inspect_bar_rejects_drill_kind(local_storage, session_store, tenant_ctx):
    drill = make_session_manifest(local_storage, session_store, tenant_ctx)
    drill.kind = SESSION_KIND_DRILL
    session_store.save(drill)
    result = inspect_bar(local_storage, drill.session, {"midi_measure": 1})
    assert "drill" in result.get("error", "")
    assert result.get("kind") == "drill"


def test_inspect_note_rejects_drill_kind(local_storage, session_store, tenant_ctx):
    drill = make_session_manifest(local_storage, session_store, tenant_ctx)
    drill.kind = SESSION_KIND_DRILL
    session_store.save(drill)
    result = inspect_note(local_storage, drill.session, {"midi_measure": 1, "beat": 1})
    assert result.get("kind") == "drill"


def test_inspect_chord_rejects_drill_kind(local_storage, session_store, tenant_ctx):
    drill = make_session_manifest(local_storage, session_store, tenant_ctx)
    drill.kind = SESSION_KIND_DRILL
    session_store.save(drill)
    result = inspect_chord(local_storage, drill.session, {"time_sec": 0.5})
    assert result.get("kind") == "drill"


# ---------------------------------------------------------------------------
# Pipeline happy-path with stubbed Gemini provider
# ---------------------------------------------------------------------------
class _StubDrillProvider:
    """Returns a fixed feedback string; doesn't touch the Gemini Files API."""
    provider_name = "stub-drill"

    def __init__(self, feedback: str = "Good pulse on the trill. Try the metronome at 80 next take.") -> None:
        self.feedback = feedback
        self.last_contents: list[Any] | None = None
        self.last_system: str | None = None

    def generate_with_tools(self, *, model, system_instruction, contents, tools, max_tool_calls, tool_executor=None):
        self.last_contents = list(contents)
        self.last_system = system_instruction
        usage = LlmUsage(provider=self.provider_name, model=model, input_tokens=400, output_tokens=80, estimated_cost_usd=0.001)
        return self.feedback, usage, []


def test_drill_pipeline_happy_path_writes_feedback_and_metrics(
    monkeypatch,
    local_storage,
    session_store,
    tenant_ctx,
):
    from masterclass.engine import drill_pipeline as dp

    drill = make_session_manifest(
        local_storage,
        session_store,
        tenant_ctx,
        artifacts={"input/source_video": b"fake-mp4-bytes"},
        metadata={
            "drill_instruction": "Play the trill from m.7 slowly with a metronome, then at top speed.",
            "parent_comment": {"measure": 7, "severity": "warn", "summary": "Trill uneven", "text": "Trill from m.7..."},
        },
    )
    drill.kind = SESSION_KIND_DRILL
    session_store.save(drill)

    # Stub each stage that touches ffmpeg / basic-pitch.
    def fake_extract_media(*, storage, store, manifest, ffmpeg, max_frames):
        audio_key = store.artifact_key(manifest.session, "artifacts/audio.wav")
        storage.write_bytes(audio_key, b"fake-wav-bytes", content_type="audio/wav")
        manifest.artifacts["artifacts/audio.wav"] = audio_key
        manifest.artifacts["artifacts/audio_16k.wav"] = audio_key
        manifest.metadata["frames"] = []
        store.save(manifest)

    def fake_transcribe(*, storage, store, manifest):
        notes = [
            {"performed_time_sec": i * 0.5, "dwell_sec": 0.4, "pitches_midi": [60 + (i % 2)], "amplitude": 0.5}
            for i in range(6)
        ]
        notes_key = store.artifact_key(manifest.session, "analysis/drill_audio_truth_notes.json")
        storage.write_json(notes_key, {"notes": notes, "n_notes": len(notes)})
        manifest.artifacts["analysis/drill_audio_truth_notes.json"] = notes_key
        store.save(manifest)
        return notes

    class FakeFfmpeg:
        pass

    monkeypatch.setattr(dp, "_stage_extract_media", fake_extract_media)
    monkeypatch.setattr(dp, "_stage_transcribe", fake_transcribe)

    provider = _StubDrillProvider()
    result = dp.run_drill_pipeline(
        storage=local_storage,
        store=session_store,
        manifest=drill,
        provider=provider,
        ffmpeg=FakeFfmpeg(),
    )

    assert result.metadata["drill_state"] == "ready"
    assert result.metadata["extract_media_state"] == "ready"
    assert result.metadata["transcribe_state"] == "ready"
    assert result.metadata["drill_metrics_state"] == "ready"
    assert result.metadata["drill_feedback_state"] == "ready"
    fb_key = result.artifacts["lesson/drill_feedback.md"]
    assert local_storage.read_bytes(fb_key).decode("utf-8") == provider.feedback
    metrics = local_storage.read_json(result.artifacts["analysis/drill_metrics.json"])
    assert metrics["n_notes"] == 6
    # The drill prompt + drill instruction should both have been sent.
    assert provider.last_system and "drill" in provider.last_system.lower()
    assert any("drill_instruction" in (c if isinstance(c, str) else "").lower() or "drill instruction" in (c if isinstance(c, str) else "").lower()
               for c in (provider.last_contents or []))


def test_drill_pipeline_failure_marks_state_and_records_error(
    monkeypatch,
    local_storage,
    session_store,
    tenant_ctx,
):
    from masterclass.engine import drill_pipeline as dp

    drill = make_session_manifest(
        local_storage,
        session_store,
        tenant_ctx,
        artifacts={"input/source_video": b"fake"},
        metadata={"drill_instruction": "anything"},
    )
    drill.kind = SESSION_KIND_DRILL
    session_store.save(drill)

    def boom(*args, **kwargs):
        raise RuntimeError("ffmpeg blew up")

    monkeypatch.setattr(dp, "_stage_extract_media", boom)

    class FakeFfmpeg: pass

    with pytest.raises(RuntimeError):
        dp.run_drill_pipeline(
            storage=local_storage,
            store=session_store,
            manifest=drill,
            provider=_StubDrillProvider(),
            ffmpeg=FakeFfmpeg(),
        )
    reloaded = session_store.load(drill.session)
    assert reloaded.metadata["drill_state"] == "failed"
    assert reloaded.metadata["extract_media_state"] == "failed"
    assert "ffmpeg blew up" in (reloaded.metadata.get("extract_media_error") or "")
    assert reloaded.errors and reloaded.errors[0]["stage"] == "drill"


def test_drill_pipeline_appends_result_to_parent_comment_thread(
    monkeypatch,
    local_storage,
    session_store,
    tenant_ctx,
):
    from masterclass.engine import drill_pipeline as dp
    from masterclass.core.chat_models import load_conversation

    parent = make_session_manifest(local_storage, session_store, tenant_ctx)

    drill = make_session_manifest(
        local_storage,
        session_store,
        tenant_ctx,
        artifacts={"input/source_video": b"fake"},
        metadata={
            "drill_instruction": "test instruction",
            "parent_session_id": parent.session.session_id,
            "parent_comment_id": "g_007",
        },
    )
    drill.kind = SESSION_KIND_DRILL
    session_store.save(drill)

    def fake_extract_media(*, storage, store, manifest, ffmpeg, max_frames):
        ak = store.artifact_key(manifest.session, "artifacts/audio.wav")
        storage.write_bytes(ak, b"x", content_type="audio/wav")
        manifest.artifacts["artifacts/audio.wav"] = ak
        manifest.metadata["frames"] = []
        store.save(manifest)

    def fake_transcribe(*, storage, store, manifest):
        notes = [{"performed_time_sec": i * 0.5, "dwell_sec": 0.3, "pitches_midi": [60], "amplitude": 0.4} for i in range(5)]
        nk = store.artifact_key(manifest.session, "analysis/drill_audio_truth_notes.json")
        storage.write_json(nk, {"notes": notes})
        manifest.artifacts["analysis/drill_audio_truth_notes.json"] = nk
        store.save(manifest)
        return notes

    monkeypatch.setattr(dp, "_stage_extract_media", fake_extract_media)
    monkeypatch.setattr(dp, "_stage_transcribe", fake_transcribe)

    dp.run_drill_pipeline(
        storage=local_storage,
        store=session_store,
        manifest=drill,
        provider=_StubDrillProvider("Nice work."),
        ffmpeg=object(),
    )

    conv = load_conversation(local_storage, session_store, parent, "cmt_g_007")
    assert any(
        msg.metadata.get("type") == "drill_result" and msg.metadata.get("drill_session_id") == drill.session.session_id
        for msg in conv.messages
    )
