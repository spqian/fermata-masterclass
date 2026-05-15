from __future__ import annotations

import json
import math
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from masterclass.core.models import JobState, SessionManifest
from masterclass.core.sessions import SessionStore
from masterclass.storage.base import ObjectStorage


@dataclass(frozen=True)
class AnalysisConfig:
    sample_rate: int = 22050
    hop_length: int = 512


def analyze_session(
    *,
    store: SessionStore,
    storage: ObjectStorage,
    manifest: SessionManifest,
    config: AnalysisConfig | None = None,
) -> SessionManifest:
    config = config or AnalysisConfig()
    audio_key = manifest.artifacts.get("artifacts/audio.wav")
    if not audio_key:
        raise ValueError("manifest is missing artifacts/audio.wav; run ingestion first")

    manifest.state = JobState.ANALYZING
    store.save(manifest)

    with tempfile.TemporaryDirectory(prefix="masterclass-analyze-") as tmp_raw:
        audio_path = Path(tmp_raw) / "audio.wav"
        storage.read_to_file(audio_key, audio_path)
        analysis, pitch_events = analyze_audio_file(audio_path, manifest, config)

    analysis_key = store.artifact_key(manifest.session, "analysis/analysis.json")
    analysis_md_key = store.artifact_key(manifest.session, "analysis/analysis.md")
    pitch_json_key = store.artifact_key(manifest.session, "analysis/pitch_events.json")
    storage.write_json(analysis_key, analysis)
    storage.write_bytes(analysis_md_key, analysis_markdown(analysis).encode("utf-8"), content_type="text/markdown")
    storage.write_json(pitch_json_key, pitch_events)
    manifest.artifacts["analysis/analysis.json"] = analysis_key
    manifest.artifacts["analysis/analysis.md"] = analysis_md_key
    manifest.artifacts["analysis/pitch_events.json"] = pitch_json_key
    manifest.metadata["analysis_summary"] = analysis["global"]
    manifest.state = JobState.AWAITING_LLM
    store.save(manifest)
    return manifest


def analyze_audio_file(audio_path: Path, manifest: SessionManifest, config: AnalysisConfig) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    import librosa
    import numpy as np

    y, sr = librosa.load(audio_path, sr=config.sample_rate, mono=True)
    duration = float(librosa.get_duration(y=y, sr=sr))
    rms = librosa.feature.rms(y=y, hop_length=config.hop_length)[0]
    rms_db = librosa.amplitude_to_db(rms, ref=np.max) if len(rms) else np.array([])
    rms_times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=config.hop_length)

    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=config.hop_length)
    onset_frames = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr, hop_length=config.hop_length, units="frames")
    onset_times = librosa.frames_to_time(onset_frames, sr=sr, hop_length=config.hop_length)
    tempo_raw, beat_frames = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr, hop_length=config.hop_length)
    tempo = float(np.asarray(tempo_raw).reshape(-1)[0]) if np.asarray(tempo_raw).size else None
    beat_times = librosa.frames_to_time(beat_frames, sr=sr, hop_length=config.hop_length)

    pitch_events: list[dict[str, Any]] = []
    try:
        f0, voiced_flag, voiced_prob = librosa.pyin(
            y,
            fmin=librosa.note_to_hz("C2"),
            fmax=librosa.note_to_hz("C7"),
            sr=sr,
            hop_length=config.hop_length,
        )
        pitch_times = librosa.frames_to_time(np.arange(len(f0)), sr=sr, hop_length=config.hop_length)
        pitch_events = _pitch_segments(f0, voiced_flag, voiced_prob, pitch_times)
        voiced_ratio = float(np.nanmean(voiced_flag)) if len(voiced_flag) else 0.0
    except Exception:
        voiced_ratio = 0.0

    loudness_stats = _loudness_stats(rms_db)
    analysis = {
        "schema_version": 1,
        "session_id": manifest.session.session_id,
        "repertoire": manifest.repertoire,
        "movement": manifest.movement,
        "instrument": manifest.instrument,
        "duration_sec": round(duration, 3),
        "global": {
            "estimated_tempo_bpm": round(tempo, 2) if tempo is not None and math.isfinite(tempo) else None,
            "detected_onsets": int(len(onset_times)),
            "detected_beats": int(len(beat_times)),
            "voiced_pitch_ratio": round(voiced_ratio, 3),
            "loudness_db": loudness_stats,
        },
        "ranked_regions": {
            "loudest_regions": _rank_loud_regions(rms_times, rms_db),
            "quietest_active_regions": _rank_quiet_active_regions(rms_times, rms_db),
            "onset_dense_regions": _rank_onset_dense_regions(onset_times, duration),
        },
        "performed_pitch_events": {
            "count": len(pitch_events),
            "preview_first_40": pitch_events[:40],
        },
        "artifacts": {
            "audio_wav": manifest.artifacts.get("artifacts/audio.wav"),
            "metadata_json": manifest.artifacts.get("artifacts/metadata.json"),
        },
    }
    return analysis, pitch_events


def _pitch_segments(f0, voiced_flag, voiced_prob, times) -> list[dict[str, Any]]:
    import librosa
    import numpy as np

    events: list[dict[str, Any]] = []
    start = None
    values = []
    probs = []
    for idx, voiced in enumerate(voiced_flag):
        if voiced and np.isfinite(f0[idx]):
            if start is None:
                start = idx
            values.append(float(f0[idx]))
            probs.append(float(voiced_prob[idx]))
            continue
        if start is not None:
            _append_pitch_event(events, start, idx, values, probs, times, librosa)
            start = None
            values = []
            probs = []
    if start is not None:
        _append_pitch_event(events, start, len(voiced_flag) - 1, values, probs, times, librosa)
    return [event for event in events if event["duration_sec"] >= 0.08]


def _append_pitch_event(events, start_idx, end_idx, values, probs, times, librosa) -> None:
    import numpy as np

    median_hz = float(np.median(values))
    midi = float(librosa.hz_to_midi(median_hz))
    nearest = round(midi)
    events.append({
        "start_sec": round(float(times[start_idx]), 3),
        "end_sec": round(float(times[min(end_idx, len(times) - 1)]), 3),
        "duration_sec": round(float(times[min(end_idx, len(times) - 1)] - times[start_idx]), 3),
        "median_hz": round(median_hz, 3),
        "midi": round(midi, 2),
        "note": librosa.midi_to_note(int(nearest)),
        "median_cents_from_equal_temperament": round((midi - nearest) * 100.0, 1),
        "confidence": "high" if float(np.mean(probs)) >= 0.85 else "medium" if float(np.mean(probs)) >= 0.65 else "low",
    })


def _loudness_stats(rms_db) -> dict[str, float | None]:
    import numpy as np

    if len(rms_db) == 0:
        return {"min": None, "max": None, "median": None, "active_range": None}
    active = rms_db[rms_db > np.percentile(rms_db, 20)]
    if len(active) == 0:
        active = rms_db
    return {
        "min": round(float(np.min(rms_db)), 2),
        "max": round(float(np.max(rms_db)), 2),
        "median": round(float(np.median(rms_db)), 2),
        "active_range": round(float(np.percentile(active, 90) - np.percentile(active, 10)), 2),
    }


def _rank_loud_regions(times, rms_db, limit: int = 8) -> list[dict[str, Any]]:
    import numpy as np

    if len(rms_db) == 0:
        return []
    indices = np.argsort(rms_db)[::-1][:limit]
    return [_region(float(times[i]), float(rms_db[i]), "loudness_peak_db") for i in indices]


def _rank_quiet_active_regions(times, rms_db, limit: int = 8) -> list[dict[str, Any]]:
    import numpy as np

    if len(rms_db) == 0:
        return []
    active_floor = np.percentile(rms_db, 25)
    indices = [i for i in np.argsort(rms_db) if rms_db[i] > active_floor][:limit]
    return [_region(float(times[i]), float(rms_db[i]), "quiet_active_db") for i in indices]


def _rank_onset_dense_regions(onset_times, duration: float, window_sec: float = 5.0, limit: int = 8) -> list[dict[str, Any]]:
    import numpy as np

    if len(onset_times) == 0:
        return []
    starts = np.arange(0, max(duration, window_sec), window_sec)
    rows = []
    for start in starts:
        end = start + window_sec
        count = int(np.sum((onset_times >= start) & (onset_times < end)))
        rows.append({"start_sec": round(float(start), 3), "end_sec": round(float(min(end, duration)), 3), "score": count})
    rows.sort(key=lambda r: r["score"], reverse=True)
    return rows[:limit]


def _region(time_sec: float, score: float, label: str) -> dict[str, Any]:
    return {
        "start_sec": round(max(0.0, time_sec - 1.0), 3),
        "end_sec": round(time_sec + 1.0, 3),
        "label": label,
        "score": round(score, 3),
    }


def analysis_markdown(analysis: dict[str, Any]) -> str:
    lines = [
        f"# Analysis - {analysis.get('repertoire') or 'Untitled'}",
        "",
        f"- Session: `{analysis.get('session_id')}`",
        f"- Movement: `{analysis.get('movement')}`",
        f"- Instrument: `{analysis.get('instrument')}`",
        f"- Duration: `{analysis.get('duration_sec')}` sec",
        "",
        "## Global audio summary",
    ]
    for key, value in analysis.get("global", {}).items():
        lines.append(f"- {key}: `{value}`")
    lines.append("")
    lines.append("## Ranked regions")
    for name, rows in analysis.get("ranked_regions", {}).items():
        lines.append("")
        lines.append(f"### {name.replace('_', ' ').title()}")
        for row in rows:
            lines.append(f"- `{row.get('start_sec')}-{row.get('end_sec')}s`: {row.get('label', name)} score `{row.get('score')}`")
    pitch = analysis.get("performed_pitch_events", {})
    lines.append("")
    lines.append(f"## Performed pitch events: `{pitch.get('count')}`")
    for event in pitch.get("preview_first_40", [])[:12]:
        lines.append(f"- `{event['start_sec']}-{event['end_sec']}s` {event.get('note')} {event.get('median_cents_from_equal_temperament')}c ({event.get('confidence')})")
    return "\n".join(lines)


def build_evidence_packet(*, store: SessionStore, storage: ObjectStorage, manifest: SessionManifest) -> SessionManifest:
    analysis_key = manifest.artifacts.get("analysis/analysis.json")
    if not analysis_key:
        raise ValueError("analysis artifact missing; run analyze first")
    analysis = storage.read_json(analysis_key)
    lines = [
        f"# Evidence Packet - {manifest.repertoire or 'Untitled'}",
        "",
        "This packet is the factual basis for the teacher agent. Measurable claims should trace to these rows or a tool call.",
        "",
        "## Session",
        f"- Session id: `{manifest.session.session_id}`",
        f"- User/Tenant: `{manifest.user_id if hasattr(manifest, 'user_id') else manifest.session.user_id}` / `{manifest.session.tenant_id}`",
        f"- Repertoire: `{manifest.repertoire}`",
        f"- Movement: `{manifest.movement}`",
        f"- Instrument: `{manifest.instrument}`",
        "",
        storage.read_bytes(manifest.artifacts["analysis/analysis.md"]).decode("utf-8"),
    ]
    if "analysis/piano_voicing.json" in manifest.artifacts:
        voicing = storage.read_json(manifest.artifacts["analysis/piano_voicing.json"])
        global_summary = voicing.get("summary", {}).get("global", {})
        lines.extend([
            "",
            "## Piano voicing summary",
            f"- Median melody margin dB: `{global_summary.get('median_melody_margin_db')}`",
            f"- Weak/buried melody events: `{global_summary.get('buried_or_weak_melody_events')}`",
            f"- Pedal blur events: `{global_summary.get('pedal_blur_events')}`",
        ])
    key = store.artifact_key(manifest.session, "analysis/evidence_packet.md")
    storage.write_bytes(key, "\n".join(lines).encode("utf-8"), content_type="text/markdown")
    manifest.artifacts["analysis/evidence_packet.md"] = key
    manifest.state = JobState.AWAITING_LLM
    store.save(manifest)
    return manifest

