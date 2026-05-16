from __future__ import annotations

import json
import math
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from masterclass.core.artifact_catalog import ArtifactCatalog
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
    catalog = ArtifactCatalog(manifest)
    audio_key = catalog.audio_wav()
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
            "audio_wav": ArtifactCatalog(manifest).audio_wav(),
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
    """Render the global-audio summary as markdown for the analysis.md artifact.

    Used to also include a 'Performed pitch events' section sourced from
    librosa.pyin, but that data is no longer relied on for teaching - the
    score-matched audio-truth notes in audio_truth_matched_notes.json are
    a better source. See build_evidence_packet().
    """
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
    return "\n".join(lines)


def _audio_truth_pitch_rows(storage: ObjectStorage, manifest: SessionManifest, limit: int | None = 40) -> list[dict[str, Any]]:
    """Return per-note rows derived from the lesson's aligned-notes timeline
    suitable for embedding in the evidence packet markdown.

    Each row carries the score-expected pitch, the detected pitch, the
    cents-from-score deviation, the measure/beat, and the match status.
    These are the same notes the technical viewer shows; using them here
    is what closes the "teacher reasons over CREPE while user reasons over
    audio-truth" inconsistency that drove the 'is the C sharp' hallucination.
    """
    from masterclass.engine.aligned_notes import load_aligned_notes
    notes = load_aligned_notes(storage, manifest)
    rows: list[dict[str, Any]] = []
    NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    for n in notes:
        if not n.pitches_midi:
            continue
        detected = int(n.pitches_midi[0])
        score_pitch = n.score_midi_pitch
        cents_off_score = None
        if score_pitch is not None:
            # If detected and matched score pitch differ by a semitone we report
            # the literal semitone offset times 100c; the teacher needs to see
            # WRONG-NOTE mismatches as clearly as in-tune-but-imprecise.
            cents_off_score = round((detected - int(score_pitch)) * 100.0, 1)
        det_name = f"{NAMES[detected % 12]}{detected // 12 - 1}"
        duration = n.dwell_sec if n.dwell_sec else (n.expected_perf_duration or 0.0)
        rows.append({
            "start_sec": round(float(n.performed_time_sec), 3),
            "duration_sec": round(float(duration), 3),
            "detected_note": det_name,
            "detected_midi": detected,
            "score_note": f"{NAMES[int(score_pitch) % 12]}{int(score_pitch) // 12 - 1}" if score_pitch is not None else None,
            "cents_off_score": cents_off_score,
            "measure": n.measure,
            "staff": n.staff_index,
            "matched": bool(n.matched),
            "confidence": n.confidence,
            "timing_offset_ms": n.timing_offset_ms,
        })
    rows.sort(key=lambda r: r["start_sec"])
    if limit is not None:
        return rows[:limit]
    return rows


def build_evidence_packet(*, store: SessionStore, storage: ObjectStorage, manifest: SessionManifest) -> SessionManifest:
    analysis_key = ArtifactCatalog(manifest).analysis_json()
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
    # Per-note audio-truth rows. We deliberately do NOT include the older
    # CREPE pitch_events.json here -- it produced confidence-laden monophonic
    # blobs that the teacher misread as intonation evidence. The audio-truth
    # rows below are score-anchored: each row tells the teacher exactly what
    # note was played, what the score expected, and the deviation.
    at_rows = _audio_truth_pitch_rows(storage, manifest, limit=80)
    # Scope rows to the player's played range: rows whose ``measure`` falls
    # outside the [first..last] window are matcher noise (snapped to a
    # nearby score event that the player never actually approached) and
    # they routinely mislead the teacher agent into commenting on measures
    # the user never played.
    from masterclass.core.played_range import derive_played_range
    played_range = derive_played_range(manifest, None)
    if at_rows:
        in_range_rows: list[dict[str, Any]] = []
        out_of_range = 0
        for r in at_rows:
            m = r.get("measure")
            if m is None or played_range.contains(m):
                in_range_rows.append(r)
            else:
                out_of_range += 1
        at_rows = in_range_rows
        header = (
            f"## Audio-truth notes (first {len(at_rows)}, score-matched, "
            f"played range: m.{played_range.first_measure}-{played_range.last_measure})"
        )
        played_note = (
            f"All rows below are scoped to the played range "
            f"m.{played_range.first_measure}-{played_range.last_measure} "
            f"(source: {played_range.source}). "
        )
        if out_of_range:
            played_note += (
                f"Dropped {out_of_range} row(s) whose detected measure fell outside this range. "
            )
        lines.extend([
            "",
            header,
            "",
            played_note
            + "Each row is one detected note matched against the reference score. "
            + "`cents_off_score` is the literal MIDI-semitone difference between the detected pitch and the score pitch "
            + "(so +100 = a full semitone sharp, often a wrong-note mistake; +35 = a sharp intonation reading; ±10 ≈ in tune). "
            + "`timing_offset_ms` is detected-onset minus score-time (positive = late).",
            "",
        ])
        for r in at_rows:
            score_part = f" vs score {r['score_note']}" if r['score_note'] else " (unmatched)"
            cents_part = ""
            if r["cents_off_score"] is not None:
                cents_part = f" cents_off_score={r['cents_off_score']:+.1f}c"
            timing_part = ""
            if r["timing_offset_ms"] is not None:
                timing_part = f" timing={r['timing_offset_ms']:+.0f}ms"
            measure_part = f" m.{r['measure']}" if r['measure'] is not None else ""
            staff_part = f" staff={r['staff']}" if r['staff'] is not None else ""
            lines.append(
                f"- `{r['start_sec']:.3f}s` {r['detected_note']}{score_part}{cents_part}{timing_part}{measure_part}{staff_part} dur={r['duration_sec']:.2f}s conf={r['confidence']}"
            )
    # Per-measure score outline. Without this the teacher hallucinates
    # pitches that aren't in the piece (e.g. "the G# in measure 7" when
    # measure 7 has only G-natural). Read MusicXML directly because
    # evidence_packet runs before score_map in the pipeline.
    try:
        from masterclass.engine.audio_truth import _load_score_notes_from_musicxml
        xml_key = None
        for k in ("masterclass/reference/musicxml.musicxml", "masterclass/reference/musicxml.mxl", "masterclass/reference/musicxml"):
            if k in manifest.artifacts and storage.exists(manifest.artifacts[k]):
                xml_key = manifest.artifacts[k]
                break
        sm_notes = _load_score_notes_from_musicxml(storage.read_bytes(xml_key)) if xml_key else []
    except Exception:
        sm_notes = []
    if sm_notes:
        import pretty_midi as _pm
        by_measure: dict[int, list[tuple[float, str]]] = {}
        for n in sm_notes:
            m = n.get("measure")
            if not m:
                continue
            name = _pm.note_number_to_name(int(n["midi_pitch"]))
            by_measure.setdefault(int(m), []).append((float(n["score_time_sec"]), name))
        if by_measure:
            lines.extend([
                "",
                "## Score pitches per measure (authoritative)",
                "",
                "This list IS the score. If a measure does NOT contain a pitch you want to discuss, that pitch is NOT in the score — do not claim it is. Cross-check every named pitch (especially accidentals like G#, C#, F#) against this list before writing a comment. Pitches are listed in score-time order.",
                "",
            ])
            for m in sorted(by_measure):
                rows = sorted(by_measure[m])
                # de-dup consecutive identical names (chord groups can repeat)
                seen: list[str] = []
                for _t, nm in rows:
                    if not seen or seen[-1] != nm:
                        seen.append(nm)
                cells = seen[:32]
                more = f" (+{len(seen) - 32} more)" if len(seen) > 32 else ""
                lines.append(f"- m.{m}: {', '.join(cells)}{more}")
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

