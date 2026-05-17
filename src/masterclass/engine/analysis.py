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


def _coerce_int(value: Any) -> int | None:
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


_PITCH_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def _midi_name(midi: int) -> str:
    return f"{_PITCH_NAMES[midi % 12]}{midi // 12 - 1}"


def _build_performance_timeline(
    matched_notes: list[Any],
    score_notes_in_range: list[dict[str, Any]],
    omr_empty_measures: set[int],
    played_range: Any,
) -> list[dict[str, Any]]:
    """One row per played measure: perf-time window + match counts + flags."""
    # Score-side density (notes-per-measure) for the played range only.
    score_density: dict[int, int] = {}
    for sn in score_notes_in_range:
        m = _coerce_int(sn.get("measure"))
        if m is None:
            continue
        score_density[m] = score_density.get(m, 0) + 1

    # Perf-side: per-measure first/last performed_time_sec + matched count.
    perf_window: dict[int, dict[str, float]] = {}
    for n in matched_notes:
        if not n.matched:
            continue
        m = _coerce_int(n.measure)
        if m is None or not played_range.contains(m):
            continue
        t = float(n.performed_time_sec or 0.0)
        win = perf_window.setdefault(m, {"first": t, "last": t, "count": 0})
        if t < win["first"]:
            win["first"] = t
        if t > win["last"]:
            win["last"] = t
        win["count"] += 1

    rows: list[dict[str, Any]] = []
    for m in played_range.measures():
        win = perf_window.get(m)
        sd = score_density.get(m, 0)
        flags: list[str] = []
        if m in omr_empty_measures:
            flags.append("OMR-gap (pitches unknown)")
        if win is None:
            rows.append({
                "measure": m,
                "starts": None,
                "ends": None,
                "matched": 0,
                "score_notes": sd,
                "flags": flags + (["no matches in this measure"] if not flags else []),
            })
            continue
        ratio = (win["count"] / sd) if sd else None
        if ratio is not None and ratio >= 1.6:
            flags.append(f"high extra-detection density ({ratio:.1f}× score)")
        rows.append({
            "measure": m,
            "starts": round(win["first"], 2),
            "ends": round(win["last"], 2),
            "matched": int(win["count"]),
            "score_notes": sd,
            "flags": flags,
        })
    return rows


def _format_perf_timeline_table(rows: list[dict[str, Any]]) -> list[str]:
    lines = [
        "| Measure | Starts at | Ends at | Notes matched | Score notes | Flags |",
        "|---|---|---|---|---|---|",
    ]
    for r in rows:
        starts = f"{r['starts']:.2f}s" if r["starts"] is not None else "—"
        ends = f"{r['ends']:.2f}s" if r["ends"] is not None else "—"
        score_notes_cell = "(OMR-empty)" if r["score_notes"] == 0 and "OMR-gap (pitches unknown)" in r["flags"] else str(r["score_notes"])
        flags = ", ".join(r["flags"]) or "—"
        lines.append(f"| m.{r['measure']} | {starts} | {ends} | {r['matched']} | {score_notes_cell} | {flags} |")
    return lines


def _collect_wrong_note_candidates(matched_notes: list[Any], played_range: Any, limit: int = 30) -> list[dict[str, Any]]:
    """Return notes the matcher accepted as exactly ±1 semitone off the score.

    These are the only rows worth flagging up-front as wrong-note candidates.
    Real intonation work needs ``inspect_intonation`` — see the anti-pattern
    note in the rendered packet.
    """
    rows: list[dict[str, Any]] = []
    for n in matched_notes:
        if not n.matched or n.score_midi_pitch is None or not n.pitches_midi:
            continue
        m = _coerce_int(n.measure)
        if m is None or not played_range.contains(m):
            continue
        detected = int(n.pitches_midi[0])
        score_pitch = int(n.score_midi_pitch)
        delta = detected - score_pitch
        if delta == 0 or abs(delta) > 1:
            continue
        rows.append({
            "perf_time": round(float(n.performed_time_sec or 0.0), 2),
            "measure": m,
            "detected": _midi_name(detected),
            "score": _midi_name(score_pitch),
            "delta_cents": delta * 100,
        })
    rows.sort(key=lambda r: r["perf_time"])
    return rows[:limit]


def _collect_suspicious_dense_regions(
    analysis: dict[str, Any],
    perf_envelope: tuple[float, float] | None,
    limit: int = 6,
) -> list[dict[str, Any]]:
    """Onset-dense windows, deduplicated and scoped to the perf envelope."""
    rows = (analysis.get("ranked_regions", {}) or {}).get("onset_dense_regions", []) or []
    out: list[dict[str, Any]] = []
    seen: set[tuple[int, int]] = set()
    for r in rows:
        start = r.get("start_sec")
        end = r.get("end_sec")
        score = r.get("score")
        if start is None or end is None or score is None:
            continue
        if perf_envelope is not None:
            lo, hi = perf_envelope
            if end < lo or start > hi:
                continue
        key = (int(start), int(end))  # collapse near-duplicates from sliding bins
        if key in seen:
            continue
        seen.add(key)
        out.append({"start": float(start), "end": float(end), "onset_count": int(score)})
        if len(out) >= limit:
            break
    return out


def _per_measure_score_pitches(
    score_notes: list[dict[str, Any]],
    played_range: Any,
    omr_empty_measures: set[int],
) -> dict[int, list[str]]:
    """Per-measure pitch outline, scoped to the played range. Wildcards excluded."""
    from masterclass.engine.audio_truth import _WILDCARD_PITCH
    by_measure: dict[int, list[tuple[float, str]]] = {}
    for n in score_notes:
        m = _coerce_int(n.get("measure"))
        if m is None or not played_range.contains(m):
            continue
        if int(n.get("midi_pitch", 0)) == _WILDCARD_PITCH or n.get("is_wildcard"):
            continue
        by_measure.setdefault(m, []).append((float(n["score_time_sec"]), _midi_name(int(n["midi_pitch"]))))
    out: dict[int, list[str]] = {}
    for m in played_range.measures():
        if m in omr_empty_measures:
            continue
        rows = sorted(by_measure.get(m, []))
        # De-dup consecutive identical names (chord groups can repeat the same pitch across voices).
        seen: list[str] = []
        for _t, nm in rows:
            if not seen or seen[-1] != nm:
                seen.append(nm)
        if seen:
            out[m] = seen
    return out


_TOOL_CATALOG_AND_ANTIPATTERNS = """## How to investigate further (tools + anti-patterns)

The packet above is a low-resolution MAP — it deliberately does NOT include per-note pitch/timing/loudness measurements because those are unreliable as static snapshots. To investigate any specific moment, USE THE TOOLS:

| Question | Tool |
|---|---|
| What does this moment sound like? | `listen(start_sec, end_sec)` |
| What does this moment look like (motion)? | `watch(start_sec, end_sec, "your question")` |
| What is the true pitch deviation at this note? | `inspect_intonation(time_sec, expected_midi_pitch)` |
| What notes are in m.X (and their measured timing)? | `inspect_bar(measure)` |
| One specific note's full record | `inspect_note(time_sec, midi_pitch)` |
| Voicing balance in a chord | `inspect_chord(time_sec)` |
| Local tempo over a window | `measure_tempo(start_sec, end_sec)` |
| Per-note peak loudness (voicing/dynamics) | `measure_dynamics(start_sec, end_sec)` |
| Trill rate / evenness | `measure_trill(start_sec, end_sec)` |
| One extra video still | `get_frames(start_sec, end_sec, fps)` |

**Critical anti-patterns** — these snapshot fields are present elsewhere in tool outputs but should NEVER ground a claim:

- **`cents_off_score` per note is meaningless.** Basic-pitch outputs integer MIDI, so the field is always 0 (perfect) or ±100 (wrong note). It cannot detect a 30-cent sharp violin note. For REAL intonation, ALWAYS call `inspect_intonation(time, expected_pitch)` — it reads the constant-Q transform and gives true cents-off-pitch.
- **`timing_offset_ms` per note is unreliable.** It's measured against a single global linear tempo model. Any rubato in the music makes the per-note value noise. For real tempo information, call `measure_tempo(start, end)` over the window of interest.
- **Do not cite a pitch not in the "Score pitches per played measure" outline above.** That outline IS the score for the played range. Wrong-note candidates are listed separately and need verification.
"""


def build_evidence_packet(*, store: SessionStore, storage: ObjectStorage, manifest: SessionManifest) -> SessionManifest:
    from masterclass.core.played_range import derive_played_range
    from masterclass.engine.aligned_notes import load_aligned_notes
    from masterclass.engine.audio_truth import _WILDCARD_PITCH, _load_score_notes_from_musicxml

    analysis_key = ArtifactCatalog(manifest).analysis_json()
    if not analysis_key:
        raise ValueError("analysis artifact missing; run analyze first")
    analysis = storage.read_json(analysis_key)

    played_range = derive_played_range(manifest, None)

    # ---- Sandbox-scope everything to the played range ONCE, here. ----
    matched_notes = load_aligned_notes(storage, manifest)
    matched_in_range = played_range.filter_by_measure(matched_notes, key="measure")
    # Also keep unmatched perf notes that have no measure (they're useful as
    # "dense burst" context but not for per-measure rows).
    matched_with_measure = [n for n in matched_notes if n.matched]

    sm_notes: list[dict[str, Any]] = []
    try:
        xml_key = None
        for k in (
            "masterclass/reference/musicxml.musicxml",
            "masterclass/reference/musicxml.mxl",
            "masterclass/reference/musicxml",
        ):
            if k in manifest.artifacts and storage.exists(manifest.artifacts[k]):
                xml_key = manifest.artifacts[k]
                break
        if xml_key:
            sm_notes = _load_score_notes_from_musicxml(storage.read_bytes(xml_key))
    except Exception:
        sm_notes = []
    score_notes_in_range = played_range.filter_by_measure(sm_notes, key="measure")

    # OMR-empty measures inside the played range only.
    omr_empty_measures: set[int] = set()
    for n in score_notes_in_range:
        m = _coerce_int(n.get("measure"))
        if m is None:
            continue
        if int(n.get("midi_pitch", 0)) == _WILDCARD_PITCH or n.get("is_wildcard"):
            omr_empty_measures.add(m)

    # Perf-time envelope of the played range (for scoping ranked-regions).
    perf_envelope: tuple[float, float] | None = None
    if matched_in_range:
        times = [float(n.performed_time_sec or 0.0) for n in matched_in_range if n.matched]
        if times:
            perf_envelope = (min(times), max(times))

    timeline_rows = _build_performance_timeline(matched_with_measure, score_notes_in_range, omr_empty_measures, played_range)
    wrong_note_candidates = _collect_wrong_note_candidates(matched_with_measure, played_range)
    dense_regions = _collect_suspicious_dense_regions(analysis, perf_envelope)
    pitch_outline = _per_measure_score_pitches(score_notes_in_range, played_range, omr_empty_measures)

    # ---- Render markdown ----
    lines: list[str] = [
        f"# Evidence packet — {manifest.repertoire or 'Untitled'}",
        f"  ({manifest.movement or '(movement unspecified)'}, {played_range.label()}, source: {played_range.source})",
        "",
        "## Lesson scope",
        f"- Piece: `{manifest.repertoire or 'Untitled'}` — {manifest.movement or '(movement unspecified)'}",
        f"- Played measures: **{played_range.label()}** (source: `{played_range.source}`)",
        f"- Audio duration: `{analysis.get('duration_sec')}` sec",
        f"- Session id: `{manifest.session.session_id}`",
    ]
    if manifest.instrument:
        lines.append(f"- Instrument: `{manifest.instrument}`")
    if omr_empty_measures:
        lines.append(
            f"- ⚠ OMR gaps in played range: {', '.join(f'm.{m}' for m in sorted(omr_empty_measures))} (pitches unknown — comment only on rhythm/timing for these)"
        )
    lines.append("")

    # Performance timeline
    lines += [
        "## Performance timeline (measures ↔ performed time)",
        "",
        "The matcher anchored each played measure to a perf-time window. Use these timestamps when calling tools (`watch`, `listen`, `inspect_*`) so you land on the right music. Boundaries can overlap slightly because chord voices and trills cross measure lines.",
        "",
    ]
    lines += _format_perf_timeline_table(timeline_rows)
    lines.append("")

    # Score pitches per played measure
    if pitch_outline or omr_empty_measures:
        lines += [
            "## Score pitches per played measure",
            "",
            "These are the ONLY pitches in the played range. **Do not cite a pitch that is not in this list** — it is not in the piece. Spell exactly as shown (e.g. use Bb in flat keys, F# in sharp keys). Pitches are listed in score-time order; consecutive duplicates are collapsed.",
            "",
        ]
        for m in played_range.measures():
            if m in omr_empty_measures:
                lines.append(f"- **m.{m}: (OMR-gap)** — pitches unknown; comment only on rhythm/timing, not pitch")
            elif m in pitch_outline:
                cells = pitch_outline[m][:32]
                more = f" (+{len(pitch_outline[m]) - 32} more)" if len(pitch_outline[m]) > 32 else ""
                lines.append(f"- m.{m}: {', '.join(cells)}{more}")
            else:
                lines.append(f"- m.{m}: (no notes extracted)")
        lines.append("")

    # Wrong-note candidates
    if wrong_note_candidates:
        lines += [
            f"## Wrong-note candidates ({len(wrong_note_candidates)})",
            "",
            "The matcher accepted these performed notes as exactly ±1 semitone off the score. They are CANDIDATES for wrong-note comments — confirm with `inspect_intonation(time, expected_pitch)` before claiming a wrong note (the apparent semitone difference may also be a chromatic neighbor, a string crossing artifact, or basic-pitch noise).",
            "",
        ]
        for r in wrong_note_candidates:
            sign = "+" if r["delta_cents"] >= 0 else "−"
            lines.append(
                f"- `{r['perf_time']:.2f}s` m.{r['measure']} played **{r['detected']}** (score expected **{r['score']}**) {sign}{abs(r['delta_cents'])}c"
            )
        lines.append("")

    # Suspicious dense regions
    if dense_regions:
        lines += [
            f"## Suspicious dense regions ({len(dense_regions)})",
            "",
            "Windows where detected onsets significantly exceed the score's note density. Often signals trills, ornaments, repeated patterns, or basic-pitch detection noise. Investigate with `watch(start, end, \"what's happening here?\")` or `measure_trill(start, end)`.",
            "",
        ]
        for r in dense_regions:
            lines.append(f"- `{r['start']:.1f}-{r['end']:.1f}s` — {r['onset_count']} onsets")
        lines.append("")

    # Optional: piano voicing summary (unchanged from old packet — still useful upfront)
    if "analysis/piano_voicing.json" in manifest.artifacts:
        voicing = storage.read_json(manifest.artifacts["analysis/piano_voicing.json"])
        global_summary = voicing.get("summary", {}).get("global", {})
        lines += [
            "## Piano voicing summary",
            "",
            f"- Median melody margin dB: `{global_summary.get('median_melody_margin_db')}`",
            f"- Weak/buried melody events: `{global_summary.get('buried_or_weak_melody_events')}`",
            f"- Pedal blur events: `{global_summary.get('pedal_blur_events')}`",
            "",
        ]

    lines.append(_TOOL_CATALOG_AND_ANTIPATTERNS)

    key = store.artifact_key(manifest.session, "analysis/evidence_packet.md")
    storage.write_bytes(key, "\n".join(lines).encode("utf-8"), content_type="text/markdown")
    manifest.artifacts["analysis/evidence_packet.md"] = key
    manifest.state = JobState.AWAITING_LLM
    store.save(manifest)
    return manifest

