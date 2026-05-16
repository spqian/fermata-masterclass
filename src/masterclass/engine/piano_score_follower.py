"""Piano/chromatic score follower using chroma DTW or bar-locked rhythm timing."""

from __future__ import annotations

import io
from dataclasses import asdict, dataclass
from typing import Any

from masterclass.core.models import SessionManifest
from masterclass.core.sessions import SessionStore
from masterclass.storage.base import ObjectStorage


@dataclass(frozen=True)
class PianoScoreFollowerConfig:
    sample_rate: int = 22050
    frame_dt: float = 0.05
    band_rad: float = 0.18
    loud_thresh_db: float = -32.0
    max_note_alignments: int = 3000
    # When True, skip the bar-locked rhythm path even if polyphonic_rhythm.json
    # exists, and always run real chroma DTW against the reference MIDI. The
    # bar-locked path is a faster approximation that snaps notes to score-grid
    # times under each bar; chroma DTW actually time-warps the score against
    # the audio chroma. The bar-locked path is the default for backward-compat
    # with the original pipeline.
    prefer_chroma_dtw: bool = False


@dataclass
class PianoScoreFollowerResult:
    summary: dict[str, Any]
    bar_starts: list[dict[str, Any]]
    note_alignments: list[dict[str, Any]]
    markdown: str
    config: dict[str, Any]


def follow_piano_score(
    *,
    storage: ObjectStorage,
    store: SessionStore,
    manifest: SessionManifest,
    midi_bytes: bytes | None = None,
    config: PianoScoreFollowerConfig | None = None,
    first_measure: int | None = None,
    last_measure: int | None = None,
) -> PianoScoreFollowerResult:
    """Run piano-oriented score following for dense/pedaled keyboard recordings."""

    del store
    config = config or PianoScoreFollowerConfig()
    audio_key = manifest.artifacts.get("artifacts/audio.wav")
    if not audio_key:
        raise ValueError("manifest is missing artifacts/audio.wav; run ingestion first")
    if midi_bytes is None:
        midi_key = manifest.artifacts.get("masterclass/reference/midi")
        if not midi_key:
            raise ValueError("manifest is missing masterclass/reference/midi")
        midi_bytes = storage.read_bytes(midi_key)

    import librosa
    import numpy as np
    import pretty_midi
    from scipy.spatial.distance import cdist

    pm = pretty_midi.PrettyMIDI(io.BytesIO(midi_bytes))
    notes = _load_midi_notes(pm)
    downbeats = list(map(float, pm.get_downbeats()))
    if not downbeats:
        raise RuntimeError("MIDI has no detectable measure structure")

    fm = int(first_measure or manifest.metadata.get("first_measure") or 1)
    lm = int(last_measure or manifest.metadata.get("last_measure") or len(downbeats))
    if fm < 1 or lm < fm or fm > len(downbeats):
        raise ValueError(f"invalid measure range: {fm}..{lm} for {len(downbeats)} measures")
    lm = min(lm, len(downbeats))
    score_start = downbeats[fm - 1]
    score_end = downbeats[lm] if lm < len(downbeats) else pm.get_end_time()

    hop = max(128, int(round(config.frame_dt * config.sample_rate)))
    y, _ = librosa.load(io.BytesIO(storage.read_bytes(audio_key)), sr=config.sample_rate, mono=True)
    rec_dur = float(len(y) / config.sample_rate)
    music_start = _detect_music_start(storage, manifest, loud_thresh_db=config.loud_thresh_db)
    y_active = y[int(music_start * config.sample_rate) :] if music_start > 0 else y

    rhythm_key = manifest.artifacts.get("analysis/polyphonic_rhythm.json")
    bar_time_map = None
    method = "piano_chroma_dtw_score_follower"
    use_bar_locked = bool(rhythm_key and storage.exists(rhythm_key)) and not config.prefer_chroma_dtw
    if use_bar_locked:
        bar_time_map = _rhythm_bar_starts(storage.read_json(rhythm_key), fm, lm)
        method = "piano_bar_locked_rhythm_follower"
        ref = np.zeros((12, 1), dtype=np.float32)
        obs = np.zeros((12, 1), dtype=np.float32)
        cost = np.zeros((1, 1), dtype=np.float32)
        ref_to_obs_time = np.asarray([music_start], dtype=np.float32)
    else:
        obs = librosa.feature.chroma_cqt(y=y_active, sr=config.sample_rate, hop_length=hop, bins_per_octave=36)
        obs = librosa.util.normalize(obs, axis=0)
        obs_times = librosa.frames_to_time(np.arange(obs.shape[1]), sr=config.sample_rate, hop_length=hop)
        ref, _ref_times = _midi_chroma(notes, score_start, score_end, config.frame_dt)
        cost = cdist(ref.T, obs.T, metric="cosine")
        cost = np.nan_to_num(cost, nan=1.0, posinf=1.0, neginf=1.0)
        _, wp = librosa.sequence.dtw(C=cost, backtrack=True, global_constraints=True, band_rad=config.band_rad)
        _, ref_to_obs_time = _interpolate_ref_to_obs(wp, ref.shape[1], obs_times, music_start)

    def map_score_time(score_time: float) -> float:
        if bar_time_map is not None:
            import bisect

            bar = max(fm, bisect.bisect_right(downbeats, score_time))
            bar = min(lm, bar)
            db0 = downbeats[bar - 1]
            db1 = downbeats[bar] if bar < len(downbeats) else score_end
            ratio = 0.0 if db1 <= db0 else (score_time - db0) / (db1 - db0)
            t0 = bar_time_map.get(bar, music_start)
            t1 = bar_time_map.get(bar + 1, rec_dur)
            return float(t0 + np.clip(ratio, 0.0, 1.0) * (t1 - t0))
        idx = np.clip((score_time - score_start) / config.frame_dt, 0, len(ref_to_obs_time) - 1)
        lo = int(np.floor(idx))
        hi = min(len(ref_to_obs_time) - 1, lo + 1)
        frac = idx - lo
        return float(ref_to_obs_time[lo] * (1.0 - frac) + ref_to_obs_time[hi] * frac)

    bar_starts = []
    for bar in range(fm, lm + 1):
        t_score = downbeats[bar - 1]
        bar_starts.append(
            {
                "measure": bar,
                "performed_time_sec": round(map_score_time(t_score), 3),
                "start": round(map_score_time(t_score), 3),
                "first_visited_state_idx": None,
                "first_visited_pitches": [],
                "is_score_bar_first_state": True,
                "bar_boundary_method": "bar_locked_rhythm" if bar_time_map is not None else "chroma_dtw",
            }
        )

    events = _score_events(notes, score_start, score_end)
    note_alignments = []
    for event in events:
        t = map_score_time(float(event["score_time_in_movement"]))
        frame = int(np.clip(round((float(event["score_time_in_movement"]) - score_start) / config.frame_dt), 0, ref.shape[1] - 1))
        if bar_time_map is not None:
            local_cost = 0.35
            confidence = "medium"
        else:
            obs_frame = int(np.clip(round((t - music_start) / config.frame_dt), 0, obs.shape[1] - 1))
            local_cost = float(cost[frame, obs_frame])
            confidence = "high" if local_cost < 0.25 else "medium" if local_cost < 0.45 else "low"
        note_alignments.append(
            {
                **event,
                "performed_time_sec": round(t, 3),
                "perf_time": round(t, 3),
                "dwell_frames": 1,
                "dwell_sec": round(config.frame_dt, 3),
                "obs_log_prob": round(1.0 - local_cost, 3),
                "confidence": confidence,
                "alignment_method": "bar_locked_rhythm" if bar_time_map is not None else "chroma_dtw",
            }
        )

    summary = {
        "method": method,
        "music_start_sec": round(music_start, 3),
        "played_measures": [fm, lm],
        "n_ref_frames": int(ref.shape[1]),
        "n_obs_frames": int(obs.shape[1]),
        "n_states": len(events),
        "states_visited": len(events),
        "note_states_visited": len(events),
        "total_note_states": len(events),
        "state_coverage": 1.0,
        "frame_dt_ms": round(config.frame_dt * 1000.0, 1),
        "method_notes": _method_notes(),
    }
    alignments = note_alignments[: config.max_note_alignments]
    return PianoScoreFollowerResult(
        summary=summary,
        bar_starts=bar_starts,
        note_alignments=alignments,
        markdown=_render_markdown(summary, bar_starts),
        config=asdict(config),
    )


def persist_piano_score_following(
    *,
    storage: ObjectStorage,
    store: SessionStore,
    manifest: SessionManifest,
    result: PianoScoreFollowerResult,
) -> None:
    json_key = store.artifact_key(manifest.session, "analysis/piano_score_following.json")
    md_key = store.artifact_key(manifest.session, "analysis/piano_score_following.md")
    storage.write_json(
        json_key,
        {
            "schema_version": 1,
            "summary": result.summary,
            "bar_starts": result.bar_starts,
            "note_alignments": result.note_alignments,
            "config": result.config,
        },
    )
    storage.write_bytes(md_key, result.markdown.encode("utf-8"), content_type="text/markdown")
    manifest.artifacts["analysis/piano_score_following.json"] = json_key
    manifest.artifacts["analysis/piano_score_following.md"] = md_key
    manifest.metadata["piano_score_following_state"] = "ready"
    manifest.metadata["piano_score_following_note_count"] = len(result.note_alignments)
    store.save(manifest)


def _detect_music_start(storage: ObjectStorage, manifest: SessionManifest, *, loud_thresh_db: float) -> float:
    for rel in ("analysis/pitch_events.json", "analysis/rich_onsets.json"):
        key = manifest.artifacts.get(rel)
        if not key or not storage.exists(key):
            continue
        data = storage.read_json(key)
        if isinstance(data, list):
            events = data
        elif isinstance(data, dict):
            events = data.get("events") or data.get("onsets") or []
        else:
            events = []
        for event in events:
            if event.get("confidence") == "low" or event.get("note") is None and event.get("note_estimate") is None:
                continue
            loud = event.get("median_loudness_db", event.get("loudness_db"))
            start = event.get("start_sec", event.get("time"))
            if loud is not None and start is not None and float(loud) >= loud_thresh_db:
                return float(start)
    return 0.0


def _load_midi_notes(pm) -> list[dict[str, Any]]:
    import pretty_midi

    notes = []
    for inst in pm.instruments:
        for note in inst.notes:
            notes.append(
                {
                    "start": float(note.start),
                    "end": float(note.end),
                    "pitch": int(note.pitch),
                    "velocity": int(note.velocity),
                    "name": pretty_midi.note_number_to_name(note.pitch),
                }
            )
    notes.sort(key=lambda n: (float(n["start"]), -int(n["pitch"])))
    return notes


def _midi_chroma(notes: list[dict[str, Any]], start_sec: float, end_sec: float, frame_dt: float):
    import librosa
    import numpy as np

    n_frames = max(2, int(np.ceil((end_sec - start_sec) / frame_dt)) + 1)
    times = start_sec + np.arange(n_frames) * frame_dt
    chroma = np.zeros((12, n_frames), dtype=np.float32)
    for note in notes:
        if note["end"] < start_sec or note["start"] > end_sec:
            continue
        pc = int(note["pitch"]) % 12
        lo = max(0, int(np.floor((note["start"] - start_sec) / frame_dt)))
        hi = min(n_frames, int(np.ceil((note["end"] - start_sec) / frame_dt)) + 1)
        if hi <= lo:
            continue
        weight = max(1.0, float(note["velocity"]) / 64.0)
        if int(note["pitch"]) < 60:
            weight *= 1.25
        chroma[pc, lo:hi] += weight
    chroma = librosa.util.normalize(chroma, axis=0)
    return chroma, times


def _score_events(notes: list[dict[str, Any]], score_start: float, score_end: float) -> list[dict[str, Any]]:
    by_start: dict[float, list[dict[str, Any]]] = {}
    for note in notes:
        if score_start - 0.001 <= float(note["start"]) < score_end - 0.001:
            by_start.setdefault(round(float(note["start"]), 4), []).append(note)
    events = []
    for idx, start in enumerate(sorted(by_start)):
        group = by_start[start]
        events.append(
            {
                "state_idx": idx,
                "score_time_in_movement": start,
                "pitches_midi": sorted({int(note["pitch"]) for note in group}),
                "names": sorted({str(note["name"]) for note in group}),
            }
        )
    return events


def _rhythm_bar_starts(data: dict[str, Any], first_measure: int, last_measure: int) -> dict[int, float]:
    import numpy as np

    summary = data.get("summary", {})
    t = float(summary.get("music_start_sec", 0.0))
    rows = summary.get("bar_durations", [])
    durations = {int(row["bar"]): float(row.get("duration_sec", 0.0)) for row in rows if "bar" in row}
    positive = [duration for duration in durations.values() if duration > 0.2]
    fallback = float(np.median(positive)) if positive else 2.5
    starts = {}
    for bar in range(first_measure, last_measure + 2):
        starts[bar] = t
        duration = durations.get(bar, fallback)
        if duration <= 0.2:
            duration = min(fallback, 1.5)
        t += duration
    return starts


def _interpolate_ref_to_obs(wp, n_ref: int, obs_times, music_start: float):
    import numpy as np

    pairs = sorted((int(r), int(o)) for r, o in wp)
    buckets: list[list[int]] = [[] for _ in range(n_ref)]
    for ref_idx, obs_idx in pairs:
        if 0 <= ref_idx < n_ref:
            buckets[ref_idx].append(obs_idx)
    ref_indices = []
    obs_t = []
    for ref_idx, obs_indices in enumerate(buckets):
        if obs_indices:
            ref_indices.append(ref_idx)
            obs_t.append(float(obs_times[int(np.median(obs_indices))] + music_start))
    if not ref_indices:
        raise RuntimeError("DTW produced no usable reference-to-observation mapping")
    all_ref = np.arange(n_ref)
    mapped = np.interp(all_ref, np.asarray(ref_indices), np.asarray(obs_t))
    return all_ref, mapped


def _method_notes() -> list[str]:
    return [
        "Piano score following is bar-first: dense pedaled chromatic texture makes note-state coverage a poor primary signal.",
        "When polyphonic_rhythm.json is available, bar starts come from onset/rhythm alignment and score events are interpolated inside each bar.",
        "Chroma-DTW remains a fallback for coarse score location, but unconstrained chroma can confuse repeated accompaniment patterns.",
        "This is location tracking, not proof that every MIDI note sounded.",
        "Use this for score-panel synchronization and phrase/bar timing; use separate voicing/chord tools for performance critique.",
    ]


def _render_markdown(summary: dict[str, Any], bar_starts: list[dict[str, Any]]) -> str:
    lines = [
        "# Piano Score Follower",
        "",
        f"- Played measures: bar `{summary['played_measures'][0]}` to bar `{summary['played_measures'][1]}`",
        f"- Reference frames: `{summary['n_ref_frames']}`; observed frames: `{summary['n_obs_frames']}`",
        f"- Frame step: `{summary['frame_dt_ms']:.1f}` ms",
        "",
        "## Per-bar start times",
        "",
        "| Bar | Performed time | Method |",
        "| --- | --- | --- |",
    ]
    for bar in bar_starts:
        lines.append(f"| {bar['measure']} | {bar['performed_time_sec']:.2f}s | {bar.get('bar_boundary_method')} |")
    lines.extend(["", "## Method"])
    lines.extend(f"- {note}" for note in summary["method_notes"])
    return "\n".join(lines) + "\n"
