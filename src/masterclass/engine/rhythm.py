"""Score-aware polyphonic rhythm analysis from HMM-aligned score notes."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from masterclass.core.artifact_catalog import ArtifactCatalog
from masterclass.core.models import SessionManifest
from masterclass.core.sessions import SessionStore
from masterclass.storage.base import ObjectStorage


PC_NAME_TO_INT = {name: i for i, name in enumerate(["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"])}


@dataclass(frozen=True)
class RhythmConfig:
    """Settings for score-aware rhythm analysis."""

    midi_quarter_bpm: float = 120.0
    local_tempo_half_window: int = 4
    min_local_tempo_points: int = 3
    onset_match_window_ms: float = 300.0
    off_pulse_threshold_ms: float = 80.0
    music_start_loud_thresh_db: float = -32.0
    music_start_sustain_count: int = 3
    music_start_sustain_drop_db: float = 6.0
    music_start_max_delta_from_alignment_sec: float = 5.0
    bar_snap_window_sec: float = 4.0
    bar_snap_min_loud_db: float = -30.0
    bar_snap_first_notes: int = 3
    bar_snap_max_safe_delta_sec: float = 1.5
    bar_snap_strong_loud_db: float = -15.0
    max_note_rows: int = 200
    max_beat_arrivals: int = 200


@dataclass
class RhythmResult:
    per_bar: list[dict[str, Any]]
    summary: dict[str, Any]
    note_rows: list[dict[str, Any]]
    local_tempos: list[dict[str, Any]]
    beat_arrivals: list[dict[str, Any]]
    bar_refinement_report: list[dict[str, Any]]
    markdown: str
    config: dict[str, Any]


def analyze_rhythm(
    *,
    storage: ObjectStorage,
    store: SessionStore,
    manifest: SessionManifest,
    config: RhythmConfig | None = None,
) -> RhythmResult:
    """Analyze polyphonic rhythm from storage-scoped HMM alignment artifacts."""

    del store
    config = config or RhythmConfig()
    _catalog = ArtifactCatalog(manifest)

    # Read everything we need through the unified aligned-notes accessor
    # (audio_truth_matched_notes.json preferred, with hmm shim as fallback)
    # and derive bar starts directly from matched notes' measure numbers.
    # The historical hmm_alignment.json bar_starts had loudness/expected_t
    # fields the old rhythm refinement read; we don't have those equivalents
    # in the audio-truth output yet, so any code path that consumed them
    # will silently fall back to its first_visited_pitches-empty default.
    from masterclass.engine.aligned_notes import load_aligned_notes, load_measure_starts
    raw_aligned = load_aligned_notes(storage, manifest)
    aligned_notes = _normalized_notes([n.to_dict() for n in raw_aligned])
    if len(aligned_notes) < 2:
        raise RuntimeError("no aligned notes available for rhythm analysis (audio_truth pipeline must run first)")

    measure_rows = load_measure_starts(storage, manifest)
    measures = _normalized_measures(measure_rows)
    if not measures:
        raise RuntimeError("no measure starts derivable from aligned notes")
    # Synthesise the minimal `alignment` shape downstream helpers expect.
    # bar_starts mirror measures; summary holds played_measures + music_start
    # so existing helpers that .get() those still work.
    played = sorted({int(m["measure"]) for m in measures})
    alignment = {
        "bar_starts": measure_rows,
        "measure_timestamps": measure_rows,
        "summary": {
            "method": "audio_truth_first_note",
            "music_start_sec": float(measures[0]["start"]),
            "played_measures": played,
            "effective_first_measure": played[0],
            "effective_last_measure": played[-1],
        },
    }
    rich_onsets = _load_rich_onsets(storage, manifest)
    music_start_sec = _music_start_sec(rich_onsets, alignment, measures, config)

    bar_score_starts = _bar_score_starts(measures, alignment.get("bar_starts") or [], aligned_notes)
    note_by_state = {int(n["state_idx"]): n for n in aligned_notes if n.get("state_idx") is not None}
    score_start = min(bar_score_starts.values()) if bar_score_starts else float(aligned_notes[0]["score_time"])
    score_end = _estimated_score_end(bar_score_starts, aligned_notes)

    score_times = _unique_score_perf_pairs(aligned_notes)
    local_tempos = _local_tempos(score_times, bar_score_starts, config)
    notes_by_bar = _notes_by_bar(aligned_notes, bar_score_starts)
    note_rows, by_bar_onset_ms = _note_onset_rows(
        aligned_notes,
        bar_score_starts,
        rich_onsets,
        config,
    )

    initial_bar_starts = {int(m["measure"]): float(m["start"]) for m in measures}
    refined_bar_starts, refinement_status = _refine_bar_starts_with_onsets(
        initial_bar_starts,
        notes_by_bar,
        rich_onsets,
        config,
    )
    bar_refinement_report = _bar_refinement_report(initial_bar_starts, refined_bar_starts, refinement_status)
    bar_durations = _bar_durations(refined_bar_starts, alignment)

    by_bar_tempo: dict[int, list[float]] = {}
    for row in local_tempos:
        if row.get("local_quarter_bpm") is not None:
            by_bar_tempo.setdefault(int(row["measure"]), []).append(float(row["local_quarter_bpm"]))

    per_bar = _per_bar_summary(
        measures=sorted(refined_bar_starts),
        refined_bar_starts=refined_bar_starts,
        bar_durations=bar_durations,
        by_bar_tempo=by_bar_tempo,
        by_bar_onset_ms=by_bar_onset_ms,
    )
    beat_arrivals = _beat_arrivals(
        score_times,
        bar_score_starts,
        score_start,
        score_end,
        per_bar,
        config,
    )
    outliers = _off_pulse_outliers(note_rows, by_bar_onset_ms, config)

    import numpy as np

    tempo_values = [float(r["local_quarter_bpm"]) for r in local_tempos if r.get("local_quarter_bpm") is not None]
    duration_values = [float(b["duration_sec"]) for b in per_bar if b.get("duration_sec") is not None and float(b["duration_sec"]) > 0]
    note_with_onsets = sum(1 for n in note_rows if n.get("onset_within_window"))
    summary = {
        "session_id": manifest.session.session_id,
        "repertoire": manifest.repertoire,
        "movement": manifest.movement,
        "method": "score_aware_polyphonic_rhythm_hmm",
        "source_alignment": _catalog.audio_truth_matched(),
        "source_notes": _catalog.audio_truth_matched() or _catalog.audio_truth_raw(),
        "source_rich_onsets": _catalog.rich_onsets() if rich_onsets else None,
        "music_start_sec": round(float(music_start_sec), 3),
        "midi_quarter_bpm_render": round(float(config.midi_quarter_bpm), 2),
        "played_measures": (alignment.get("summary") or {}).get("played_measures")
        or [min(initial_bar_starts), max(initial_bar_starts)],
        "total_score_notes_in_window": len(aligned_notes),
        "score_notes_with_nearby_onset_within_window": note_with_onsets,
        "onset_alignment_rate": round(note_with_onsets / max(1, len(note_rows)), 3),
        "overall_player_quarter_bpm_median": round(float(np.median(tempo_values)), 2) if tempo_values else None,
        "bar_duration_median_sec": round(float(np.median(duration_values)), 3) if duration_values else None,
        "bar_duration_pct_range": round(float((max(duration_values) - min(duration_values)) / np.median(duration_values) * 100.0), 1)
        if len(duration_values) >= 2 and float(np.median(duration_values)) > 0
        else None,
        "bar_count": len(per_bar),
        "beat_arrivals_count": len(beat_arrivals),
        "off_pulse_outliers": outliers,
        "alignment_summary": alignment.get("summary") or {},
        "method_notes": [
            "Inputs are storage-scoped HMM score-note alignments and HMM measure timestamps.",
            "Local quarter-BPM = score-seconds/performance-seconds slope over a note window multiplied by RhythmConfig.midi_quarter_bpm.",
            "Per-bar tempo is the median of local-tempo samples assigned by inferred score bar boundaries.",
            "Bar duration comes from HMM measure timestamps, optionally snapped to nearby strong rich onsets that match early score-note pitch classes.",
            "Onset placement compares each expected HMM note time to the nearest rich onset within the configured tolerance.",
            f"Off-pulse outliers differ from their bar's median onset placement by >= {config.off_pulse_threshold_ms:.0f} ms.",
        ],
    }
    markdown = _render_markdown(summary, per_bar, outliers)
    return RhythmResult(
        per_bar=per_bar,
        summary=summary,
        note_rows=note_rows[: config.max_note_rows],
        local_tempos=local_tempos,
        beat_arrivals=beat_arrivals[: config.max_beat_arrivals],
        bar_refinement_report=bar_refinement_report,
        markdown=markdown,
        config=asdict(config),
    )


def persist_rhythm(
    *,
    storage: ObjectStorage,
    store: SessionStore,
    manifest: SessionManifest,
    result: RhythmResult,
) -> None:
    """Persist rhythm JSON/Markdown artifacts and stamp the session manifest."""

    json_key = store.artifact_key(manifest.session, "analysis/polyphonic_rhythm.json")
    md_key = store.artifact_key(manifest.session, "analysis/polyphonic_rhythm.md")
    storage.write_json(
        json_key,
        {
            "schema_version": 1,
            "summary": result.summary,
            "per_bar": result.per_bar,
            "note_rows": result.note_rows,
            "local_tempos": result.local_tempos,
            "beat_arrivals": result.beat_arrivals,
            "bar_refinement_report": result.bar_refinement_report,
            "config": result.config,
        },
    )
    storage.write_bytes(md_key, result.markdown.encode("utf-8"), content_type="text/markdown")
    manifest.artifacts["analysis/polyphonic_rhythm.json"] = json_key
    manifest.artifacts["analysis/polyphonic_rhythm.md"] = md_key
    manifest.metadata["polyphonic_rhythm_summary"] = {
        "overall_player_quarter_bpm_median": result.summary.get("overall_player_quarter_bpm_median"),
        "bar_duration_median_sec": result.summary.get("bar_duration_median_sec"),
        "bar_count": result.summary.get("bar_count"),
        "music_start_sec": result.summary.get("music_start_sec"),
    }
    store.save(manifest)


def _normalized_notes(raw: list[dict[str, Any]]) -> list[dict[str, Any]]:
    notes: list[dict[str, Any]] = []
    for row in raw:
        score_t = _as_float(row.get("score_time_in_movement", row.get("score_time_local", row.get("score_time_sec"))), default=None)
        perf_t = _as_float(row.get("performed_time_sec", row.get("perf_time")), default=None)
        if score_t is None or perf_t is None:
            continue
        pitches = row.get("pitches_midi") or row.get("pitches") or []
        names = row.get("names") or row.get("expected_pitch") or []
        if isinstance(names, str):
            names = [names]
        notes.append(
            {
                **row,
                "score_time": float(score_t),
                "perf_time": float(perf_t),
                "pitches_midi": [int(p) for p in pitches if p is not None],
                "names": list(names),
            }
        )
    notes.sort(key=lambda n: (float(n["score_time"]), float(n["perf_time"])))
    return notes


def _normalized_measures(raw: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in raw:
        measure = row.get("measure")
        start = row.get("start", row.get("performed_time_sec"))
        if measure is None or start is None:
            continue
        out.append({"measure": int(measure), "start": float(start)})
    out.sort(key=lambda m: int(m["measure"]))
    return out


def _load_rich_onsets(storage: ObjectStorage, manifest: SessionManifest) -> list[dict[str, Any]]:
    key = ArtifactCatalog(manifest).rich_onsets()
    if not key or not storage.exists(key):
        return []
    doc = storage.read_json(key)
    events = doc.get("onsets") or doc.get("events") or []
    return [e for e in events if isinstance(e, dict)]


def _bar_score_starts(
    measures: list[dict[str, Any]],
    bar_starts: list[dict[str, Any]],
    notes: list[dict[str, Any]],
) -> dict[int, float]:
    by_state = {int(n["state_idx"]): float(n["score_time"]) for n in notes if n.get("state_idx") is not None}
    starts: dict[int, float] = {}
    for row in bar_starts:
        measure = row.get("measure")
        state_idx = row.get("first_visited_state_idx")
        if measure is not None and state_idx is not None and int(state_idx) in by_state:
            starts[int(measure)] = by_state[int(state_idx)]
    for row in measures:
        measure = int(row["measure"])
        if measure in starts:
            continue
        perf_start = float(row["start"])
        candidate = min(notes, key=lambda n: abs(float(n["perf_time"]) - perf_start))
        starts[measure] = float(candidate["score_time"])
    return dict(sorted(starts.items()))


def _estimated_score_end(bar_score_starts: dict[int, float], notes: list[dict[str, Any]]) -> float:
    if len(bar_score_starts) >= 2:
        ordered = sorted(bar_score_starts.items())
        score_durs = [ordered[i + 1][1] - ordered[i][1] for i in range(len(ordered) - 1) if ordered[i + 1][1] > ordered[i][1]]
        if score_durs:
            import numpy as np

            return float(ordered[-1][1] + float(np.median(score_durs)))
    return max(float(n["score_time"]) for n in notes)


def _unique_score_perf_pairs(notes: list[dict[str, Any]]) -> list[tuple[float, float]]:
    by_score: dict[float, list[float]] = {}
    for note in notes:
        by_score.setdefault(round(float(note["score_time"]), 6), []).append(float(note["perf_time"]))
    import numpy as np

    return [(score, float(np.median(times))) for score, times in sorted(by_score.items())]


def _measure_for_score(score_t: float, bar_score_starts: dict[int, float]) -> int:
    ordered = sorted(bar_score_starts.items(), key=lambda kv: kv[1])
    measure = ordered[0][0]
    for m, start in ordered:
        if score_t + 1e-9 >= float(start):
            measure = int(m)
        else:
            break
    return measure


def _local_tempos(
    pairs: list[tuple[float, float]],
    bar_score_starts: dict[int, float],
    config: RhythmConfig,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for idx, (score_t, perf_t) in enumerate(pairs):
        lo = max(0, idx - config.local_tempo_half_window)
        hi = min(len(pairs) - 1, idx + config.local_tempo_half_window)
        if hi - lo + 1 < config.min_local_tempo_points:
            continue
        score_span = float(pairs[hi][0] - pairs[lo][0])
        perf_span = float(pairs[hi][1] - pairs[lo][1])
        if score_span <= 1e-9 or perf_span <= 1e-9:
            continue
        bpm = (score_span / perf_span) * float(config.midi_quarter_bpm)
        rows.append(
            {
                "score_time_in_movement": round(float(score_t), 3),
                "performed_time": round(float(perf_t), 3),
                "measure": _measure_for_score(float(score_t), bar_score_starts),
                "local_quarter_bpm": round(float(bpm), 2),
            }
        )
    return rows


def _notes_by_bar(notes: list[dict[str, Any]], bar_score_starts: dict[int, float]) -> dict[int, list[dict[str, Any]]]:
    by_bar: dict[int, list[dict[str, Any]]] = {}
    for note in notes:
        by_bar.setdefault(_measure_for_score(float(note["score_time"]), bar_score_starts), []).append(note)
    for rows in by_bar.values():
        rows.sort(key=lambda n: (float(n["score_time"]), float(n["perf_time"])))
    return by_bar


def _note_onset_rows(
    notes: list[dict[str, Any]],
    bar_score_starts: dict[int, float],
    rich_onsets: list[dict[str, Any]],
    config: RhythmConfig,
) -> tuple[list[dict[str, Any]], dict[int, list[float]]]:
    import numpy as np

    onset_times = np.array(sorted({_event_time(e) for e in rich_onsets if _event_time(e) is not None}), dtype=float)
    by_bar_ms: dict[int, list[float]] = {}
    rows: list[dict[str, Any]] = []
    for note in notes:
        perf_t = float(note["perf_time"])
        if len(onset_times):
            idx = int(np.argmin(np.abs(onset_times - perf_t)))
            nearest = float(onset_times[idx])
            delta_ms = (nearest - perf_t) * 1000.0
            within = abs(delta_ms) <= float(config.onset_match_window_ms)
        else:
            nearest = None
            delta_ms = None
            within = False
        measure = _measure_for_score(float(note["score_time"]), bar_score_starts)
        if delta_ms is not None and within:
            by_bar_ms.setdefault(measure, []).append(float(delta_ms))
        rows.append(
            {
                "expected_pitch": ",".join(str(n) for n in note.get("names", [])),
                "score_time_in_movement": round(float(note["score_time"]), 3),
                "measure": measure,
                "expected_performed_time": round(perf_t, 3),
                "nearest_detected_onset": round(float(nearest), 3) if nearest is not None else None,
                "onset_delta_ms": round(float(delta_ms), 1) if delta_ms is not None else None,
                "onset_within_window": bool(within),
            }
        )
    return rows, by_bar_ms


def _detect_music_start(events: list[dict[str, Any]], config: RhythmConfig) -> float | None:
    for i, event in enumerate(events):
        loud = _event_loudness(event)
        t = _event_time(event)
        if loud is None or t is None:
            continue
        if event.get("confidence") == "low":
            continue
        if loud < config.music_start_loud_thresh_db:
            continue
        sustained = 0
        for follow in events[i : i + config.music_start_sustain_count + 4]:
            f_loud = _event_loudness(follow)
            if f_loud is not None and f_loud > config.music_start_loud_thresh_db - config.music_start_sustain_drop_db:
                sustained += 1
        if sustained >= config.music_start_sustain_count:
            return float(t)
    return None


def _music_start_sec(
    events: list[dict[str, Any]],
    alignment: dict[str, Any],
    measures: list[dict[str, Any]],
    config: RhythmConfig,
) -> float:
    hmm_start = _as_float((alignment.get("summary") or {}).get("music_start_sec"), default=None)
    first_bar_start = float(measures[0]["start"]) if measures else hmm_start
    rich_start = _detect_music_start(events, config)
    if rich_start is not None and first_bar_start is not None:
        if abs(float(rich_start) - float(first_bar_start)) <= config.music_start_max_delta_from_alignment_sec:
            return float(rich_start)
    if hmm_start is not None:
        return float(hmm_start)
    if first_bar_start is not None:
        return float(first_bar_start)
    return float(rich_start or 0.0)


def _refine_bar_starts_with_onsets(
    initial_starts: dict[int, float],
    notes_by_bar: dict[int, list[dict[str, Any]]],
    events: list[dict[str, Any]],
    config: RhythmConfig,
) -> tuple[dict[int, float], dict[int, str]]:
    if not events:
        return dict(initial_starts), {bar: "no_rich_onsets" for bar in initial_starts}

    import numpy as np

    refined: dict[int, float] = {}
    status: dict[int, str] = {}
    prev_t = -1.0e9
    for bar in sorted(initial_starts):
        predicted = float(initial_starts[bar])
        expected_pcs = set()
        for note in notes_by_bar.get(bar, [])[: config.bar_snap_first_notes]:
            for pitch in note.get("pitches_midi", []):
                expected_pcs.add(int(pitch) % 12)
        if not expected_pcs:
            refined[bar] = predicted
            status[bar] = "no_score"
            prev_t = predicted
            continue

        candidates: list[tuple[float, float, float]] = []
        for event in events:
            t = _event_time(event)
            loud = _event_loudness(event)
            pc = _event_pc(event)
            if t is None or loud is None or pc is None:
                continue
            if t <= prev_t + 0.20:
                continue
            if abs(t - predicted) > config.bar_snap_window_sec:
                continue
            if pc not in expected_pcs or loud < config.bar_snap_min_loud_db:
                continue
            strength = _as_float(event.get("onset_strength"), default=0.0) or 0.0
            score = float(loud) + float(strength) - abs(t - predicted) * 6.0
            candidates.append((score, float(t), float(loud)))

        if not candidates:
            refined[bar] = predicted
            status[bar] = "fallback"
            prev_t = predicted
            continue
        candidates.sort(reverse=True)
        _, best_t, best_loud = candidates[0]
        delta = abs(best_t - predicted)
        if delta <= config.bar_snap_max_safe_delta_sec:
            refined[bar] = best_t
            status[bar] = "snapped"
            prev_t = best_t
        else:
            second_loud = candidates[1][2] if len(candidates) > 1 else -1e9
            if best_loud >= config.bar_snap_strong_loud_db and (best_loud - second_loud) >= 5.0:
                refined[bar] = best_t
                status[bar] = "snapped_strong"
                prev_t = best_t
            else:
                refined[bar] = predicted
                status[bar] = "rejected_far_snap"
                prev_t = predicted

    bars = sorted(refined)
    if len(bars) >= 3:
        durs = [refined[bars[i + 1]] - refined[bars[i]] for i in range(len(bars) - 1)]
        med = float(np.median([d for d in durs if d > 0] or durs))
        for i in range(len(bars) - 1):
            next_bar = bars[i + 1]
            this_dur = refined[next_bar] - refined[bars[i]]
            if status.get(next_bar, "").startswith("snapped") and this_dur < 0.6 * med:
                refined[next_bar] = float(initial_starts[next_bar])
                status[next_bar] = "reverted_short_bar"
    return refined, status


def _bar_durations(refined_starts: dict[int, float], alignment: dict[str, Any]) -> dict[int, float]:
    bars = sorted(refined_starts)
    audio_end = _as_float(alignment.get("audio_total_seconds"), default=None)
    durations: dict[int, float] = {}
    prev_durs: list[float] = []
    for i, bar in enumerate(bars):
        start = float(refined_starts[bar])
        if i + 1 < len(bars):
            end = float(refined_starts[bars[i + 1]])
        elif audio_end is not None and audio_end > start:
            end = float(audio_end)
        elif prev_durs:
            import numpy as np

            end = start + float(np.median(prev_durs))
        else:
            end = start
        dur = max(0.0, end - start)
        durations[bar] = dur
        if dur > 0 and i + 1 < len(bars):
            prev_durs.append(dur)
    return durations


def _per_bar_summary(
    *,
    measures: list[int],
    refined_bar_starts: dict[int, float],
    bar_durations: dict[int, float],
    by_bar_tempo: dict[int, list[float]],
    by_bar_onset_ms: dict[int, list[float]],
) -> list[dict[str, Any]]:
    import numpy as np

    rows: list[dict[str, Any]] = []
    all_bars = sorted(set(measures) | set(by_bar_tempo) | set(by_bar_onset_ms))
    for bar in all_bars:
        tempos = by_bar_tempo.get(bar, [])
        onsets = by_bar_onset_ms.get(bar, [])
        median_tempo = float(np.median(tempos)) if tempos else None
        duration = bar_durations.get(bar)
        rows.append(
            {
                "bar": int(bar),
                "measure": int(bar),
                "perf_start_sec": round(float(refined_bar_starts[bar]), 3) if bar in refined_bar_starts else None,
                "duration_sec": round(float(duration), 3) if duration is not None else None,
                "tempo_count": len(tempos),
                "median_quarter_bpm": round(median_tempo, 2) if median_tempo is not None else None,
                "tempo_min": round(float(np.min(tempos)), 2) if tempos else None,
                "tempo_max": round(float(np.max(tempos)), 2) if tempos else None,
                "tempo_range_pct": round(float((np.max(tempos) - np.min(tempos)) / median_tempo * 100.0), 1)
                if tempos and median_tempo and median_tempo > 0
                else None,
                "onset_count_within_window": len(onsets),
                "median_onset_delta_ms": round(float(np.median(onsets)), 1) if onsets else None,
                "onset_abs_max_ms": round(float(np.max(np.abs(onsets))), 1) if onsets else None,
            }
        )
    return rows


def _beat_arrivals(
    pairs: list[tuple[float, float]],
    bar_score_starts: dict[int, float],
    score_start: float,
    score_end: float,
    per_bar: list[dict[str, Any]],
    config: RhythmConfig,
) -> list[dict[str, Any]]:
    import numpy as np

    tempos = [float(b["median_quarter_bpm"]) for b in per_bar if b.get("median_quarter_bpm") is not None]
    if not tempos:
        return []
    overall_bpm = float(np.median(tempos))
    if overall_bpm <= 0:
        return []
    score = np.array([p[0] for p in pairs], dtype=float)
    perf = np.array([p[1] for p in pairs], dtype=float)
    ordered_bar_scores = sorted(bar_score_starts.items(), key=lambda kv: kv[1])
    if len(ordered_bar_scores) >= 2:
        score_bar_durs = [ordered_bar_scores[i + 1][1] - ordered_bar_scores[i][1] for i in range(len(ordered_bar_scores) - 1)]
        beat_step_score = float(np.median([d for d in score_bar_durs if d > 0])) / 4.0
    else:
        beat_step_score = 60.0 / float(config.midi_quarter_bpm)
    if beat_step_score <= 0:
        return []

    first_perf = float(np.interp(score_start, score, perf))
    rows: list[dict[str, Any]] = []
    t = float(score_start)
    while t <= float(score_end) + 1e-6 and len(rows) < config.max_beat_arrivals:
        quarters_from_start = (t - score_start) / (60.0 / float(config.midi_quarter_bpm))
        ideal_perf = first_perf + quarters_from_start * (60.0 / overall_bpm)
        actual_perf = float(np.interp(t, score, perf))
        rows.append(
            {
                "score_time_in_movement": round(t, 3),
                "measure": _measure_for_score(t, bar_score_starts),
                "ideal_performed_time": round(float(ideal_perf), 3),
                "actual_performed_time": round(float(actual_perf), 3),
                "deviation_ms": round(float((actual_perf - ideal_perf) * 1000.0), 1),
                "is_downbeat": any(abs(t - s) < 0.01 for s in bar_score_starts.values()),
            }
        )
        t += beat_step_score
    return rows


def _off_pulse_outliers(
    note_rows: list[dict[str, Any]],
    by_bar_onset_ms: dict[int, list[float]],
    config: RhythmConfig,
) -> list[dict[str, Any]]:
    import numpy as np

    outliers: list[dict[str, Any]] = []
    for row in note_rows:
        if row.get("onset_delta_ms") is None or not row.get("onset_within_window"):
            continue
        bar = int(row["measure"])
        local = by_bar_onset_ms.get(bar, [])
        if len(local) < 2:
            continue
        median = float(np.median(local))
        delta = float(row["onset_delta_ms"])
        if abs(delta - median) >= config.off_pulse_threshold_ms:
            outliers.append(
                {
                    "expected_pitch": row.get("expected_pitch"),
                    "measure": bar,
                    "score_time_in_movement": row.get("score_time_in_movement"),
                    "onset_delta_ms": round(delta, 1),
                    "local_bar_median_ms": round(median, 1),
                    "deviation_from_local_ms": round(delta - median, 1),
                }
            )
    return outliers


def _bar_refinement_report(initial: dict[int, float], refined: dict[int, float], status: dict[int, str]) -> list[dict[str, Any]]:
    return [
        {
            "bar": int(bar),
            "dtw_predicted_start": round(float(initial.get(bar, 0.0)), 3),
            "refined_start": round(float(refined[bar]), 3),
            "delta_sec": round(float(refined[bar] - initial.get(bar, 0.0)), 3),
            "method": status.get(bar, "?"),
        }
        for bar in sorted(refined)
    ]


def _render_markdown(summary: dict[str, Any], per_bar: list[dict[str, Any]], outliers: list[dict[str, Any]]) -> str:
    lines = [
        "# Polyphonic Score-Aware Rhythm",
        "",
        f"- Session: `{summary.get('session_id')}`",
        f"- Repertoire: `{summary.get('repertoire')}`",
        f"- Played measures: `{summary.get('played_measures')}`",
        f"- Music start: `{summary.get('music_start_sec')}` sec",
        f"- MIDI render quarter BPM: `{summary.get('midi_quarter_bpm_render')}`",
        f"- **Player overall quarter BPM (median local tempo): `{summary.get('overall_player_quarter_bpm_median')}`**",
        f"- Bar duration median: `{summary.get('bar_duration_median_sec')}` sec; range: `{summary.get('bar_duration_pct_range')}`%",
        f"- Score notes with nearby rich onset: `{summary.get('score_notes_with_nearby_onset_within_window')}` "
        f"(`{float(summary.get('onset_alignment_rate') or 0) * 100:.1f}%`)",
        "",
        "## Per-bar tempo and onset precision",
        "",
        "| Bar | start | duration | tempo events | median BPM | min | max | range % | onsets | median onset Δms |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in per_bar:
        lines.append(
            f"| {row['bar']} | {row.get('perf_start_sec')} | {row.get('duration_sec')} | "
            f"{row.get('tempo_count')} | {row.get('median_quarter_bpm')} | {row.get('tempo_min')} | "
            f"{row.get('tempo_max')} | {row.get('tempo_range_pct')} | "
            f"{row.get('onset_count_within_window')} | {row.get('median_onset_delta_ms')} |"
        )
    lines.extend(["", "## Off-pulse outliers", ""])
    if outliers:
        lines.extend(["| Bar | Pitch | Onset Δms | Bar median Δms | Deviation Δms |", "| --- | --- | --- | --- | --- |"])
        for row in outliers[:30]:
            lines.append(
                f"| {row['measure']} | {row.get('expected_pitch')} | {row.get('onset_delta_ms')} | "
                f"{row.get('local_bar_median_ms')} | {row.get('deviation_from_local_ms')} |"
            )
    else:
        lines.append("- No notes deviate beyond the configured off-pulse threshold, or rich onset data is unavailable.")
    lines.extend(["", "## Method"])
    for note in summary.get("method_notes", []):
        lines.append(f"- {note}")
    return "\n".join(lines) + "\n"


def _event_time(event: dict[str, Any]) -> float | None:
    return _as_float(event.get("time", event.get("start_sec")), default=None)


def _event_loudness(event: dict[str, Any]) -> float | None:
    return _as_float(event.get("loudness_db", event.get("median_loudness_db")), default=None)


def _event_pc(event: dict[str, Any]) -> int | None:
    pc = event.get("pc_top1")
    if isinstance(pc, str):
        return PC_NAME_TO_INT.get(pc.replace("♯", "#").replace("♭", "b"))
    note = event.get("note_estimate", event.get("note"))
    if isinstance(note, str):
        base = "".join(c for c in note.replace("♯", "#") if c.isalpha() or c == "#")
        return PC_NAME_TO_INT.get(base)
    peak_midi = _as_float(event.get("peak_midi"), default=None)
    if peak_midi is not None:
        return int(round(peak_midi)) % 12
    return None


def _as_float(value: Any, *, default: float | None) -> float | None:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default
