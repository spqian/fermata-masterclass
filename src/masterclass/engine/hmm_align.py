"""HMM/Viterbi note-level score follower.

Ports the PoC score follower into the v2 storage-scoped engine.  The model has
one state per unique score-time event (simultaneous notes are merged into one
chord state) plus a leading silence state.  Observations are high-resolution CQT
pitch energies with harmonic support; transitions are sparse, forward-only
self/+1/+2/+3 Viterbi transitions so the path never drifts backward.
"""

from __future__ import annotations

import io
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from masterclass.agent_tools._common import session_key
from masterclass.core.models import SessionManifest
from masterclass.core.sessions import SessionStore
from masterclass.storage.base import ObjectStorage


@dataclass(frozen=True)
class HmmAlignConfig:
    sample_rate: int = 22050
    hop_length: int = 512
    bins_per_octave: int = 60
    n_octaves: int = 7
    search_bins: int = 2
    harmonic_weight: float = 0.3
    hesitation_factor: float = 2.5
    skip_penalty: float = 0.005
    max_note_alignments: int = 2000
    refine_with_onsets: bool = True
    auto_detect_played_range: bool = True
    played_range_confidence_threshold: float = 0.4
    played_range_trailing_silence_sec: float = 8.0


@dataclass
class HmmAlignResult:
    measure_timestamps: list[dict[str, float]]
    measure_count: int
    midi_total_seconds: float
    audio_total_seconds: float
    notes: list[dict[str, Any]]
    debug: dict[str, Any]
    summary: dict[str, Any]
    bar_starts: list[dict[str, Any]]
    markdown_report: str
    refinement_applied: bool = False
    notes_with_onset_correction: int = 0
    mean_onset_correction_ms: float = 0.0
    bars_anchored_to_onsets: int = 0
    bars_no_onset_match: int = 0


def _load_pitch_events(storage: ObjectStorage, session) -> list[dict]:
    key = session_key(session, "analysis/pitch_events.json")
    if storage.exists(key):
        return storage.read_json(key)
    return []


def _load_rich_onsets_as_events(storage: ObjectStorage, session) -> list[dict]:
    key = session_key(session, "analysis/rich_onsets.json")
    if not storage.exists(key):
        return []
    data = storage.read_json(key)
    out = []
    for o in data.get("onsets", []):
        out.append({
            "start_sec": float(o.get("time", 0.0)),
            "note": o.get("note_estimate"),
            "median_loudness_db": float(o.get("loudness_db", -99.0)),
            "confidence": "high" if o.get("is_strong") else "medium",
            "source": "rich_onset",
        })
    return out


def _merge_event_sources(storage: ObjectStorage, session) -> list[dict]:
    return _load_pitch_events(storage, session) + _load_rich_onsets_as_events(storage, session)


def detect_effective_last_measure(
    note_alignments: list[dict],
    *,
    score_first_measure: int,
    score_last_measure: int,
    audio_duration_sec: float,
    confidence_threshold: float = 0.4,
    trailing_silence_sec: float = 8.0,
    score_duration_sec: float | None = None,
    observed_events: list[dict] | None = None,
) -> int:
    """Find the last score measure that appears to have been played."""

    if score_last_measure <= score_first_measure:
        return int(score_first_measure)

    by_measure: dict[int, list[dict]] = {}
    for note in note_alignments:
        measure = _as_int(note.get("measure"), None, default=0)
        if score_first_measure <= measure <= score_last_measure:
            by_measure.setdefault(measure, []).append(note)

    if not by_measure:
        return int(score_last_measure)

    high_conf = {"high", "medium"}
    ratios: dict[int, float] = {}
    raw_played: dict[int, bool] = {}
    last_high_time: float | None = None
    for measure in range(score_first_measure, score_last_measure + 1):
        notes = by_measure.get(measure, [])
        if not notes:
            ratios[measure] = 0.0
            raw_played[measure] = False
            continue
        strong = [n for n in notes if n.get("confidence") in high_conf]
        ratios[measure] = len(strong) / max(1, len(notes))
        raw_played[measure] = ratios[measure] >= confidence_threshold
        for note in strong:
            try:
                t = float(note.get("performed_time_sec"))
            except (TypeError, ValueError):
                continue
            last_high_time = t if last_high_time is None else max(last_high_time, t)

    # Smooth over a +/-1-bar window: bridge isolated low-confidence bars between
    # played neighbors, but do not invent an extra trailing bar unless the
    # last-onset check below supports it.
    played = {
        measure: raw_played.get(measure, False)
        or (raw_played.get(measure - 1, False) and raw_played.get(measure + 1, False))
        for measure in range(score_first_measure, score_last_measure + 1)
    }

    effective_last = int(score_first_measure)
    for measure in range(score_first_measure, score_last_measure + 1):
        if not played.get(measure, False):
            break
        effective_last = int(measure)

    # If the final score bars also have confidence, avoid false-trimming a full
    # performance even when earlier bars had a small local gap.
    if played.get(score_last_measure) or raw_played.get(score_last_measure) or raw_played.get(score_last_measure - 1):
        return int(score_last_measure)

    # Preserve a low-confidence final attempted bar when the last detected onset
    # lands inside that bar's HMM-assigned time window.
    last_onset_time: float | None = None
    for event in observed_events or []:
        if event.get("note") is None:
            continue
        t_raw = event.get("start_sec", event.get("time", event.get("t")))
        try:
            t = float(t_raw)
        except (TypeError, ValueError):
            continue
        last_onset_time = t if last_onset_time is None else max(last_onset_time, t)
    if last_onset_time is not None:
        windows: dict[int, tuple[float, float]] = {}
        for measure, notes in by_measure.items():
            times = []
            for note in notes:
                try:
                    times.append(float(note.get("performed_time_sec")))
                except (TypeError, ValueError):
                    pass
            if times:
                windows[measure] = (min(times), max(times))
        sorted_measures = sorted(windows)
        for idx, measure in enumerate(sorted_measures):
            start, end = windows[measure]
            next_start = windows[sorted_measures[idx + 1]][0] if idx + 1 < len(sorted_measures) else end + trailing_silence_sec
            if start - 1.0 <= last_onset_time <= max(end + 1.0, next_start + 0.5):
                if measure <= effective_last + 2:
                    effective_last = max(effective_last, int(measure))

    # Long trailing silence is strong evidence that the confidence-derived range
    # is the true ending.
    if last_high_time is not None and last_high_time + trailing_silence_sec < float(audio_duration_sec):
        return max(int(score_first_measure), min(int(effective_last), int(score_last_measure)))

    # If confident HMM notes continue to the end of the recording, avoid
    # false-trimming a complete take just because of a long low-confidence gap.
    if last_high_time is not None and (
        score_duration_sec is None or float(audio_duration_sec) <= float(score_duration_sec) * 1.1
    ):
        return int(score_last_measure)

    return max(int(score_first_measure), min(int(effective_last), int(score_last_measure)))


def _estimate_music_start_sec(storage: ObjectStorage, session, fallback: float) -> float:
    onset_key = session_key(session, "analysis/onsets.json")
    if storage.exists(onset_key):
        data = storage.read_json(onset_key)
        raw_onsets = data if isinstance(data, list) else data.get("onsets", [])
        candidates = [
            o for o in raw_onsets
            if o.get("is_strong") or o.get("confidence") == "high" or o.get("strength_rank") == "strong"
        ]
        for o in (candidates or raw_onsets):
            t = o.get("time", o.get("start_sec", o.get("t")))
            if t is not None:
                return float(t)

    rich_key = session_key(session, "analysis/rich_onsets.json")
    if storage.exists(rich_key):
        data = storage.read_json(rich_key)
        raw_onsets = data.get("onsets", [])
        candidates = [o for o in raw_onsets if o.get("is_strong")]
        for o in (candidates or raw_onsets):
            t = o.get("time", o.get("start_sec"))
            if t is not None:
                return float(t)

    return float(fallback)


def refine_with_onsets(note_alignments: list, observed_events: list,
                        max_search_sec: float = 2.5, min_loud_db: float = -28.0,
                        smoothing_window: int = 7) -> list:
    """Post-process HMM alignments by snapping each note to its nearest matching
    detected onset, then applying a smoothed offset across the whole sequence.

    Strategy:
      1. For each high/medium-confidence HMM-aligned note, find the nearest detected
         pitch event with the same pitch-class within max_search_sec.
      2. The delta between HMM time and onset time is a per-note residual.
      3. Smooth the residuals with a centered moving median over `smoothing_window`
         neighboring high/medium-conf notes (filters noise from same-pitch ambiguity).
      4. Subtract the smoothed residual from each note's perf_time. The net effect is
         that HMM's relative timing is preserved, but global drift (and small consistent
         offsets) are corrected toward observed onsets.
      5. Low-confidence and unvisited notes get the smoothed residual interpolated
         from their neighbors and the same correction applied.
    """
    events = list(observed_events)
    # Build pitch-name -> sorted list of (start_sec, loud_db) for high/medium events
    pc_events: dict[str, list] = {}
    for e in events:
        if e.get("confidence") == "low": continue
        if not e.get("note"): continue
        loud = e.get("median_loudness_db")
        if loud is None or loud < min_loud_db: continue
        pc_events.setdefault(e["note"], []).append((float(e["start_sec"]), float(loud)))
    for k in pc_events: pc_events[k].sort()

    import bisect
    residuals: list[float | None] = []
    for n in note_alignments:
        if n.get("confidence") not in ("high", "medium") and n.get("timestamp_source") != "expected_tempo_low_coverage_fallback":
            residuals.append(None)
            continue
        names = n.get("names", [])
        if not names:
            residuals.append(None)
            continue
        target = names[0]
        candidates = pc_events.get(target, [])
        if not candidates:
            residuals.append(None)
            continue
        t = n["performed_time_sec"]
        # bisect for nearest
        starts = [c[0] for c in candidates]
        idx = bisect.bisect_left(starts, t)
        best_dt = None
        for i in (idx - 1, idx):
            if 0 <= i < len(candidates):
                dt = candidates[i][0] - t
                if abs(dt) > max_search_sec: continue
                if best_dt is None or abs(dt) < abs(best_dt):
                    best_dt = dt
        residuals.append(best_dt)

    # Smooth residuals with median over window of neighboring non-None values
    n_total = len(note_alignments)
    smoothed = [None] * n_total
    half = smoothing_window // 2
    for i in range(n_total):
        # Collect non-None residuals in window
        bucket = []
        for j in range(max(0, i - half), min(n_total, i + half + 1)):
            if residuals[j] is not None:
                bucket.append(residuals[j])
        if bucket:
            bucket.sort()
            smoothed[i] = bucket[len(bucket) // 2]  # median

    # Forward/backward fill for stretches with no residual at all
    last_known = 0.0
    for i in range(n_total):
        if smoothed[i] is not None:
            last_known = smoothed[i]
        else:
            smoothed[i] = last_known

    # Apply correction: new_perf_time = perf_time + smoothed_residual
    refined = []
    for n, corr in zip(note_alignments, smoothed):
        new_n = dict(n)
        new_n["performed_time_sec"] = round(float(n["performed_time_sec"]) + float(corr), 3)
        new_n["perf_time"] = new_n["performed_time_sec"]
        new_n["onset_correction_ms"] = round(float(corr) * 1000.0, 1)
        refined.append(new_n)
    return refined


def find_bar_boundaries_global(
    bar_starts_hint: list,
    observed_events: list,
    recording_duration: float,
    music_start_sec: float,
    n_bars: int,
    loud_thresh_db: float = -22.0,
) -> list[dict]:
    """Globally re-derive bar boundaries by optimizing over detected onsets.

    Idea: bar boundaries should (a) start at music_start, (b) land on STRONG attacks,
    (c) be roughly evenly spaced (within rubato tolerance), and (d) sum to the
    played duration. This is a chain-DP problem: for each bar in turn, pick the
    onset that best satisfies (b) and (c) given the previous boundary.

    Returns a list of dicts with the same shape as HMM bar_starts.
    """
    events = list(observed_events)
    onsets = sorted([
        (float(e["start_sec"]), float(e.get("median_loudness_db") or -99.0), e.get("note"))
        for e in events
        if e.get("confidence") in ("high", "medium") and e.get("note")
        and (e.get("median_loudness_db") or -99) >= loud_thresh_db
    ])
    if not onsets:
        return list(bar_starts_hint)

    played_dur = recording_duration - music_start_sec
    expected_bar_dur = played_dur / max(1, n_bars)

    # Map HMM hints by bar number for the distance prior
    hmm_by_bar = {b["measure"]: b["performed_time_sec"] for b in bar_starts_hint}
    bar_numbers = sorted(hmm_by_bar.keys())
    if not bar_numbers:
        return list(bar_starts_hint)

    out = []
    # Bar 1 is anchored at music_start (or HMM bar 1 if close)
    first_bar = bar_numbers[0]
    bar1_t = music_start_sec
    out.append({
        "measure": first_bar,
        "performed_time_sec": round(bar1_t, 3),
        "start": round(bar1_t, 3),
        "first_visited_pitches": ["(music start)"],
        "is_score_bar_first_state": True,
        "method": "music_start",
    })

    prev_t = bar1_t
    for i, bar in enumerate(bar_numbers[1:], start=1):
        # Where do we EXPECT this bar to start?
        expected_t = prev_t + expected_bar_dur
        # Allow shrink/stretch up to 50% of expected duration
        lo = prev_t + 0.5 * expected_bar_dur
        hi = prev_t + 1.6 * expected_bar_dur
        # Among onsets in [lo, hi], score = loudness - distance_penalty - hmm_distance_penalty
        candidates = []
        for t, loud, note in onsets:
            if t < lo or t > hi: continue
            dist_pen = abs(t - expected_t) * 4.0  # 4 dB per second from even spacing
            hmm_pen = 0.0
            if bar in hmm_by_bar:
                hmm_pen = abs(t - hmm_by_bar[bar]) * 1.5
            score = loud - dist_pen - hmm_pen
            candidates.append((score, t, loud, note))
        if candidates:
            candidates.sort(reverse=True)
            _, t, loud, note = candidates[0]
            out.append({
                "measure": bar,
                "performed_time_sec": round(t, 3),
                "start": round(t, 3),
                "first_visited_pitches": [note] if note else [],
                "is_score_bar_first_state": True,
                "method": "global_dp",
                "loudness_db": round(loud, 1),
                "expected_t_sec": round(expected_t, 3),
                "delta_from_expected_ms": round((t - expected_t) * 1000, 0),
            })
            prev_t = t
        else:
            # No onset in window; fall back to expected_t
            out.append({
                "measure": bar,
                "performed_time_sec": round(expected_t, 3),
                "start": round(expected_t, 3),
                "first_visited_pitches": ["(no onset)"],
                "is_score_bar_first_state": False,
                "method": "expected_only",
            })
            prev_t = expected_t
    return out


def align_lesson_with_midi_hmm(
    *,
    storage: ObjectStorage,
    store: SessionStore,
    manifest: SessionManifest,
    midi_bytes: bytes,
    config: HmmAlignConfig | None = None,
    music_start_sec: float | None = None,
    first_measure: int | None = None,
    last_measure: int | None = None,
) -> HmmAlignResult:
    """Run HMM Viterbi note-level alignment for a lesson audio and MIDI.

    Inputs are storage-scoped: audio is read from ``manifest.artifacts`` and MIDI
    bytes are already loaded.  ``first_measure`` / ``last_measure`` override the
    manifest metadata range when supplied; otherwise the manifest range is used.
    """

    del store  # Kept in the signature for parity with the v2 aligner API.
    config = config or HmmAlignConfig()
    audio_key = manifest.artifacts.get("artifacts/audio.wav")
    if not audio_key:
        raise ValueError("manifest is missing artifacts/audio.wav")

    import librosa
    import numpy as np
    import pretty_midi

    start_sec = (
        float(music_start_sec)
        if music_start_sec is not None
        else (_estimate_music_start_sec(storage, manifest.session, 0.0) if config.refine_with_onsets else 0.0)
    )

    midi = pretty_midi.PrettyMIDI(io.BytesIO(midi_bytes))
    all_notes = _load_midi_notes(midi)
    if not all_notes:
        raise RuntimeError("MIDI contains no notes")

    downbeats = _measure_times_from_midi(midi)
    if not downbeats:
        raise RuntimeError("MIDI has no detectable measure structure")

    user_supplied_last_measure = last_measure is not None or manifest.metadata.get("last_measure") not in (None, "")
    fm = _as_int(first_measure, manifest.metadata.get("first_measure"), default=1)
    lm = _as_int(last_measure, manifest.metadata.get("last_measure"), default=len(downbeats))
    if fm < 1 or lm < fm or fm > len(downbeats):
        raise ValueError(f"invalid measure range: {fm}..{lm} for {len(downbeats)} measures")
    lm = min(lm, len(downbeats))
    effective_first_measure = int(fm)
    effective_last_measure = int(lm)
    played_range_auto_detected = False
    played_range_method = "user_supplied" if user_supplied_last_measure else "auto_confidence"

    score_start = float(downbeats[fm - 1])
    score_end = float(downbeats[lm]) if lm < len(downbeats) else float(midi.get_end_time())
    midi_played_dur = max(0.001, score_end - score_start)

    def measure_for_score_time(score_time: float) -> int:
        measure = fm
        for idx in range(fm - 1, min(lm, len(downbeats))):
            start = float(downbeats[idx])
            end = float(downbeats[idx + 1]) if idx + 1 < len(downbeats) else float(midi.get_end_time()) + 0.001
            if start - 0.001 <= float(score_time) < end - 0.001:
                return int(idx + 1)
            if float(score_time) >= start:
                measure = int(idx + 1)
        return int(min(max(measure, fm), lm))

    frame_dt = config.hop_length / config.sample_rate

    with tempfile.TemporaryDirectory(prefix="mc-hmm-align-") as tmp_raw:
        audio_path = Path(tmp_raw) / "lesson.wav"
        storage.read_to_file(audio_key, audio_path)
        y, _ = librosa.load(str(audio_path), sr=config.sample_rate, mono=True)

    audio_total_seconds = float(len(y) / config.sample_rate)
    perf_active_dur = max(0.001, audio_total_seconds - max(0.0, start_sec))
    tempo_factor = perf_active_dur / midi_played_dur

    states = _build_states(
        all_notes,
        score_start,
        score_end,
        tempo_factor=tempo_factor,
        frame_dt=frame_dt,
        leading_silence_sec=max(0.0, start_sec),
    )
    for state in states:
        if state.get("kind") == "note":
            state["measure"] = measure_for_score_time(float(state["score_time_in_movement"]))
    if len(states) <= 1:
        raise RuntimeError("measure range contains no note events")

    fmin_hz = librosa.note_to_hz("C2")
    n_bins = config.bins_per_octave * config.n_octaves
    cqt = np.abs(
        librosa.cqt(
            y,
            sr=config.sample_rate,
            hop_length=config.hop_length,
            fmin=fmin_hz,
            n_bins=n_bins,
            bins_per_octave=config.bins_per_octave,
        )
    )
    n_frames = int(cqt.shape[1])
    log_obs = _compute_log_obs(
        cqt,
        fmin_hz,
        config.bins_per_octave,
        states,
        search_bins=config.search_bins,
        harmonic_weight=config.harmonic_weight,
        music_start_sec=max(0.0, start_sec),
        frame_dt=frame_dt,
    )

    n_states = len(states)
    log_init = np.full(n_states, -1e18, dtype=np.float64)
    log_init[0] = 0.0

    log_self, log_next, log_skip2, log_skip3 = _build_log_transitions(
        states,
        frame_dt,
        hesitation_factor=config.hesitation_factor,
        skip_penalty=config.skip_penalty,
    )
    path, total_logp = _viterbi_sparse(log_obs, log_self, log_next, log_skip2, log_skip3, log_init)

    state_first_frame: dict[int, int] = {}
    state_dwell: dict[int, int] = {}
    prev = -1
    prev_start = 0
    for t, state_idx in enumerate(path):
        s = int(state_idx)
        if s != prev:
            if prev >= 0:
                state_dwell[prev] = t - prev_start
            if s not in state_first_frame:
                state_first_frame[s] = t
            prev_start = t
            prev = s
    if prev >= 0:
        state_dwell[prev] = n_frames - prev_start

    def frame_to_time(frame: int) -> float:
        return float(frame * frame_dt)

    note_alignments: list[dict[str, Any]] = []
    for s_idx, state in enumerate(states):
        if state["kind"] != "note" or s_idx not in state_first_frame:
            continue
        first_f = state_first_frame[s_idx]
        dwell = int(state_dwell.get(s_idx, 0))
        obs_lp = float(log_obs[s_idx, first_f])
        t_perf = frame_to_time(first_f)
        note_alignments.append(
            {
                "state_idx": int(s_idx),
                "pitches_midi": list(state["pitches"]),
                "names": list(state["names"]),
                "measure": int(state.get("measure", fm)),
                "score_time_in_movement": round(float(state["score_time_in_movement"]), 3),
                "score_time_local": round(float(state["score_time_local"]), 3),
                "expected_perf_duration": round(float(state["expected_perf_duration"]), 3),
                "performed_time_sec": round(t_perf, 3),
                "perf_time": round(t_perf, 3),
                "dwell_frames": dwell,
                "dwell_sec": round(float(dwell) * frame_dt, 3),
                "obs_log_prob": round(obs_lp, 2),
                "confidence": _classify_confidence(dwell, obs_lp),
                "timestamp_source": "hmm_viterbi",
            }
        )

    raw_note_alignments_for_detection = list(note_alignments)
    raw_hmm_coverage = len(note_alignments) / max(1, len(states) - 1)
    used_expected_fallback = False
    if raw_hmm_coverage < 0.5:
        # The PoC HMM was tuned for monophonic violin.  On dense/noisy v2 smoke
        # assets it can legitimately fail to visit enough states; keep the run
        # useful by preserving the MIDI event sequence with the inferred global
        # tempo so downstream measure timestamps are complete and monotone.
        note_alignments = _expected_note_alignments(states, start_sec, tempo_factor)
        used_expected_fallback = True

    note_align_by_state = {int(n["state_idx"]): n for n in note_alignments}
    bar_starts: list[dict[str, Any]] = []
    for measure in range(fm, lm + 1):
        db_start = float(downbeats[measure - 1])
        db_end = float(downbeats[measure]) if measure < len(downbeats) else score_end + 0.001
        candidates = []
        for s_idx, state in enumerate(states):
            if state["kind"] != "note" or s_idx not in note_align_by_state:
                continue
            if db_start - 0.001 <= float(state["score_time_in_movement"]) < db_end - 0.001:
                candidates.append((float(note_align_by_state[s_idx]["performed_time_sec"]), s_idx, state))
        if not candidates:
            continue
        candidates.sort(key=lambda row: row[0])
        first_t, s_idx, first_state = candidates[0]
        score_first_idx = min(
            (
                idx
                for idx, state in enumerate(states)
                if state["kind"] == "note"
                and db_start - 0.001 <= float(state["score_time_in_movement"]) < db_end - 0.001
            ),
            default=None,
        )
        bar_starts.append(
            {
                "measure": int(measure),
                "performed_time_sec": round(first_t, 3),
                "start": round(first_t, 3),
                "first_visited_state_idx": int(s_idx),
                "first_visited_pitches": list(first_state["names"]),
                "is_score_bar_first_state": bool(s_idx == score_first_idx),
                "method": "hmm_viterbi",
            }
        )

    merged_events = _merge_event_sources(storage, manifest.session)
    if config.auto_detect_played_range and not user_supplied_last_measure:
        effective_last_measure = detect_effective_last_measure(
            raw_note_alignments_for_detection,
            score_first_measure=fm,
            score_last_measure=lm,
            audio_duration_sec=audio_total_seconds,
            confidence_threshold=config.played_range_confidence_threshold,
            trailing_silence_sec=config.played_range_trailing_silence_sec,
            score_duration_sec=midi_played_dur,
            observed_events=merged_events,
        )
        played_range_auto_detected = True
        if effective_last_measure < lm:
            note_alignments = [
                n for n in note_alignments if int(n.get("measure", 0) or 0) <= effective_last_measure
            ]
            bar_starts = [
                b for b in bar_starts if int(b.get("measure", 0) or 0) <= effective_last_measure
            ]
            lm = int(effective_last_measure)
    else:
        effective_last_measure = int(lm)

    refinement_applied = False
    if merged_events and config.refine_with_onsets:
        note_alignments = refine_with_onsets(note_alignments, merged_events)
        bar_music_start_sec = _estimate_music_start_sec(
            storage=storage,
            session=manifest.session,
            fallback=float(bar_starts[0]["performed_time_sec"]) if bar_starts else start_sec,
        )
        bar_starts = find_bar_boundaries_global(
            bar_starts_hint=bar_starts,
            observed_events=merged_events,
            recording_duration=audio_total_seconds,
            music_start_sec=bar_music_start_sec,
            n_bars=len(bar_starts),
        )
        refinement_applied = True

    visited_note_states = len(note_alignments)
    total_note_states = len(states) - 1
    state_coverage = visited_note_states / max(1, total_note_states)
    measure_timestamps = [
        {"measure": int(b["measure"]), "start": float(b["performed_time_sec"])} for b in bar_starts
    ]
    onset_corrections = [
        float(n.get("onset_correction_ms") or 0.0)
        for n in note_alignments
        if abs(float(n.get("onset_correction_ms") or 0.0)) > 0.0
    ]
    notes_with_onset_correction = len(onset_corrections)
    mean_onset_correction_ms = (
        sum(onset_corrections) / notes_with_onset_correction if notes_with_onset_correction else 0.0
    )
    bars_anchored_to_onsets = sum(1 for b in bar_starts if b.get("method") == "global_dp")
    bars_no_onset_match = sum(1 for b in bar_starts if b.get("method") == "expected_only")

    summary = {
        "method": "hmm_score_follower",
        "music_start_sec": round(start_sec, 3),
        "tempo_factor_perf_over_midi": round(float(tempo_factor), 3),
        "n_states": len(states),
        "n_frames": n_frames,
        "viterbi_log_prob": round(float(total_logp), 2),
        "played_measures": [fm, lm],
        "effective_first_measure": int(effective_first_measure),
        "effective_last_measure": int(effective_last_measure),
        "played_range_auto_detected": bool(played_range_auto_detected),
        "played_range_method": played_range_method,
        "states_visited": len(state_first_frame),
        "note_states_visited": visited_note_states,
        "total_note_states": total_note_states,
        "state_coverage": round(float(state_coverage), 3),
        "frame_dt_ms": round(frame_dt * 1000, 2),
        "cqt_settings": {
            "bins_per_octave": config.bins_per_octave,
            "n_bins": n_bins,
            "cents_per_bin": 1200.0 / config.bins_per_octave,
            "search_bins": config.search_bins,
        },
        "method_notes": [
            "HMM with one state per unique score-time event (chords merged), plus a leading silence state.",
            "Observation: log of max CQT energy in +/- search_bins around expected pitch, plus 2nd/3rd harmonics.",
            "Transitions: forward-only sparse (self, +1, +2, +3); no backward transitions.",
            "Inference: log-domain Viterbi over the full recording at frame_dt resolution.",
            "Per-bar start = recording time of the first visited state in that bar, optionally refined to strong onsets.",
        ],
        "raw_hmm_state_coverage": round(float(raw_hmm_coverage), 3),
        "used_expected_fallback": used_expected_fallback,
        "refinement_applied": refinement_applied,
        "refine_with_onsets_enabled": bool(config.refine_with_onsets),
        "merged_event_count": len(merged_events),
        "note_count": len(note_alignments),
        "notes_with_onset_correction": notes_with_onset_correction,
        "mean_onset_correction_ms": round(float(mean_onset_correction_ms), 1),
        "bars_anchored_to_onsets": bars_anchored_to_onsets,
        "bars_no_onset_match": bars_no_onset_match,
    }
    debug = {
        "audio_total_seconds": audio_total_seconds,
        "midi_total_seconds": float(midi.get_end_time()),
        "midi_played_seconds": float(midi_played_dur),
        "perf_active_seconds": float(perf_active_dur),
        "measure_count": len(measure_timestamps),
        "note_count": len(note_alignments),
        "cqt_shape": [int(cqt.shape[0]), int(cqt.shape[1])],
    }
    markdown = _render_markdown(summary, bar_starts)

    return HmmAlignResult(
        measure_timestamps=measure_timestamps,
        measure_count=len(measure_timestamps),
        midi_total_seconds=float(midi.get_end_time()),
        audio_total_seconds=audio_total_seconds,
        notes=note_alignments[: config.max_note_alignments],
        debug=debug,
        summary=summary,
        bar_starts=bar_starts,
        markdown_report=markdown,
        refinement_applied=refinement_applied,
        notes_with_onset_correction=notes_with_onset_correction,
        mean_onset_correction_ms=round(float(mean_onset_correction_ms), 1),
        bars_anchored_to_onsets=bars_anchored_to_onsets,
        bars_no_onset_match=bars_no_onset_match,
    )


def persist_hmm_alignment(
    *,
    storage: ObjectStorage,
    store: SessionStore,
    manifest: SessionManifest,
    result: HmmAlignResult,
) -> None:
    """Persist HMM alignment JSON, aligned notes JSON, report markdown, and manifest stamps."""

    align_key = store.artifact_key(manifest.session, "analysis/hmm_alignment.json")
    storage.write_json(
        align_key,
        {
            "schema_version": 1,
            "summary": result.summary,
            "measure_timestamps": result.measure_timestamps,
            "measure_count": result.measure_count,
            "midi_total_seconds": result.midi_total_seconds,
            "audio_total_seconds": result.audio_total_seconds,
            "bar_starts": result.bar_starts,
            "notes": result.notes,
            "note_alignments": result.notes,
            "debug": result.debug,
            "refinement_applied": result.refinement_applied,
            "notes_with_onset_correction": result.notes_with_onset_correction,
            "mean_onset_correction_ms": result.mean_onset_correction_ms,
            "bars_anchored_to_onsets": result.bars_anchored_to_onsets,
            "bars_no_onset_match": result.bars_no_onset_match,
        },
    )
    notes_key = store.artifact_key(manifest.session, "analysis/hmm_aligned_notes.json")
    storage.write_json(notes_key, {"schema_version": 1, "notes": result.notes})
    md_key = store.artifact_key(manifest.session, "analysis/hmm_alignment.md")
    storage.write_bytes(md_key, result.markdown_report.encode("utf-8"), content_type="text/markdown")

    manifest.artifacts["analysis/hmm_alignment.json"] = align_key
    manifest.artifacts["analysis/hmm_aligned_notes.json"] = notes_key
    manifest.artifacts["analysis/hmm_alignment.md"] = md_key
    manifest.metadata["hmm_alignment_state"] = "ready"
    manifest.metadata["hmm_alignment_measure_count"] = result.measure_count
    manifest.metadata["hmm_alignment_note_count"] = len(result.notes)
    manifest.metadata["hmm_alignment_state_coverage"] = result.summary.get("state_coverage")
    manifest.metadata["auto_detected_last_measure"] = result.summary.get("effective_last_measure")
    manifest.metadata["auto_detected_first_measure"] = result.summary.get("effective_first_measure")
    manifest.metadata["played_range_method"] = result.summary.get("played_range_method")
    store.save(manifest)


def _load_midi_notes(midi) -> list[dict[str, Any]]:
    import pretty_midi

    notes: list[dict[str, Any]] = []
    for inst in midi.instruments:
        for n in inst.notes:
            notes.append(
                {
                    "start": float(n.start),
                    "end": float(n.end),
                    "pitch": int(n.pitch),
                    "name": pretty_midi.note_number_to_name(int(n.pitch)),
                    "velocity": int(n.velocity),
                    "is_drum": bool(inst.is_drum),
                }
            )
    notes.sort(key=lambda n: (float(n["start"]), -int(n["pitch"])))
    return notes


def _build_states(
    midi_notes: list[dict[str, Any]],
    score_start_sec: float,
    score_end_sec: float,
    *,
    tempo_factor: float,
    frame_dt: float,
    leading_silence_sec: float,
) -> list[dict[str, Any]]:
    by_start: dict[float, list[dict[str, Any]]] = {}
    for note in midi_notes:
        if not (score_start_sec - 0.001 <= float(note["start"]) < score_end_sec - 0.001):
            continue
        key = round(float(note["start"]) - score_start_sec, 4)
        by_start.setdefault(key, []).append(note)

    states: list[dict[str, Any]] = [
        {
            "kind": "silence",
            "score_time_in_movement": float(score_start_sec),
            "score_time_local": 0.0,
            "pitches": [],
            "names": ["(silence)"],
            "midi_duration": 0.0,
            "expected_perf_duration": float(max(frame_dt * 4.0, leading_silence_sec or frame_dt * 4.0)),
        }
    ]
    sorted_starts = sorted(by_start.keys())
    for i, start in enumerate(sorted_starts):
        notes_here = by_start[start]
        pitches = sorted({int(n["pitch"]) for n in notes_here})
        names = sorted({str(n["name"]) for n in notes_here})
        notated_end = max(float(n["end"]) - score_start_sec for n in notes_here)
        next_start = sorted_starts[i + 1] if i + 1 < len(sorted_starts) else (score_end_sec - score_start_sec)
        eff_midi_dur = max(0.04, min(notated_end, next_start) - start)
        states.append(
            {
                "kind": "note",
                "score_time_in_movement": float(start + score_start_sec),
                "score_time_local": float(start),
                "pitches": pitches,
                "names": names,
                "midi_duration": float(eff_midi_dur),
                "expected_perf_duration": float(eff_midi_dur * tempo_factor),
            }
        )
    return states


def _midi_to_log_bin(midi_pitch: float, fmin_hz: float, bins_per_octave: int) -> float:
    import numpy as np

    freq = 440.0 * 2 ** ((midi_pitch - 69.0) / 12.0)
    return float(np.log2(freq / fmin_hz) * bins_per_octave)


def _compute_log_obs(
    cqt,
    fmin_hz: float,
    bins_per_octave: int,
    states: list[dict[str, Any]],
    *,
    search_bins: int = 2,
    harmonic_weight: float = 0.3,
    music_start_sec: float = 0.0,
    frame_dt: float,
):
    import numpy as np

    n_bins, n_frames = cqt.shape
    floor = np.median(cqt, axis=0) + 1e-6
    log_obs = np.zeros((len(states), n_frames), dtype=np.float32)

    frame_energy = np.median(cqt, axis=0) + 1e-6
    global_floor = float(np.median(frame_energy) + 1e-6)
    log_obs[0] = np.clip(np.log(global_floor / frame_energy), -5.0, 5.0)
    for s_idx, state in enumerate(states):
        if state.get("kind") != "note":
            continue
        contributions = []
        for pitch in state["pitches"]:
            bin_pos = _midi_to_log_bin(float(pitch), fmin_hz, bins_per_octave)
            ctr = int(round(bin_pos))
            lo = max(0, ctr - search_bins)
            hi = min(n_bins - 1, ctr + search_bins)
            fund = cqt[lo : hi + 1, :].max(axis=0)
            score_p = np.log((fund + 1e-6) / floor)
            for h in (2, 3):
                h_freq = 440.0 * 2 ** ((float(pitch) - 69.0) / 12.0) * h
                hbin_pos = np.log2(h_freq / fmin_hz) * bins_per_octave
                hctr = int(round(hbin_pos))
                if 0 <= hctr < n_bins:
                    hlo = max(0, hctr - search_bins)
                    hhi = min(n_bins - 1, hctr + search_bins)
                    h_e = cqt[hlo : hhi + 1, :].max(axis=0)
                    score_p = score_p + harmonic_weight * np.log((h_e + 1e-6) / floor)
            contributions.append(score_p)
        if contributions:
            stacked = np.stack(contributions, axis=0)
            log_obs[s_idx] = stacked.sum(axis=0) / np.sqrt(len(contributions))
    if music_start_sec > 0:
        start_frame = min(n_frames, max(0, int(round(music_start_sec / frame_dt))))
        log_obs[0, :start_frame] += 6.0
        log_obs[0, start_frame:] -= 6.0
        log_obs[1:, :start_frame] -= 50.0
    return log_obs


def _build_log_transitions(states: list[dict[str, Any]], frame_dt: float, *, hesitation_factor: float = 2.5, skip_penalty: float = 0.005):
    import numpy as np

    n = len(states)
    p_self = np.zeros(n)
    for i, state in enumerate(states):
        d = max(frame_dt * 1.2, float(state["expected_perf_duration"])) * hesitation_factor
        p_self[i] = float(np.exp(-frame_dt / d))
    p_self = np.clip(p_self, 1e-3, 1.0 - 1e-5)
    p_move = 1.0 - p_self
    p_next = p_move * (1.0 - skip_penalty)
    p_skip2 = p_move * (skip_penalty * 0.75)
    p_skip3 = p_move * (skip_penalty * 0.25)
    eps = 1e-30
    return np.log(p_self + eps), np.log(p_next + eps), np.log(p_skip2 + eps), np.log(p_skip3 + eps)


def _viterbi_sparse(log_obs, log_self, log_next, log_skip2, log_skip3, log_init):
    import numpy as np

    n_states, n_frames = log_obs.shape
    delta = np.full((n_states, n_frames), -np.inf, dtype=np.float64)
    psi = np.zeros((n_states, n_frames), dtype=np.int32)
    delta[:, 0] = log_init + log_obs[:, 0]

    neg = -1e18
    for t in range(1, n_frames):
        c_self = delta[:, t - 1] + log_self
        c_prev = np.full(n_states, neg, dtype=np.float64)
        c_prev[1:] = delta[:-1, t - 1] + log_next[:-1]
        c_skip2 = np.full(n_states, neg, dtype=np.float64)
        c_skip2[2:] = delta[:-2, t - 1] + log_skip2[:-2]
        c_skip3 = np.full(n_states, neg, dtype=np.float64)
        c_skip3[3:] = delta[:-3, t - 1] + log_skip3[:-3]
        cand = np.stack([c_self, c_prev, c_skip2, c_skip3], axis=0)
        best = np.argmax(cand, axis=0)
        delta[:, t] = np.take_along_axis(cand, best[None, :], axis=0).squeeze(0) + log_obs[:, t]
        idx = np.arange(n_states, dtype=np.int32)
        psi[:, t] = idx - best.astype(np.int32)

    path = np.zeros(n_frames, dtype=np.int32)
    path[-1] = int(np.argmax(delta[:, -1]))
    for t in range(n_frames - 2, -1, -1):
        path[t] = psi[path[t + 1], t + 1]
    return path, float(delta[path[-1], -1])


def _expected_note_alignments(states: list[dict[str, Any]], music_start_sec: float, tempo_factor: float) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for s_idx, state in enumerate(states):
        if state.get("kind") != "note":
            continue
        t_perf = float(music_start_sec) + float(state["score_time_local"]) * float(tempo_factor)
        out.append(
            {
                "state_idx": int(s_idx),
                "pitches_midi": list(state["pitches"]),
                "names": list(state["names"]),
                "measure": int(state.get("measure", 1)),
                "score_time_in_movement": round(float(state["score_time_in_movement"]), 3),
                "score_time_local": round(float(state["score_time_local"]), 3),
                "expected_perf_duration": round(float(state["expected_perf_duration"]), 3),
                "performed_time_sec": round(t_perf, 3),
                "perf_time": round(t_perf, 3),
                "dwell_frames": 0,
                "dwell_sec": 0.0,
                "obs_log_prob": None,
                "confidence": "low",
                "timestamp_source": "expected_tempo_low_coverage_fallback",
            }
        )
    return out


def _classify_confidence(dwell_frames: int, obs_log_prob: float) -> str:
    if dwell_frames >= 8 and obs_log_prob >= 4.0:
        return "high"
    if dwell_frames >= 4 and obs_log_prob >= 1.5:
        return "medium"
    return "low"


def _measure_times_from_midi(midi) -> list[float]:
    try:
        downbeats = list(midi.get_downbeats())
    except Exception:
        downbeats = []
    if downbeats:
        return [float(t) for t in downbeats]

    import math

    end = float(midi.get_end_time())
    tempo = midi.estimate_tempo() or 120.0
    seconds_per_measure = (60.0 / tempo) * 4.0
    n = max(1, math.ceil(end / seconds_per_measure))
    return [i * seconds_per_measure for i in range(n)]


def _as_int(primary: int | None, secondary: Any, *, default: int) -> int:
    if primary is not None:
        return int(primary)
    if isinstance(secondary, int):
        return int(secondary)
    return default


def _render_markdown(summary: dict[str, Any], bar_starts: list[dict[str, Any]]) -> str:
    md: list[str] = []
    md.append("# HMM Score Follower Alignment")
    md.append("")
    md.append("- Method: probabilistic HMM with note-level states, forward-only sparse Viterbi")
    md.append(f"- Played measures: bar `{summary['played_measures'][0]}` to bar `{summary['played_measures'][1]}`")
    md.append(f"- States: `{summary['n_states']}` (1 silence + `{summary['total_note_states']}` note events)")
    md.append(
        f"- State coverage: `{summary['note_states_visited']}/{summary['total_note_states']}` "
        f"(`{float(summary['state_coverage']) * 100:.1f}%`)"
    )
    md.append(f"- Music start: `{float(summary['music_start_sec']):.2f}s`; tempo factor: `{float(summary['tempo_factor_perf_over_midi']):.2f}`x of MIDI")
    md.append(f"- Frame resolution: `{float(summary['frame_dt_ms']):.1f} ms`")
    md.append(f"- Viterbi log-prob: `{float(summary['viterbi_log_prob']):.1f}`")
    md.append("")
    md.append("## Per-bar start times")
    md.append("")
    md.append("| Bar | Performed time | First visited state | Score-bar-first? |")
    md.append("| --- | --- | --- | --- |")
    for bar in bar_starts:
        names = ", ".join(str(n) for n in bar.get("first_visited_pitches", []))
        first_marker = "yes" if bar.get("is_score_bar_first_state") else "no (skipped to next event)"
        state_label = (
            f"state {bar['first_visited_state_idx']} ({names})"
            if "first_visited_state_idx" in bar
            else f"{bar.get('method', 'refined')} ({names})"
        )
        md.append(
            f"| {bar['measure']} | {float(bar['performed_time_sec']):.2f}s | "
            f"{state_label} | {first_marker} |"
        )
    md.append("")
    md.append("## Method")
    for note in summary["method_notes"]:
        md.append(f"- {note}")
    return "\n".join(md) + "\n"


