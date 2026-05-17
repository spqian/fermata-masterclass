"""Summary statistics over basic-pitch transcribed notes for a drill clip.

The drill pipeline doesn't have a score to align against, so we don't have
per-note ``timing_offset_ms`` like the lesson pipeline. Instead we compute
self-referential metrics — onset evenness, tempo estimate, pitch-class
distribution, intonation drift on the most common pitch — and hand the
summary to the LLM along with the audio so it can weigh "what the student
prescribed themselves to practice" against "what the recording shows".
"""
from __future__ import annotations

import math
import statistics
from typing import Any


_PITCH_CLASS_NAMES = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")


def _midi_to_freq(midi: float) -> float:
    return 440.0 * (2.0 ** ((midi - 69.0) / 12.0))


def compute_drill_metrics(notes: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute summary statistics over a list of basic-pitch notes.

    Input shape mirrors :func:`audio_truth._transcribe_basic_pitch`: each
    note carries ``performed_time_sec``, ``dwell_sec``, ``pitches_midi``,
    ``amplitude``.

    The returned dict always carries ``low_signal: bool`` so the LLM can
    be told to comment on audio quality when there are too few detected
    notes to draw conclusions.
    """
    n = len(notes)
    if n == 0:
        return {
            "schema_version": 1,
            "n_notes": 0,
            "low_signal": True,
            "low_signal_reason": "no notes detected by basic-pitch",
        }

    onsets = sorted(float(note.get("performed_time_sec") or 0.0) for note in notes)
    first_onset = onsets[0]
    last_onset = onsets[-1]

    durations = [float(note.get("dwell_sec") or 0.0) for note in notes]
    amplitudes = [float(note.get("amplitude") or 0.0) for note in notes]

    iois = [b - a for a, b in zip(onsets, onsets[1:]) if (b - a) > 0]
    ioi_mean = statistics.fmean(iois) if iois else None
    ioi_stdev = statistics.pstdev(iois) if len(iois) >= 2 else 0.0 if iois else None
    ioi_median = statistics.median(iois) if iois else None

    # Tempo estimate: assume the median IOI is one beat. This is wrong for
    # ornamented passages but gives the LLM a sane order-of-magnitude.
    tempo_bpm = None
    if ioi_median and ioi_median > 0:
        tempo_bpm = round(60.0 / ioi_median, 1)

    # Pitch class histogram and per-pitch intonation drift.
    pitch_counts: dict[int, int] = {}
    for note in notes:
        pitches = note.get("pitches_midi") or []
        for pitch in pitches:
            try:
                midi = int(pitch)
            except (TypeError, ValueError):
                continue
            pitch_counts[midi] = pitch_counts.get(midi, 0) + 1

    pitch_class_hist: dict[str, int] = {}
    for midi, count in pitch_counts.items():
        pc = _PITCH_CLASS_NAMES[midi % 12]
        pitch_class_hist[pc] = pitch_class_hist.get(pc, 0) + count

    top_pitch_midi = None
    top_pitch_count = 0
    if pitch_counts:
        top_pitch_midi = max(pitch_counts.items(), key=lambda kv: kv[1])[0]
        top_pitch_count = pitch_counts[top_pitch_midi]

    metrics = {
        "schema_version": 1,
        "n_notes": n,
        "first_onset_sec": round(first_onset, 3),
        "last_onset_sec": round(last_onset, 3),
        "duration_sec": round(max(0.0, last_onset - first_onset), 3),
        "ioi_mean_sec": round(ioi_mean, 4) if ioi_mean is not None else None,
        "ioi_median_sec": round(ioi_median, 4) if ioi_median is not None else None,
        "ioi_stdev_sec": round(ioi_stdev, 4) if ioi_stdev is not None else None,
        "ioi_cv": (
            round(ioi_stdev / ioi_mean, 3)
            if ioi_mean and ioi_mean > 0 and ioi_stdev is not None
            else None
        ),
        "tempo_bpm_estimate": tempo_bpm,
        "mean_dwell_sec": round(statistics.fmean(durations), 4) if durations else None,
        "mean_amplitude": round(statistics.fmean(amplitudes), 4) if amplitudes else None,
        "pitch_class_histogram": dict(sorted(pitch_class_hist.items(), key=lambda kv: -kv[1])),
        "n_unique_pitches": len(pitch_counts),
        "top_pitch_midi": top_pitch_midi,
        "top_pitch_count": top_pitch_count,
        "low_signal": n < 4,
    }
    if n < 4:
        metrics["low_signal_reason"] = (
            f"only {n} note(s) detected — audio quality, mic distance, or "
            "an unpitched percussive clip can cause this; the LLM should "
            "call this out rather than draw conclusions"
        )
    return metrics
