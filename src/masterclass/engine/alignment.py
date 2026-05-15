"""Audio↔MIDI alignment via chroma DTW, plus measure timing extraction.

Given a lesson audio file and a reference MIDI:
  1) load MIDI, infer tempo + measures
  2) synthesize MIDI to audio at the lesson's sample rate
  3) compute chroma features for both and run DTW to map performance-time
     to MIDI-time
  4) project measure boundaries from MIDI-time onto performance-time

This restores deterministic time->measure alignment without depending on the
LLM's listening estimates.
"""

from __future__ import annotations

import math
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from masterclass.core.models import SessionManifest
from masterclass.core.sessions import SessionStore
from masterclass.storage.base import ObjectStorage
from masterclass.toolchain.ffmpeg import FfmpegToolchain


@dataclass(frozen=True)
class AlignmentConfig:
    sample_rate: int = 22050
    hop_length: int = 512
    chroma_n_fft: int = 4096
    synth_program: int = 0  # acoustic grand piano
    use_cens: bool = True   # tempo-invariant chroma; better cross-version alignment


@dataclass
class AlignmentResult:
    measure_timestamps: list[dict[str, float]]   # [{measure, start}]
    measure_count: int
    midi_total_seconds: float
    audio_total_seconds: float
    notes: list[dict[str, Any]]                  # per-MIDI-note times mapped to performance-time
    debug: dict[str, Any]


def align_lesson_with_midi(
    *,
    storage: ObjectStorage,
    store: SessionStore,
    ffmpeg: FfmpegToolchain,
    manifest: SessionManifest,
    midi_bytes: bytes,
    config: AlignmentConfig | None = None,
) -> AlignmentResult:
    """Run chroma DTW alignment and return per-measure / per-note performance times.

    If the lesson manifest specifies a measure range (first_measure / last_measure
    in metadata), the MIDI is trimmed to that range before synthesis so the DTW
    isn't asked to squeeze the entire piece into a short excerpt.
    """

    config = config or AlignmentConfig()
    audio_key = manifest.artifacts.get("artifacts/audio.wav")
    if not audio_key:
        raise ValueError("manifest is missing artifacts/audio.wav")

    import librosa
    import pretty_midi
    import soundfile

    midi = pretty_midi.PrettyMIDI(_io_bytes(midi_bytes))
    full_measure_times = _measure_times_from_midi(midi)
    if not full_measure_times:
        raise RuntimeError("MIDI has no detectable measure structure")

    first_measure = manifest.metadata.get("first_measure")
    last_measure = manifest.metadata.get("last_measure")
    measure_offset = 0
    crop_start = None
    crop_end = None
    if isinstance(first_measure, int) and isinstance(last_measure, int) and 1 <= first_measure <= last_measure <= len(full_measure_times):
        crop_start = float(full_measure_times[first_measure - 1])
        crop_end = float(full_measure_times[last_measure]) if last_measure < len(full_measure_times) else float(midi.get_end_time())
        midi = _crop_midi(midi, crop_start, crop_end)
        measure_offset = first_measure - 1
        measure_times = [t - crop_start for t in full_measure_times[first_measure - 1:last_measure]]
    else:
        measure_times = full_measure_times

    with tempfile.TemporaryDirectory(prefix="mc-align-") as tmp_raw:
        tmp = Path(tmp_raw)
        audio_path = tmp / "lesson.wav"
        storage.read_to_file(audio_key, audio_path)
        perf_y, _ = librosa.load(str(audio_path), sr=config.sample_rate, mono=True)

        synth = _synthesize_midi(midi, config)
        # pretty_midi.synthesize returns a numpy float in [-1, 1] at fs sample rate
        synth_y = librosa.resample(synth.astype(np.float32), orig_sr=midi.resolution * 0 + config.sample_rate if False else config.sample_rate, target_sr=config.sample_rate) if False else synth.astype(np.float32)

    if config.use_cens:
        perf_chroma = librosa.feature.chroma_cens(y=perf_y, sr=config.sample_rate, hop_length=config.hop_length)
        midi_chroma = librosa.feature.chroma_cens(y=synth_y, sr=config.sample_rate, hop_length=config.hop_length)
    else:
        perf_chroma = librosa.feature.chroma_cqt(y=perf_y, sr=config.sample_rate, hop_length=config.hop_length)
        midi_chroma = librosa.feature.chroma_cqt(y=synth_y, sr=config.sample_rate, hop_length=config.hop_length)

    # Guard against silent regions producing zero-magnitude chroma columns
    # which would yield NaN cosine distances downstream.
    perf_chroma = _normalize_chroma(perf_chroma)
    midi_chroma = _normalize_chroma(midi_chroma)

    # DTW. CENS chroma is tempo-invariant which helps cross-version alignment;
    # we use the default step pattern (any of (1,1), (0,1), (1,0)) so a wide
    # range of tempo deviations stays valid.
    D, wp = librosa.sequence.dtw(
        X=midi_chroma,
        Y=perf_chroma,
        metric="cosine",
        subseq=False,
    )
    wp = np.asarray(wp[::-1])  # ascending order

    # Build a function mapping midi-time to performance-time using the warping path
    midi_to_perf = _build_time_map(wp, hop_length=config.hop_length, sr=config.sample_rate)

    measures_out: list[dict[str, float]] = []
    for idx, t_midi in enumerate(measure_times, start=1):
        t_perf = midi_to_perf(t_midi)
        measures_out.append({"measure": idx + measure_offset, "start": float(round(max(0.0, t_perf), 3))})

    notes_out: list[dict[str, Any]] = []
    for instrument in midi.instruments:
        for note in instrument.notes:
            t_perf = midi_to_perf(float(note.start))
            notes_out.append({
                "midi_time": float(note.start),
                "perf_time": float(round(max(0.0, t_perf), 3)),
                "duration": float(note.end - note.start),
                "pitch": int(note.pitch),
                "velocity": int(note.velocity),
                "is_drum": bool(instrument.is_drum),
            })

    debug = {
        "perf_frames": int(perf_chroma.shape[1]),
        "midi_frames": int(midi_chroma.shape[1]),
        "warp_points": int(wp.shape[0]),
        "midi_total_seconds": float(midi.get_end_time()),
        "audio_total_seconds": float(perf_y.shape[0] / config.sample_rate),
        "measure_count": len(measures_out),
        "note_count": len(notes_out),
    }

    return AlignmentResult(
        measure_timestamps=measures_out,
        measure_count=len(measures_out),
        midi_total_seconds=float(midi.get_end_time()),
        audio_total_seconds=float(perf_y.shape[0] / config.sample_rate),
        notes=notes_out,
        debug=debug,
    )


def persist_alignment(
    *,
    storage: ObjectStorage,
    store: SessionStore,
    manifest: SessionManifest,
    result: AlignmentResult,
) -> None:
    """Write the alignment result onto the lesson session."""

    align_key = store.artifact_key(manifest.session, "analysis/alignment.json")
    storage.write_json(align_key, {
        "schema_version": 1,
        "measure_timestamps": result.measure_timestamps,
        "measure_count": result.measure_count,
        "midi_total_seconds": result.midi_total_seconds,
        "audio_total_seconds": result.audio_total_seconds,
        "debug": result.debug,
    })
    notes_key = store.artifact_key(manifest.session, "analysis/aligned_notes.json")
    storage.write_json(notes_key, {
        "schema_version": 1,
        "notes": result.notes,
    })
    manifest.artifacts["analysis/alignment.json"] = align_key
    manifest.artifacts["analysis/aligned_notes.json"] = notes_key
    manifest.metadata["alignment_measure_count"] = result.measure_count
    manifest.metadata["alignment_note_count"] = len(result.notes)
    store.save(manifest)


def _crop_midi(midi, start_sec: float, end_sec: float):
    """Return a new PrettyMIDI restricted to the [start_sec, end_sec] window.

    Note timings are shifted so the crop window starts at t=0.
    """

    import pretty_midi

    cropped = pretty_midi.PrettyMIDI(initial_tempo=midi.estimate_tempo() or 120.0)
    # Preserve time signatures relative to the new origin
    for ts in midi.time_signature_changes:
        if start_sec <= ts.time <= end_sec:
            cropped.time_signature_changes.append(
                pretty_midi.TimeSignature(ts.numerator, ts.denominator, ts.time - start_sec)
            )
    for ks in midi.key_signature_changes:
        if start_sec <= ks.time <= end_sec:
            cropped.key_signature_changes.append(
                pretty_midi.KeySignature(ks.key_number, ks.time - start_sec)
            )

    for inst in midi.instruments:
        new_inst = pretty_midi.Instrument(program=inst.program, is_drum=inst.is_drum, name=inst.name)
        for note in inst.notes:
            if note.end <= start_sec or note.start >= end_sec:
                continue
            ns = max(start_sec, note.start) - start_sec
            ne = min(end_sec, note.end) - start_sec
            if ne > ns:
                new_inst.notes.append(pretty_midi.Note(velocity=note.velocity, pitch=note.pitch, start=ns, end=ne))
        if new_inst.notes:
            cropped.instruments.append(new_inst)
    return cropped


def _io_bytes(data: bytes):
    import io
    return io.BytesIO(data)


def _normalize_chroma(chroma: np.ndarray) -> np.ndarray:
    """Replace any all-zero columns with a uniform vector and L2-normalize.

    Chroma columns from silent audio frames produce zero-magnitude vectors,
    which then yield NaN cosine distances and break DTW. We treat such frames
    as "no preference" by injecting a uniform pitch-class distribution and
    then L2-normalize all columns so cosine distance stays well-defined.
    """

    chroma = np.asarray(chroma, dtype=np.float32)
    if chroma.ndim != 2:
        return chroma
    norms = np.linalg.norm(chroma, axis=0)
    zero_cols = norms < 1e-9
    if zero_cols.any():
        chroma[:, zero_cols] = 1.0 / np.sqrt(chroma.shape[0])
        norms = np.linalg.norm(chroma, axis=0)
    norms[norms < 1e-9] = 1.0
    return chroma / norms[np.newaxis, :]


def _measure_times_from_midi(midi) -> list[float]:
    """Return seconds at which each measure starts (1-indexed by position).

    Uses pretty_midi.get_downbeats() if available; falls back to a uniform
    grid based on time-signature changes when not.
    """

    try:
        downbeats = list(midi.get_downbeats())
    except Exception:
        downbeats = []
    if downbeats:
        return [float(t) for t in downbeats]

    # Fallback: assume 4/4 at the inferred tempo
    end = float(midi.get_end_time())
    tempo = midi.estimate_tempo() or 120.0
    seconds_per_beat = 60.0 / tempo
    seconds_per_measure = seconds_per_beat * 4
    n = max(1, math.ceil(end / seconds_per_measure))
    return [i * seconds_per_measure for i in range(n)]


def _synthesize_midi(midi, config: AlignmentConfig) -> np.ndarray:
    """Synthesize MIDI to a mono float waveform at config.sample_rate."""

    # pretty_midi.synthesize uses a basic sine-additive synth; perfectly fine
    # for chroma matching (we only care about pitch class distribution over time).
    return midi.fluidsynth(fs=config.sample_rate) if False else midi.synthesize(fs=config.sample_rate)


def _build_time_map(wp: np.ndarray, *, hop_length: int, sr: int):
    """Construct a callable that maps MIDI time (seconds) to performance time (seconds).

    ``wp`` is a 2-column array of (midi_frame_index, perf_frame_index) pairs in
    ascending order. We collapse duplicates and linearly interpolate.
    """

    midi_frames = wp[:, 0]
    perf_frames = wp[:, 1]
    # Collapse to monotonically non-decreasing midi axis
    midi_seconds = midi_frames * hop_length / sr
    perf_seconds = perf_frames * hop_length / sr

    # Unique midi times for interpolation
    sort_order = np.argsort(midi_seconds, kind="stable")
    ms = midi_seconds[sort_order]
    ps = perf_seconds[sort_order]
    # average performance times at duplicate midi times
    unique_ms, idx_start = np.unique(ms, return_index=True)
    averaged_ps = np.array([ps[start:next_idx].mean() for start, next_idx in zip(idx_start, list(idx_start[1:]) + [len(ps)])])
    # Force monotone non-decreasing performance axis
    averaged_ps = np.maximum.accumulate(averaged_ps)

    def f(t_midi: float) -> float:
        if not len(unique_ms):
            return float(t_midi)
        if t_midi <= unique_ms[0]:
            return float(averaged_ps[0])
        if t_midi >= unique_ms[-1]:
            return float(averaged_ps[-1])
        return float(np.interp(t_midi, unique_ms, averaged_ps))

    return f
