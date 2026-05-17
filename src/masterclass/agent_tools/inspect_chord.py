from __future__ import annotations

from typing import Any

from masterclass.core.models import SessionRef
from masterclass.storage.base import ObjectStorage
from ._common import load_audio, midi_pitch_to_name

INSPECT_CHORD_SCHEMA = {"type": "object", "properties": {"time_sec": {"type": "number"}, "window_ms": {"type": "number"}}, "required": ["time_sec"]}
DESCRIPTION = "Spectral peak voicing at a moment (NUMERIC; not perceptual). args: {time_sec, window_ms?}"


def inspect_chord(storage: ObjectStorage, session: SessionRef, args: dict[str, Any]) -> dict[str, Any]:
    from ._drill_guard import reject_if_drill
    rejection = reject_if_drill(storage, session)
    if rejection is not None:
        return rejection
    import librosa
    import numpy as np
    from scipy.signal import find_peaks

    t = float(args["time_sec"]); window_ms = float(args.get("window_ms", 200.0))
    y, sr = load_audio(storage, session)
    hop = 512; fmin = librosa.note_to_hz("C2"); bpo = 60; n_bins = bpo * 7
    cqt = np.abs(librosa.cqt(y, sr=sr, hop_length=hop, fmin=fmin, n_bins=n_bins, bins_per_octave=bpo))
    cqt_times = librosa.frames_to_time(np.arange(cqt.shape[1]), sr=sr, hop_length=hop)
    f_lo = int(np.searchsorted(cqt_times, t)); f_hi = min(int(np.searchsorted(cqt_times, t + window_ms / 1000.0)), cqt.shape[1])
    if f_hi <= f_lo:
        return {"error": "time out of audio range"}
    profile = cqt[:, f_lo:f_hi].mean(axis=1)
    if profile.max() <= 0:
        return {"error": "no energy in window"}
    profile_db = 20.0 * np.log10(np.maximum(profile, 1e-9)); profile_db -= profile_db.max()
    peaks, _ = find_peaks(profile_db, height=-30.0, distance=int(bpo * 0.5))
    out = []
    for p in sorted(peaks, key=lambda idx: -profile_db[idx])[:8]:
        midi = 36 + p * 12.0 / bpo
        out.append({"midi": round(float(midi), 2), "note_name": midi_pitch_to_name(int(round(midi))), "freq_hz": round(float(librosa.midi_to_hz(midi)), 1), "rel_db": round(float(profile_db[p]), 1)})
    return {"time_sec": t, "window_ms": window_ms, "peaks": out, "interpretation": "Peaks listed by descending energy. The loudest peak is the dominant voice. Adjacent peaks within 2 semitones may be partials of the same note rather than distinct notes."}
