"""Tool: measure how sharp/flat a specific note actually was in the recording.

The teacher must call this before making any "<pitch> tends sharp/flat by X cents"
claim. The polyphonic_intonation summary in the evidence packet aggregates by
pitch class across the whole take and can mislead - one buzzy sustained note can
skew the median for that pitch class. This tool grounds the claim in one
specific moment.
"""
from __future__ import annotations

from typing import Any

from masterclass.core.models import SessionRef
from masterclass.storage.base import ObjectStorage
from ._common import load_audio, midi_pitch_to_name


INSPECT_INTONATION_SCHEMA = {
    "type": "object",
    "properties": {
        "time_sec": {"type": "number", "description": "performed time the note was attacked"},
        "expected_pitch": {
            "type": "string",
            "description": "score-expected pitch as MIDI number (as a string, e.g. '60') or note name (e.g. 'C5')",
        },
        "window_ms": {"type": "number", "description": "averaging window after attack, default 200 ms"},
    },
    "required": ["time_sec", "expected_pitch"],
}

DESCRIPTION = (
    "Measure cents-off-from-score for ONE specific note in the recording. "
    "Use BEFORE any 'this note was sharp/flat by N cents' claim. "
    "args: {time_sec, expected_pitch (midi int or note name like 'C5'), window_ms?}"
)


def _coerce_midi(value: Any) -> int | None:
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(round(value))
    if isinstance(value, str):
        s = value.strip()
        if not s:
            return None
        if s.lstrip("-").isdigit():
            return int(s)
        import re
        m = re.match(r"^([A-Ga-g])([#b]?)(-?\d+)$", s)
        if not m:
            return None
        letter, accidental, octv = m.group(1).upper(), m.group(2), int(m.group(3))
        base = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}[letter]
        if accidental == "#":
            base += 1
        elif accidental == "b":
            base -= 1
        return base + (octv + 1) * 12
    return None


def inspect_intonation(storage: ObjectStorage, session: SessionRef, args: dict[str, Any]) -> dict[str, Any]:
    import librosa
    import numpy as np

    t = float(args["time_sec"])
    window_ms = float(args.get("window_ms", 200.0))
    expected_midi = _coerce_midi(args.get("expected_pitch"))
    if expected_midi is None:
        return {"error": "expected_pitch must be a MIDI integer or note name like 'C5'"}

    y, sr = load_audio(storage, session)
    hop = 512
    fmin_note = "C2"
    fmin = librosa.note_to_hz(fmin_note)
    bpo = 60  # 20 cents per bin
    n_bins = bpo * 7
    cqt = np.abs(librosa.cqt(y, sr=sr, hop_length=hop, fmin=fmin, n_bins=n_bins, bins_per_octave=bpo))
    cqt_times = librosa.frames_to_time(np.arange(cqt.shape[1]), sr=sr, hop_length=hop)

    f_lo = int(np.searchsorted(cqt_times, t))
    f_hi = min(int(np.searchsorted(cqt_times, t + window_ms / 1000.0)), cqt.shape[1])
    if f_hi <= f_lo:
        return {"error": "time out of audio range"}
    profile = cqt[:, f_lo:f_hi].mean(axis=1)
    if profile.max() <= 0:
        return {"error": "no energy in window"}

    cents_per_bin = 1200.0 / bpo  # 20.0
    # Bin index corresponding to expected pitch
    base_midi = librosa.note_to_midi(fmin_note)  # 36
    expected_bin_f = (expected_midi - base_midi) * (bpo / 12.0)
    expected_bin = int(round(expected_bin_f))
    if expected_bin < 0 or expected_bin >= n_bins:
        return {"error": f"expected_pitch {expected_midi} out of CQT range"}

    # Search +/- 50 cents around expected pitch
    search_bins = int(round(50.0 / cents_per_bin))
    lo = max(0, expected_bin - search_bins)
    hi = min(n_bins, expected_bin + search_bins + 1)
    local = profile[lo:hi]
    if local.max() <= 0:
        return {"error": "no energy near expected pitch"}
    peak_local_idx = int(np.argmax(local))
    peak_bin = lo + peak_local_idx
    # Parabolic interpolation for sub-bin accuracy
    if 0 < peak_bin < n_bins - 1:
        a = profile[peak_bin - 1]; b = profile[peak_bin]; c = profile[peak_bin + 1]
        denom = (a - 2 * b + c)
        offset = 0.5 * (a - c) / denom if denom != 0 else 0.0
        offset = max(-1.0, min(1.0, float(offset)))
    else:
        offset = 0.0
    peak_bin_f = peak_bin + offset
    cents_off = (peak_bin_f - expected_bin_f) * cents_per_bin

    profile_db = 20.0 * np.log10(np.maximum(profile, 1e-9))
    profile_db -= profile_db.max()
    expected_db = float(profile_db[expected_bin])
    peak_db = float(profile_db[peak_bin])

    # Also report neighboring +/-100 cents window in case real fundamental is
    # one semitone away (i.e. wrong note, not just out-of-tune).
    wider_lo = max(0, expected_bin - int(round(150.0 / cents_per_bin)))
    wider_hi = min(n_bins, expected_bin + int(round(150.0 / cents_per_bin)) + 1)
    wider = profile[wider_lo:wider_hi]
    wider_peak_local = int(np.argmax(wider))
    wider_peak_bin = wider_lo + wider_peak_local
    wider_cents_off = (wider_peak_bin - expected_bin_f) * cents_per_bin

    return {
        "time_sec": round(t, 3),
        "window_ms": window_ms,
        "expected_pitch_midi": expected_midi,
        "expected_pitch_name": midi_pitch_to_name(expected_midi),
        "cents_off_score": round(float(cents_off), 1),
        "peak_rel_db": round(peak_db, 1),
        "expected_pitch_energy_rel_db": round(expected_db, 1),
        "wider_search_cents_off": round(float(wider_cents_off), 1),
        "interpretation": (
            "cents_off_score is the deviation of the loudest spectral peak within +/-50 cents of the expected pitch. "
            "Positive = sharp, negative = flat. |cents_off_score| < 15 means in tune. "
            "If wider_search_cents_off differs from cents_off_score by more than ~50 cents, "
            "the player likely produced a different pitch entirely (use that as evidence of a wrong note, not out-of-tune)."
        ),
    }
