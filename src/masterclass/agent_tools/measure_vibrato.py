from __future__ import annotations

from typing import Any

from masterclass.core.models import SessionRef
from masterclass.storage.base import ObjectStorage
from ._common import load_audio

MEASURE_VIBRATO_SCHEMA = {"type": "object", "properties": {"start_sec": {"type": "number"}, "end_sec": {"type": "number"}}, "required": ["start_sec", "end_sec"]}
DESCRIPTION = "Vibrato rate+depth in a window (numeric). args: {start_sec, end_sec}"


def measure_vibrato(storage: ObjectStorage, session: SessionRef, args: dict[str, Any]) -> dict[str, Any]:
    import librosa
    import numpy as np

    start = float(args["start_sec"]); end = float(args["end_sec"])
    if end <= start or end - start > 5.0:
        return {"error": "window must be 0.1-5.0s"}
    y, sr = load_audio(storage, session); seg = y[int(start * sr):int(end * sr)]
    if len(seg) < sr // 10:
        return {"error": "segment too short"}
    f0, _, _ = librosa.pyin(seg, sr=sr, fmin=float(librosa.note_to_hz("G3")), fmax=float(librosa.note_to_hz("E7")), frame_length=2048)
    f0_voiced = f0[~np.isnan(f0)]
    if len(f0_voiced) < 20:
        return {"verdict": "none", "reason": "insufficient voiced frames"}
    cents = 1200.0 * np.log2(f0_voiced / np.median(f0_voiced)); cents -= np.mean(cents)
    ac = np.correlate(cents, cents, mode="full")[len(cents)-1:]; ac = ac / (ac[0] + 1e-9)
    hop = 512; frame_dt = hop / sr; lo_lag = int(1.0 / 9.0 / frame_dt); hi_lag = min(int(1.0 / 3.0 / frame_dt), len(ac) - 1)
    if lo_lag >= hi_lag:
        return {"verdict": "none", "reason": "window too short for vibrato analysis"}
    peak_lag = lo_lag + int(np.argmax(ac[lo_lag:hi_lag])); peak_strength = float(ac[peak_lag])
    rate_hz = 1.0 / (peak_lag * frame_dt) if peak_lag > 0 else 0.0; depth_cents = float(np.std(cents) * np.sqrt(2.0))
    if peak_strength < 0.25 or depth_cents < 8.0:
        verdict = "none"
    elif depth_cents > 100.0:
        verdict = "non-vibrato pitch motion (probably a slur, shift, or chord arpeggiation)"
    elif rate_hz < 4.5 and depth_cents > 35:
        verdict = "wide-and-slow"
    elif rate_hz > 7.5 and depth_cents < 25:
        verdict = "fast-and-narrow"
    else:
        verdict = "present"
    return {"window_sec": [start, end], "rate_hz": round(rate_hz, 2), "depth_cents": round(depth_cents, 1), "ac_peak_strength": round(peak_strength, 2), "n_voiced_frames": int(len(f0_voiced)), "verdict": verdict}
