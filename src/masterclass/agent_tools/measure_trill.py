from __future__ import annotations

from typing import Any

from masterclass.core.models import SessionRef
from masterclass.storage.base import ObjectStorage
from ._common import load_audio

MEASURE_TRILL_SCHEMA = {"type": "object", "properties": {"start_sec": {"type": "number"}, "end_sec": {"type": "number"}}, "required": ["start_sec", "end_sec"]}
DESCRIPTION = "Trill rate+evenness in a window (numeric). args: {start_sec, end_sec}"


def measure_trill(storage: ObjectStorage, session: SessionRef, args: dict[str, Any]) -> dict[str, Any]:
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
        return {"verdict": "no trill detected", "reason": "insufficient voiced frames"}
    cents = 1200.0 * np.log2(f0_voiced / np.median(f0_voiced)); cents -= np.mean(cents)
    swing = float(np.percentile(cents, 90) - np.percentile(cents, 10))
    ac = np.correlate(cents, cents, mode="full")[len(cents)-1:]; ac = ac / (ac[0] + 1e-9)
    hop = 512; frame_dt = hop / sr; lo_lag = int(1.0 / 14.0 / frame_dt); hi_lag = min(int(1.0 / 4.0 / frame_dt), len(ac) - 1)
    if lo_lag >= hi_lag:
        return {"verdict": "no trill detected", "reason": "window too short"}
    peak_lag = lo_lag + int(np.argmax(ac[lo_lag:hi_lag])); peak_strength = float(ac[peak_lag]); rate_hz = 1.0 / (peak_lag * frame_dt) if peak_lag > 0 else 0.0
    zc = np.where(np.diff(np.sign(cents)) != 0)[0]
    evenness = 0.0
    if len(zc) >= 4:
        intervals = np.diff(zc).astype(float); evenness = 1.0 - min(1.0, float(np.std(intervals) / (np.mean(intervals) + 1e-9)))
    if peak_strength < 0.20 or swing < 40.0:
        verdict = "no trill detected"
    elif rate_hz < 5.0:
        verdict = "slow trill (may sound deliberate or laboured)"
    elif rate_hz > 11.0:
        verdict = "very fast trill"
    elif evenness > 0.7:
        verdict = "regular trill"
    else:
        verdict = "trill present but uneven"
    return {"window_sec": [start, end], "rate_hz": round(rate_hz, 2), "alternation_depth_cents": round(swing, 1), "evenness": round(evenness, 2), "ac_peak_strength": round(peak_strength, 2), "verdict": verdict}
