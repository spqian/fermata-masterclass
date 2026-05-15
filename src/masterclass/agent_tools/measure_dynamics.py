from __future__ import annotations

from typing import Any

from masterclass.core.models import SessionRef
from masterclass.storage.base import ObjectStorage
from ._common import load_audio, note_perf_time, read_json

MEASURE_DYNAMICS_SCHEMA = {"type": "object", "properties": {"start_sec": {"type": "number"}, "end_sec": {"type": "number"}}, "required": ["start_sec", "end_sec"]}
DESCRIPTION = "Loudness contour + per-note peak loudness across a phrase. args: {start_sec, end_sec}"


def measure_dynamics(storage: ObjectStorage, session: SessionRef, args: dict[str, Any]) -> dict[str, Any]:
    import librosa
    import numpy as np
    from scipy.signal import find_peaks

    start = float(args["start_sec"]); end = float(args["end_sec"])
    if end <= start or end - start < 0.1 or end - start > 30.0:
        return {"error": "window must be 0.1-30s"}
    y, sr = load_audio(storage, session); seg = y[int(start * sr):int(end * sr)]
    if len(seg) < sr // 10:
        return {"error": "segment too short"}
    hop = 512; rms = librosa.feature.rms(y=seg, hop_length=hop)[0]; rms_db = 20.0 * np.log10(np.maximum(rms, 1e-6)); times = librosa.frames_to_time(np.arange(len(rms_db)), sr=sr, hop_length=hop) + start
    smooth_n = max(1, int(0.05 * sr / hop)); rms_s = np.convolve(rms_db, np.ones(smooth_n) / smooth_n, mode="same") if smooth_n > 1 else rms_db
    peak_db = float(np.max(rms_s)); active_floor = max(-60.0, peak_db - 50.0); valid = rms_s[rms_s > active_floor]
    floor_db = float(np.percentile(valid, 10)) if len(valid) else active_floor; loud_db = float(np.percentile(valid, 90)) if len(valid) else peak_db
    samples = [{"t_sec": round(float(times[i]), 3), "rel_db": round(float(rms_s[i] - peak_db), 1)} for i in range(0, len(rms_s), max(1, int(0.25 * sr / hop)))]
    peaks, _ = find_peaks(rms_s, distance=max(1, int(0.15 * sr / hop)), prominence=2.0)
    peaks_out = [{"t_sec": round(float(t), 3), "rel_db": round(float(d), 1)} for t, d in sorted([(times[p], rms_s[p] - peak_db) for p in peaks], key=lambda x: -x[1])[:8]]
    note_peaks = []
    sm = read_json(storage, session, "score/score_map.json", {}) or {}
    for n in [n for n in sm.get("notes", []) if note_perf_time(n) is not None and start <= note_perf_time(n) <= end]:
        t = note_perf_time(n); mask = (times >= t) & (times <= t + 0.5)
        if not np.any(mask):
            continue
        curve = rms_s[mask]; ntimes = times[mask]; idx = int(np.argmax(curve)); peak_in_note = float(curve[idx])
        note_peaks.append({"note_id": n.get("note_id"), "names": n.get("names"), "perf_time": round(float(t), 3), "peak_db_rel": round(peak_in_note - peak_db, 1), "peak_offset_ms": round(float((ntimes[idx] - t) * 1000.0), 0), "onset_db_rel": round(float(curve[0] - peak_db), 1), "is_chord": bool(n.get("is_chord"))})
    note_peaks.sort(key=lambda x: x["perf_time"])
    return {"window_sec": [start, end], "duration_sec": round(end - start, 2), "dynamic_range_db": round(float(loud_db - floor_db), 1), "peak_db_absolute": round(peak_db, 1), "floor_db_absolute": round(floor_db, 1), "range_method": "90th_percentile_minus_10th_percentile_active_rms", "envelope_samples": samples, "loudest_peaks": peaks_out, "per_note_peaks": note_peaks, "interpretation": "All dB values are relative to the loudest moment in this window (peak = 0). dynamic_range_db uses active-frame percentiles, not silence-to-peak range."}
