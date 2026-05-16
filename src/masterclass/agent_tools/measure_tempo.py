from __future__ import annotations

from typing import Any

from masterclass.core.models import SessionRef
from masterclass.storage.base import ObjectStorage
from ._common import read_json

MEASURE_TEMPO_SCHEMA = {"type": "object", "properties": {"midi_measure": {"type": "integer"}, "start_measure": {"type": "integer"}, "end_measure": {"type": "integer"}}, "required": []}
DESCRIPTION = "Tempo/rubato evidence for one bar or measure window (numeric). args: {midi_measure} or {start_measure,end_measure}"


def _bar_num(row):
    return int(row.get("bar", row.get("measure", -999)))


def measure_tempo(storage: ObjectStorage, session: SessionRef, args: dict[str, Any]) -> dict[str, Any]:
    rhy = read_json(storage, session, "analysis/polyphonic_rhythm.json", None)
    if rhy is None:
        return {"error": "analysis/polyphonic_rhythm.json missing"}
    summary = rhy.get("summary", {}); median_dur = summary.get("bar_duration_median_sec")
    per_bar = summary.get("bar_durations") or rhy.get("per_bar") or []
    if "start_measure" in args or "end_measure" in args:
        start = int(args.get("start_measure", args.get("midi_measure", 1))); end = int(args.get("end_measure", start))
        rows = [b for b in per_bar if start <= _bar_num(b) <= end]
        return {"measure_window": [start, end], "median_bar_duration_sec": median_dur, "bars": [_compact_bar(b, median_dur, summary) for b in rows], "bar_count": len(rows)}
    midi_measure = int(args.get("midi_measure", args.get("measure", 1)))
    bar = next((b for b in per_bar if _bar_num(b) == midi_measure), None)
    if not bar:
        available = sorted({_bar_num(b) for b in per_bar if _bar_num(b) >= 0})
        return {
            "error": f"no per-bar tempo data for measure {midi_measure}",
            "reason": "audio-truth matching produced no notes inside this measure (likely a rolling chord, fermata, or tacet at the start/end of the played range)",
            "available_measures": available,
            "median_bar_duration_sec": median_dur,
            "overall_player_quarter_bpm_median": summary.get("overall_player_quarter_bpm_median"),
            "interpretation": "Pick the closest available_measure instead, or use {start_measure, end_measure} with a wider window.",
        }
    out = _compact_bar(bar, median_dur, summary); out["midi_measure"] = midi_measure
    return out


def _compact_bar(bar: dict[str, Any], median_dur: float | None, summary: dict[str, Any]) -> dict[str, Any]:
    duration = bar.get("duration_sec")
    pct = ((float(duration) - float(median_dur)) / float(median_dur) * 100.0) if duration is not None and median_dur else None
    num = _bar_num(bar)
    outliers = [o for o in summary.get("off_pulse_outliers", []) if int(o.get("measure", -999)) == num]
    return {"bar": num, "duration_sec": round(float(duration), 3) if duration is not None else None, "median_bar_duration_sec": round(float(median_dur), 3) if median_dur else None, "pct_vs_median": round(float(pct), 1) if pct is not None else None, "tempo_bpm_quarter": bar.get("tempo_bpm_quarter", bar.get("median_quarter_bpm")), "off_pulse_outliers": outliers, "n_outliers": len(outliers)}
