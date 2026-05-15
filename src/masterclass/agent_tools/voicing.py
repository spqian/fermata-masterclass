from __future__ import annotations

from typing import Any

from masterclass.core.models import SessionRef, session_prefix
from masterclass.storage.base import ObjectStorage


INSPECT_VOICING_SCHEMA = {
    "type": "object",
    "properties": {
        "midi_measure": {"type": "integer"},
        "measure": {"type": "integer"},
        "beat": {"type": "number"},
        "start_sec": {"type": "number"},
        "end_sec": {"type": "number"},
    },
}


def inspect_voicing(storage: ObjectStorage, session: SessionRef, args: dict[str, Any]) -> dict[str, Any]:
    data = _read_first_existing(
        storage,
        [
            f"{session_prefix(session)}/analysis/piano_voicing.json",
            f"{session_prefix(session)}/piano_voicing.json",
            f"{session_prefix(session)}/artifacts/piano_voicing.json",
        ],
    )
    if data is None:
        return {"error": "piano_voicing.json missing; run piano voicing analysis first"}

    rows = data.get("rows", [])
    measure = args.get("midi_measure") or args.get("measure")
    beat = args.get("beat")
    start = args.get("start_sec")
    end = args.get("end_sec")
    selected = rows
    if measure is not None:
        selected = [r for r in selected if int(r.get("measure", -999)) == int(measure)]
    if beat is not None:
        selected = [r for r in selected if r.get("beat") is not None and abs(float(r["beat"]) - float(beat)) <= 0.35]
    if start is not None:
        selected = [r for r in selected if float(r.get("perf_time", -1)) >= float(start)]
    if end is not None:
        selected = [r for r in selected if float(r.get("perf_time", 1e9)) <= float(end)]
    selected = sorted(selected, key=lambda r: float(r.get("perf_time", 0.0)))

    return {
        "query": args,
        "events_returned": min(len(selected), 24),
        "events_matched": len(selected),
        "method": "recording-derived CQT energy at score-aligned written piano pitches; not a reference MIDI visualization",
        "global_summary": data.get("summary", {}).get("global", {}),
        "measure_summary": [
            m for m in data.get("summary", {}).get("by_measure", [])
            if measure is None or int(m.get("measure", -999)) == int(measure)
        ][:8],
        "events": [_compact_event(r) for r in selected[:24]],
    }


def _read_first_existing(storage: ObjectStorage, keys: list[str]) -> dict[str, Any] | None:
    for key in keys:
        if storage.exists(key):
            return storage.read_json(key)
    return None


def _compact_event(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "measure": row.get("measure"),
        "beat": row.get("beat"),
        "time_sec": row.get("perf_time"),
        "notes": row.get("names"),
        "melody_note": row.get("melody_note"),
        "melody_margin_db": row.get("melody_margin_db"),
        "melody_projection": row.get("melody_projection"),
        "attack_spread_ms": row.get("attack_spread_ms"),
        "pedal_residue_db_rel": row.get("pedal_residue_db_rel"),
        "pedal_blur": row.get("pedal_blur"),
        "members": [
            {
                "name": m.get("name"),
                "role": m.get("role"),
                "onset_db_rel": m.get("onset_db_rel"),
                "present": m.get("present"),
                "attack_offset_ms": m.get("attack_offset_ms"),
            }
            for m in row.get("members", [])
        ],
    }

