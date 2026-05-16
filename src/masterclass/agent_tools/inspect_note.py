from __future__ import annotations

from typing import Any

from masterclass.core.models import SessionRef
from masterclass.storage.base import ObjectStorage
from ._common import find_score_note, note_perf_time, note_score_time, read_json

INSPECT_NOTE_SCHEMA = {"type": "object", "properties": {"midi_measure": {"type": "integer"}, "beat": {"type": "number"}, "pitch": {"type": "string"}}, "required": ["midi_measure"]}
DESCRIPTION = "Full evidence for one note. args: {midi_measure, beat?, pitch?}"


def inspect_note(storage: ObjectStorage, session: SessionRef, args: dict[str, Any]) -> dict[str, Any]:
    midi_measure = int(args["midi_measure"]); beat = float(args["beat"]) if "beat" in args else None; pitch = args.get("pitch")
    sm = read_json(storage, session, "score/score_map.json", None)
    if sm is None:
        return {"error": "score/score_map.json missing"}
    note = find_score_note(sm, midi_measure, beat, pitch)
    if not note:
        return {"error": f"no note at m{midi_measure} beat={beat} pitch={pitch}"}
    out: dict[str, Any] = {"score_map_note": note}
    st = note_score_time(note)
    # Audio-truth match: find the detected note whose matched score-time
    # closest matches the requested score-time. The matcher stores the
    # score time the note was matched against under score_time_sec; legacy
    # consumers also looked under score_time_in_movement / score_time_local
    # which the shim aliases.
    at = read_json(storage, session, "analysis/audio_truth_matched_notes.json", {}) or {}
    if not at:
        at = read_json(storage, session, "analysis/audio_truth_notes.json", {}) or {}
    at_notes = at.get("notes") if isinstance(at, dict) else None
    if at_notes and st is not None:
        # Prefer matched-by-measure-and-pitch; fall back to nearest-score-time.
        target_pitch = note.get("pitch_midi") or note.get("midi_pitch")
        match = None
        for n in at_notes:
            if not isinstance(n, dict):
                continue
            if int(n.get("measure") or -999) != midi_measure:
                continue
            score_t = n.get("score_time_sec") or n.get("score_time_in_movement")
            if score_t is None or abs(float(score_t) - st) > 0.05:
                continue
            if target_pitch is not None and n.get("score_midi_pitch") is not None:
                if int(n["score_midi_pitch"]) != int(target_pitch):
                    continue
            match = n
            break
        if match:
            out["audio_truth"] = {k: match.get(k) for k in (
                "performed_time_sec", "dwell_sec", "amplitude", "confidence",
                "names", "pitches_midi", "state_idx",
                "score_time_sec", "score_midi_pitch", "timing_offset_ms",
                "matched", "staff_index",
            ) if k in match}
    intn = read_json(storage, session, "analysis/polyphonic_intonation.json", {}) or {}
    rows = intn.get("rows") or intn.get("events") or []
    if st is not None:
        found = [r for r in rows if int(r.get("measure", r.get("midi_measure", -999))) == midi_measure and abs(float(r.get("score_time", r.get("score_time_sec", r.get("score_time_in_movement", -1)))) - st) < 0.02]
        if found:
            out["intonation"] = found[0]
    pt = note_perf_time(note)
    ro = read_json(storage, session, "analysis/rich_onsets.json", {}) or {}
    if pt is not None:
        nearby = [o for o in ro.get("onsets", []) if abs(float(o.get("time", o.get("time_sec", -999))) - pt) < 0.2]
        if nearby:
            out["nearby_rich_onsets"] = nearby
    return out
