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
    # Scope to the played range: refuse to look up a note in a measure the
    # player never played. Without this gate the teacher agent can ask
    # about m.36 in an m.1-8 lesson and get back stale matcher noise.
    from masterclass.core.played_range import derive_played_range_from_score_map
    played_range = derive_played_range_from_score_map(sm)
    if played_range is not None and not played_range.contains(midi_measure):
        return {
            "error": (
                f"measure {midi_measure} is outside the played range "
                f"{played_range.label()} (source={played_range.source}); "
                "inspect_note only accepts measures the player actually played."
            ),
            "played_range": {
                "first_measure": played_range.first_measure,
                "last_measure": played_range.last_measure,
                "source": played_range.source,
            },
        }
    note = find_score_note(sm, midi_measure, beat, pitch)
    if not note:
        return {"error": f"no note at m{midi_measure} beat={beat} pitch={pitch}"}
    out: dict[str, Any] = {"score_map_note": note}
    st = note_score_time(note)
    # Audio-truth match: find the detected note whose matched score-time
    # closest matches the requested score-time. Routed through the typed
    # aligned-notes accessor (matched > raw > hmm shim) so we see the
    # canonical schema (score_time_sec / score_midi_pitch) regardless of
    # which on-disk artifact won.
    from masterclass.engine.aligned_notes import load_aligned_notes_for_session
    at_notes = [n.to_dict() for n in load_aligned_notes_for_session(storage, session)]
    if at_notes and st is not None:
        # Prefer matched-by-measure-and-pitch; fall back to nearest-score-time.
        target_pitch = note.get("pitch_midi") or note.get("midi_pitch")
        match = None
        for n in at_notes:
            if int(n.get("measure") or -999) != midi_measure:
                continue
            score_t = n.get("score_time_sec")
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
