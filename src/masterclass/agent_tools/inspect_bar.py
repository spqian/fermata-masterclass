from __future__ import annotations

from typing import Any

from masterclass.core.models import SessionRef
from masterclass.storage.base import ObjectStorage
from ._common import read_json

INSPECT_BAR_SCHEMA = {"type": "object", "properties": {"midi_measure": {"type": "integer"}}, "required": ["midi_measure"]}
DESCRIPTION = "All evidence for one bar. args: {midi_measure}"


def inspect_bar(storage: ObjectStorage, session: SessionRef, args: dict[str, Any]) -> dict[str, Any]:
    midi_measure = int(args["midi_measure"])
    sm = read_json(storage, session, "score/score_map.json", None)
    if sm is None:
        return {"error": "score/score_map.json missing"}
    bar = next((b for b in sm.get("bars", []) if int(b.get("midi_measure", b.get("measure", -999))) == midi_measure), None)
    if not bar:
        return {"error": f"no bar {midi_measure} in score_map"}
    notes = [n for n in sm.get("notes", []) if int(n.get("midi_measure", n.get("measure", -999))) == midi_measure]
    out: dict[str, Any] = {"bar": bar, "notes": notes, "n_notes": len(notes)}
    intn = read_json(storage, session, "analysis/polyphonic_intonation.json", {}) or {}
    bm = next((m for m in intn.get("summary", {}).get("by_measure", []) if int(m.get("measure", -999)) == midi_measure), None)
    if bm:
        out["intonation_summary"] = bm
    rows = [r for r in (intn.get("rows") or intn.get("events") or []) if int(r.get("measure", r.get("midi_measure", -999))) == midi_measure]
    if rows:
        out["intonation_rows"] = rows
    rhy = read_json(storage, session, "analysis/polyphonic_rhythm.json", {}) or {}
    bd = next((b for b in (rhy.get("summary", {}).get("bar_durations") or rhy.get("per_bar") or []) if int(b.get("bar", b.get("measure", -999))) == midi_measure), None)
    if bd:
        out["rhythm_bar_duration"] = bd
    outliers = [o for o in rhy.get("summary", {}).get("off_pulse_outliers", []) if int(o.get("measure", -999)) == midi_measure]
    if outliers:
        out["off_pulse_outliers"] = outliers
    # Audio-truth: the per-note timeline. Filter by measure so the teacher
    # sees only the notes inside this bar. The shape carries detected vs
    # score-expected pitch, timing offsets, and matched/unmatched status -
    # everything inspect_bar used to read from hmm.note_alignments plus the
    # score-matched cents-off-score field.
    audio_truth = read_json(session_storage_aware(storage), session, "analysis/audio_truth_matched_notes.json", {}) or {}
    if not audio_truth:
        audio_truth = read_json(session_storage_aware(storage), session, "analysis/audio_truth_notes.json", {}) or {}
    audio_notes = audio_truth.get("notes") if isinstance(audio_truth, dict) else None
    ps = bar.get("perf_start_sec", bar.get("performed_start_sec"))
    pe = bar.get("perf_end_sec", bar.get("performed_end_sec"))
    if audio_notes and ps is not None and pe is not None:
        in_bar = [
            n for n in audio_notes
            if isinstance(n, dict)
            and (int(n.get("measure") or -999) == midi_measure
                 or (float(ps) - 0.2 <= float(n.get("performed_time_sec", n.get("perf_time", -1))) <= float(pe) + 0.2))
        ]
        if in_bar:
            out["audio_truth_notes"] = in_bar
        ro = read_json(storage, session, "analysis/rich_onsets.json", {}) or {}
        ons = [o for o in ro.get("onsets", []) if float(ps) <= float(o.get("time", o.get("time_sec", -999))) <= float(pe)]
        if ons:
            out["rich_onsets"] = ons
    return out


# Compat shim: read_json takes a storage and a session; the upstream helper
# already does the right thing, this just keeps the inspect_bar signature
# self-explanatory after the rename.
def session_storage_aware(storage):  # pragma: no cover - trivial passthrough
    return storage
