from __future__ import annotations

from typing import Any


def candidate_notes_for_window(
    notes: list[dict],
    start_sec: float,
    end_sec: float,
    *,
    pad_sec: float = 0.5,
    max_count: int = 24,
) -> list[dict]:
    """Return score notes that fall in or near a performance-time window."""

    start = float(start_sec) - float(pad_sec)
    end = float(end_sec) + float(pad_sec)
    out: list[dict[str, Any]] = []
    for note in notes:
        perf_time = note.get("perf_time")
        if perf_time is None:
            continue
        perf_time_f = float(perf_time)
        if start <= perf_time_f <= end:
            out.append({
                "note_id": note.get("note_id"),
                "midi_measure": note.get("midi_measure"),
                "beat": note.get("beat_in_bar"),
                "names": note.get("names"),
                "perf_time": round(perf_time_f, 2),
                "is_chord": bool(note.get("is_chord")),
                "hmm_confidence": note.get("hmm_confidence"),
            })
    return out[:max_count]


def attach_candidate_notes(
    comments: list[dict],
    score_map: dict,
    *,
    pad_sec: float = 0.5,
) -> list[dict]:
    """Return copied comments with PoC-compatible candidate_notes attached."""

    notes = score_map.get("notes", [])
    out: list[dict] = []
    for comment in comments:
        copied = dict(comment)
        start = _comment_time(comment, "start_sec", "start", default=0.0)
        end = _comment_time(comment, "end_sec", "end", default=start + 1.0)
        copied["candidate_notes"] = candidate_notes_for_window(notes, start, end, pad_sec=pad_sec)
        out.append(copied)
    return out


def _comment_time(comment: dict, primary: str, fallback: str, *, default: float) -> float:
    value = comment.get(primary, comment.get(fallback, default))
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)
