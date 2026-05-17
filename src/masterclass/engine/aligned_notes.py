"""Single accessor for "the lesson's per-note timeline".

After the audio-truth refactor there is exactly one canonical source for
per-note data: analysis/audio_truth_matched_notes.json (or analysis/
audio_truth_notes.json if score-matching couldn't run). Every consumer
that used to read analysis/hmm_aligned_notes.json or analysis/
hmm_alignment.json should call :func:`load_aligned_notes` here instead,
so there is one chokepoint to evolve when the schema changes.

The returned type is the typed :class:`AlignedNote` dataclass. Consumers
read attributes (``note.score_time_sec``) instead of dict keys, which
gives them static checking and removes the need for the historic field
alias shim. :meth:`AlignedNote.from_dict` still accepts the legacy field
names (``score_time_in_movement``, ``score_time_local``, ``expected_pitch``,
``perf_time``) so older on-disk artifacts written by
:func:`audio_truth._build_legacy_hmm_artifacts` keep loading during the
transition window.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Optional

from masterclass.core.models import SessionManifest, SessionRef, session_prefix
from masterclass.storage.base import ObjectStorage


# Order matters: prefer the score-matched output, fall back to the raw
# transcription, then the new vocabulary-clean legacy-shim
# (``analysis/aligned_notes.json``), and only as a last resort the original
# HMM-named legacy shim. The production pipeline currently writes the same
# document to both ``aligned_notes.json`` (new) and ``hmm_aligned_notes.json``
# (old) -- see ``engine/audio_truth.py::_build_legacy_hmm_artifacts``.
#
# Deprecation timeline:
#   * R+0 (current): write both shim names, read both shim names.
#   * R+1: stop writing ``analysis/hmm_aligned_notes.json``; keep reading
#          it so older sessions on disk still load.
#   * R+2: drop ``analysis/hmm_aligned_notes.json`` from this tuple; the
#          legacy artifact and ``_build_legacy_hmm_artifacts`` go away in
#          the same release.
_CANDIDATE_KEYS: tuple[str, ...] = (
    "analysis/audio_truth_matched_notes.json",
    "analysis/audio_truth_notes.json",
    "analysis/aligned_notes.json",
    "analysis/hmm_aligned_notes.json",
)


@dataclass(frozen=True)
class AlignedNote:
    """Typed contract for one row of the lesson's per-note timeline.

    Fields below the ``matched`` line are populated only when the matcher
    paired this detected note with a score event. Unmatched notes leave
    them ``None``.
    """

    state_idx: int
    pitches_midi: list[int]
    names: list[str]
    performed_time_sec: float
    dwell_sec: float
    amplitude: float
    confidence: str  # "high" / "medium" / "low"
    matched: bool
    # Score-matched fields (None when matched=False)
    measure: Optional[int] = None
    staff_index: Optional[int] = None
    track_name: Optional[str] = None
    score_time_sec: Optional[float] = None
    score_midi_pitch: Optional[int] = None
    timing_offset_ms: Optional[float] = None
    expected_perf_duration: Optional[float] = None
    # True when the matcher anchored this note to an OMR-empty-measure ghost
    # (synthesized because Audiveris dropped notes from a dense ornament
    # measure). We know the measure, but the score pitch is unknown, so
    # downstream consumers must skip cents-off / intonation comparisons.
    matched_to_wildcard: bool = False

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "AlignedNote":
        """Parse one row, accepting both audio-truth and legacy HMM field names.

        Legacy aliases:
            ``score_time_in_movement`` / ``score_time_local`` -> ``score_time_sec``
            ``expected_pitch``                                -> ``score_midi_pitch``
            ``perf_time``                                     -> ``performed_time_sec``
        """
        score_time_sec = _first_not_none(
            d.get("score_time_sec"),
            d.get("score_time_in_movement"),
            d.get("score_time_local"),
        )
        score_midi_pitch = _first_not_none(
            d.get("score_midi_pitch"),
            d.get("expected_pitch"),
        )
        performed = _first_not_none(
            d.get("performed_time_sec"),
            d.get("perf_time"),
        )

        pitches_raw = d.get("pitches_midi") or d.get("pitches") or []
        if isinstance(pitches_raw, (int, float)):
            pitches_raw = [pitches_raw]
        pitches_midi = [int(p) for p in pitches_raw if p is not None]

        names_raw = d.get("names") or []
        if isinstance(names_raw, str):
            names_raw = [names_raw]
        names = [str(n) for n in names_raw]

        return cls(
            state_idx=int(d.get("state_idx") or 0),
            pitches_midi=pitches_midi,
            names=names,
            performed_time_sec=float(performed) if performed is not None else 0.0,
            dwell_sec=float(d.get("dwell_sec") or 0.0),
            amplitude=float(d.get("amplitude") or 0.0),
            confidence=str(d.get("confidence") or "low"),
            matched=bool(d.get("matched", False)),
            measure=int(d["measure"]) if d.get("measure") is not None else None,
            staff_index=int(d["staff_index"]) if d.get("staff_index") is not None else None,
            track_name=str(d["track_name"]) if d.get("track_name") is not None else None,
            score_time_sec=float(score_time_sec) if score_time_sec is not None else None,
            score_midi_pitch=int(score_midi_pitch) if score_midi_pitch is not None else None,
            timing_offset_ms=float(d["timing_offset_ms"]) if d.get("timing_offset_ms") is not None else None,
            expected_perf_duration=float(d["expected_perf_duration"]) if d.get("expected_perf_duration") is not None else None,
            matched_to_wildcard=bool(d.get("matched_to_wildcard", False)),
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize back to a plain dict using canonical field names.

        Legacy aliases (``score_time_in_movement`` etc.) are *not* emitted;
        consumers that still want them should read the canonical names.
        """
        return asdict(self)


def _first_not_none(*values: Any) -> Any:
    for v in values:
        if v is not None:
            return v
    return None


def _parse_notes(doc: Any) -> list[AlignedNote]:
    """Extract a typed note list from either a top-level array or a
    ``{"notes": [...]}`` envelope. Non-dict rows are skipped silently."""
    rows: Any
    if isinstance(doc, list):
        rows = doc
    elif isinstance(doc, dict):
        rows = doc.get("notes")
    else:
        return []
    if not isinstance(rows, list):
        return []
    return [AlignedNote.from_dict(r) for r in rows if isinstance(r, dict)]


def _load_via_resolver(
    storage: ObjectStorage, resolver: Callable[[str], Optional[str]]
) -> tuple[str, list[AlignedNote]]:
    """Try each candidate key in priority order via ``resolver`` (which
    maps a relative artifact key to a full storage key, or returns None).
    Returns ``(winning_rel_key, notes)`` or ``("", [])``."""
    for rel_key in _CANDIDATE_KEYS:
        full_key = resolver(rel_key)
        if not full_key:
            continue
        if not storage.exists(full_key):
            continue
        try:
            doc = storage.read_json(full_key)
        except (FileNotFoundError, ValueError, TypeError):
            continue
        notes = _parse_notes(doc)
        # Even an empty list counts as "this artifact won"; an empty
        # artifact is informative (the matcher ran and found nothing).
        # But to match the old behaviour we keep walking only on read
        # failures, not on empties.
        return rel_key, notes
    return "", []


def load_aligned_notes(storage: ObjectStorage, manifest: SessionManifest) -> list[AlignedNote]:
    """Return the canonical per-note list for this lesson.

    Always returns a list (possibly empty). Raises nothing; callers that
    need to fail when there are no notes should check the result.
    """
    _, notes = _load_via_resolver(storage, manifest.artifacts.get)
    return notes


def load_aligned_notes_source(
    storage: ObjectStorage, manifest: SessionManifest
) -> tuple[str, list[AlignedNote]]:
    """Same as :func:`load_aligned_notes` but also returns which artifact
    won. Useful for logging and for the technical-viewer "method" badge."""
    return _load_via_resolver(storage, manifest.artifacts.get)


def load_aligned_notes_for_session(
    storage: ObjectStorage, session: SessionRef
) -> list[AlignedNote]:
    """Session-scoped variant for callers that hold a :class:`SessionRef`
    but no manifest (the ``agent_tools/*`` inspectors)."""
    prefix = session_prefix(session)

    def _resolver(rel: str) -> Optional[str]:
        return f"{prefix}/{rel}"

    _, notes = _load_via_resolver(storage, _resolver)
    return notes


def load_measure_starts(
    storage: ObjectStorage, manifest: SessionManifest
) -> list[dict[str, Any]]:
    """Return ``[{measure: int, start: float}]`` derived from the aligned
    notes. Replaces the old read-from-``hmm_alignment.bar_starts`` path;
    we use the first matched-note in each measure as the bar anchor.
    """
    notes = load_aligned_notes(storage, manifest)
    by_measure: dict[int, float] = {}
    for n in notes:
        if n.measure is None:
            continue
        t = n.performed_time_sec
        if n.measure not in by_measure or t < by_measure[n.measure]:
            by_measure[n.measure] = float(t)
    return [{"measure": m, "start": t} for m, t in sorted(by_measure.items())]
