"""Single accessor for "the lesson's per-note timeline".

After the audio-truth refactor there is exactly one canonical source for
per-note data: analysis/audio_truth_matched_notes.json (or analysis/
audio_truth_notes.json if score-matching couldn't run). Every consumer
that used to read analysis/hmm_aligned_notes.json or analysis/
hmm_alignment.json should call load_aligned_notes() here instead, so
there is one chokepoint to evolve when the schema changes.

The returned shape is the *enriched* note dict produced by
audio_truth._enrich_for_legacy_consumers (or, for the score_matched
file directly, the matcher's output). Keys present on every row:

    state_idx           int      stable identifier for cross-artifact join
    pitches_midi        [int]    detected MIDI pitches (basic-pitch is mono-per-note;
                                 PTI is mono-per-note; both wrap in a single-elt list)
    names               [str]    NOTE_NAMES-formatted, e.g. "C5"
    performed_time_sec  float    onset time in seconds within the recording
    perf_time           float    alias of performed_time_sec for legacy callers
    dwell_sec           float    note duration
    confidence          str      "high" / "medium" / "low"
    timestamp_source    str      tells you which transcriber produced the row
    matched             bool     true when the note was matched against the score

When matched=True (the common case), these additional fields are populated:

    measure             int      score measure number (1-based)
    staff_index         int      0 = treble/right hand, 1 = bass/left hand
    track_name          str      MIDI/MusicXML track label
    score_time_sec      float    score-time the note was supposed to land at
    score_midi_pitch    int      score-expected pitch
    timing_offset_ms    float    performed - score time, signed (positive = late)
    score_time_in_movement  float  alias for score_time_sec (legacy callers)
    score_time_local        float  alias for score_time_sec
    expected_pitch          int    alias for score_midi_pitch
    expected_pitch_name     str   e.g. "C5"
    obs_log_prob            float alias for amplitude/velocity (legacy quality gate)
"""
from __future__ import annotations

from typing import Any

from masterclass.core.models import SessionManifest
from masterclass.storage.base import ObjectStorage


# Order matters: prefer the score-matched output, fall back to the raw
# transcription, and only as a last resort the old HMM-shim from
# audio_truth._build_legacy_hmm_artifacts (which the production pipeline
# also writes for now). When the shim is deleted, this list shrinks.
_CANDIDATE_KEYS: tuple[str, ...] = (
    "analysis/audio_truth_matched_notes.json",
    "analysis/audio_truth_notes.json",
    "analysis/hmm_aligned_notes.json",
)


def _enrich_for_legacy_consumers(notes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Add legacy field aliases consumers (rhythm, intonation, inspect_*) read.

    The audio-truth schema uses ``score_time_sec`` / ``score_midi_pitch``,
    while the old HMM code used ``score_time_in_movement`` / ``expected_pitch``.
    Rather than touch every reader, normalise here so every consumer sees
    both names. Idempotent — if both names already exist we keep the values.
    """
    out: list[dict[str, Any]] = []
    for n in notes:
        if not isinstance(n, dict):
            continue
        enriched = dict(n)
        st = enriched.get("score_time_sec")
        if st is not None:
            enriched.setdefault("score_time_in_movement", float(st))
            enriched.setdefault("score_time_local", float(st))
        sp = enriched.get("score_midi_pitch")
        if sp is not None:
            enriched.setdefault("expected_pitch", int(sp))
        enriched.setdefault("perf_time", enriched.get("performed_time_sec"))
        out.append(enriched)
    return out


def load_aligned_notes(storage: ObjectStorage, manifest: SessionManifest) -> list[dict[str, Any]]:
    """Return the canonical per-note list for this lesson.

    Always returns a list (possibly empty). Raises nothing; callers that
    need to fail when there are no notes should check the result.
    """
    for key_name in _CANDIDATE_KEYS:
        key = manifest.artifacts.get(key_name)
        if not key or not storage.exists(key):
            continue
        try:
            doc = storage.read_json(key)
        except (FileNotFoundError, ValueError, TypeError):
            continue
        if isinstance(doc, list):
            return _enrich_for_legacy_consumers([n for n in doc if isinstance(n, dict)])
        notes = doc.get("notes") if isinstance(doc, dict) else None
        if isinstance(notes, list):
            return _enrich_for_legacy_consumers([n for n in notes if isinstance(n, dict)])
    return []


def load_aligned_notes_source(storage: ObjectStorage, manifest: SessionManifest) -> tuple[str, list[dict[str, Any]]]:
    """Same as :func:`load_aligned_notes` but also returns which artifact won.

    Useful for logging and for the technical-viewer "method" badge.
    """
    for key_name in _CANDIDATE_KEYS:
        key = manifest.artifacts.get(key_name)
        if not key or not storage.exists(key):
            continue
        try:
            doc = storage.read_json(key)
        except (FileNotFoundError, ValueError, TypeError):
            continue
        if isinstance(doc, list):
            return key_name, _enrich_for_legacy_consumers([n for n in doc if isinstance(n, dict)])
        notes = doc.get("notes") if isinstance(doc, dict) else None
        if isinstance(notes, list):
            return key_name, _enrich_for_legacy_consumers([n for n in notes if isinstance(n, dict)])
    return "", []


def load_measure_starts(storage: ObjectStorage, manifest: SessionManifest) -> list[dict[str, Any]]:
    """Return [{measure: int, start: float}] from the aligned notes.

    Replaces the old read-from-hmm_alignment.bar_starts path. We derive
    bar starts from the first matched-note in each measure -- this is the
    same approach the audio_truth legacy shim took, and it's good enough
    for the consumers that need a per-measure anchor (rhythm, score_map).
    """
    notes = load_aligned_notes(storage, manifest)
    by_measure: dict[int, float] = {}
    for n in notes:
        m = n.get("measure")
        if m is None:
            continue
        t = n.get("performed_time_sec") or n.get("perf_time")
        if t is None:
            continue
        if m not in by_measure or float(t) < by_measure[m]:
            by_measure[m] = float(t)
    return [{"measure": m, "start": t} for m, t in sorted(by_measure.items())]
