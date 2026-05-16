"""Tests for the typed aligned-notes contract.

Covers:
* :meth:`AlignedNote.from_dict` round-trips both the canonical audio-truth
  schema and the legacy HMM field names.
* :func:`load_aligned_notes` priority order (matched > raw > hmm shim).
* Empty / missing manifest returns ``[]`` instead of raising.
* Malformed JSON survives (returns ``[]``).
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from masterclass.core.models import SessionManifest, SessionRef
from masterclass.engine.aligned_notes import (
    AlignedNote,
    load_aligned_notes,
    load_aligned_notes_source,
    load_measure_starts,
)
from masterclass.storage.local import LocalObjectStorage


CANONICAL_ROW = {
    "state_idx": 5,
    "pitches_midi": [60, 64],
    "names": ["C4", "E4"],
    "performed_time_sec": 1.234,
    "dwell_sec": 0.5,
    "amplitude": 0.7,
    "confidence": "high",
    "matched": True,
    "measure": 3,
    "staff_index": 0,
    "track_name": "RH",
    "score_time_sec": 1.0,
    "score_midi_pitch": 60,
    "timing_offset_ms": 234.0,
    "expected_perf_duration": 0.5,
}

LEGACY_ROW = {
    "state_idx": 5,
    "pitches_midi": [60, 64],
    "names": ["C4", "E4"],
    "perf_time": 1.234,
    "dwell_sec": 0.5,
    "amplitude": 0.7,
    "confidence": "high",
    "matched": True,
    "measure": 3,
    "staff_index": 0,
    "track_name": "RH",
    "score_time_in_movement": 1.0,
    "score_time_local": 1.0,
    "expected_pitch": 60,
    "timing_offset_ms": 234.0,
}


def _manifest_for(storage_root: Path, artifacts: dict[str, str]) -> SessionManifest:
    session = SessionRef(tenant_id="t", user_id="u", session_id="s1")
    return SessionManifest(
        schema_version=1,
        session=session,
        artifacts=dict(artifacts),
    )


def _write_notes_doc(storage: LocalObjectStorage, key: str, rows: list[dict]) -> None:
    storage.write_json(key, {"notes": rows})


def test_from_dict_canonical_schema():
    note = AlignedNote.from_dict(CANONICAL_ROW)
    assert note.state_idx == 5
    assert note.score_time_sec == 1.0
    assert note.score_midi_pitch == 60
    assert note.performed_time_sec == 1.234
    assert note.matched is True
    assert note.measure == 3


def test_from_dict_legacy_aliases():
    note = AlignedNote.from_dict(LEGACY_ROW)
    # Legacy field names are normalised to canonical.
    assert note.score_time_sec == 1.0
    assert note.score_midi_pitch == 60
    assert note.performed_time_sec == 1.234


def test_from_dict_roundtrip_canonical():
    note = AlignedNote.from_dict(CANONICAL_ROW)
    again = AlignedNote.from_dict(note.to_dict())
    assert again == note


def test_from_dict_roundtrip_legacy_to_canonical():
    """Legacy input -> canonical dict -> identical AlignedNote."""
    note = AlignedNote.from_dict(LEGACY_ROW)
    assert "score_time_in_movement" not in note.to_dict()
    again = AlignedNote.from_dict(note.to_dict())
    assert again == note


def test_from_dict_unmatched_note_has_none_score_fields():
    raw = {
        "state_idx": 1,
        "pitches_midi": [72],
        "names": ["C5"],
        "performed_time_sec": 0.1,
        "dwell_sec": 0.2,
        "amplitude": 0.5,
        "confidence": "low",
        "matched": False,
    }
    note = AlignedNote.from_dict(raw)
    assert note.matched is False
    assert note.score_time_sec is None
    assert note.score_midi_pitch is None
    assert note.measure is None
    assert note.staff_index is None


def test_load_aligned_notes_priority_matched_over_raw(tmp_path: Path):
    storage = LocalObjectStorage(tmp_path)
    matched_key = "tenant/user/session/analysis/audio_truth_matched_notes.json"
    raw_key = "tenant/user/session/analysis/audio_truth_notes.json"
    _write_notes_doc(storage, matched_key, [CANONICAL_ROW])
    _write_notes_doc(storage, raw_key, [{**CANONICAL_ROW, "state_idx": 999}])

    manifest = _manifest_for(tmp_path, {
        "analysis/audio_truth_matched_notes.json": matched_key,
        "analysis/audio_truth_notes.json": raw_key,
    })
    notes = load_aligned_notes(storage, manifest)
    assert len(notes) == 1
    assert notes[0].state_idx == 5  # matched, not 999


def test_load_aligned_notes_priority_raw_over_hmm_shim(tmp_path: Path):
    storage = LocalObjectStorage(tmp_path)
    raw_key = "tenant/user/session/analysis/audio_truth_notes.json"
    hmm_key = "tenant/user/session/analysis/hmm_aligned_notes.json"
    _write_notes_doc(storage, raw_key, [CANONICAL_ROW])
    _write_notes_doc(storage, hmm_key, [LEGACY_ROW])

    manifest = _manifest_for(tmp_path, {
        "analysis/audio_truth_notes.json": raw_key,
        "analysis/hmm_aligned_notes.json": hmm_key,
    })
    src, notes = load_aligned_notes_source(storage, manifest)
    assert src == "analysis/audio_truth_notes.json"
    assert len(notes) == 1


def test_load_aligned_notes_falls_back_to_hmm_shim(tmp_path: Path):
    """When only the legacy hmm artifact exists, load_aligned_notes still
    returns notes (now typed) instead of an empty list."""
    storage = LocalObjectStorage(tmp_path)
    hmm_key = "tenant/user/session/analysis/hmm_aligned_notes.json"
    _write_notes_doc(storage, hmm_key, [LEGACY_ROW])

    manifest = _manifest_for(tmp_path, {
        "analysis/hmm_aligned_notes.json": hmm_key,
    })
    src, notes = load_aligned_notes_source(storage, manifest)
    assert src == "analysis/hmm_aligned_notes.json"
    assert len(notes) == 1
    # Legacy aliases normalised away.
    assert notes[0].score_time_sec == 1.0


def test_load_aligned_notes_empty_manifest_returns_empty_list(tmp_path: Path):
    storage = LocalObjectStorage(tmp_path)
    manifest = _manifest_for(tmp_path, {})
    assert load_aligned_notes(storage, manifest) == []
    src, notes = load_aligned_notes_source(storage, manifest)
    assert src == ""
    assert notes == []


def test_load_aligned_notes_malformed_json_returns_empty(tmp_path: Path):
    storage = LocalObjectStorage(tmp_path)
    key = "tenant/user/session/analysis/audio_truth_matched_notes.json"
    storage.write_bytes(key, b"{not valid json", content_type="application/json")
    manifest = _manifest_for(tmp_path, {
        "analysis/audio_truth_matched_notes.json": key,
    })
    # Should not raise.
    assert load_aligned_notes(storage, manifest) == []


def test_load_aligned_notes_top_level_array_supported(tmp_path: Path):
    storage = LocalObjectStorage(tmp_path)
    key = "tenant/user/session/analysis/audio_truth_matched_notes.json"
    storage.write_json(key, [CANONICAL_ROW])
    manifest = _manifest_for(tmp_path, {
        "analysis/audio_truth_matched_notes.json": key,
    })
    notes = load_aligned_notes(storage, manifest)
    assert len(notes) == 1
    assert notes[0].state_idx == 5


def test_load_measure_starts(tmp_path: Path):
    storage = LocalObjectStorage(tmp_path)
    key = "tenant/user/session/analysis/audio_truth_matched_notes.json"
    _write_notes_doc(storage, key, [
        {**CANONICAL_ROW, "measure": 3, "performed_time_sec": 1.5},
        {**CANONICAL_ROW, "measure": 3, "performed_time_sec": 1.2},  # earlier
        {**CANONICAL_ROW, "measure": 5, "performed_time_sec": 3.0},
        # unmatched note (no measure) is ignored
        {**CANONICAL_ROW, "measure": None, "matched": False, "performed_time_sec": 2.0},
    ])
    manifest = _manifest_for(tmp_path, {
        "analysis/audio_truth_matched_notes.json": key,
    })
    starts = load_measure_starts(storage, manifest)
    assert starts == [
        {"measure": 3, "start": 1.2},
        {"measure": 5, "start": 3.0},
    ]
