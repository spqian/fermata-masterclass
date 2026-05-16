"""Shared pytest helpers for the music-masterclass-v2 regression suite.

These fixtures intentionally use *real* engine code paths against tiny
synthetic inputs so that the tests catch the kind of plumbing/contract
bugs that produced the 7-bug audio-truth migration.
"""
from __future__ import annotations

import io
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from scipy.io import wavfile

from masterclass.core.masterclasses import MasterclassStore
from masterclass.core.models import (
    MasterclassManifest,
    MasterclassRef,
    SessionManifest,
    SessionRef,
    TenantContext,
)
from masterclass.core.sessions import SessionStore
from masterclass.storage.local import LocalObjectStorage


# --------------------------------------------------------------------------
# Storage / store fixtures
# --------------------------------------------------------------------------

@pytest.fixture
def local_storage(tmp_path: Path) -> LocalObjectStorage:
    root = tmp_path / "storage"
    root.mkdir(parents=True, exist_ok=True)
    return LocalObjectStorage(root)


@pytest.fixture
def session_store(local_storage: LocalObjectStorage) -> SessionStore:
    return SessionStore(local_storage)


@pytest.fixture
def masterclass_store(local_storage: LocalObjectStorage) -> MasterclassStore:
    return MasterclassStore(local_storage)


@pytest.fixture
def tenant_ctx() -> TenantContext:
    return TenantContext(tenant_id="t1", user_id="u1")


# --------------------------------------------------------------------------
# Manifest builders
# --------------------------------------------------------------------------

def make_session_manifest(
    storage: LocalObjectStorage,
    store: SessionStore,
    ctx: TenantContext,
    *,
    artifacts: dict[str, Any] | None = None,
    metadata: dict[str, Any] | None = None,
    repertoire: str = "Test Piece",
    movement: str | None = "I",
    instrument: str | None = "violin",
) -> SessionManifest:
    """Create + persist a SessionManifest with the supplied artifacts.

    ``artifacts`` maps a relative key (e.g. ``"artifacts/audio.wav"``)
    to bytes, or to a JSON-serialisable object when the key ends in
    ``.json``.
    """
    ref = SessionRef.new(ctx)
    manifest = SessionManifest(
        schema_version=1,
        session=ref,
        repertoire=repertoire,
        movement=movement,
        instrument=instrument,
        metadata=dict(metadata or {}),
    )
    for rel, data in (artifacts or {}).items():
        key = store.artifact_key(ref, rel)
        if rel.endswith(".json") and isinstance(data, (dict, list)):
            storage.write_json(key, data)
        else:
            storage.write_bytes(key, data)
        manifest.artifacts[rel] = key
    store.save(manifest)
    return manifest


def make_masterclass_manifest(
    storage: LocalObjectStorage,
    masterclass_store: MasterclassStore,
    ctx: TenantContext,
    *,
    artifacts: dict[str, Any] | None = None,
    piece_name: str = "Test Piece",
    movement: str | None = "I",
    instrument: str | None = "violin",
) -> MasterclassManifest:
    ref = MasterclassRef.new(ctx)
    manifest = MasterclassManifest(
        schema_version=1,
        masterclass=ref,
        piece_name=piece_name,
        movement=movement,
        instrument=instrument,
    )
    for rel, data in (artifacts or {}).items():
        key = masterclass_store.artifact_key(ref, rel)
        if rel.endswith(".json") and isinstance(data, (dict, list)):
            storage.write_json(key, data)
        else:
            storage.write_bytes(key, data)
        manifest.artifacts[rel] = key
    masterclass_store.save(manifest)
    return manifest


# --------------------------------------------------------------------------
# Audio / score fixtures (plain helpers, callable from tests)
# --------------------------------------------------------------------------

def tiny_audio_wav(
    *,
    freq_hz: float = 440.0,
    duration_sec: float = 2.0,
    sample_rate: int = 22050,
    amplitude: float = 0.5,
) -> bytes:
    """Return WAV bytes for a constant sine tone."""
    n = int(round(duration_sec * sample_rate))
    t = np.arange(n) / sample_rate
    y = (amplitude * np.sin(2.0 * np.pi * freq_hz * t)).astype(np.float32)
    buf = io.BytesIO()
    wavfile.write(buf, sample_rate, y)
    return buf.getvalue()


def tiny_musicxml(
    *,
    pitches: list[tuple[str, int, int]] | None = None,
    divisions: int = 4,
    tempo_qpm: float = 120.0,
    measures: int = 1,
) -> bytes:
    """Return MusicXML bytes for a single-part stub piece.

    ``pitches`` is a list of ``(step, alter, octave)`` tuples placed as
    consecutive quarter notes split across ``measures`` measures.
    Defaults to a C-major fragment.
    """
    notes = pitches or [("C", 0, 4), ("D", 0, 4), ("E", 0, 4), ("F", 0, 4)]
    # split notes evenly across measures
    per_measure = max(1, len(notes) // measures)
    measure_chunks: list[list[tuple[str, int, int]]] = []
    for i in range(measures):
        start = i * per_measure
        end = (i + 1) * per_measure if i < measures - 1 else len(notes)
        measure_chunks.append(notes[start:end] or [("C", 0, 4)])

    def _note_xml(step: str, alter: int, octave: int) -> str:
        alter_xml = f"<alter>{alter}</alter>" if alter else ""
        return (
            f"        <note>\n"
            f"          <pitch><step>{step}</step>{alter_xml}<octave>{octave}</octave></pitch>\n"
            f"          <duration>{divisions}</duration>\n"
            f"          <voice>1</voice>\n"
            f"          <type>quarter</type>\n"
            f"          <staff>1</staff>\n"
            f"        </note>"
        )

    measures_xml = []
    for idx, chunk in enumerate(measure_chunks, start=1):
        notes_xml = "\n".join(_note_xml(s, a, o) for (s, a, o) in chunk)
        if idx == 1:
            attrs = (
                f"      <attributes>\n"
                f"        <divisions>{divisions}</divisions>\n"
                f"        <key><fifths>0</fifths></key>\n"
                f"        <time><beats>4</beats><beat-type>4</beat-type></time>\n"
                f"        <clef><sign>G</sign><line>2</line></clef>\n"
                f"      </attributes>\n"
                f"      <direction placement=\"above\">\n"
                f"        <direction-type><metronome><beat-unit>quarter</beat-unit><per-minute>{int(tempo_qpm)}</per-minute></metronome></direction-type>\n"
                f"        <sound tempo=\"{tempo_qpm}\"/>\n"
                f"      </direction>\n"
            )
        else:
            attrs = ""
        measures_xml.append(
            f"    <measure number=\"{idx}\">\n{attrs}{notes_xml}\n    </measure>"
        )
    measures_block = "\n".join(measures_xml)
    xml = (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<!DOCTYPE score-partwise PUBLIC "-//Recordare//DTD MusicXML 3.1 Partwise//EN" '
        '"http://www.musicxml.org/dtds/partwise.dtd">\n'
        '<score-partwise version="3.1">\n'
        '  <part-list>\n'
        '    <score-part id="P1"><part-name>Test</part-name></score-part>\n'
        '  </part-list>\n'
        '  <part id="P1">\n'
        f'{measures_block}\n'
        '  </part>\n'
        '</score-partwise>\n'
    )
    return xml.encode("utf-8")


def audio_truth_matched_notes(
    *,
    matched_pitches_by_measure: dict[int, list[int]] | None = None,
    include_unmatched: bool = True,
    perf_step_sec: float = 0.5,
) -> list[dict[str, Any]]:
    """Build a list of audio-truth-shaped matched notes.

    Uses the *new* schema: ``score_time_sec``, ``score_midi_pitch``,
    ``matched``. Deliberately omits HMM-era aliases like
    ``score_time_in_movement`` so consumers must handle the new field
    name (this is exactly the gap that produced Bugs #1 and #2).
    """
    matched = matched_pitches_by_measure or {1: [60, 62, 64, 65]}
    rows: list[dict[str, Any]] = []
    state = 0
    t = 0.5
    score_t = 0.0
    for measure, pitches in sorted(matched.items()):
        for midi in pitches:
            rows.append({
                "state_idx": state,
                "pitches_midi": [int(midi)],
                "names": [_midi_name(int(midi))],
                "performed_time_sec": round(t, 3),
                "dwell_sec": 0.4,
                "confidence": "high",
                "timestamp_source": "basic_pitch",
                "matched": True,
                "measure": int(measure),
                "staff_index": 0,
                "track_name": "part0_voice1_staff0",
                "score_time_sec": round(score_t, 3),
                "score_midi_pitch": int(midi),
                "timing_offset_ms": 0.0,
                "amplitude": 0.7,
            })
            state += 1
            t += perf_step_sec
            score_t += perf_step_sec
    if include_unmatched:
        rows.append({
            "state_idx": state,
            "pitches_midi": [80],
            "names": [_midi_name(80)],
            "performed_time_sec": round(t, 3),
            "dwell_sec": 0.3,
            "confidence": "low",
            "timestamp_source": "basic_pitch",
            "matched": False,
            "staff_index": None,
        })
    return rows


def _midi_name(midi: int) -> str:
    names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    return f"{names[midi % 12]}{midi // 12 - 1}"
