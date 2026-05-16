"""Unit tests for masterclass.core.artifact_catalog."""

from __future__ import annotations

import pytest

from masterclass.core.artifact_catalog import ArtifactCatalog, ArtifactMissingError
from masterclass.core.models import (
    MasterclassManifest,
    MasterclassRef,
    SessionManifest,
    SessionRef,
)


def _session(artifacts: dict[str, str] | None = None) -> SessionManifest:
    return SessionManifest(
        schema_version=1,
        session=SessionRef(tenant_id="t1", user_id="u1", session_id="s1"),
        artifacts=dict(artifacts or {}),
    )


def _masterclass(artifacts: dict[str, str] | None = None) -> MasterclassManifest:
    return MasterclassManifest(
        schema_version=1,
        masterclass=MasterclassRef(tenant_id="t1", user_id="u1", masterclass_id="m1"),
        piece_name="Test Piece",
        artifacts=dict(artifacts or {}),
    )


# Priority ordering ----------------------------------------------------------


def test_musicxml_prefers_musicxml_extension_over_mxl_and_bare():
    session = _session({
        "masterclass/reference/musicxml.musicxml": "K_xml",
        "masterclass/reference/musicxml.mxl": "K_mxl",
        "masterclass/reference/musicxml": "K_bare",
    })
    assert ArtifactCatalog(session).musicxml() == "K_xml"


def test_musicxml_falls_back_to_mxl_when_no_musicxml():
    session = _session({
        "masterclass/reference/musicxml.mxl": "K_mxl",
        "masterclass/reference/musicxml": "K_bare",
    })
    assert ArtifactCatalog(session).musicxml() == "K_mxl"


def test_musicxml_falls_back_to_bare_when_only_bare_present():
    session = _session({"masterclass/reference/musicxml": "K_bare"})
    assert ArtifactCatalog(session).musicxml() == "K_bare"


def test_musicxml_dual_lookup_consults_masterclass_manifest():
    session = _session({})
    masterclass = _masterclass({"reference/musicxml.mxl": "MC_mxl"})
    assert ArtifactCatalog(session, masterclass).musicxml() == "MC_mxl"


def test_musicxml_session_wins_over_masterclass_when_both_present():
    session = _session({"masterclass/reference/musicxml.musicxml": "S_xml"})
    masterclass = _masterclass({"reference/musicxml.musicxml": "MC_xml"})
    assert ArtifactCatalog(session, masterclass).musicxml() == "S_xml"


def test_aligned_notes_priority_matched_then_raw_then_hmm():
    s = _session({
        "analysis/audio_truth_matched_notes.json": "K_matched",
        "analysis/audio_truth_notes.json": "K_raw",
        "analysis/hmm_aligned_notes.json": "K_hmm",
    })
    assert ArtifactCatalog(s).aligned_notes() == "K_matched"

    s = _session({
        "analysis/audio_truth_notes.json": "K_raw",
        "analysis/hmm_aligned_notes.json": "K_hmm",
    })
    assert ArtifactCatalog(s).aligned_notes() == "K_raw"

    s = _session({"analysis/hmm_aligned_notes.json": "K_hmm"})
    assert ArtifactCatalog(s).aligned_notes() == "K_hmm"


# Missing artifacts ----------------------------------------------------------


def test_lookup_returns_none_when_no_candidate_present():
    catalog = ArtifactCatalog(_session())
    assert catalog.musicxml() is None
    assert catalog.audio_wav() is None
    assert catalog.aligned_notes() is None


def test_required_raises_artifact_missing_with_helpful_message():
    catalog = ArtifactCatalog(_session())
    with pytest.raises(ArtifactMissingError) as excinfo:
        catalog.musicxml_required()
    msg = str(excinfo.value)
    assert "musicxml" in msg
    assert "s1" in msg
    assert "masterclass/reference/musicxml.musicxml" in msg


def test_required_returns_value_when_present():
    s = _session({"artifacts/audio.wav": "K_audio"})
    assert ArtifactCatalog(s).audio_wav_required() == "K_audio"


# Stable path constants (snapshot) ------------------------------------------


def test_paths_constant_snapshot():
    """If you change this snapshot, also update every consumer that reads
    the listed artifacts (and bump pipeline migrations as appropriate)."""
    assert ArtifactCatalog._PATHS == {
        "audio_wav": ("artifacts/audio.wav",),
        "aligned_notes": (
            "analysis/audio_truth_matched_notes.json",
            "analysis/audio_truth_notes.json",
            "analysis/hmm_aligned_notes.json",
        ),
        "musicxml": (
            "masterclass/reference/musicxml.musicxml",
            "masterclass/reference/musicxml.mxl",
            "masterclass/reference/musicxml",
        ),
        "score_prep": ("reference/score_prep.json",),
        "score_map": ("score/score_map.json",),
        "evidence_packet": (
            "analysis/evidence_packet.md",
            "evidence_packet.md",
        ),
        "analysis_json": ("analysis/analysis.json",),
        "analysis_md": ("analysis/analysis.md",),
        "audio_truth_matched": ("analysis/audio_truth_matched_notes.json",),
        "audio_truth_raw": ("analysis/audio_truth_notes.json",),
        "polyphonic_intonation": ("analysis/polyphonic_intonation.json",),
        "polyphonic_rhythm": ("analysis/polyphonic_rhythm.json",),
        "piano_voicing": ("analysis/piano_voicing.json",),
        "rich_onsets": ("analysis/rich_onsets.json",),
    }
