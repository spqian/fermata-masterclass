"""Regression test for Bug #3: score_map needed MIDI artifact.

Pre-fix, ``build_score_map`` insisted on a ``reference/midi`` artifact
even though masterclasses created from PDF+OMR never produce one. The
fix is to parse MusicXML directly via
``audio_truth._load_score_notes_from_musicxml``. This test feeds a tiny
MusicXML through the real ``build_score_map`` -- no MIDI artifact in
the manifest -- and asserts the result has ``notes`` and ``bars``.
"""
from __future__ import annotations

from tests.conftest import make_session_manifest, make_masterclass_manifest, tiny_musicxml


def test_build_score_map_works_without_midi(
    local_storage, session_store, masterclass_store, tenant_ctx
):
    from masterclass.engine.score_map import build_score_map

    xml_bytes = tiny_musicxml(
        pitches=[
            ("C", 0, 4), ("D", 0, 4), ("E", 0, 4), ("F", 0, 4),
            ("G", 0, 4), ("A", 0, 4), ("B", 0, 4), ("C", 0, 5),
        ],
        measures=2,
    )
    # score_prep: a minimal layout with 1 page, 1 system covering both
    # measures. Movement spans m.1..m.2.
    score_prep = {
        "total_measures": 2,
        "key": "c_major",
        "instrument": "violin",
        "movements": [
            {"id": 1, "title": "I", "first_measure": 1, "last_measure": 2,
             "key_signature": "c_major", "measure_count": 2},
        ],
        "pages": [
            {
                "page": 1,
                "kind": "music",
                "first_measure": 1,
                "last_measure": 2,
                "systems": [
                    {
                        "system_index": 1,
                        "movement_id": 1,
                        "first_measure": 1,
                        "last_measure": 2,
                        "bbox": [0.0, 0.0, 1.0, 0.2],
                    },
                ],
            },
        ],
    }

    mc = make_masterclass_manifest(
        local_storage, masterclass_store, tenant_ctx,
        artifacts={
            "reference/musicxml.musicxml": xml_bytes,
            "reference/score_prep.json": score_prep,
        },
        piece_name="Tiny",
        movement="I",
    )

    manifest = make_session_manifest(
        local_storage, session_store, tenant_ctx,
        artifacts={},
        metadata={
            "masterclass_id": mc.masterclass.masterclass_id,
            "first_measure": 1,
            "last_measure": 2,
            "played_movement_id": 1,
        },
        movement="I",
    )

    result = build_score_map(
        storage=local_storage,
        masterclass_store=masterclass_store,
        store=session_store,
        manifest=manifest,
    )

    # Bug #3: pre-fix this raised because no MIDI artifact existed.
    assert isinstance(result.score_map, dict)
    assert "notes" in result.score_map and result.score_map["notes"], (
        "score_map.notes empty -- MusicXML fallback didn't run (Bug #3)"
    )
    assert "bars" in result.score_map and result.score_map["bars"], (
        "score_map.bars empty"
    )
    # Confirm the notes carry pitches from the MusicXML.
    pitches = {int(n["pitch_midi"] if isinstance(n.get("pitch_midi"), int) else n["midi_pitches"][0])
               for n in result.score_map["notes"]}
    assert 60 in pitches  # C4
    assert 72 in pitches  # C5
    # And no reference/midi artifact was required: the source key in
    # the result metadata must point at the MusicXML, not a MIDI file.
    src_key = result.score_map.get("_meta", {}).get("score_source_key", "")
    assert "musicxml" in src_key.lower(), (
        f"score source should be MusicXML, got {src_key!r}"
    )
