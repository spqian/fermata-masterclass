"""Regression test for Bug #4: evidence packet missing score inventory.

Without an authoritative "this is what's actually in the score" listing,
the teacher LLM hallucinated pitches (e.g. a G# in m.7 when m.7 only
contained G-natural). The fix added a "## Score pitches per measure"
section to the evidence packet, sourced directly from the MusicXML so
that the LLM can cross-check claims.
"""
from __future__ import annotations

from tests.conftest import make_session_manifest, tiny_musicxml


def test_evidence_packet_includes_score_inventory_from_musicxml(
    local_storage, session_store, tenant_ctx
):
    from masterclass.engine.analysis import build_evidence_packet

    xml = tiny_musicxml(
        pitches=[
            ("C", 0, 4), ("D", 0, 4), ("E", 0, 4), ("F", 0, 4),
            ("G", 0, 4), ("A", 0, 4), ("B", 0, 4), ("C", 0, 5),
        ],
        measures=2,
    )
    # build_evidence_packet expects analysis.json + analysis.md already
    # produced by analyze_session. We stub minimal versions.
    analysis_json = {
        "session_id": "test",
        "repertoire": "Test Piece",
        "movement": "I",
        "instrument": "violin",
        "duration_sec": 10.0,
        "global": {"rms_mean_db": -20.0},
        "ranked_regions": {},
    }
    analysis_md = "# Analysis - Test\n\n- placeholder\n"

    manifest = make_session_manifest(
        local_storage, session_store, tenant_ctx,
        artifacts={
            "analysis/analysis.json": analysis_json,
            "analysis/analysis.md": analysis_md.encode("utf-8"),
            "masterclass/reference/musicxml.musicxml": xml,
        },
        metadata={"first_measure": 1, "last_measure": 2},
    )

    updated = build_evidence_packet(
        store=session_store, storage=local_storage, manifest=manifest,
    )

    packet_key = updated.artifacts["analysis/evidence_packet.md"]
    text = local_storage.read_bytes(packet_key).decode("utf-8")

    # Bug #4: the score-inventory section did not exist; LLM had no
    # ground truth and hallucinated accidentals. The rebuilt packet
    # calls this section 'Score pitches per played measure' because it
    # is now scoped to the played-measure sandbox by design.
    assert "## Score pitches per played measure" in text, (
        "evidence packet is missing the 'Score pitches per played measure' section"
    )
    # The section must contain the *actual* pitches from the MusicXML.
    # Measure 1 contains C4..F4; measure 2 contains G4..C5.
    assert "m.1:" in text
    assert "m.2:" in text
    # And it must NOT contain accidentals we never put in the score.
    inventory_start = text.index("## Score pitches per played measure")
    inventory = text[inventory_start:]
    # Only check the per-measure bullet rows (they start with "- m.").
    bullet_lines = "\n".join(line for line in inventory.splitlines() if line.startswith("- m."))
    for fake in ("G#", "C#", "F#"):
        assert fake not in bullet_lines, (
            f"score inventory bullets contain hallucinated accidental {fake!r}"
        )
    # Sanity: real pitches show up.
    for real in ("C4", "E4", "G4", "C5"):
        assert real in bullet_lines, f"score inventory missing real pitch {real!r}"
