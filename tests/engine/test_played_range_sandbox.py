"""Sandbox invariant: evidence packet must NOT leak data from outside the played range.

This is the structural guarantee the user requested after the m.1-199
score-pitch dump appeared in the packet for a lesson where only m.1-8
were played. Any consumer that uses ``PlayedRange.filter_by_measure``
should automatically satisfy this; the test exists to catch regressions
in future code paths that bypass the helper.
"""
from __future__ import annotations

from tests.conftest import make_session_manifest, tiny_musicxml


def test_packet_does_not_mention_measures_outside_played_range(
    local_storage, session_store, tenant_ctx
):
    from masterclass.engine.analysis import build_evidence_packet

    # Score has 8 measures of C-major scale; student only "played" m.2-4.
    xml = tiny_musicxml(
        pitches=[("C", 0, 4), ("D", 0, 4), ("E", 0, 4), ("F", 0, 4)],
        measures=8,
    )
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
        metadata={"first_measure": 2, "last_measure": 4},
    )

    updated = build_evidence_packet(
        store=session_store, storage=local_storage, manifest=manifest,
    )
    text = local_storage.read_bytes(updated.artifacts["analysis/evidence_packet.md"]).decode("utf-8")

    # In-range measures must appear...
    for in_range in ("m.2", "m.3", "m.4"):
        assert in_range in text, f"in-range measure {in_range} missing from packet"

    # ...out-of-range measures must NOT appear as standalone bullet rows in any
    # measure-data section. We allow them to occur inside body prose / table
    # cells only when there's no bullet "- m.X:" form, which is how the packet
    # surfaces each measure's data.
    for out_of_range in ("m.1:", "m.5:", "m.6:", "m.7:", "m.8:"):
        assert out_of_range not in text, (
            f"sandbox violation: bullet for {out_of_range} appeared in packet "
            "for lesson scoped to m.2-4"
        )
    # The performance-timeline table row for out-of-range measures must
    # also be absent.
    for out_of_range_row in ("| m.1 |", "| m.5 |", "| m.6 |", "| m.7 |", "| m.8 |"):
        assert out_of_range_row not in text, (
            f"sandbox violation: timeline table contains {out_of_range_row} "
            "for lesson scoped to m.2-4"
        )


def test_played_range_filter_helpers():
    """PlayedRange.filter_by_measure works for both dict and object items."""
    from masterclass.core.played_range import PlayedRange

    pr = PlayedRange(first_measure=2, last_measure=4, source="user_specified")
    items = [
        {"measure": 1, "name": "A"},
        {"measure": 2, "name": "B"},
        {"measure": 3, "name": "C"},
        {"measure": 5, "name": "D"},
        {"measure": None, "name": "E"},
        {"measure": "not-a-number", "name": "F"},
    ]
    kept = pr.filter_by_measure(items)
    assert [k["name"] for k in kept] == ["B", "C"]

    # Also works for object attributes.
    class N:
        def __init__(self, m):
            self.measure = m
    obj_items = [N(1), N(2), N(3), N(4), N(5)]
    kept_obj = pr.filter_by_measure(obj_items)
    assert [n.measure for n in kept_obj] == [2, 3, 4]


def test_played_range_filter_time_window():
    from masterclass.core.played_range import PlayedRange

    pr = PlayedRange(first_measure=1, last_measure=8, source="user_specified")
    items = [
        {"start": 0.0, "end": 5.0},
        {"start": 10.0, "end": 12.0},
        {"start": 50.0, "end": 55.0},
        {"start": 80.0, "end": 85.0},  # outside envelope
    ]
    kept = pr.filter_time_window(items, perf_start_sec=2.0, perf_end_sec=60.0)
    # The first three overlap [2, 60]; the last does not.
    assert len(kept) == 3
