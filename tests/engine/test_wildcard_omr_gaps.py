"""Wildcard ghost-note tests for OMR-empty-measure handling."""
from __future__ import annotations

from masterclass.engine.audio_truth import (
    _WILDCARD_PITCH,
    _load_score_notes_from_musicxml,
    match_to_score,
)


def _mxl_with_empty_m4() -> bytes:
    """Two-measure-then-empty-then-two-measure score in raw MusicXML."""
    return ("""<?xml version="1.0"?>
<score-partwise version="3.1"><part-list><score-part id="P1"/></part-list>
<part id="P1">
  <measure number="1"><attributes><divisions>4</divisions><time><beats>4</beats><beat-type>4</beat-type></time></attributes>
    <note><pitch><step>C</step><octave>4</octave></pitch><duration>16</duration><voice>1</voice></note>
  </measure>
  <measure number="2">
    <note><pitch><step>D</step><octave>4</octave></pitch><duration>16</duration><voice>1</voice></note>
  </measure>
  <measure number="3"></measure>
  <measure number="4">
    <note><pitch><step>E</step><octave>4</octave></pitch><duration>16</duration><voice>1</voice></note>
  </measure>
  <measure number="5">
    <note><pitch><step>F</step><octave>4</octave></pitch><duration>16</duration><voice>1</voice></note>
  </measure>
</part></score-partwise>""").encode("utf-8")


def test_wildcard_notes_injected_for_empty_measures():
    notes = _load_score_notes_from_musicxml(_mxl_with_empty_m4())
    by_measure: dict[int, list[dict]] = {}
    for n in notes:
        by_measure.setdefault(n["measure"], []).append(n)
    # m.1, m.2, m.4, m.5 have one real note each
    assert len(by_measure[1]) == 1 and by_measure[1][0]["midi_pitch"] == 60
    assert len(by_measure[2]) == 1
    assert len(by_measure[4]) == 1
    assert len(by_measure[5]) == 1
    # m.3 should have wildcards (default 4 per measure)
    assert 3 in by_measure
    wildcards = by_measure[3]
    assert len(wildcards) == 4
    for w in wildcards:
        assert w["midi_pitch"] == _WILDCARD_PITCH
        assert w.get("is_wildcard") is True
    # Wildcard times should span the m.3 region (score_time 8s -> 12s at 120bpm, 4/4)
    times = sorted(w["score_time_sec"] for w in wildcards)
    assert 3.9 <= times[0] <= 4.1  # measure 3 starts at qtr=8 -> 4s @ 120bpm
    assert 5.4 <= times[-1] <= 5.6  # last beat at qtr=11 -> 5.5s


def test_wildcards_not_injected_when_measure_has_notes():
    """Sanity: non-empty measures shouldn't get wildcard padding."""
    notes = _load_score_notes_from_musicxml(_mxl_with_empty_m4())
    assert all(not n.get("is_wildcard") for n in notes if n["measure"] != 3)


def test_matcher_uses_wildcards_to_anchor_empty_measure():
    """A perf sequence with notes in the OMR-gap region should get
    matched_to_wildcard=True on those notes, NOT bleed into m.4 or m.5."""
    score_notes = _load_score_notes_from_musicxml(_mxl_with_empty_m4())
    # Perf plays: C4, D4, then a chromatic run (G4 G4 G4 G4) in the m.3 gap,
    # then E4, F4. The G4s have no pitch counterpart in the score (only wildcards),
    # so the wildcard reward should claim them and tag them as matched_to_wildcard.
    perf = [
        {"state_idx": 0, "pitches_midi": [60], "performed_time_sec": 0.0, "dwell_sec": 0.2, "amplitude": 0.7, "confidence": "high", "names": []},
        {"state_idx": 1, "pitches_midi": [62], "performed_time_sec": 2.0, "dwell_sec": 0.2, "amplitude": 0.7, "confidence": "high", "names": []},
        {"state_idx": 2, "pitches_midi": [67], "performed_time_sec": 4.0, "dwell_sec": 0.2, "amplitude": 0.7, "confidence": "high", "names": []},
        {"state_idx": 3, "pitches_midi": [67], "performed_time_sec": 4.5, "dwell_sec": 0.2, "amplitude": 0.7, "confidence": "high", "names": []},
        {"state_idx": 4, "pitches_midi": [67], "performed_time_sec": 5.0, "dwell_sec": 0.2, "amplitude": 0.7, "confidence": "high", "names": []},
        {"state_idx": 5, "pitches_midi": [67], "performed_time_sec": 5.5, "dwell_sec": 0.2, "amplitude": 0.7, "confidence": "high", "names": []},
        {"state_idx": 6, "pitches_midi": [64], "performed_time_sec": 6.0, "dwell_sec": 0.2, "amplitude": 0.7, "confidence": "high", "names": []},
        {"state_idx": 7, "pitches_midi": [65], "performed_time_sec": 8.0, "dwell_sec": 0.2, "amplitude": 0.7, "confidence": "high", "names": []},
    ]
    matched = match_to_score(perf, score_notes)
    by_measure: dict[int, list[dict]] = {}
    for m in matched:
        if m.get("matched"):
            by_measure.setdefault(m["measure"], []).append(m)

    # All measures present
    assert 1 in by_measure and 2 in by_measure
    assert 4 in by_measure and 5 in by_measure
    # m.3 should have at least one wildcard match for the gap-region G4s
    assert 3 in by_measure, f"m.3 missing from matches: {sorted(by_measure)}"
    wc_matches = [m for m in by_measure[3] if m.get("matched_to_wildcard")]
    assert len(wc_matches) >= 1, f"expected wildcard matches in m.3; got {by_measure[3]}"
    for m in wc_matches:
        assert m["score_midi_pitch"] is None
    # m.4 and m.5 each get their real-pitch matches (not stolen by G4s)
    real_m4 = [m for m in by_measure[4] if not m.get("matched_to_wildcard")]
    real_m5 = [m for m in by_measure[5] if not m.get("matched_to_wildcard")]
    assert real_m4, "m.4 missed its real-pitch match"
    assert real_m5, "m.5 missed its real-pitch match"


def test_wildcard_match_serializes_with_aligned_note_dataclass():
    """AlignedNote.from_dict / to_dict should round-trip the matched_to_wildcard flag."""
    from masterclass.engine.aligned_notes import AlignedNote
    src = {
        "state_idx": 0,
        "pitches_midi": [67],
        "names": ["G4"],
        "performed_time_sec": 5.0,
        "dwell_sec": 0.2,
        "amplitude": 0.7,
        "confidence": "high",
        "matched": True,
        "measure": 4,
        "staff_index": 0,
        "track_name": "part0_wildcard",
        "score_time_sec": 6.0,
        "score_midi_pitch": None,
        "timing_offset_ms": 12.3,
        "matched_to_wildcard": True,
    }
    note = AlignedNote.from_dict(src)
    assert note.matched_to_wildcard is True
    assert note.score_midi_pitch is None
    round_tripped = note.to_dict()
    assert round_tripped["matched_to_wildcard"] is True
