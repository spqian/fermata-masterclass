"""Regression test for Bug #1: NameError ``notes_key`` in intonation.py.

The audio-truth schema renamed fields, breaking ``analyze_intonation`` when
it tried to read score timestamps and assemble a notes_key reference. This
test runs the real ``analyze_intonation`` against a tiny WAV + 4-note
matched-notes fixture (new schema: ``score_time_sec``) and asserts that
events are produced for matched notes only.
"""
from __future__ import annotations

from tests.conftest import (
    audio_truth_matched_notes,
    make_session_manifest,
    tiny_audio_wav,
)


def test_analyze_intonation_runs_on_audio_truth_matched_notes(
    local_storage, session_store, tenant_ctx
):
    from masterclass.engine.intonation import analyze_intonation, IntonationConfig

    matched = audio_truth_matched_notes(
        matched_pitches_by_measure={1: [69, 69, 69, 69]},  # A4 (440 Hz)
        include_unmatched=True,
    )
    matched_count = sum(1 for n in matched if n.get("matched"))
    audio_bytes = tiny_audio_wav(freq_hz=440.0, duration_sec=3.0)
    manifest = make_session_manifest(
        local_storage, session_store, tenant_ctx,
        artifacts={
            "artifacts/audio.wav": audio_bytes,
            "analysis/audio_truth_matched_notes.json": {"notes": matched},
        },
    )

    # The real bug was a NameError on ``notes_key`` -- analyze_intonation
    # would crash before producing any events. Asserting len(events) > 0
    # would have caught that immediately.
    result = analyze_intonation(
        storage=local_storage, store=session_store, manifest=manifest,
        config=IntonationConfig(max_events=50),
    )

    assert isinstance(result.events, list)
    # Bug #1 was a NameError before any events were appended -- so the
    # primary signal is that we get at least one event per matched note.
    assert len(result.events) >= matched_count, (
        f"expected at least {matched_count} events (one per matched note); "
        f"got {len(result.events)}"
    )
    # Filter to the matched-note events: those are the rows the teacher
    # consumes ("the A4 in m.1 was X cents flat"). They must carry the
    # expected fields populated by intonation_from_wav_file.
    matched_events = [r for r in result.events if r.get("score_time_sec") is not None]
    assert len(matched_events) == matched_count, (
        f"expected {matched_count} matched-note events with score_time_sec; "
        f"got {len(matched_events)}"
    )
    for row in matched_events:
        assert row["expected_pitch_midi"] == 69
        assert row["expected_note"] == "A4"
        assert row["performed_time_sec"] is not None
        # 440 Hz sine on a 440 Hz expected pitch -> roughly in tune.
        if row.get("present"):
            assert abs(float(row["cents_offset"])) < 10.0

    # Summary must record the notes-key (Bug #1 left ``notes_key`` undefined).
    assert result.summary.get("hmm_aligned_notes"), (
        "summary.hmm_aligned_notes missing; the notes_key wiring is broken"
    )
