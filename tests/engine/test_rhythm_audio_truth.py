"""Regression test for Bug #2: rhythm.py silently dropped all notes.

``_normalized_notes`` only looked at the HMM-era field names
(``score_time_in_movement`` / ``score_time_local``) so the audio-truth
output (which uses ``score_time_sec``) yielded an empty list and the
``per_bar`` summary collapsed. This test feeds the new schema directly
and asserts that ``per_bar`` contains rows with non-zero duration_sec.
"""
from __future__ import annotations

from tests.conftest import (
    audio_truth_matched_notes,
    make_session_manifest,
)


def test_analyze_rhythm_handles_audio_truth_schema(
    local_storage, session_store, tenant_ctx
):
    from masterclass.engine.rhythm import analyze_rhythm, RhythmConfig

    # Two measures, 4 notes each. perf_step_sec controls bar duration.
    matched = audio_truth_matched_notes(
        matched_pitches_by_measure={1: [60, 62, 64, 65], 2: [67, 69, 71, 72]},
        include_unmatched=False,
        perf_step_sec=0.5,
    )
    manifest = make_session_manifest(
        local_storage, session_store, tenant_ctx,
        artifacts={
            "analysis/audio_truth_matched_notes.json": {"notes": matched},
        },
    )

    result = analyze_rhythm(
        storage=local_storage, store=session_store, manifest=manifest,
        config=RhythmConfig(),
    )

    # Bug #2: pre-fix, per_bar would be empty because every row was
    # silently dropped by _normalized_notes. Post-fix, we expect at
    # least the played measures to appear.
    assert isinstance(result.per_bar, list)
    assert len(result.per_bar) >= 1, (
        "per_bar is empty -- normalised notes dropped silently (Bug #2)"
    )
    nonzero = [b for b in result.per_bar if b.get("duration_sec") and float(b["duration_sec"]) > 0]
    assert nonzero, (
        f"no per_bar entry has duration_sec > 0; per_bar={result.per_bar}"
    )
    # And summary should reflect that we actually counted bars.
    assert result.summary.get("bar_count") == len(result.per_bar)
    assert result.summary.get("bar_duration_median_sec"), (
        "bar_duration_median_sec is None -- rhythm could not derive any "
        "tempo, indicating notes were dropped"
    )
