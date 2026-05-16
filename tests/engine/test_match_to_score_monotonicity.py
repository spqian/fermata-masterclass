"""Regression test for Bug #5: match_to_score temporal monotonicity.

In the polyphonic_rhythm output for one production lesson, bar 6 had
a perf_start_sec *earlier* than bar 5 -- impossible. The cause was the
score matcher aliasing performed notes to repeated-pitch passages
later in the score because no soft monotonicity penalty was in place.
``audio_truth.match_to_score`` now adds ``backward_penalty_sec`` and a
``last_matched_score_time`` ratchet. This test is a property check on
that behaviour using synthetic perf notes that arrive in score order.
"""
from __future__ import annotations


def _score_with_repeats() -> list[dict]:
    """Score with repeated pitches across measures (m.1 C4 D4, m.2 C4 D4, ...)."""
    score: list[dict] = []
    t = 0.0
    state = 0
    for measure in (1, 2, 3, 4, 5, 6):
        for midi in (60, 62):  # C4, D4 repeats every measure
            score.append({
                "score_time_sec": t,
                "midi_pitch": midi,
                "staff_index": 0,
                "track_name": "t0",
                "duration_sec": 0.5,
                "measure": measure,
            })
            t += 0.5
            state += 1
    return score


def test_match_to_score_preserves_score_time_monotonicity():
    from masterclass.engine.audio_truth import match_to_score

    score = _score_with_repeats()
    # Simulate a faithful performance: perf times follow score order but
    # with a constant +0.2s lag.
    perf = []
    for i, s in enumerate(score):
        perf.append({
            "state_idx": i,
            "pitches_midi": [s["midi_pitch"]],
            "performed_time_sec": s["score_time_sec"] + 0.2,
            "dwell_sec": 0.4,
            "amplitude": 0.7,
        })

    matched = match_to_score(perf, score)
    # Every perf note must match to *some* score note.
    matched_rows = [m for m in matched if m.get("matched")]
    assert len(matched_rows) == len(perf), (
        f"expected all {len(perf)} perf notes matched, got {len(matched_rows)}"
    )

    # Property: for matched rows, score_time_sec must be non-decreasing
    # in the order the perf notes arrive. If bar 6 was assigned before
    # bar 5 (Bug #5), the sequence would dip.
    score_times = [float(m["score_time_sec"]) for m in matched_rows]
    for prev, curr in zip(score_times, score_times[1:]):
        # Allow tiny float fuzz, but no real backward jumps.
        assert curr >= prev - 1e-6, (
            f"score_time_sec went backwards: prev={prev}, curr={curr}; "
            f"matcher is mis-aliasing to repeated-pitch later passages"
        )

    # And measure assignments should likewise not skip backwards
    # between consecutive matched perf notes when the perf order is
    # strictly increasing.
    measures = [int(m["measure"]) for m in matched_rows]
    for prev, curr in zip(measures, measures[1:]):
        assert curr >= prev, (
            f"matched measures went backwards: prev m.{prev}, curr m.{curr}"
        )
