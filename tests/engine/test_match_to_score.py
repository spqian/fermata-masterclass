from __future__ import annotations

import json
from pathlib import Path

import pytest

from masterclass.engine.audio_truth import match_to_score


def _score(pitches: list[int], *, step: float = 1.0, measures: list[int] | None = None) -> list[dict]:
    return [
        {
            "score_time_sec": round(i * step, 3),
            "midi_pitch": pitch,
            "staff_index": 0,
            "track_name": "part0_voice1_staff0",
            "duration_sec": step,
            "measure": measures[i] if measures else i + 1,
        }
        for i, pitch in enumerate(pitches)
    ]


def _perf(pitches: list[int], *, ratio: float = 1.0, offset: float = 0.0, step: float = 1.0) -> list[dict]:
    return [
        {
            "state_idx": i,
            "pitches_midi": [pitch],
            "names": [],
            "performed_time_sec": round(offset + ratio * i * step, 3),
            "dwell_sec": 0.4,
            "amplitude": 0.8,
            "confidence": "high",
        }
        for i, pitch in enumerate(pitches)
    ]


def _matched(rows: list[dict]) -> list[dict]:
    return [r for r in rows if r.get("matched")]


def _rank(values: list[float]) -> list[float]:
    indexed = sorted(enumerate(values), key=lambda item: item[1])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(indexed):
        j = i + 1
        while j < len(indexed) and indexed[j][1] == indexed[i][1]:
            j += 1
        avg = (i + j - 1) / 2.0 + 1.0
        for k in range(i, j):
            ranks[indexed[k][0]] = avg
        i = j
    return ranks


def _spearman(xs: list[float], ys: list[float]) -> float:
    rx = _rank(xs)
    ry = _rank(ys)
    mx = sum(rx) / len(rx)
    my = sum(ry) / len(ry)
    num = sum((x - mx) * (y - my) for x, y in zip(rx, ry))
    den_x = sum((x - mx) ** 2 for x in rx)
    den_y = sum((y - my) ** 2 for y in ry)
    return num / ((den_x * den_y) ** 0.5)


def test_identical_sequences_perfect_match():
    score = _score([60, 62, 64, 65])
    matched = match_to_score(_perf([60, 62, 64, 65]), score)
    assert [m["score_midi_pitch"] for m in matched] == [60, 62, 64, 65]
    assert all(m["matched"] for m in matched)
    assert all(abs(m["timing_offset_ms"]) < 1e-6 for m in matched)


def test_repeated_motif_uses_sequence_position_not_later_alias():
    motif = [60, 62, 64]
    score = _score(motif + motif, measures=[1, 1, 1, 2, 2, 2])
    perf = _perf(motif + motif, ratio=2.0, step=1.0)
    matched = match_to_score(perf, score)
    assert [m["measure"] for m in matched] == [1, 1, 1, 2, 2, 2]
    assert [m["score_time_sec"] for m in matched] == pytest.approx([0, 1, 2, 3, 4, 5])


def test_extra_perf_notes_become_perf_gaps():
    score = _score([60, 62, 64])
    perf = _perf([60, 72, 62, 64])
    perf[1]["performed_time_sec"] = 0.5
    matched = match_to_score(perf, score)
    assert [m.get("matched") for m in matched] == [True, False, True, True]
    assert matched[1]["staff_index"] is None


def test_missing_score_notes_become_score_gaps():
    score = _score([60, 62, 64, 65])
    perf = _perf([60, 64, 65])
    perf[1]["performed_time_sec"] = 2.0
    perf[2]["performed_time_sec"] = 3.0
    matched = match_to_score(perf, score)
    assert [m["score_midi_pitch"] for m in _matched(matched)] == [60, 64, 65]


def test_off_by_one_semitone_still_matches():
    score = _score([60, 62, 64])
    perf = _perf([61, 62, 63])
    matched = match_to_score(perf, score)
    assert all(m["matched"] for m in matched)
    assert [m["score_midi_pitch"] for m in matched] == [60, 62, 64]


def test_early_stop_does_not_penalize_trailing_score_gaps():
    score = _score([60, 62, 64, 65, 67, 69])
    perf = _perf([60, 62, 64])
    matched = match_to_score(perf, score)
    assert all(m["matched"] for m in matched)
    assert [m["score_midi_pitch"] for m in matched] == [60, 62, 64]


def test_empty_inputs():
    assert match_to_score([], _score([60])) == []
    perf = _perf([60])
    matched = match_to_score(perf, [])
    assert len(matched) == 1
    assert matched[0]["matched"] is False
    assert matched[0]["staff_index"] is None


def test_default_tempo_same_tempo_perf_vs_score():
    score = _score([60, 62, 64], step=0.5)
    perf = _perf([60, 62, 64], step=0.5)
    matched = match_to_score(perf, score)
    assert [m["score_time_sec"] for m in matched] == pytest.approx([0.0, 0.5, 1.0])
    assert all(abs(m["timing_offset_ms"]) < 1e-6 for m in matched)


def test_no_pitch_perf_note_preserves_output_contract():
    score = _score([60])
    perf = [{"state_idx": 1, "pitches_midi": [], "performed_time_sec": 0.0, "amplitude": 0.1}]
    matched = match_to_score(perf, score)
    assert len(matched) == 1
    assert matched[0]["matched"] is False
    assert matched[0]["state_idx"] == 1


def test_bach_fixture_measure_assignments_are_temporally_coherent():
    fixture = Path(__file__).with_name("fixture_bach_matcher.json")
    data = json.loads(fixture.read_text())
    matched = match_to_score(data["perf_notes"], data["score_notes"])
    matched_rows = _matched(matched)

    assert len(matched) == len(data["perf_notes"])
    assert len(matched_rows) / len(matched) >= 0.60

    measure_1_times = [
        float(row["performed_time_sec"])
        for row in matched_rows
        if int(row.get("measure") or 0) == 1
    ]
    assert measure_1_times
    assert max(measure_1_times) <= 25.0

    timed_measures = [
        (float(row["performed_time_sec"]), int(row["measure"]))
        for row in matched_rows
        if row.get("measure") is not None
    ]
    for (prev_t, prev_m), (curr_t, curr_m) in zip(timed_measures, timed_measures[1:]):
        if curr_m < prev_m - 1:
            assert curr_t - prev_t <= 3.0, (
                f"large backwards measure jump from m.{prev_m} at {prev_t:.2f}s "
                f"to m.{curr_m} at {curr_t:.2f}s"
            )

    rho = _spearman(
        [float(row["performed_time_sec"]) for row in matched_rows],
        [float(row["score_time_sec"]) for row in matched_rows],
    )
    assert rho > 0.95
