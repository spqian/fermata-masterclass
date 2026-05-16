"""Regression test for Bug #6: measure_tempo bare error.

Pre-fix, ``measure_tempo`` returned ``{"error": "no bar X duration found"}``
which gave the LLM nothing actionable. The fix returns
``available_measures`` and an interpretation hint. This test ensures
that contract holds.
"""
from __future__ import annotations


def test_measure_tempo_invalid_bar_returns_available_measures(
    local_storage, session_store, tenant_ctx
):
    from masterclass.agent_tools.measure_tempo import measure_tempo
    from masterclass.core.models import SessionRef

    # Write a polyphonic_rhythm.json artifact with per_bar entries for
    # measures 3, 4, 5 only.
    ref = SessionRef(tenant_ctx.tenant_id, tenant_ctx.user_id, "session1")
    rhy = {
        "summary": {
            "bar_duration_median_sec": 2.0,
            "overall_player_quarter_bpm_median": 120.0,
            "off_pulse_outliers": [],
        },
        "per_bar": [
            {"measure": 3, "duration_sec": 2.0, "median_quarter_bpm": 120.0},
            {"measure": 4, "duration_sec": 2.1, "median_quarter_bpm": 115.0},
            {"measure": 5, "duration_sec": 1.9, "median_quarter_bpm": 125.0},
        ],
    }
    from masterclass.core.models import session_prefix
    key = f"{session_prefix(ref)}/analysis/polyphonic_rhythm.json"
    local_storage.write_json(key, rhy)

    result = measure_tempo(local_storage, ref, {"midi_measure": 99})

    # Bug #6: the bare error message had no fallback information.
    assert "error" in result
    assert "available_measures" in result, (
        "measure_tempo did not return available_measures for invalid bar (Bug #6)"
    )
    assert result["available_measures"] == [3, 4, 5]
    # And the structured response should include the overall medians so
    # the LLM can still ground a tempo claim.
    assert result.get("median_bar_duration_sec") == 2.0
    assert result.get("overall_player_quarter_bpm_median") == 120.0
    # Suggests a remedy path.
    assert "interpretation" in result


def test_measure_tempo_valid_bar_still_works(
    local_storage, session_store, tenant_ctx
):
    """Sanity: the happy path is unchanged by the helpful-error refactor."""
    from masterclass.agent_tools.measure_tempo import measure_tempo
    from masterclass.core.models import SessionRef, session_prefix

    ref = SessionRef(tenant_ctx.tenant_id, tenant_ctx.user_id, "session2")
    rhy = {
        "summary": {
            "bar_duration_median_sec": 2.0,
            "overall_player_quarter_bpm_median": 120.0,
            "off_pulse_outliers": [],
        },
        "per_bar": [
            {"measure": 4, "duration_sec": 2.1, "median_quarter_bpm": 115.0},
        ],
    }
    key = f"{session_prefix(ref)}/analysis/polyphonic_rhythm.json"
    local_storage.write_json(key, rhy)

    result = measure_tempo(local_storage, ref, {"midi_measure": 4})
    assert result.get("bar") == 4
    assert result.get("duration_sec") == 2.1
    assert "error" not in result
