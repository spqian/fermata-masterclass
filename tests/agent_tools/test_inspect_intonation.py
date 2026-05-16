"""Regression test for inspect_intonation tool contract.

A 440 Hz sine should be reported as in-tune (~0 cents) against an
A4 (MIDI 69) expected pitch. The tool must accept both note-name
strings and MIDI ints (as strings) for ``expected_pitch``.
"""
from __future__ import annotations

from tests.conftest import tiny_audio_wav


def _write_audio(local_storage, session_store, tenant_ctx, freq_hz: float = 440.0):
    from masterclass.core.models import SessionRef, session_prefix
    ref = SessionRef(tenant_ctx.tenant_id, tenant_ctx.user_id, "intonation_session")
    audio_key = f"{session_prefix(ref)}/artifacts/audio.wav"
    local_storage.write_bytes(audio_key, tiny_audio_wav(freq_hz=freq_hz, duration_sec=2.0))
    return ref


def test_inspect_intonation_in_tune_name_form(local_storage, session_store, tenant_ctx):
    from masterclass.agent_tools.inspect_intonation import inspect_intonation

    ref = _write_audio(local_storage, session_store, tenant_ctx, freq_hz=440.0)
    result = inspect_intonation(local_storage, ref, {"time_sec": 0.5, "expected_pitch": "A4"})

    assert "error" not in result, result
    assert result["expected_pitch_midi"] == 69
    assert result["expected_pitch_name"].startswith("A4"), result["expected_pitch_name"]
    assert abs(float(result["cents_off_score"])) < 5.0, (
        f"440 Hz sine should be ~0c off A4, got {result['cents_off_score']}c"
    )


def test_inspect_intonation_in_tune_midi_int_form(local_storage, session_store, tenant_ctx):
    from masterclass.agent_tools.inspect_intonation import inspect_intonation

    ref = _write_audio(local_storage, session_store, tenant_ctx, freq_hz=440.0)
    # MIDI int form must work just as well as the note-name form.
    result = inspect_intonation(local_storage, ref, {"time_sec": 0.5, "expected_pitch": "69"})

    assert "error" not in result, result
    assert result["expected_pitch_midi"] == 69
    assert abs(float(result["cents_off_score"])) < 5.0, (
        f"440 Hz sine should be ~0c off MIDI 69, got {result['cents_off_score']}c"
    )
